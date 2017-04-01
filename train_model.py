import time
import numpy as np
import read_data as rd
import tensorflow as tf
import tensorlayer as tl


NUM_EPOCHS = 20
BATCH_SIZE = 200
LEARNING_RATE = 0.001

CLASS_NUM = 6
PATCH_SIZE = 35
CHANNEL_NUM = 3

LABEL_SET_SHAPE = [BATCH_SIZE, CLASS_NUM]
IMAGE_SET_SHAPE = [BATCH_SIZE, PATCH_SIZE,
                   PATCH_SIZE, CHANNEL_NUM]


def weight(shape):
    sd = 1 / np.sqrt(np.prod(shape[0:3]) * CLASS_NUM)
    return tf.random_normal_initializer(stddev=sd)


def conv2d(net, shape, act=tf.nn.relu, name=None):
    return tl.layers.Conv2dLayer(net,
                                 act=act,
                                 shape=shape,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID',
                                 W_init=weight(shape),
                                 b_init=None,
                                 name=name)


def max_pool(net, name=None):
    return tl.layers.PoolLayer(net,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               pool=tf.nn.max_pool,
                               name=name)


def sub2ind(shape, rows, cols):
    return rows * shape[1] + cols


def reshape_labels(labels):
    lc = np.zeros(LABEL_SET_SHAPE).flatten()
    index = sub2ind(LABEL_SET_SHAPE,
                    np.arange(BATCH_SIZE),
                    np.reshape(labels, [1, BATCH_SIZE]))
    lc[index] = 1

    return np.reshape(lc, LABEL_SET_SHAPE)


def build_network(x):
    net = tl.layers.InputLayer(inputs=x, name='input_layer')
    net = conv2d(net, [6, 6, 3, 30], name='conv1')
    net = max_pool(net, 'maxpool1')
    net = conv2d(net, [6, 6, 30, 50], name='conv2')
    net = max_pool(net, 'maxpool2')
    net = conv2d(net, [4, 4, 50, 500], name='conv3')
    net = conv2d(net, [2, 2, 500, 6], tf.identity, name='conv4')

    return net


def train_model(train_set_path,
                validation_set_path,
                save_model_path):
    x = tf.placeholder(tf.float32, shape=IMAGE_SET_SHAPE)
    y = tf.placeholder(tf.float32, shape=LABEL_SET_SHAPE)

    net = build_network(x)

    y_out = net.outputs
    y_out = tf.reshape(y_out, shape=LABEL_SET_SHAPE)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                logits=y_out))

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    y_arg = tf.reshape(tf.argmax(y_out, 1), shape=[BATCH_SIZE])
    correct_prediction = tf.equal(y_arg, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tri_img, tri_lbl = rd.inputs(path=train_set_path,
                                 batch_size=BATCH_SIZE,
                                 num_epochs=NUM_EPOCHS,
                                 patch_size=PATCH_SIZE,
                                 channel_num=CHANNEL_NUM)

    val_img, val_lbl = rd.inputs(path=validation_set_path,
                                 batch_size=BATCH_SIZE,
                                 num_epochs=NUM_EPOCHS,
                                 patch_size=PATCH_SIZE,
                                 channel_num=CHANNEL_NUM)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    sess = tf.InteractiveSession()
    sess.run(init)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            [tris, tril] = sess.run([tri_img, tri_lbl])
            fd_train = {x: tris, y: reshape_labels(tril)}

            if step % 10 == 0:
                [vals, vall] = sess.run([val_img, val_lbl])
                fd_val = {x: vals, y: reshape_labels(vall)}

                print("----------\nStep {}:\n----------".format(step))

                tri_accuracy = accuracy.eval(feed_dict=fd_train)
                print("Training accuracy {0:.6f}".format(tri_accuracy))
                tri_cost = loss.eval(feed_dict=fd_train)
                print("Training cost is {0:.6f}".format(tri_cost))

                val_accuracy = accuracy.eval(feed_dict=fd_val)
                print("Validation accuracy {0:.6f}".format(val_accuracy))
                val_cost = loss.eval(feed_dict=fd_val)
                print("Validation cost is {0:.6f}".format(val_cost))

            sess.run(train_step, feed_dict=fd_train)
            step += 1
            time.sleep(1)

    except tf.errors.OutOfRangeError:
        print('---------\nTraining has stopped.')
    finally:
        coord.request_stop()

    tl.files.save_npz(net.all_params, save_model_path)
    coord.join(thread)
    sess.close()


if __name__ == '__main__':
    train_set_path = 'TFRecords/train.tfrecords'
    validation_set_path = 'TFRecords/validation.tfrecords'
    save_model_path = 'model.npz'
    train_model(train_set_path,
                validation_set_path,
                save_model_path)

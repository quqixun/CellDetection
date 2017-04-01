import os
import numpy as np
from PIL import Image
import tensorflow as tf


def create_record(path, classes, filename, patch_size):
    writer = tf.python_io.TFRecordWriter(filename)
    for index, name in enumerate(classes):
        class_path = path + str(name) + '/'
        print(class_path, index)
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((patch_size, patch_size))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[index])),
                'image': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def decode_record(filename_queue, patch_size,
                  channel_num=3):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        })

    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [patch_size, patch_size, channel_num])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


def inputs(path, batch_size, num_epochs,
           patch_size, channel_num=3,
           capacity=50000, mad=30000):
    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [path], num_epochs=num_epochs)
        image, label = decode_record(filename_queue,
                                     patch_size,
                                     channel_num)

        images, labels = \
            tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=4,
                capacity=capacity,
                min_after_dequeue=mad)

    return images, labels


if __name__ == '__main__':
    path = os.getcwd() + '/ImageSet/Train/'
    classes = np.arange(1, 6 + 1, 1)
    filename = 'TFRecords/train.tfrecords'
    patch_size = 35
    create_record(path, classes, filename, patch_size)

    channel_num = 3
    images, labels = inputs(path=filename,
                            batch_size=10,
                            num_epochs=2,
                            patch_size=patch_size,
                            channel_num=channel_num,
                            capacity=500,
                            mad=100)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            [val, l] = sess.run([images, labels])
            print(val.shape, l)
    except tf.errors.OutOfRangeError:
        print('Out of range.')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

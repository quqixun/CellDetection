import numpy as np
from PIL import Image
import tensorflow as tf
import tensorlayer as tl
import train_model as tm
import scipy.ndimage as sn
from matplotlib import pyplot as plt


HIGH_PROB = 0.7
PATCH_RADIUS = int((tm.PATCH_SIZE - 1) / 2)


def load_image(img_path):
    img = Image.open(img_path)
    img_raw = np.asarray(img, dtype=np.uint8)
    img_data = img_raw * (1. / 255) - 0.5
    pad_width = ((PATCH_RADIUS, PATCH_RADIUS),
                 (PATCH_RADIUS, PATCH_RADIUS), (0, 0))
    img_pad = np.lib.pad(img_data, pad_width, 'symmetric')

    return img_raw, img_pad


def strict_local_maximum(prob_map):
    prob_gau = np.zeros(prob_map.shape)
    sn.gaussian_filter(prob_map, 2,
                       output=prob_gau,
                       mode='mirror')

    prob_fil = np.zeros(prob_map.shape)
    sn.rank_filter(prob_gau, -2,
                   output=prob_fil,
                   footprint=np.ones([3, 3]))

    temp = np.logical_and(prob_gau > prob_fil,
                          prob_map > HIGH_PROB) * 1.
    idx = np.where(temp > 0)

    return idx


def plot_result(img_raw, idx):
    plt.imshow(img_raw)
    plt.scatter(idx[1], idx[0], c='r', s=10)
    plt.axis('off')
    plt.show()

    return


def test_image(img_path, model_path):
    img_raw, img_pad = load_image(img_path)

    rows = img_raw.shape[0]
    cols = img_raw.shape[1]
    test_set_shape = [cols, tm.PATCH_SIZE,
                      tm.PATCH_SIZE, tm.CHANNEL_NUM]
    print(test_set_shape)

    x = tf.placeholder(tf.float32, test_set_shape)
    net = tm.build_network(x)
    y_out = tf.reshape(net.outputs, shape=[cols, tm.CLASS_NUM])
    y_stm = tf.nn.softmax(y_out)
    print(y_stm.shape)

    sess = tf.InteractiveSession()
    load_params = tl.files.load_npz(path='', name=model_path)
    tl.files.assign_params(sess, load_params, net)

    prob_map = np.zeros([rows, cols])
    for r in range(rows):
        print("Processing NO.{} rows.".format(r + 1))
        x_ = np.zeros(test_set_shape)
        for c in range(cols):
            x_[c] = img_pad[r:r + tm.PATCH_SIZE,
                            c:c + tm.PATCH_SIZE, :]

        prob = y_stm.eval(feed_dict={x: x_})
        temp = np.where(prob[:, 5] > HIGH_PROB)[0]
        prob_map[r, temp] = prob[temp, 5]

    sess.close()

    idx = strict_local_maximum(prob_map)
    plot_result(img_raw, idx)

    return


if __name__ == '__main__':
    test_image('ImageSet/Test/img_41.png', 'model.npz')

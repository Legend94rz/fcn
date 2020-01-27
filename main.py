import tensorflow as tf
import cv2
from dataset.helper import ADE20k, random_crop, fixed_crop
from layers import conv2d, conv2d_transpose
from metrics import categorical_accuracy
from model.utils import get_session, initialize


# 用data API有两个缺点：
# 1. 不方便划分训练/验证
# 2. 模型init函数与dataset耦合
# 也有好处：
# 1. 无需手动写padding、并行数据流。与session.run配合
# 2. 无需手动迭代

class FCN():
    def __init__(self, train_data, val_data, num_class):
        self.num_class = num_class

        re_iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        self.train_init_op = re_iter.make_initializer(train_data)
        self.val_init_op = re_iter.make_initializer(val_data)
        X, Y = re_iter.get_next()
        X_and_offset = tf.map_fn(lambda t: random_crop(t, (320, 480)), X, dtype=(tf.int32, tf.int32))
        X = X_and_offset[0]
        self.Y = tf.map_fn(lambda t: fixed_crop(t[0], t[1], (320, 480)), (Y, X_and_offset[1]), dtype=tf.int32)

        self.X = tf.cast(X, tf.float32)
        # backbone: VGG16
        # todo: stride param of conv layers.
        f = conv2d(self.X, [3, 3, 3, 64], "conv1", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 64, 64], "conv2", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = tf.nn.max_pool(f, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

        f = conv2d(f, [3, 3, 64, 128], "conv3", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 128, 128], "conv4", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = tf.nn.max_pool(f, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

        f = conv2d(f, [3, 3, 128, 256], "conv5", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 256, 256], "conv6", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 256, 256], "conv7", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        p3 = tf.nn.max_pool(f, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

        f = conv2d(p3, [3, 3, 256, 512], "conv8", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 512, 512], "conv9", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 512, 512], "conv10", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        p4 = tf.nn.max_pool(f, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

        f = conv2d(p4, [3, 3, 512, 512], "conv11", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 512, 512], "conv12", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        f = conv2d(f, [3, 3, 512, 512], "conv13", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        p5 = tf.nn.max_pool(f, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

        conv6 = conv2d(p5, [1, 1, 512, 4096], "conv14", activation=tf.nn.relu, padding="SAME", strides=(1, 1, 1, 1))
        conv7 = conv2d(conv6, [1, 1, 4096, 4096], "conv15", activation=tf.nn.relu, padding='SAME', strides=(1, 1, 1, 1))
        f32 = conv2d(conv7, [1, 1, 4096, num_class], "f32", padding='SAME', strides=(1, 1, 1, 1))  # /32

        f32 = conv2d_transpose(f32, [4, 4, 512, num_class], "convt1", padding='SAME', strides=(1, 2, 2, 1),
                               output_shape=tf.shape(p4))
        f16 = conv2d_transpose(f32 + p4, [4, 4, 256, 512], "convt2", padding='SAME', strides=(1, 2, 2, 1),
                               output_shape=tf.shape(p3))
        output_shape = tf.concat([tf.shape(self.X)[:-1], [num_class]], axis=-1)
        self.logits = f8 = conv2d_transpose(f16 + p3, [7, 7, num_class, 256], "convt3", padding="SAME",
                                            strides=(1, 8, 8, 1), output_shape=output_shape)
        self.y_pred = tf.cast(tf.argmax(f8, axis=-1), tf.int32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.squeeze(self.Y, axis=[3])))
        self.pxacc = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.Y, axis=[3]), self.y_pred), tf.float32))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.session = get_session(debug=False)
        initialize()

    def train(self, epochs=10):
        for e in range(epochs):
            self.session.run(self.train_init_op)
            while True:
                try:
                    _, loss, pxacc = self.session.run([self.train_op, self.loss, self.pxacc])
                    print(f'[{e}/{epochs}], Loss={loss}, pxacc={pxacc}')
                except tf.errors.OutOfRangeError as e:
                    break


if __name__ == "__main__":
    ds_train = ADE20k('training', 320, 480).padded_batch(5, ([None, None, 3], [None, None, 1])) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ADE20k('validation', 320, 480).padded_batch(5, ([None, None, 3], [None, None, 1])) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    m = FCN(ds_train, ds_val, 151)
    m.train()

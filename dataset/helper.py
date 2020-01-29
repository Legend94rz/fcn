import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.python.ops import control_flow_ops

from PIL import Image
import tarfile as tar
import pandas as pd


def random_crop(value, size):
    with tf.name_scope(None, "random_crop", [value, size]) as name:
        value = tf.convert_to_tensor(value, name="value")
        size = tf.concat([size, tf.shape(value)[-1:]], axis=-1, name="size")
        shape = tf.shape(value)
        check = control_flow_ops.Assert(
            tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size], summarize=1000)
        shape = control_flow_ops.with_dependencies([check], shape)
        limit = shape - size + 1
        offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max) % limit
        return tf.slice(value, offset, size, name=name), offset[0:2]


def fixed_crop(value, offset, size):
    # 这里的size如果不是一个由调用者指定的常量，将导致后面无法确定输出形状，因此random_crop只返回offset，不以Tensor的形式返回size。
    with tf.name_scope(None, "fixed_crop", [value, offset, size]) as name:
        value = tf.convert_to_tensor(value, name="value")
        offset = tf.concat([offset, [0]], axis=-1, name="size")
        size = tf.concat([size, tf.shape(value)[-1:]], axis=-1, name="size")
        # todo: assert
        return tf.slice(value, offset, size, name=name)


class ADE20k(tf.data.Dataset):
    root = "."

    @staticmethod
    def _gen(usage, minh, minw):
        usage = usage.decode()
        xtrain_path = f'{ADE20k.root}/images/{usage}'
        ytrain_path = f'{ADE20k.root}/annotations/{usage}'
        xlst = list(map(lambda t: f'{xtrain_path}/{t}', os.listdir(xtrain_path)))
        ylst = list(map(lambda t: f'{ytrain_path}/{t}', os.listdir(ytrain_path)))
        assert (len(xlst) == len(ylst))
        for i in range(len(xlst)):
            # print(xlst[i], ylst[i])
            x = cv2.imread(xlst[i], cv2.IMREAD_COLOR)
            if x.shape[0] < minh or x.shape[1] < minw:
                continue
            y = np.expand_dims(cv2.imread(ylst[i], -1), axis=-1)
            yield x, y

    def __new__(cls, usage, min_height, min_width):
        return tf.data.Dataset.from_generator(cls._gen, output_types=(tf.int32, tf.int32),
                                              output_shapes=([None, None, 3], [None, None, 1]),
                                              args=(usage, min_height, min_width))

    def __init__(self, usage, min_height, min_width):
        super(ADE20k, self).__init__()
        self.usage = usage
        self.out_height = min_height
        self.out_width = min_width


class PascalVOL12(tf.data.Dataset):
    tarfilepath = './VOCtrainval_11-May-2012.tar'
    color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
             [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
             [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
             [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
             [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    color2cls = np.zeros(256 ** 3, dtype='uint8')
    for i, c in enumerate(color):
        color2cls[c[0] * 256 ** 2 + c[1] * 256 + c[2]] = i

    def __new__(cls, usage, min_size):
        return tf.data.Dataset.from_generator(cls._gen, output_types=(tf.int32, tf.int32),
                                              output_shapes=([None, None, 3], [None, None, 1]),
                                              args=(usage, min_size[0], min_size[1]))

    def __init__(self, usage, min_size):
        r"""
        :param usage: {'train','val'}
        :param min_size: (min height, min width)
        """
        super().__init__()
        self.usage = usage
        self.min_size = min_size

    @staticmethod
    def color2label(img):
        return PascalVOL12.color2cls[(img[:, :, 0] * 256 + img[:, :, 1]) * 256 + img[:, :, 2]]

    @staticmethod
    def _gen(usage, minh, minw):
        usage = usage.decode()
        with tar.open(PascalVOL12.tarfilepath, 'r') as t:
            with t.extractfile(f'VOCdevkit/VOC2012/ImageSets/Segmentation/{usage}.txt') as csv:
                lst = pd.read_csv(csv, header=None)
            lst = lst.values.reshape(-1, ).tolist()
            for p in lst:
                with t.extractfile(f'VOCdevkit/VOC2012/JPEGImages/{p}.jpg') as img:
                    jpg = Image.open(img).convert('RGB')
                    x = cv2.cvtColor(np.array(jpg), cv2.COLOR_RGB2BGR)
                if x.shape[0] < minh or x.shape[1] < minw:
                    continue
                with t.extractfile(f'VOCdevkit/VOC2012/SegmentationClass/{p}.png') as img:
                    png = Image.open(img).convert('RGB')  # drop alpha channel
                    y = np.array(np.array(png))
                y = np.expand_dims(PascalVOL12.color2label(y), axis=-1)
                yield x, y


if __name__ == "__main__":
    print(PascalVOL12.color2cls.sum())

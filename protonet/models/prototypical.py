import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, MaxPool2D, Add, \
    GlobalAveragePooling2D, LeakyReLU, Layer
from tensorflow.keras import Model


def calc_euclidian_dists(x, y):
    """数值稳定的距离计算"""
    # 输入x和y已经是L2归一化的
    # x形状: [batch, n_query, 1, z_dim]
    # y形状: [batch, 1, n_way, z_dim]
    cosine_similarity = tf.reduce_sum(x * y, axis=-1)  # [batch, n_query, n_way]
    return 2 - 2 * cosine_similarity  # 转换为欧氏距离平方


class DropBlock2D(Layer):
    def __init__(self, keep_prob, block_size, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = keep_prob
        self.block_size = block_size

    def call(self, inputs, training=False):
        if not training:
            return inputs
        gamma = (1. - self.keep_prob) * tf.math.reduce_prod(tf.cast(inputs.shape[1:3], tf.float32)) / (
                    self.block_size ** 2)
        mask = tf.cast(tf.random.uniform(tf.shape(inputs)) < gamma, tf.float32)
        mask = tf.nn.max_pool2d(mask, ksize=self.block_size, strides=1, padding='SAME')
        return inputs * (1 - mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'keep_prob': self.keep_prob,
            'block_size': self.block_size
        })
        return config


class ResidualBlock(Layer):
    def __init__(self, filters, downsample=False, kernel_initializer='he_normal', use_bias=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.downsample = downsample
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias

        # 主路径
        self.conv1 = Conv2D(filters, 3, padding='same', strides=2 if downsample else 1,
                            kernel_initializer=kernel_initializer, use_bias=use_bias)
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, 3, padding='same',
                            kernel_initializer=kernel_initializer, use_bias=use_bias)
        self.bn2 = BatchNormalization()

        # 跳跃连接
        self.shortcut = tf.keras.Sequential()
        if downsample:
            self.shortcut.add(Conv2D(filters, 1, strides=2,
                                     kernel_initializer=kernel_initializer, use_bias=use_bias))
            self.shortcut.add(BatchNormalization())

        # 正则化
        self.dropblock = DropBlock2D(keep_prob=0.5, block_size=5)
        self.leaky_relu = LeakyReLU(alpha=0.1)

    def call(self, inputs):
        # 主路径
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 跳跃连接
        shortcut = self.shortcut(inputs)

        # 合并路径
        x = Add()([x, shortcut])
        x = self.leaky_relu(x)

        if self.downsample:
            x = self.dropblock(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'downsample': self.downsample,
            'kernel_initializer': self.kernel_initializer,
            'use_bias': self.use_bias
        })
        return config


class Prototypical(Model):
    def __init__(self, n_support, n_query, w, h, c, encoder_type='resnet12', nb_layers=4, nb_filters=64):
        super(Prototypical, self).__init__()
        self.w, self.h, self.c = w, h, c
        self.n_support = n_support
        self.n_query = n_query
        self.tau = tf.Variable(0.5, trainable=True, name='temperature')  # 可学习温度参数

        # 特征编码器
        if encoder_type == 'conv64F':
            layers = []
            for _ in range(nb_layers):
                layers += [
                    Conv2D(nb_filters, 3, padding='same'),
                    BatchNormalization(),
                    ReLU(),
                    MaxPool2D(2)
                ]
            layers.append(Flatten())
            self.encoder = tf.keras.Sequential(layers)
        elif encoder_type == 'resnet12':
            filters = [64, 160, 320, 640]
            self.encoder = tf.keras.Sequential([
                *[ResidualBlock(filters[i], downsample=True) for i in range(nb_layers)],
                GlobalAveragePooling2D(),
                DropBlock2D(0.5, 5),
                Flatten()
            ])
        else:
            raise ValueError("Unsupported encoder type")

    def call(self, support_batch, query_batch):
        # 合并输入
        combined = tf.concat([
            tf.reshape(support_batch, [-1, self.w, self.h, self.c]),
            tf.reshape(query_batch, [-1, self.w, self.h, self.c])
        ], axis=0)

        # 特征提取
        z_all = self.encoder(combined)
        z_all = tf.math.l2_normalize(z_all, axis=-1)  # L2归一化

        # 分割支持集和查询集
        batch_size = tf.shape(support_batch)[0]
        n_way = tf.shape(support_batch)[1]

        # 原型计算
        z_support = z_all[:batch_size * n_way * self.n_support]
        z_support = tf.reshape(z_support, [batch_size, n_way, self.n_support, -1])
        prototypes = tf.math.reduce_mean(z_support, axis=2)
        prototypes = tf.math.l2_normalize(prototypes, axis=-1)  # 原型归一化

        # 查询样本
        z_query = z_all[batch_size * n_way * self.n_support:]
        z_query = tf.reshape(z_query, [batch_size, n_way * self.n_query, -1])
        z_query = tf.math.l2_normalize(z_query, axis=-1)

        # 计算距离
        dists = calc_euclidian_dists(
            tf.expand_dims(z_query, 2),  # [batch, nq, 1, d]
            tf.expand_dims(prototypes, 1)  # [batch, 1, nw, d]
        )
        dists = tf.clip_by_value(dists, 0.0, 5.0)  # 限制距离范围

        # 计算损失
        logits = -dists / (self.tau + 1e-8)
        labels = tf.tile(tf.range(n_way, dtype=tf.int64)[None, :],
                         [batch_size, self.n_query])
        labels = tf.reshape(labels, [batch_size, n_way * self.n_query])

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )

        # 计算准确率
        preds = tf.argmax(logits, axis=-1, output_type=tf.int64)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

        return loss, acc

    def save(self, model_path):
        self.encoder.save(model_path, save_format='tf')

    def load(self, model_path):
        self.encoder = tf.keras.models.load_model(model_path,
                                                  custom_objects={
                                                      'ResidualBlock': ResidualBlock,
                                                      'DropBlock2D': DropBlock2D,
                                                      'LeakyReLU': LeakyReLU
                                                  })
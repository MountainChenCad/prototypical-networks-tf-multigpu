import os
import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE


class DataLoader(object):
    def __init__(self, data_dir, split, n_way, n_support, n_query, num_parallel=8, prefetch=10):
        # 保持原有初始化代码
        self.n_way = n_way  # 新增属性
        self.n_support = n_support
        self.n_query = n_query
        self.data_dir = data_dir
        self.split = split
        self.n_way = n_way
        self.split_file = os.path.join('..', data_dir, 'splits', 'default', f'{split}.csv')

        # 预加载元数据
        self.img_paths, self.labels = self._load_split()
        self.unique_labels = np.unique(self.labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}

        # 创建TensorFlow数据集
        self.dataset = self._create_dataset(num_parallel, prefetch)
        self.episode_generator = self._episode_generation()

    def _load_split(self):
        with open(self.split_file, 'r') as f:
            lines = f.readlines()[1:]
            filenames = [line.split(',')[0] for line in lines]
            labels = [line.split(',')[1].strip() for line in lines]
        return [os.path.join('..', self.data_dir, 'data', f) for f in filenames], labels

    def _parse_function(self, img_path, label):
        # 使用TF原生图像处理
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [84, 84])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def _create_dataset(self, num_parallel, prefetch):
        # 创建基础数据集
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # 并行化预处理
        dataset = dataset.shuffle(buffer_size=len(self.img_paths), reshuffle_each_iteration=True)
        dataset = dataset.map(self._parse_function,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=prefetch)
        return dataset

    def _sample_episode(self):
        # 使用TF操作生成episode
        selected_classes = tf.random.shuffle(self.unique_labels)[:self.n_way]

        def gather_class_examples(class_label):
            mask = tf.equal(self.labels, class_label)
            class_dataset = self.dataset.filter(lambda img, lbl: tf.equal(lbl, class_label))
            return class_dataset.take(self.n_support + self.n_query)

        episode_ds = tf.data.Dataset.from_tensor_slices(selected_classes)
        episode_ds = episode_ds.interleave(
            gather_class_examples,
            cycle_length=self.n_way,
            num_parallel_calls=AUTOTUNE
        )

        # 重组support和query集
        all_examples = list(episode_ds.batch(self.n_support + self.n_query).as_numpy_iterator())
        support = np.stack([ex[0][:self.n_support] for ex in all_examples])
        query = np.stack([ex[0][self.n_support:] for ex in all_examples])
        return support, query

    def _episode_generation(self):
        while True:
            yield self._sample_episode()

    def get_next_episode(self):
        return next(self.episode_generator)


def load_dogs(data_dir, config, splits):
    ret = {}
    for split in splits:
        num_parallel = config.get('data.num_parallel', 8)
        prefetch = config.get('data.prefetch', 10)

        if split in ['val', 'test']:
            n_way = config['data.test_way']
            n_support = config['data.test_support']
            n_query = config['data.test_query']
        else:
            n_way = config['data.train_way']
            n_support = config['data.train_support']
            n_query = config['data.train_query']

        data_loader = DataLoader(
            data_dir, split, n_way, n_support, n_query,
            num_parallel=num_parallel,
            prefetch=prefetch
        )
        ret[split] = data_loader
    return ret
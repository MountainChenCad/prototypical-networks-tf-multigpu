import os
import numpy as np
from PIL import Image
import tensorflow as tf
from multiprocessing import Pool
from tensorflow.keras.preprocessing.image import apply_affine_transform

class DataLoader(object):
    def __init__(self, data_dir, split, n_way, n_support, n_query, batch_size=4, 
                 num_workers=4, augment=False):
        self.data_dir = data_dir
        self.split = split
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        
        # 修正文件路径处理
        self.split_file = os.path.join('..', data_dir, 'splits', 'default', split + '.csv')
        self._verify_paths()
        
        self.img_paths, self.labels = self._load_split()
        self.unique_labels = np.unique(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] 
                                for label in self.unique_labels}
        self._preload_images()

    def _verify_paths(self):
        """验证关键文件路径是否存在"""
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split文件不存在: {self.split_file}")
        sample_img = os.path.join('..', self.data_dir, 'data', 'n02085620_473.jpg')
        if not os.path.exists(sample_img):
            raise FileNotFoundError(f"数据目录结构异常，示例文件不存在: {sample_img}")

    def _load_split(self):
        """加载CSV分割文件并进行完整性检查"""
        with open(self.split_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()[1:] if line.strip()]
            
        # 解析并验证数据
        filenames, labels = [], []
        for line in lines:
            parts = line.split(',')
            if len(parts) != 2:
                continue
            filename, label = parts
            full_path = os.path.join('..', self.data_dir, 'data', filename)
            if os.path.exists(full_path):
                filenames.append(filename)
                labels.append(label)
            else:
                print(f"警告：缺失文件 {filename}，已跳过")
                
        return filenames, np.array(labels)

    def _preload_images(self):
        """多进程预加载图像到内存"""
        print(f"预加载 {len(self.img_paths)} 张图像...")
        with Pool(self.num_workers) as pool:
            results = pool.imap(self._load_and_preprocess_image, 
                              [os.path.join('..', self.data_dir, 'data', f) for f in self.img_paths],
                              chunksize=16)
            self.image_cache = np.stack(list(results), axis=0)
        print(f"图像预加载完成，缓存形状: {self.image_cache.shape}")

    def _load_and_preprocess_image(self, img_path):
        """加载并预处理单个图像"""
        try:
            img = Image.open(img_path).convert('RGB')
            # 保持长宽比的缩放
            img = self._aspect_preserving_resize(img, target_size=92)
            # 随机裁剪到84x84
            img = self._random_crop(np.array(img), target_size=84)
            # 数值标准化
            img = self._normalize_image(img)
            return img
        except Exception as e:
            print(f"加载图像错误 {img_path}: {str(e)}")
            return np.zeros((84, 84, 3), dtype=np.float32)

    def _aspect_preserving_resize(self, img, target_size):
        """保持长宽比的缩放"""
        width, height = img.size
        scale = target_size / min(width, height)
        new_size = (int(width * scale), int(height * scale))
        return img.resize(new_size, Image.BILINEAR)

    def _random_crop(self, img, target_size):
        """随机裁剪到目标尺寸"""
        height, width = img.shape[:2]
        dy = np.random.randint(0, height - target_size)
        dx = np.random.randint(0, width - target_size)
        return img[dy:dy+target_size, dx:dx+target_size]

    def _normalize_image(self, img):
        """应用ImageNet标准化"""
        img = img.astype(np.float32) / 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return (img - mean) / std

    def _augment_image(self, img):
        """应用随机数据增强"""
        if np.random.rand() < 0.5:
            img = np.fliplr(img)  # 水平翻转
            
        # 随机仿射变换
        rotation = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.9, 1.1)
        shear = np.random.uniform(-0.1, 0.1)
        return apply_affine_transform(
            img, 
            theta=rotation,
            zx=scale, zy=scale,
            shear=shear,
            row_axis=0, 
            col_axis=1, 
            channel_axis=2
        )

    def get_batch(self):
        """生成一个批次的增强数据"""
        support = np.zeros([self.batch_size, self.n_way, self.n_support, 84, 84, 3], 
                          dtype=np.float32)
        query = np.zeros([self.batch_size, self.n_way, self.n_query, 84, 84, 3], 
                        dtype=np.float32)
        
        for b in range(self.batch_size):
            # 确保每个episode选择唯一的类别
            selected_classes = np.random.choice(
                self.unique_labels, 
                size=self.n_way, 
                replace=False
            )
            
            for i, class_label in enumerate(selected_classes):
                indices = self.label_to_indices[class_label]
                selected = np.random.choice(
                    indices, 
                    size=self.n_support + self.n_query, 
                    replace=len(indices) >= (self.n_support + self.n_query)
                )
                
                # 支持集处理
                support_images = self.image_cache[selected[:self.n_support]]
                if self.augment:
                    support_images = np.stack([self._augment_image(img) for img in support_images])
                support[b, i] = support_images
                
                # 查询集处理
                query_images = self.image_cache[selected[self.n_support:]]
                if self.augment:
                    query_images = np.stack([self._augment_image(img) for img in query_images])
                query[b, i] = query_images
        
        # 转换为TensorFlow张量
        return (tf.convert_to_tensor(support), 
                tf.convert_to_tensor(query))

def load_dogs(data_dir, config, splits):
    ret = {}
    for split in splits:
        is_train = (split == 'train')
        loader = DataLoader(
            data_dir=data_dir,
            split=split,
            n_way=config['data.train_way' if is_train else 'data.test_way'],
            n_support=config['data.train_support' if is_train else 'data.test_support'],
            n_query=config['data.train_query' if is_train else 'data.test_query'],
            batch_size=config['data.batch_size'],
            num_workers=config.get('data.num_workers', 8),
            augment=is_train and config.get('data.augment', True)
        )
        ret[split] = loader
    return ret
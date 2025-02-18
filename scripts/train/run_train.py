import argparse
import configparser
import re
import tensorflow as tf
from train_setup import train


def sanitize_value(value):
    """增强型配置值清洗"""
    if isinstance(value, str):
        value = value.strip(" '\";\t\n\r")  # 移除多余字符

        # 处理布尔型
        lower_val = value.lower()
        if lower_val in {'true', 'yes', 't', 'y', '1'}:
            return True
        if lower_val in {'false', 'no', 'f', 'n', '0'}:
            return False

        # 处理科学计数法
        if re.match(r'^-?\d+\.?\d*[eE][+-]?\d+$', value):
            return float(value)

        # 处理百分数
        if '%' in value:
            return float(value.replace('%', '')) / 100

        # 处理空值
        if value in {'', 'None', 'null'}:
            return None

    return value


def preprocess_config(c):
    """强化类型安全配置处理"""
    conf_dict = {}

    # 动态类型推断参数列表
    int_params = {
        'data.batch_size', 'data.train_way', 'data.test_way',
        'data.train_support', 'data.test_support', 'data.train_query',
        'data.test_query', 'data.episodes', 'data.gpu', 'data.cuda',
        'model.z_dim', 'train.epochs', 'train.patience', 'model.nb_layers',
        'model.nb_filters', 'data.num_workers', 'data.val_batches',
        'data.batches_per_epoch', 'data.prefetch_buffer'
    }

    float_params = {
        'data.train_size', 'data.test_size', 'train.lr',
        'data.rotation_range', 'data.width_shift_range',
        'data.height_shift_range', 'data.horizontal_flip'
    }

    bool_params = {
        'data.horizontal_flip'
    }

    for param in c:
        raw_value = c[param]
        cleaned_value = sanitize_value(raw_value)

        try:
            if param in int_params:
                # 支持浮点字符串转换（如"32.0" -> 32）
                conf_dict[param] = int(float(cleaned_value)) if cleaned_value else 0
            elif param in float_params:
                conf_dict[param] = float(cleaned_value) if cleaned_value else 0.0
            elif param in bool_params:
                # 确保布尔类型
                conf_dict[param] = bool(cleaned_value)
            else:
                # 保留原始类型
                conf_dict[param] = cleaned_value if cleaned_value is not None else ''
        except (ValueError, TypeError) as e:
            raise ValueError(f"参数转换失败: {param}={raw_value} (处理后: {cleaned_value})") from e

    # 后处理验证
    if 'data.horizontal_flip' in conf_dict and isinstance(conf_dict['data.horizontal_flip'], float):
        conf_dict['data.horizontal_flip'] = conf_dict['data.horizontal_flip'] > 0.5

    return conf_dict


# 增强型命令行解析
parser = argparse.ArgumentParser(description='元学习训练脚本',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 基础配置
parser.add_argument("--config", type=str, default="./src/config/config_omniglot.conf",
                    help="主配置文件路径")

# 数据配置组
data_group = parser.add_argument_group('数据参数')
data_group.add_argument("--data.dataset", type=str,
                        help="数据集名称 (omniglot/miniImagenet)")
data_group.add_argument("--data.split", type=str,
                        help="数据划分方式 (default/version)")
data_group.add_argument("--data.batch_size", type=int,
                        help="每批任务数量")
data_group.add_argument("--data.train_way", type=int,
                        help="训练时类别数")
data_group.add_argument("--data.train_support", type=int,
                        help="每类支持样本数")
data_group.add_argument("--data.train_query", type=int,
                        help="每类查询样本数")
data_group.add_argument("--data.test_way", type=int,
                        help="测试时类别数")
data_group.add_argument("--data.test_support", type=int,
                        help="测试支持样本数")
data_group.add_argument("--data.test_query", type=int,
                        help="测试查询样本数")

# 数据增强组
aug_group = parser.add_argument_group('数据增强')
aug_group.add_argument("--data.rotation_range", type=float,
                       help="旋转角度范围 (度)")
aug_group.add_argument("--data.width_shift_range", type=float,
                       help="宽度平移比例")
aug_group.add_argument("--data.height_shift_range", type=float,
                       help="高度平移比例")
aug_group.add_argument("--data.horizontal_flip", type=lambda x: x.lower() in ['true', '1'],
                       help="是否水平翻转")

# 训练参数组
train_group = parser.add_argument_group('训练参数')
train_group.add_argument("--train.epochs", type=int,
                         help="训练总epoch数")
train_group.add_argument("--train.patience", type=int,
                         help="早停等待epoch数")
train_group.add_argument("--train.lr", type=float,
                         help="初始学习率")

# 模型参数组
model_group = parser.add_argument_group('模型参数')
model_group.add_argument("--model.type", type=str,
                         choices=['conv64F', 'resnet12'],
                         help="编码器类型")
model_group.add_argument("--model.x_dim", type=str,
                         help="输入维度 (格式: width,height,channel)")
model_group.add_argument("--model.z_dim", type=int,
                         help="特征维度")
model_group.add_argument("--model.nb_layers", type=int,
                         help="网络层数")
model_group.add_argument("--model.nb_filters", type=int,
                         help="初始卷积核数量")

if __name__ == "__main__":
    # 解析参数
    args = parser.parse_args()
    args_dict = vars(args)

    # 加载基础配置
    config = configparser.ConfigParser(strict=False, interpolation=None)
    config.read(args.config, encoding='utf-8')

    # 合并配置（支持多section）
    merged_config = {}
    for section in ['TRAIN', 'EVAL', 'DATA', 'MODEL']:
        if config.has_section(section):
            merged_config.update(dict(config.items(section)))

    # 用命令行参数覆盖配置
    for arg_key, arg_val in args_dict.items():
        if arg_val is not None and arg_key in merged_config:
            # 保持原始类型
            orig_type = type(sanitize_value(merged_config[arg_key]))
            try:
                merged_config[arg_key] = orig_type(arg_val)
            except ValueError:
                merged_config[arg_key] = arg_val
        elif arg_val is not None:
            merged_config[arg_key] = arg_val

    # 最终配置处理
    final_config = preprocess_config(merged_config)

    # 必要参数检查
    required_params = [
        'data.dataset', 'data.train_way', 'data.test_way',
        'model.type', 'model.x_dim', 'train.lr'
    ]
    for param in required_params:
        if param not in final_config or final_config[param] is None:
            raise ValueError(f"缺失必要参数: {param}")

    # GPU可用性检查
    if final_config.get('data.cuda', False):
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError("配置要求使用GPU但未检测到可用GPU设备!")
        # 显存优化设置
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 启动训练流程
    train(final_config)
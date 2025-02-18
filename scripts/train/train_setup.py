"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from protonet_tf2.protonet import TrainEngine
from protonet_tf2.protonet.models import Prototypical
from protonet_tf2.protonet.datasets import load


def _create_distributed_dataset(strategy, data_loader, config):
    """Create distributed dataset pipeline"""
    n_way = data_loader.n_way
    n_support = data_loader.n_support
    n_query = data_loader.n_query

    def _episode_generator():
        while True:
            support, query = data_loader.get_next_episode()
            # 确保形状符合预期
            support = np.reshape(support, (n_way, n_support, 84, 84, 3))
            query = np.reshape(query, (n_way, n_query, 84, 84, 3))
            yield (support.astype(np.float32), query.astype(np.float32))

    dataset = tf.data.Dataset.from_generator(
        _episode_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([n_way, n_support, 84, 84, 3]),  # 明确指定形状
            tf.TensorShape([n_way, n_query, 84, 84, 3])
        )
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return strategy.experimental_distribute_dataset(dataset)

def train(config):
    # 初始化多GPU策略
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    np.random.seed(2019)
    tf.random.set_seed(2019)

    # 创建模型和优化器在策略范围内
    with strategy.scope():
        # 模型定义
        n_support = config['data.train_support']
        n_query = config['data.train_query']
        w, h, c = list(map(int, config['model.x_dim'].split(',')))
        model = Prototypical(n_support, n_query, w, h, c,
                           nb_layers=config['model.nb_layers'],
                           nb_filters=config['model.nb_filters'])

        # 优化器
        optimizer = tf.keras.optimizers.Adam(config['train.lr'])

        # 分布式指标
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_acc = tf.keras.metrics.Mean(name='train_accuracy')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_acc = tf.keras.metrics.Mean(name='val_accuracy')

    # 文件路径配置
    now_as_str = datetime.now().strftime('%Y_%m_%d-%H:%M:%S')
    model_file = config['model.save_path'].format(config['model.type'], now_as_str)
    config_file = config['output.config_path'].format(config['model.type'], now_as_str)
    csv_output_file = config['output.train_path'].format(config['model.type'], now_as_str)

    # 创建目录
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

    # 加载数据集
    data_dir = f"../datasets/{config['data.dataset']}"
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # 创建分布式数据集
    train_dist_dataset = _create_distributed_dataset(strategy, train_loader, config)
    val_dist_dataset = _create_distributed_dataset(strategy, val_loader, config)

    # 定义训练步骤
    @tf.function
    def train_step(iterator):
        def step_fn(inputs):
            support, query = inputs
            with tf.GradientTape() as tape:
                loss, acc = model(support, query)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss.update_state(loss)
            train_acc.update_state(acc)
            return loss

        for _ in tf.range(tf.convert_to_tensor(config['data.batch_size'])):
            strategy.run(step_fn, args=(next(iterator),))

    # 定义验证步骤
    @tf.function
    def test_step(iterator):
        def step_fn(inputs):
            support, query = inputs
            loss, acc = model(support, query)
            val_loss.update_state(loss)
            val_acc.update_state(acc)
            return loss

        for _ in tf.range(tf.convert_to_tensor(config['data.batch_size'])):
            strategy.run(step_fn, args=(next(iterator),))

    # 创建训练引擎
    train_engine = TrainEngine()

    # 训练过程钩子函数
    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_states()
        train_acc.reset_states()

    def on_end_epoch(state):
        # 验证阶段
        val_iterator = iter(val_dist_dataset)
        for _ in range(config['data.episodes'] // config['data.batch_size']):
            test_step(val_iterator)

        # 打印指标
        template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
        print(template.format(
            state['epoch'],
            train_loss.result(),
            train_acc.result() * 100,
            val_loss.result(),
            val_acc.result() * 100
        ))

        # 记录到CSV
        with open(csv_output_file, 'a') as f:
            f.write(f"{state['epoch']}, {train_loss.result():.4f}, {train_acc.result() * 100:.2f}, "
                    f"{val_loss.result():.4f}, {val_acc.result() * 100:.2f}\n")

        # 保存最佳模型
        current_val_loss = val_loss.result().numpy()
        if current_val_loss < state['best_val_loss']:
            state['best_val_loss'] = current_val_loss
            model.save(model_file)

        # 早停机制
        state['val_losses'].append(current_val_loss)
        if len(state['val_losses']) > config['train.patience'] and \
                all(v >= state['val_losses'][-config['train.patience']]
                    for v in state['val_losses'][-config['train.patience']:]):
            state['early_stopping_triggered'] = True

        val_loss.reset_states()
        val_acc.reset_states()

    # 注册钩子
    train_engine.hooks.update({
        'on_start_epoch': on_start_epoch,
        'on_end_epoch': on_end_epoch
    })

    # 初始化训练状态
    state = {
        'epoch': 1,
        'best_val_loss': np.inf,
        'val_losses': [],
        'early_stopping_triggered': False,
        'epochs': config['train.epochs']
    }

    # 启动训练循环
    train_iterator = iter(train_dist_dataset)
    for epoch in range(config['train.epochs']):
        train_engine.hooks['on_start_epoch'](state)

        # 训练阶段
        for _ in range(config['data.episodes'] // config['data.batch_size']):
            train_step(train_iterator)

        train_engine.hooks['on_end_epoch'](state)
        state['epoch'] += 1

        if state['early_stopping_triggered']:
            print("Early stopping triggered!")
            break

    print(f"Training completed. Best validation loss: {state['best_val_loss']:.4f}")

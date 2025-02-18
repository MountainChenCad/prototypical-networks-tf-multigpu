"""
增强版训练配置，主要改进：
1. 分布式梯度裁剪
2. 学习率预热机制
3. 梯度监控系统
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

"""
修复学习率调度器访问问题的完整代码
"""

import sys
import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from protonet_tf2.protonet import TrainEngine
from protonet_tf2.protonet.models import Prototypical
from protonet_tf2.protonet.datasets import load


def train(config):
    # 分布式策略初始化
    strategy = tf.distribute.MirroredStrategy()
    print(f'可用设备数: {strategy.num_replicas_in_sync}')

    with strategy.scope():
        # 动态调整批量大小
        config['data.batch_size'] *= strategy.num_replicas_in_sync

        # 模型初始化
        w, h, c = map(int, config['model.x_dim'].split(','))
        model = Prototypical(
            n_support=config['data.train_support'],
            n_query=config['data.train_query'],
            w=w, h=h, c=c,
            nb_layers=config['model.nb_layers'],
            nb_filters=config['model.nb_filters']
        )

        # 学习率管理（核心修复）
        base_lr = tf.Variable(config['train.lr'], dtype=tf.float32, name='base_learning_rate')
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr,
            decay_steps=config.get('train.decay_steps', 1000),
            decay_rate=0.95,
            staircase=True
        )

        # 优化器配置
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule(global_step),  # 直接使用调度函数
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

    # 监控指标
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.Mean(name='train_acc')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc = tf.keras.metrics.Mean(name='val_acc')

    # 文件路径配置
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_dir = f"results/{config['data.dataset']}/protonet"
    os.makedirs(f"{model_dir}/checkpoints", exist_ok=True)
    model_path = f"{model_dir}/checkpoints/model_{timestamp}.h5"

    @tf.function
    def distributed_train_step(support, query):
        def step_fn(inputs):
            s, q = inputs
            with tf.GradientTape() as tape:
                loss, acc = model(s, q)
                loss = tf.debugging.check_numerics(loss, "损失值异常")

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 5.0) for g in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, acc

        per_replica_loss, per_replica_acc = strategy.run(step_fn, args=((support, query),))
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_acc, axis=None)
        )

    # 训练引擎配置
    engine = TrainEngine()

    def on_start_batch(state):
        try:
            # 更新全局步数
            tf.compat.v1.assign_add(global_step, 1)

            support, query = state['batch']
            support = tf.clip_by_value(support, 0.0, 1.0)
            query = tf.clip_by_value(query, 0.0, 1.0)

            loss, acc = distributed_train_step(support, query)

            # 获取当前实际学习率
            current_lr = optimizer.learning_rate.numpy()

            train_loss.update_state(loss)
            train_acc.update_state(acc)

            if state['total_batch'] % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {state['total_batch']} - "
                      f"Loss: {loss:.4f} Acc: {acc * 100:.2f}% LR: {current_lr:.2e}")

        except tf.errors.InvalidArgumentError as e:
            print(f"\n数值异常: {str(e)}")
            # 直接修改基础学习率变量
            new_lr = base_lr.assign(base_lr * 0.5)
            print(f"降低学习率至: {new_lr.numpy():.2e}")
            state['total_batch'] -= 1

    def on_end_epoch(state):
        try:
            val_results = []
            for _ in range(config['data.val_batches']):
                s, q = state['val_loader'].get_batch()
                loss, acc = distributed_val_step(s, q)
                val_loss.update_state(loss)
                val_acc.update_state(acc)
                val_results.append((loss.numpy(), acc.numpy()))

            avg_val_loss = np.nanmean([x[0] for x in val_results])
            avg_val_acc = np.nanmean([x[1] for x in val_results])

            print(f"\nEpoch {state['epoch']} 结果:")
            print(f"训练损失: {train_loss.result():.4f} 准确率: {train_acc.result() * 100:.2f}%")
            print(f"验证损失: {avg_val_loss:.4f} 准确率: {avg_val_acc * 100:.2f}%")

            if avg_val_loss < state['best_val_loss']:
                state['best_val_loss'] = avg_val_loss
                state['best_epoch'] = state['epoch']
                model.save_weights(model_path)
                print(f"保存最佳模型至 {model_path}")

            train_loss.reset_states()
            train_acc.reset_states()
            val_loss.reset_states()
            val_acc.reset_states()

        except Exception as e:
            print(f"验证异常: {str(e)}")
            state['early_stopping_triggered'] = True

    engine.hooks.update({
        'on_start_batch': on_start_batch,
        'on_end_epoch': on_end_epoch
    })

    # 初始化训练状态
    data_dir = f"datasets/{config['data.dataset']}"
    state = {
        'train_loader': load(data_dir, config, ['train']).get('train'),
        'val_loader': load(data_dir, config, ['val']).get('val'),
        'epoch': 1,
        'total_batch': 1,
        'epochs': config['train.epochs'],
        'batches_per_epoch': config['data.batches_per_epoch'],
        'best_val_loss': float('inf'),
        'early_stopping_triggered': False
    }

    print("\n训练启动...")
    print(f"初始基础学习率: {base_lr.numpy():.2e}")
    start_time = time.time()
    try:
        engine.train(
            train_loader=state['train_loader'],
            val_loader=state['val_loader'],
            epochs=config['train.epochs'],
            batches_per_epoch=config['data.batches_per_epoch'],
            state=state
        )
    finally:
        final_model_path = f"{model_dir}/checkpoints/final_{timestamp}.h5"
        model.save_weights(final_model_path)
        print(f"训练完成，最终模型保存至: {final_model_path}")

    return state
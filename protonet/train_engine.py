import numpy as np
import tensorflow as tf


class TrainEngine(object):
    def __init__(self):
        self.hooks = {
            'on_start': lambda state: None,
            'on_start_epoch': lambda state: None,
            'on_end_epoch': lambda state: None,
            'on_start_batch': lambda state: None,
            'on_end_batch': lambda state: None,
            'on_end': lambda state: None
        }

    def train(self, train_loader, val_loader, epochs, batches_per_epoch, state=None):
        # 初始化状态跟踪
        if state is None:
            state = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'batch': None,
                'epoch': 1,
                'total_batch': 1,
                'epochs': epochs,
                'batches_per_epoch': batches_per_epoch,
                'best_val_loss': float('inf'),
                'best_epoch': 0,
                'early_stopping_triggered': False,
                'train_loss_history': [],
                'val_loss_history': [],
                'train_acc_history': [],
                'val_acc_history': [],
                'grad_norms': [],
                'update_magnitudes': []
            }
        else:
            state.setdefault('grad_norms', [])
            state.setdefault('update_magnitudes', [])

        self.hooks['on_start'](state)
        try:
            while state['epoch'] <= state['epochs'] and not state['early_stopping_triggered']:
                # epoch开始处理
                self.hooks['on_start_epoch'](state)

                # 批次训练循环
                for batch_idx in range(state['batches_per_epoch']):
                    if state['early_stopping_triggered']:
                        break

                    # 获取数据
                    support_batch, query_batch = state['train_loader'].get_batch()
                    state['batch'] = (support_batch, query_batch)
                    state['current_batch'] = batch_idx + 1

                    # 单个批次处理
                    self.hooks['on_start_batch'](state)
                    self.hooks['on_end_batch'](state)

                    state['total_batch'] += 1

                # epoch结束处理
                if not state['early_stopping_triggered']:
                    self.hooks['on_end_epoch'](state)
                    state['epoch'] += 1

        except Exception as e:
            print(f"训练异常终止: {str(e)}")
            state['early_stopping_triggered'] = True
            raise
        finally:
            self.hooks['on_end'](state)

        return state
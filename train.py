# coding=utf-8
# =============================================
# @Time      : 2022-04-21 14:01
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utils import parameter,data_help
import tensorflow as tf
import time
import numpy as np
from utils.networks import MyModel
from utils import gpu_git


# 损失计算
def loss_fun(y_ture, y_pred,loss_object):
    loss_ = loss_object(y_ture, y_pred)
    return tf.reduce_mean(loss_)


# 自适应学习率
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_size, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embedding_size = tf.cast(embedding_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)




def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
    tf.device = gpu_git.check_gpus(mode=args.mode, logger=args.logger)

    # 数据加载
    train_dataset, args = data_help.data_set_alphamind(args)




    # 构建模型
    network = MyModel(
        network_name = args.network_name,
        vocab_size=args.input_vocab_size,
        embedding_size=args.embedding_size,
        rnn_size=args.rnn_size,
        drop_rate=args.dropout_rate,
        num_classes=args.num_calss,
        batch_size=args.batch_size,
        max_len=args.max_seq_length,
    )
    # 优化器
    learing_rate = CustomSchedule(args.embedding_size)  # 自适应学习率
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # 损失
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # 准确率
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


    #ckpt管理器
    ckpt = tf.train.Checkpoint(network=network, optimizer=optimizer)

    # 模型保存
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt,
        directory=args.output_dir,
        max_to_keep=2,
        checkpoint_name=args.model_ckpt_name)
    # 模型恢复
    ckpt.restore(tf.train.latest_checkpoint(args.output_dir))

    args.logger.info(f'epoch num {args.num_epochs}, bach size num {len(train_dataset)*args.num_epochs}')
    # 获取迭代数据
    for epoch in range(args.num_epochs):
        start = time.time()
        # # 重置记录项
        train_loss.reset_states()
        train_accuracy.reset_states()
        for batch, (inputs, targets) in enumerate(train_dataset):
            # 开始训练
            # 基于计算图的求导
            with tf.GradientTape() as tape:
                predictions,scores = network(inputs,training=True)
                loss = loss_fun(targets, predictions,loss_object)
            # 求梯度
            gradients = tape.gradient(loss, network.trainable_variables)
            # 反向传播
            optimizer.apply_gradients(zip(gradients, network.trainable_variables))
            # 记录loss和准确率
            train_loss(loss)
            train_accuracy(targets, predictions)

            if batch % 50 == 0:
                args.logger.info('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()
                ))

        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            args.logger.info('epoch {}, save model at {}'.format(
                epoch + 1, ckpt_save_path
            ))

        args.logger.info('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()
        ))

        args.logger.info('time in 1 epoch:{} secs\n'.format(time.time() - start))



if __name__ == '__main__':
    args = parameter.parser_opt('train')
    train(args)
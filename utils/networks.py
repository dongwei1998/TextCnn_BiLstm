# coding=utf-8
# =============================================
# @Time      : 2022-04-18 17:43
# @Author    : DongWei1998
# @FileName  : networks.py
# @Software  : PyCharm
# =============================================
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, GRU, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D



class RnnLayer(layers.Layer):
    def __init__(self, rnn_size, drop_rate):
        super().__init__()
        # 前向
        fwd_lstm = LSTM(rnn_size, return_sequences=True, go_backwards=False, dropout=drop_rate, name="fwd_lstm")
        # 后向
        bwd_lstm = LSTM(rnn_size, return_sequences=True, go_backwards=True, dropout=drop_rate, name="bwd_lstm")

        self.bilstm = Bidirectional(merge_mode="concat", layer=fwd_lstm, backward_layer=bwd_lstm, name="bilstm")
        # self.bilstm = Bidirectional(LSTM(rnn_size, activation= "relu", return_sequences = True, dropout = drop_rate))

        self.fla = layers.Flatten()

    def call(self, inputs, training):
        out_list = []
        for i in inputs:
            cell_inputs = tf.squeeze(i, axis=2)
            outputs = self.bilstm(cell_inputs, training=training)
            out_list.append(self.fla(outputs))
        return out_list


class GruLayer(layers.Layer):
    def __init__(self, gru_size, drop_rate):
        super().__init__()
        # 前向
        fwd_GRU = GRU(gru_size, return_sequences=True, go_backwards=False, dropout=drop_rate, name="fwd_gru")
        # 后向
        bwd_GRU = GRU(gru_size, return_sequences=True, go_backwards=True, dropout=drop_rate, name="bwd_gru")
        self.bigru = Bidirectional(merge_mode="concat", layer=fwd_GRU, backward_layer=bwd_GRU, name="bigru")
        self.fla = layers.Flatten()

    def call(self, inputs, training):
        out_list = []
        for i in inputs:
            cell_inputs = tf.squeeze(i, axis=2)
            outputs = self.bigru(cell_inputs, training=training)
            out_list.append(self.fla(outputs))
        return out_list


class TextCnnLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.fla = layers.Flatten()  # 一般为卷积网络最近全连接的前一层，用于将数据压缩成一维

        self.l_conv1 = Conv2D(filters=128, kernel_size=(2, 128), activation='relu',
                              padding="valid")  # 现长度 = 1+（原长度-卷积核大小+2*填充层大小） /步长 卷积核的形状（fsz，embedding_size）
        self.l_pool1 = MaxPooling2D(pool_size=(1, 1), padding='same')  # 这里面最大的不同 池化层核的大小与卷积完的数据长度一样

        self.l_conv2 = Conv2D(filters=128, kernel_size=(3, 128), activation='relu', padding="valid")
        self.l_pool2 = MaxPooling2D(pool_size=(1, 1), padding='same')

        self.l_conv3 = Conv2D(filters=128, kernel_size=(4, 128), activation='relu', padding="valid")
        self.l_pool3 = MaxPooling2D(pool_size=(1, 1), padding='same')

    def call(self, inputs):
        embed = tf.expand_dims(inputs, -1)
        convs = []
        l_conv1 = self.l_conv1(embed)
        l_pool1 = self.l_pool1(l_conv1)
        convs.append(l_pool1)

        l_conv2 = self.l_conv2(embed)
        l_pool2 = self.l_pool2(l_conv2)
        convs.append(l_pool2)
        l_conv3 = self.l_conv3(embed)
        l_pool3 = self.l_pool3(l_conv3)
        convs.append(l_pool3)

        # N: 样本数目(批次大小)
        # H: 卷积之后的高度: h = length - filter_height + 1
        # W: 1
        # C: self.num_filters[i]

        return convs


class MyModel(tf.keras.Model):
    def __init__(self, network_name,vocab_size, embedding_size, rnn_size, drop_rate, num_classes, batch_size, max_len):
        super().__init__()
        self.network_name = network_name
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.max_len = max_len

        # 创建embedding层
        self.embedding_layer = Embedding(
            vocab_size,
            embedding_size,
            embeddings_initializer="uniform",
            name="embeding_0")

        # 构建CNN
        self.cnn_layers = TextCnnLayer()

        # 构建BI-LSTM
        self.rnn_layer = RnnLayer(rnn_size, drop_rate)

        self.dropout_layer = Dropout(rate=drop_rate)

        self.dense_layer = Dense(num_classes)
        self.softmax_layer = Dense(num_classes, activation="softmax",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="scores")


    def call(self, input_x, training):
        embedding_output = self.embedding_layer(input_x)
        cnn_layers = self.cnn_layers(embedding_output, training=training)


        outputs = self.rnn_layer(cnn_layers, training=training)


        # 做一个合并
        output = tf.concat(outputs, -1)
        h_drop = self.dropout_layer(output)

        predictions = self.dense_layer(h_drop)
        scores = self.softmax_layer(predictions)



        return predictions,scores




if __name__ == '__main__':
    batch_size = 12
    max_len = 256
    model = MyModel(
        network_name='text_cnn_lstm',
        vocab_size=9000,
        embedding_size=128,
        rnn_size=128,
        drop_rate=0.5,
        num_classes=5,
        batch_size=batch_size,
        max_len=max_len
    )
    inputs = np.array(tf.ones(shape=[batch_size, max_len]))
    training = True
    scores = model(inputs, training)
    print(scores)
    model.summary()

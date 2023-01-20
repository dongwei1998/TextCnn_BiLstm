# coding=utf-8
# =============================================
# @Time      : 2022-04-21 16:15
# @Author    : DongWei1998
# @FileName  : server.py
# @Software  : PyCharm
# =============================================
from flask import Flask, jsonify, request
from utils import parameter,data_help,networks,token_tool
import json
import tensorflow as tf
import jieba

class Predictor(object):
    def __init__(self, args):
        self.args = args

        # 加载class_2_id
        with open(args.label_2_id_dir, 'r', encoding='utf-8') as r:
            label_2_id = json.loads(r.read())
            self.id_2_label = {v:k for k,v in label_2_id.items()}
        # 参数更新
        with open(args.vocab_file, 'r', encoding='utf-8') as r:
            self.input_vocab = r.readlines()
            jieba.load_userdict(self.input_vocab)
        args.input_vocab_size = len(self.input_vocab)
        args.num_calss = len(label_2_id)

        # 加载模型类
        self.network = networks.MyModel(
            network_name=args.network_name,
            vocab_size=args.input_vocab_size,
            embedding_size=args.embedding_size,
            rnn_size=args.rnn_size,
            drop_rate=args.dropout_rate,
            num_classes=args.num_calss,
            batch_size=args.batch_size,
            max_len=args.max_seq_length,
        )
        # 模型恢复
        ckpt = tf.train.Checkpoint(network=self.network)
        ckpt.restore(tf.train.latest_checkpoint(args.output_dir))



    def predict_(self,queries):

        # 数据格式化
        inputs = [word for word in jieba.cut(queries)]
        # 序列截断predictions,scores
        if len(inputs) >= self.args.max_seq_length:
            inputs = inputs[0:self.args.max_seq_length]
        tokenizer = token_tool.FullTokenizer(vocab_file=self.args.vocab_file)
        text = tokenizer.convert_tokens_to_ids(inputs)
        # 序列填充
        while len(text) < self.args.max_seq_length:
            text.append(0)

        data = tf.expand_dims(text, axis=0)

        predictions,scores = self.network(data, False)

        labels_lists = tf.squeeze(tf.cast(scores, dtype=tf.float32),axis=0).numpy().tolist()
        label_idx = labels_lists.index(max(labels_lists))
        label = self.id_2_label[label_idx]
        return label,labels_lists[label_idx]




if __name__ == '__main__':

    app = Flask(__name__)

    app.config['JSON_AS_ASCII'] = False

    model = 'server'
    args = parameter.parser_opt(model)
    detector = Predictor(args)
    args.logger.info("API 启动 开始预测 ...")
    @app.route('/api/v1/classification', methods=['POST'])
    def predict():
        try:

            # 参数获取
            data = request.files
            if 'input' not in data:
                return 'input not exsit', 500
            file = data['input']
            queries = file.read().decode('utf-8')
            data_dict = {
                'text': queries
            }

            # for k, v in infos.items():
            #     data_dict[k] = v

            queries = data_dict['text'].replace('\n', '').replace('\r', '')
            # 参数检查
            if queries is None:
                return jsonify({
                    'code': 500,
                    'msg': '请给定参数text！！！'
                })
            # 直接调用预测的API
            label,predict = detector.predict_(queries)
            return jsonify({
                'code': 200,
                'msg': '成功',
                'text': queries,
                'label':str(label),
                'prob':str(predict)
            })
        except Exception as e:
            args.logger.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 500,
                'msg': '预测数据失败!!!'
            })
    # 启动
    app.run(host='0.0.0.0',port=5557)
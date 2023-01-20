# coding=utf-8
# =============================================
# @Time      : 2022-04-21 14:02
# @Author    : DongWei1998
# @FileName  : data_help.py
# @Software  : PyCharm
# =============================================
from utils import token_tool
import os
import json
import jieba
import tensorflow as tf

def data_pretreatment(text,args):
    vocab_word = []
    if not os.path.exists(args.vocab_file):
        with open(args.vocab_file,'a+',encoding='utf-8') as w:
            for word in jieba.cut(text):
                if word not in vocab_word:
                    vocab_word.append(word)
    else:
        with open(args.vocab_file,'r',encoding='utf-8') as r:
            vocab_word = r.readlines()
            jieba.load_userdict(vocab_word)
    return [word for word in jieba.cut(text)]


def read_data_alphamind(args):
    data_files = args.train_data_files
    lines = []
    label_2_id = {}
    vocab_word = []
    if not os.path.exists(args.vocab_file):
        vocab_obj = open(args.vocab_file, 'w', encoding='utf-8')
        for i, label in enumerate(os.listdir(data_files)):
            label_2_id[label] = i
            f_3 = os.path.join(data_files, label)
            for file in os.listdir(f_3):
                with open(os.path.join(f_3, file), 'r', encoding='utf-8') as r:
                    # 分词
                    w_list = []
                    for word in jieba.cut(r.read()):
                        w_list.append(word)
                        if word not in vocab_word:
                            vocab_word.append(word)
                            vocab_obj.write(word + '\n')
                lines.append([w_list, label])
        return lines,label_2_id
    else:
        with open(args.vocab_file, 'r', encoding='utf-8') as r:
            jieba.load_userdict(r.readlines())
            for i,label in enumerate(os.listdir(data_files)):
                label_2_id[label] = i
                f_3 = os.path.join(data_files, label)
                for file in os.listdir(f_3):
                    with open(os.path.join(f_3, file), 'r', encoding='utf-8') as r:
                        # 分词
                        w_list = []
                        for word in jieba.cut(r.read()):
                            w_list.append(word)
                    lines.append([w_list, label])

        return lines,label_2_id


def data_set_alphamind(args):
    # 读取数据
    lines, label_2_id = read_data_alphamind(args)
    # 文本序列化工具
    tokenizer = token_tool.FullTokenizer(vocab_file=args.vocab_file)
    # label_map 持久化
    if os.path.exists(args.label_2_id_dir):
        with open(args.label_2_id_dir, 'r', encoding='utf-8') as r:
            label_2_id = json.loads(r.read())
    else:
        with open(args.label_2_id_dir, 'w', encoding='utf-8') as w:
            w.write(json.dumps(label_2_id))

    # 文本数据序列化
    data, label = token_tool.text_to_features(lines, label_2_id, args.max_seq_length, tokenizer)
    # 将数据转换为dataset格式
    train_dataset = tf.data.Dataset.from_tensor_slices((data, label))
    # 对数据进行打乱 批次话 buffer_size 总数据量   batch 批次大小
    train_dataset = train_dataset.shuffle(buffer_size=len(data)).batch(args.batch_size)

    # # 参数更新
    with open(args.vocab_file,'r',encoding='utf-8') as r:
        input_vocab = r.readlines()
    args.input_vocab_size = len(input_vocab)
    args.num_calss = len(label_2_id)

    return train_dataset, args


if __name__ == '__main__':
    from utils import parameter
    args = parameter.parser_opt('train')
    train_dataset, args = data_set_alphamind(args)


# coding=utf-8
# =============================================
# @Time      : 2022-04-21 14:18
# @Author    : DongWei1998
# @FileName  : token_tool.py
# @Software  : PyCharm
# =============================================
import collections


class FullTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        output = []
        for item in tokens:
            output.append(self.vocab.get(item, 100))
        return output


    def convert_ids_to_tokens(self, ids):
        output = []
        for item in ids:
            output.append(self.inv_vocab.get(item, 100))
        return output



def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab.get(item, 100))
  return output


def load_vocab(vocab_file):
    # 字典对象 有序的
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf-8') as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


# 文本序列化
def text_to_features(examples, label_2_id, max_seq_length, tokenizer):
    label_data = []
    data = []
    # 遍历训练数据
    for i, (textlist, label) in enumerate(examples):
        # 序列截断
        if len(textlist) >= max_seq_length:
            textlist = textlist[0:max_seq_length]
        input_ids = tokenizer.convert_tokens_to_ids(textlist)  # 将序列中的词(ntokens)转化为ID形式
        # 序列填充
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
        data.append(input_ids)
        label_data.append(label_2_id[label])
    return data, label_data

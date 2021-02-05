import os
import pickle
import json

# Todo: convert data -> dataset


#load data
def load_data_ATIS(dir):
    file = open(dir, 'rb')
    file_json = json.load(file)['rasa_nlu_data']["common_examples"]
    tokens = []
    tags = []
    seqlen = []
    for i in range(len(file_json)):
        tokens.append(file_json[i]['text'].split())
        tags.append(file_json[i]["intent"])
        seqlen.append(len(file_json[i]['text'].split()))
    return tokens, tags, seqlen

def get_maxlen(all_tokens):
    max = 0
    for i in range(len(all_tokens)):
        tmp = len(all_tokens[i])
        if tmp > max:
            max = tmp
    return max


def create_dict(tokens, tags):
    word2idx = {}
    idx2word = {}
    tag2idx = {}
    idx2tag = {}
    idx = 1
    idy = 1
    word2idx['<pad>'] = 0
    idx2word[0] = '<pad>'
    tag2idx['<pad>'] = 0
    idx2tag[0] = ['<pad>']
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            word = tokens[i][j]
            if not word in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
        tag = tags[i]
        if not tag in tag2idx:
            tag2idx[tag] = idy
            idx2tag[idy] = tag
            idy += 1
    return word2idx, idx2word, tag2idx, idx2tag


def add_padding(tokens, maxlen):
    for i in range(len(tokens)):
        for j in range(maxlen - len(tokens[i])):
            tokens[i].append('<pad>')
    return tokens


def convert2vec(tokens, tags, word2idx, tag2idx):
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            tokens[i][j] = word2idx[tokens[i][j]]
        tags[i] = tag2idx[tags[i]]
    return tokens, tags


def convertback(vec_tokens, vec_tags, idx2word, idx2tag):
    for i in range(len(vec_tokens)):
        for j in range(len(vec_tokens[i])):
            vec_tokens[i][j] = idx2word[vec_tokens[i][j]]
        vec_tags[i] = idx2tag[vec_tags[i]]
    return vec_tokens, vec_tags

def get_all(tokens1, tokens2, tags1, tags2):
    all_tokens = []
    all_tags = []
    for i in range(len(tokens1)):
        all_tokens.append(tokens1[i])
        all_tags.append(tags1[i])
    for j in range(len(tokens2)):
        all_tokens.append(tokens2[j])
        all_tags.append(tags2[j])
    return all_tokens, all_tags


if __name__ == '__main__':
    dir = '/p300/MerryUp_RE2RNN/dataset/ATIS/train.json'
    tokens, tags, seqlen = load_data_ATIS(dir)
    print(tokens[0])
    print(tags[0])
    print(seqlen[0])

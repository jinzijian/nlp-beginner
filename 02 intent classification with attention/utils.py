import torch
from copy import deepcopy
import numpy as np
import random

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def collate_fn(batch):
    for i in range(len(batch)):
        token = batch[i]['input']
        label = batch[i]['label'][i]
    return token, label


def flatten(input, length):
    """
    :param input: B x L x ?
    :param length: B
    :return:
    """

    B, L = input.size()[0], input.size()[1]
    # fraction = 1
    flattened = torch.cat([input[i,:length[i]] for i in range(B)], dim=0)
    return flattened


class Best_Model_Recorder():
    def __init__(self, selector='f', level='token-level', init_results_train=None, init_results_dev=None, init_results_test=None, save_model=False):

        selector_pool = ['p', 'r', 'f']
        assert selector in selector_pool
        assert level in ['token-level', 'entity-level']

        self.best_dev_results = init_results_dev
        self.best_dev_train_results = init_results_train
        self.best_dev_test_results = init_results_test
        self.selector = selector_pool.index(selector) + 1
        self.level = level
        self.best_selector = self.best_dev_results[self.level][self.selector]
        self.best_model_state_dict = None
        self.save_model = save_model

    def update_and_record(self, results_train, results_dev, results_test, model_state_dict,):
        temp_selector_value = results_dev[self.level][self.selector]
        if temp_selector_value > self.best_selector:
            self.best_selector = temp_selector_value
            self.best_dev_results = results_dev
            self.best_dev_test_results = results_test
            self.best_dev_train_results = results_train

            if self.save_model:
                self.best_model_state_dict = deepcopy(model_state_dict)


# -*- encoding:utf-8 -*-
import torch


def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_model(model, model_path):
    if hasattr(model, "module"):  #moudle 模型框架
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    else:  # 单纯存参数
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    return model

def load_embedding(path, idx2word):
    words = []
    values = []
    embdict = {}
    with open(path, encoding='utf-8') as embs:
        for line in embs:
            if line:
                line = line.rstrip()
                parts = line.split(' ')
                words.append(parts[0])
                embdict[parts[0]] = parts[1:]
    for i in range(len(idx2word)):
        inp = idx2word[i]
        if inp in embdict:
            value = embdict[inp]
            values.append([float(x) for x in value])
        # todo
        else:
            values.append(np.random.rand(100))
    return np.asarray(values)


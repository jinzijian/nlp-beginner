from model import baseModel
from torch import nn
import torch
import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import eval


class trainer():
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, epochs, word2idx, tag2idx, idx2word,
                 idx2tag, use_gpu):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2word = idx2word
        self.idx2tag = idx2tag
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = self.model.cuda()

    def train(self):
        for i in range(self.epochs):
            avg_loss = 0
            self.model.train()
            for idx, batch in enumerate(self.train_dataloader):
                self.model.zero_grad()
                input = batch['input']  # B * L
                label = batch['label']  # B
                seqlen = batch['length']
                # 要把list of tensor 转化为tensor
                input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                # label = torch.vstack(label)  # B  tensor
                if self.use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                output = self.model.forward(input, seqlen)  # B  * C
                loss_fn = nn.CrossEntropyLoss()
                # cross entropy needs flatten
                #output = utils.flatten(output, seqlen)
                #label = utils.flatten(label, seqlen)
                loss = loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            avg_loss = avg_loss / len(self.train_dataloader)
            print('epoch %d' % i)
            print("loss is %f" % avg_loss)

            #  evaluate train result
            self.model.eval()
            all_pred_label = []
            all_true_label = []
            with torch.no_grad():
                for idx, batch in enumerate(self.train_dataloader):
                    input = batch['input']
                    label = batch['label']
                    seqlen = batch['length']
                    # 要把list of tensor 转化为tensor
                    input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                    #label = torch.vstack(label)
                    if self.use_gpu:
                        input = input.cuda()
                        label = label.cuda()
                    output = self.model.predict(input, seqlen)  # B * L
                    # 为什么需要flatten
                    # label = utils.flatten(label, seqlen)
                    # output = utils.flatten(output, seqlen)
                    all_true_label.append(label)
                    all_pred_label.append(output)
            all_pred_label = torch.hstack(all_pred_label)
            all_true_label = torch.hstack(all_true_label)
            acc = eval.evaluate(all_pred=all_pred_label, all_label=all_true_label)
            print('Train, ACC: {} '.format(acc))
            print(all_pred_label[:5])
            print(all_true_label[:5])

            #  evaluate test result
            self.model.eval()
            all_pred_label = []
            all_true_label = []
            with torch.no_grad():
                for idx, batch in enumerate(self.test_dataloader):
                    input = batch['input']
                    label = batch['label']
                    seqlen = batch['length']
                    # 要把list of tensor 转化为tensor
                    input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                    # label = torch.vstack(label)
                    if self.use_gpu:
                        input = input.cuda()
                        label = label.cuda()
                    output = self.model.predict(input, seqlen)  # B * L
                    # 为什么需要flatten
                    # label = utils.flatten(label, seqlen)
                    # output = utils.flatten(output, seqlen)
                    all_true_label.append(label)
                    all_pred_label.append(output)
            all_pred_label = torch.hstack(all_pred_label)
            all_true_label = torch.hstack(all_true_label)
            acc = eval.evaluate(all_pred=all_pred_label, all_label=all_true_label)
            print('Test, ACC: {} '.format(acc))
            print(all_pred_label[:5])
            print(all_true_label[:5])
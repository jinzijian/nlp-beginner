from model import bilstm_crf
from torch import nn
import torch
import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import utils
import eval

class trainer():
    def __init__(self, model, train_dataloader, dev_dataloader, test_dataloader, optimizer, epochs, word2idx, tag2idx, idx2word, idx2tag, use_gpu, o_idx):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2word = idx2word
        self.idx2tag = idx2tag
        self.use_gpu = use_gpu
        self.o_idx = o_idx
        if self.use_gpu:
            self.model = self.model.cuda()

    def train(self):
        for i in range(self.epochs):
            all_pred_label = []
            all_true_label = []
            avg_loss = 0
            for idx, batch in enumerate(self.train_dataloader):
                self.model.zero_grad()
                input = batch['input']
                label = batch['label']
                seqlen = batch['length']
                # 要把list of tensor 转化为tensor
                input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                label = torch.vstack(label).transpose(0,1)  # B * L tensor
                if self.use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                output = self.model.forward(input, seqlen) # B * L * C
                loss_fn = nn.CrossEntropyLoss()
                # cross entropy needs flatten
                output = utils.flatten(output, seqlen)
                label = utils.flatten(label, seqlen)
                loss = loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            avg_loss = avg_loss / len(self.train_dataloader)
            print('epoch %d' % i)
            print("loss is %f" % avg_loss)

            #train
            all_pred_label = []
            all_true_label = []
            for idx, batch in enumerate(self.train_dataloader):
                input = batch['input']
                label = batch['label']
                seqlen = batch['length']
                # 要把list of tensor 转化为tensor
                input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                label = torch.vstack(label).transpose(0,1)  # B * L tensor
                if self.use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                output = self.model.predict(input, seqlen)  # B * L
                label = utils.flatten(label, seqlen)
                output = utils.flatten(output, seqlen)
                all_true_label.append(label)
                all_pred_label.append(output)
            all_pred_label = torch.hstack(all_pred_label)
            all_true_label = torch.hstack(all_true_label)
            # evaluate
            i2s = None
            acc, p, r, f = eval.eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label, o_idx=self.o_idx)
            print("result in train ,acc '{0}', precision '{1}', recall '{2}', f1 score '{3}'".format(acc, p, r, f))

            # dev
            all_pred_label = []
            all_true_label = []
            for idx, batch in enumerate(self.dev_dataloader):
                input = batch['input']
                label = batch['label']
                seqlen = batch['length']
                # 要把list of tensor 转化为tensor
                input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                label = torch.vstack(label).transpose(0, 1)  # B * L tensor
                if self.use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                output = self.model.predict(input, seqlen)  # B * L
                label = utils.flatten(label, seqlen)
                output = utils.flatten(output, seqlen)
                all_true_label.append(label)
                all_pred_label.append(output)
            all_pred_label = torch.hstack(all_pred_label)
            all_true_label = torch.hstack(all_true_label)
            # evaluate
            i2s = None
            acc, p, r, f = eval.eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label,
                                               o_idx=self.o_idx)
            print("result in dev ,acc '{0}', precision '{1}', recall '{2}', f1 score '{3}'".format(acc, p, r, f))

            # test

            all_pred_label = []
            all_true_label = []
            for idx, batch in enumerate(self.test_dataloader):
                input = batch['input']
                label = batch['label']
                seqlen = batch['length']
                # 要把list of tensor 转化为tensor
                input = torch.vstack(input).transpose(0, 1)  # B * L tensor
                label = torch.vstack(label).transpose(0, 1)  # B * L tensor
                if self.use_gpu:
                    input = input.cuda()
                    label = label.cuda()
                with torch.no_grad():
                    output = self.model.predict(input, seqlen)  # B * L
                label = utils.flatten(label, seqlen)
                output = utils.flatten(output, seqlen)
                all_true_label.append(label)
                all_pred_label.append(output)
            all_pred_label = torch.hstack(all_pred_label)
            all_true_label = torch.hstack(all_true_label)
            # evaluate
            i2s = None
            acc, p, r, f = eval.eval_seq_token(seq_label_pred=all_pred_label, seq_label_true=all_true_label,
                                               o_idx=self.o_idx)
            print("result in test ,acc '{0}', precision '{1}', recall '{2}', f1 score '{3}'".format(acc, p, r, f))




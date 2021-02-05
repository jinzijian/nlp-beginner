from torch.utils.data import DataLoader
import data
from model import bilstm_crf
from dataset import nerDataSet
import argparse
from torch import optim
from train import trainer
import utils
import torch


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=20, type=int, help='should be int')
    parser.add_argument('--hidden_dim', default=150, type=int, help='should be int')
    parser.add_argument('--embedding_dim', default=100, type=int, help='should be int')
    parser.add_argument('--epochs', default=30, type=int, help='should be int')

    args = parser.parse_args()
    # assert args
    pass


    # gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("using gpu")
        print('changed')
    # embeddding
    emb_path = '/p300/TensorFSARNN/data/emb/glove.6B/glove.6B.100d.txt'

    # data file
    dir = '/p300/nlp-beginner-by-jzj/dataset/conll03/'
    train_file = dir + 'eng.train'
    dev_file = dir + 'eng.testa'
    test_file = dir + 'eng.testb'

    # convert data
    data.create_list(test_file)
    train_tokens, train_tags = data.create_list(train_file)
    dev_tokens, dev_tags = data.create_list(dev_file)
    test_tokens, test_tags = data.create_list(test_file)

    # get all tokens
    all_tokens = []
    all_tags = []
    all_tokens = data.add_all(all_tokens, train_tokens)
    all_tokens = data.add_all(all_tokens, dev_tokens)
    all_tokens = data.add_all(all_tokens, test_tokens)
    # get all tags
    all_tags = data.add_all(all_tags, train_tags)
    all_tags = data.add_all(all_tags, dev_tags)
    all_tags = data.add_all(all_tags, test_tags)

    # get dict
    word2idx, idx2word, tag2idx, idx2tag = data.create_dic(all_tokens, all_tags)
    o_idx = tag2idx['O']
    
    train_tokens, train_seqlen = data.add_padding(train_tokens)
    test_tokens, test_seqlen = data.add_padding(test_tokens)
    dev_tokens, dev_seqlen = data.add_padding(dev_tokens)
    train_tags, train_tag_seqlen = data.add_padding(train_tags)
    dev_tags, dev_tag_seqlen = data.add_padding(dev_tags)
    test_tags, test_tag_seqlen = data.add_padding(test_tags)

    train_tokens, train_tags = data.convert2vec(train_tokens, train_tags, word2idx, tag2idx)
    dev_tokens, dev_tags = data.convert2vec(dev_tokens, dev_tags, word2idx, tag2idx)
    test_tokens, test_tags = data.convert2vec(test_tokens, test_tags, word2idx, tag2idx)

    # create dataset and dataloader
    train_dataset = nerDataSet(train_tokens, train_tags, train_seqlen)
    dev_dataset = nerDataSet(dev_tokens, dev_tags, dev_seqlen)
    test_dataset = nerDataSet(test_tokens, test_tags, test_seqlen)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size)
    dev_data = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size)

    # init model
    nerModel = bilstm_crf(vocab_size=len(word2idx), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                          tag2idx=tag2idx, batch_size=args.batch_size, use_gpu = use_gpu, idx2word=idx2word, emb_path=emb_path)
    optimizer = optim.SGD(nerModel.parameters(), lr=0.01, weight_decay=1e-4)

    # train
    nerTrainer = trainer(model=nerModel, train_dataloader=train_data, dev_dataloader=dev_data, test_dataloader=test_data, optimizer=optimizer, epochs=args.epochs, word2idx=word2idx,
                         idx2word=idx2word, tag2idx=tag2idx, idx2tag=idx2tag, use_gpu=use_gpu, o_idx=o_idx)
    nerTrainer.train()

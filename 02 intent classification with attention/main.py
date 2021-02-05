from dataset import myDataSet
from torch.utils.data import DataLoader
import data
import argparse
import torch
from torch import optim
from model import baseModel, AttentionModel
from train import trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--batch_size', default=16, type=int, help='should be int')
    parser.add_argument('--epochs', default=150, type=int, help='should be int')
    # model
    parser.add_argument('--embedding_dim', default=100, type=int, help='should be int')
    parser.add_argument('--hidden_dim', default=200, type=int, help='should be int')
    parser.add_argument('--mode', default='attention', type=str, help='should be str')
    parser.add_argument('--lr', default='0.001', type=float, help='should be float')

    args = parser.parse_args()
    pass

    # Cuda
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using gpu")
    else:
        print('Using cpu')

    # embedding path
    emb_path = '/p300/TensorFSARNN/data/emb/glove.6B/glove.6B.100d.txt'
    #  load data
    dir = '/p300/MerryUp_RE2RNN/dataset/ATIS/'
    train_tokens, train_tags, train_seqlen = data.load_data_ATIS(dir + 'train.json')
    test_tokens, test_tags, test_seqlen = data.load_data_ATIS(dir + 'test.json')
    # get all
    all_tokens, all_tags = data.get_all(train_tokens, test_tokens, train_tags, test_tags)
    #  get max
    max = data.get_maxlen(all_tokens)
    # get dict
    word2idx, idx2word, tag2idx, idx2tag = data.create_dict(all_tokens, all_tags)
    vocab_size = len(word2idx) # vocab_size = 942
    tag_size = len(tag2idx)  # C = 27
    # add padding
    train_tokens = data.add_padding(train_tokens, max)
    test_tokens = data.add_padding(test_tokens, max)
    # convert2vec
    train_tokens, train_tags = data.convert2vec(train_tokens, train_tags, word2idx, tag2idx)
    test_tokens, test_tags = data.convert2vec(test_tokens, test_tags, word2idx=word2idx, tag2idx = tag2idx)
    # dataset
    train_dataset = myDataSet(train_tokens, train_tags, train_seqlen)
    test_dataset = myDataSet(test_tokens, test_tags, test_seqlen)
    # dataloader
    train_data = DataLoader(train_dataset, batch_size=args.batch_size)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size)

    # model
    baseModel = baseModel(vocab_size=vocab_size, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, tag2idx=tag2idx,
                          batch_size=args.batch_size, use_gpu=use_gpu, idx2word=idx2word, emb_path=emb_path)
    attentionModel = AttentionModel(vocab_size=vocab_size, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, tag2idx=tag2idx,
                          batch_size=args.batch_size, use_gpu=use_gpu, idx2word=idx2word, emb_path=emb_path)
    optimizer = optim.Adam(attentionModel.parameters(), lr=args.lr)

    # trainer
    if args.mode == 'base':
        myTrainer = trainer(model=baseModel, train_dataloader=train_data, test_dataloader=test_data, optimizer=optimizer,
                            epochs=args.epochs, word2idx=word2idx, tag2idx=tag2idx, idx2word=idx2word, idx2tag=idx2tag, use_gpu=use_gpu)
    if args.mode == 'attention':
        myTrainer = trainer(model=attentionModel, train_dataloader=train_data, test_dataloader=test_data, optimizer=optimizer,
                            epochs=args.epochs, word2idx=word2idx, tag2idx=tag2idx, idx2word=idx2word, idx2tag=idx2tag, use_gpu=use_gpu)
    else:
        print('not right mode')
    myTrainer.train()

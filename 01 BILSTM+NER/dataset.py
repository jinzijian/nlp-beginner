from torch.utils.data import Dataset, DataLoader
import data


class nerDataSet(Dataset):
    def __init__(self, tokens, tags, seqlen):
        self.tokens = tokens
        self.tags = tags
        self.seqlen = seqlen

    def __getitem__(self, idx):
        return {
            'input': self.tokens[idx],
            'label': self.tags[idx],
            'length': self.seqlen[idx]
        }

    def __len__(self):
        return len(self.tokens)




if __name__ == '__main__':
    dir = '/Users/jinlinlin/Desktop/nlp-beginner-by-jzj/dataset/conll03/'
    file_name = dir + 'eng.train'
    samples = data.create_list(file_name)
    samples = data.add_padding(samples)
    word2idx, idx2word = data.create_dic(samples[:3])
    nerDataSet = nerDataSet(samples)
    print(nerDataSet.__getitem__(4))

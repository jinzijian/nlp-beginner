from torch.utils.data import Dataset, DataLoader
import data


class myDataSet(Dataset):
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
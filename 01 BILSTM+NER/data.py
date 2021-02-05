def create_list(file_name):
    data = []
    tokens = []
    tags = []
    token = []
    tag = []
    with open(file_name, 'r') as f:
        line = f.readline()
        while line:
            data.append(line)
            line = f.readline()
    for i in range(len(data)):
        tmp = data[i]
        if tmp == '\n' and len(token) != 0:
            tokens.append(token)
            tags.append(tag)
            token = []
            tag = []
        else:
            x = tmp.split()
            token.append(x[0])
            tag.append(x[-1])
    return tokens[1:], tags[1:]


def create_dic(tokens, tags):
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
        for k in range(len(tags[i])):
            tag = tags[i][k]
            if not tag in tag2idx:
                tag2idx[tag] = idy
                idx2tag[idy] = tag
                idy += 1
    return word2idx, idx2word, tag2idx, idx2tag


def get_seqlen():
    lens = []
    for i in range(len(tokens)):
        lens.append(len(tokens))
    return lens


def add_padding(tokens):
    seq_len = []
    max = 0
    for i in range(len(tokens)):
        if len(tokens[i]) > max:
            max = len(tokens[i])
        seq_len.append(len(tokens[i]))
    for i in range(len(tokens)):
        for j in range(max - len(tokens[i])):
            tokens[i].append('<pad>')
    return tokens, seq_len


def convert2vec(tokens, tags, word2idx, tag2idx):
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            tokens[i][j] = word2idx[tokens[i][j]]
        for k in range(len(tags[i])):
            tags[i][k] = tag2idx[tags[i][k]]
    return tokens, tags


def convertback(vec_tokens, vec_tags, idx2word, idx2tag):
    for i in range(len(vec_tokens)):
        for j in range(len(vec_tokens[i])):
            vec_tokens[i][j] = idx2word[vec_tokens[i][j]]
        for k in range(len(vec_tags[i])):
            vec_tags[i][k] = idx2tag[vec_tags[i][k]]
    return vec_tokens, vec_tags


def add_all(all_tokens, tokens):
    for token in tokens:
        all_tokens.append(token)
    return all_tokens


if __name__ == '__main__':
    dir = '/Users/jinlinlin/Desktop/nlp-beginner-by-jzj/dataset/conll03/'
    file_name = dir + 'eng.train'
    # data = []
    # with open(file_name, 'r') as f:
    #     line = f.readline()
    #     while line:
    #         data.append(line)
    #         line = f.readline()
    tokens, tags = create_list(file_name)
    print(tokens[:3])
    print(tags[:3])
    tokens = add_padding(tokens)
    print(tokens[:3])

    word2idx, idx2word, tag2idx, idx2tag = create_dic(tokens, tags)
    tokens, tags = convert2vec(tokens, tags, word2idx, tag2idx)
    print(tokens[:3])
    print(tags[:4])
    tokens, tags = convertback(tokens, tags, idx2word, idx2tag)
    print(tokens[:3], tags[:3])
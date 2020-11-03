import pickle
import numpy as np
# from mxnet.contrib import text
import torch.utils.data as data_utils
import torch


def build_vec(file_in, file_out):
    fp = open(file_in, 'r', encoding='utf-8')
    op = open(file_out, 'wb')

    lines = fp.readlines()
    vectors = []
    word_idx = 0
    word2id = {}
    for line in lines:

        tokens = line.rstrip().split(' ')
        word = tokens[0]

        vectors.append(np.asarray([float(x) for x in tokens[1:]]))
        word2id[word] = word_idx
        word_idx += 1
    vectors = np.asarray(vectors)
    pickle.dump(vectors, op)
    pickle.dump(word2id, op)


def load_data(batch_size=64):
    X_tst = np.load("./data/AAPD/X_test.npy")
    X_trn = np.load("./data/AAPD/X_train.npy")
    Y_trn = np.load("./data/AAPD/y_train.npy")
    Y_tst = np.load("./data/AAPD/y_test.npy")
    label_embed = np.load("./data/AAPD/label_embed.npy")
    # embed = text.embedding.CustomEmbedding('/data/AAPD/word_embed.txt')
    fp = open('./data/AAPD/word_embed.pkl','rb')
    embed = pickle.load(fp)
    word2id = pickle.load(fp)

    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    return train_loader, test_loader, label_embed, embed, X_tst, word2id, Y_tst, Y_trn

def load_my_data(batch_size=64):
    dev_fp = open("./data/mul_v0_0_3/dev.pkl", 'rb')
    train_fp = open("./data/mul_v0_0_3/train.pkl", 'rb')
    X_tst = pickle.load(dev_fp)
    Y_tst = pickle.load(dev_fp)
    X_trn = pickle.load(train_fp)
    Y_trn = pickle.load(train_fp)
    fp = open('./data/mul_v0_0_3/vec.pkl', 'rb')
    embed = pickle.load(fp)

    dict_fp = open('./data/mul_v0_0_3/label.dict', 'rb')
    label2id = pickle.load(dict_fp)
    id2label = pickle.load(dict_fp)
    word2id = pickle.load(dict_fp)
    id2word = pickle.load(dict_fp)

    class_num = len(label2id)
    label_embed = torch.randn([class_num, 300]).float()

    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)

    return train_loader, test_loader, label_embed, embed, X_tst, word2id, Y_tst, Y_trn, id2label


build_vec('./data/AAPD/word_embed.txt','./data/AAPD/word_embed.pkl')
# load_data(batch_size=64)
load_my_data(batch_size=64)



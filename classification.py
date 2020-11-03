from attention.model import StructuredSelfAttention
from attention.train import train
import torch
import utils
import data_got
config = utils.read_config("config.yml")
if config.GPU:
    torch.cuda.set_device(0)
print('loading data...\n')
label_num = 9
train_loader, test_loader, label_embed,embed,X_tst,word_to_id,Y_tst,Y_trn,id2label = data_got.load_my_data(batch_size=config.batch_size)
# label_embed = torch.from_numpy(label_embed).float()  # [L*256]
embed = torch.from_numpy(embed).float()
print("load done")

def multilabel_classification(attention_model, train_loader, test_loader, epochs, GPU=True):
    loss = torch.nn.BCELoss()
    opt = torch.optim.Adam(attention_model.parameters(), lr=0.001, betas=(0.9, 0.99))
    train(attention_model, train_loader, test_loader, loss, opt, epochs, GPU,id2label)

attention_model = StructuredSelfAttention(batch_size=config.batch_size, lstm_hid_dim=config['lstm_hidden_dimension'],
                                          d_a=config["d_a"], n_classes=label_num, label_embed=label_embed,embeddings=embed)
if config.use_cuda:
    attention_model.cuda()
multilabel_classification(attention_model, train_loader, test_loader, epochs=config["epochs"])

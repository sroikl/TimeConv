import torch
from torch.utils import data
from Configuration import parse_args,DATA_DIR,LABEL_DIR
from DataLoader import DataLoader
from Model import TemporalSpatialModel
from Training import TCNTrainer
import numpy as np
from torch.utils.data.sampler import BatchSampler,SequentialSampler

def runTCN(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============================== DATA LOADING =========================

    data = DataLoader(DATA_DIR=DATA_DIR, LABEL_DIR=LABEL_DIR, device=device, padsize=args.padsize, usefull=True)
    X, y = data.LoadData()

    # NumSamples= X.shape[0]
    # num_full_batches= NumSamples//args.batch_size
    # X= X[:num_full_batches * args.batch_size,:] ; y= y[:num_full_batches * args.batch_size,:]

    # X=X.view(-1, *X.shape[2:]) ; y=y.view(-1) #This line is for flattening the Tensor
    # X=X[100:400,:]; y=y[100:400,:]
    validation_split = 0.3;

    dataset = torch.utils.data.TensorDataset(X, y)
    dl_train, dl_test = Split_Test_Train(dataset=dataset, validation_split=validation_split,
                                         batch_size=args.batch_size)

    print(f'Sampels Shape is:%s' % {next(iter(dl_train))[0].shape})
    print(f'Label Shape is:%s' % {next(iter(dl_train))[1].shape})

    model= TemporalSpatialModel(num_levels=10,num_hidden=1000,embedding_size=160,kernel_size=3,dropout=0.2)
    optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)
    loss_fn = loss()

    Trainer= TCNTrainer(model=model,loss_fn=loss_fn,optimizer=optimizer,device=device)
    Trainer.fit(dl_train=dl_train,dl_test=dl_test,num_epochs=args.Epochs)

def Split_Test_Train(dataset,validation_split,batch_size):

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    num_full_batches_train= len(train_indices)//batch_size
    num_full_batches_test= len(val_indices)//batch_size

    train_indices= train_indices[:int(num_full_batches_train*batch_size)]
    val_indices= val_indices[:int(num_full_batches_test*batch_size)]

    # Creating PT data samplers and loaders:
    train_sampler = SequentialSampler(train_indices)
    valid_sampler = SequentialSampler(val_indices)

    dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, shuffle=False)
    dl_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          sampler=valid_sampler, shuffle=False)

    return dl_train,dl_test



def loss():
    loss = torch.nn.MSELoss(reduction='mean')

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss
if __name__ == '__main__':
    args = parse_args()
    runTCN(args)
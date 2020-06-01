import torch
from torch.utils import data
from OldVersionCode.Configuration import parse_args,DATA_DIR,LABEL_DIR,SAVE_DIR
from OldVersionCode.DataLoader import DataLoader
from OldVersionCode.Model import TemporalSpatialModel
from OldVersionCode.Training import TCNTrainer
import numpy as np
from torch.utils.data.sampler import BatchSampler,SequentialSampler
import os
from random import choices
import itertools

def runTCN(args):

    RunInference=True
    # RunInference=False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(SAVE_DIR)
    your_data = {"Epochs":args.Epochs, 'lr':args.lr,'Weight_Decay':args.weight_decay,'TCN_LEVELS':args.num_levels,'Hidden Dimension':args.num_hidden,
                 'Kernel_Size':args.kernel_size,'Dropout':args.dropout,'embedding_Size':args.embedding_size2}

    print(your_data, file=open(os.path.join(SAVE_DIR,'RunData.txt'), 'w'))

    # ============================== DATA LOADING =========================

    data = DataLoader(DATA_DIR=DATA_DIR, LABEL_DIR=LABEL_DIR, device=device, padsize=args.padsize, usefull=True)
    X, y = data.LoadData()

    validation_split = 0.3;

    dataset = torch.utils.data.TensorDataset(X, y)
    dl_train, dl_test = Split_Test_Train(dataset=dataset, validation_split=validation_split,
                                         batch_size=args.batch_size)

    print(f'Sampels Shape is:%s' % {next(iter(dl_train))[0].shape})
    print(f'Label Shape is:%s' % {next(iter(dl_train))[1].shape})

    model= TemporalSpatialModel(num_levels=args.num_levels,num_hidden=args.num_hidden,embedding_size2=args.embedding_size2,
                                kernel_size=args.kernel_size,dropout=args.dropout,numplants=args.NumPlants
                                ,batch_size=args.batch_size,embedding_size1=args.embedding_size1).to(device=device)

    optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.999), lr=args.lr)

    loss_fn = loss()

    Trainer= TCNTrainer(model=model,loss_fn=loss_fn,optimizer=optimizer,device=device)
    Trainer.fit(dl_train=dl_train,dl_test=dl_test,num_epochs=args.Epochs)
    torch.save(model.sa,f'{os.getcwd()}/Model.pt')
def Split_Test_Train(dataset,validation_split,batch_size):

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    num_full_batches= dataset_size//batch_size

    indices= np.arange(num_full_batches*batch_size)
    indices= np.reshape(indices,(num_full_batches,batch_size))

    num_full_batches_test= int(num_full_batches*validation_split)

    batch_indices_test= choices(np.arange(0,num_full_batches),k=num_full_batches_test)
    batch_indices_train= [ val for val in np.arange(0,num_full_batches) if val not in batch_indices_test]

    train_indices= indices[batch_indices_train]
    test_indices= indices[batch_indices_test]

    # Creating PT data samplers and loaders:
    train_sampler = BatchSampler(SequentialSampler(list(itertools.chain(*train_indices))),batch_size=batch_size,drop_last=True)
    valid_sampler = BatchSampler(SequentialSampler(list(itertools.chain(*test_indices))),batch_size=batch_size,drop_last=True)

    dl_train = torch.utils.data.DataLoader(dataset,
                                           sampler=train_sampler, shuffle=False)
    dl_test = torch.utils.data.DataLoader(dataset,
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
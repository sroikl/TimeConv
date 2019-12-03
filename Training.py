from torch.utils.data import DataLoader
import abc
import tqdm
import torch
import os
import numpy as np
from Configuration import SAVE_DIR

class Trainer:
    def __init__(self,SAVE_DIR):
        self.save_dir= os.path.join(SAVE_DIR,'Results.pt')

    def fit(self,dl_train:DataLoader,dl_test:DataLoader,num_epochs:int,**kwargs):

        loss_train,loss_test = [],[]

        for Epoch in range(num_epochs):

            print(f'--- EPOCH {Epoch + 1}/{num_epochs} ---')

            loss_tr_epoch = self._train_epoch(dl_train,**kwargs)
            loss_train.append(np.mean(loss_tr_epoch))

            loss_ts_epoch = self._test_epoch(dl_test,**kwargs)
            loss_test.append(np.mean(loss_ts_epoch))

            #Saving data
            if Epoch%5==0:
                saved_state= dict(loss_train=loss_train,loss_test=loss_test)
                torch.save(saved_state,os.path.join(SAVE_DIR,'Results.pt'))




        #TODO: add features such as checkpoints, early stopping etc.


    def _train_epoch(self,dl_train,**kwargs):
        return self._ForBatch(dl_train,self.train_batch,**kwargs)
    def _test_epoch(self,dl_test,**kwargs):
        return self._ForBatch(dl_test, self.test_batch, **kwargs)

    @abc.abstractmethod
    def train_batch(self, batch):
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch):
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _ForBatch(dl:DataLoader,forward_fn):

        losses = []
        num_batches = len(dl.batch_sampler)

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_loss = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_loss:.3f})')
                pbar.update()

                losses.append(batch_loss)

            avg_loss = sum(losses) / num_batches
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f},')

        return losses

class TCNTrainer(Trainer):

    def __init__(self, model,loss_fn, optimizer ,device=None):
        super(Trainer, self).__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_batch(self, batch):
        X, y = batch
        x = X.squeeze(dim=0).transpose(0, 1).to(self.device)
        y = y.squeeze(dim=0).transpose(0, 1).to(self.device)

        self.optimizer.zero_grad()
        #Forward Pass
        output= self.model(x)

        # Compute Loss
        loss= self.loss_fn(output,y)

        #Back Prop
        loss.backward()

        #Update params
        self.optimizer.step()


        return loss.item()

    def test_batch(self, batch):
        X, y = batch
        x = X.squeeze(dim=0).transpose(0, 1).to(self.device)
        y = y.squeeze(dim=0).transpose(0, 1).to(self.device)

        with torch.no_grad():
            # Forward Pass
            output = self.model(x)

            # Zero Grad
            loss = self.loss_fn(output, y)


        return loss.item()
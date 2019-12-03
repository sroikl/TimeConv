from matplotlib import pyplot as plt
from Configuration import SAVE_DIR
import os
import torch
import numpy as np
def PlotResults():
    LoadPath= os.path.join('/Users/roiklein/Dropbox/Msc Project/Deep Learning Project/TimeSeriesAnalysis','Results.pt')
    LoadPath2= os.path.join('/Users/roiklein/Dropbox/Msc Project/Deep Learning Project/TimeSeriesAnalysis','Results2.pt')

    saved_results1= torch.load(LoadPath)
    train_loss= np.sqrt(saved_results1['loss_train']); test_loss= np.sqrt(saved_results1['loss_test'])

    saved_results2 = torch.load(LoadPath2)
    train_loss2 = np.sqrt(saved_results2['loss_train']); test_loss2 = np.sqrt(saved_results2['loss_test'])

    plt.figure()
    plt.title('Exp 1000 - Transpiration\n Training on Regression Task')
    plt.plot(train_loss[:len(train_loss2)],label='With 1st Normalization')
    plt.plot(train_loss2,label='With Erez Normalization')
    # plt.plot(test_loss,label= 'test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()

    plt.figure()
    plt.title('Exp 1000 - Transpiration\n Training on Regression Task - with Erez Normalization')
    plt.plot(train_loss2, label='Train')
    plt.plot(test_loss2,label= 'test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()


if __name__ == '__main__':
    PlotResults()
    plt.show()
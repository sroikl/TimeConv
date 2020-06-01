import torch
import numpy as np

def Inference(model,dl_test):

    for x,_ in dl_test:
        output= model()



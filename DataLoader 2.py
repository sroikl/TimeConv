import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import torch
from matplotlib import pyplot
from Configuration import pixelfulldict,dry_wet_cloth,start_date
from torchvision import transforms
import os
from datetime import datetime

class ImageLoader:

    def __init__(self,DATA_DIR:str,LABEL_DIR:str
                       ,device:str,padsize:int,list_dates_depth:list):

        self.LABEL_DIR= LABEL_DIR
        self.device= device
        self.padsize = padsize
        self.DATA_DIR = DATA_DIR
        self.list_dates_depth= list_dates_depth
        self.pixeldict = pixelfulldict

    def ImageCrop(self,date:str):

        ListOfCroppedImages = []
        file = glob.glob(self.DATA_DIR + date + '**/*.jpg')
        CatTensor= None

        if file:


            data= pyplot.imread(file[0])
            norm_dry= np.median(data[dry_wet_cloth['dry'][1][0]:dry_wet_cloth['dry'][1][1],dry_wet_cloth['dry'][0][0]:dry_wet_cloth['dry'][0][1]])
            norm_wet= np.median(data[dry_wet_cloth['wet'][1][0]:dry_wet_cloth['wet'][1][1],dry_wet_cloth['wet'][0][0]:dry_wet_cloth['wet'][0][1]])

            #normalization
            range= np.max(data) - np.min(data)
            data= (data-np.min(data))/range

            range2= norm_dry-norm_wet
            data= data*range2 + norm_wet

            for key in self.pixeldict.keys():
                img = data[self.pixeldict[key][1][0]:self.pixeldict[key][1][1],self.pixeldict[key][0][0]:self.pixeldict[key][0][1]]

                xsize,ysize = img.shape

                padx = self.padsize - xsize ; pady = self.padsize - ysize
                Padded_im = np.pad(img,((padx//2,padx//2),(pady//2,pady//2)),constant_values=0,mode='constant')
                ListOfCroppedImages.append(Padded_im)

            CatTensor = torch.stack([img for img in ListOfCroppedImages])

        return CatTensor


class DataLoader(ImageLoader):
    def __init__(self,DATA_DIR:str,LABEL_DIR:str,device:str,padsize:int,**kwargs):
        super().__init__(DATA_DIR=DATA_DIR,LABEL_DIR=LABEL_DIR,device=device,padsize=padsize)

    def LoadData(self):
        labeldict = self.GetDateTimeLabel(self.LABEL_DIR,self.pixeldict)
        with tqdm(total=len(labeldict['lys1']),desc='Loading Data') as pbar:
            label_list,data_list = [],[]
            for i in range(len(labeldict['lys1'])):
                labels = [torch.Tensor(np.asarray(labeldict[key][i][0]),require_grad=False) for key in labeldict.keys()] ; date = labeldict['lys1'][i][1]
                date_object= self.GetDateObject(date)

                img_label= torch.stack([label for label in labels])
                img_tensor= self.ImageCrop(date=date)

                if img_tensor is not None:
                    data_list.append(img_tensor)
                    label_list.append(img_label)

                pbar.update(1)
            pbar.close()

            X= torch.stack([tensor for tensor in data_list],dim=0)
            y= torch.stack([label for label in label_list],dim=0)

        self.X = X ; self.y= y
        return X,y

    @staticmethod
    def GetDateTimeLabel(LABEL_DIR:str,pixeldict:dict) -> dict:

        #this function takes the label input dir and based on the date&time returns a dictionary with keys lys1,lys2...and values
        #(label,date)

        # === initialize ===
        labeldict= create_labels_dict(pixeldict)
        labels_csv = pd.read_csv(LABEL_DIR)
        # === initialize dates and times of data ====
        dates = np.asarray('2019', dtype='datetime64[Y]') + np.asarray(labels_csv['day of year'],
                                                                       dtype='timedelta64[D]') - 1
        hours = np.asarray(labels_csv['hour'],dtype='int8') ; minutes = np.asarray(labels_csv['minute'],dtype='int8')

        # === Uploading the labels ===
        labels= labels_csv['ET'] ;labels= labels.interpolate(method='linear')
        plant_labels= labels_csv['lysimeter']

        for i,label in enumerate(labels):
            labeldict[plant_labels[i]].append((label,str(dates[i]).replace('-','_') + '_{num:02d}_'.format(num=hours[i]) + '{num:02d}'.format(num=minutes[i])))
        return labeldict


    @staticmethod
    def GetDateObject(date):
        date_object= datetime.now()
        try:
            date_object=datetime(int(date.split('_')[0]), int(date.split('_')[1]), int(date.split('_')[2]),
                    int(date.split('_')[3]), int(date.split('_')[4]))
        except ValueError:
            pass
        return date_object

def create_image_dict(pixeldict:dict) -> dict:
    imagedict = {}
    for key in pixeldict.keys():
        imagedict[key] = []
    return imagedict

def create_labels_dict(pixeldict:dict) -> dict:
    imagedict = {}
    for key in pixeldict.keys():
        imagedict['lys'+str(int(key))] = []
    return imagedict



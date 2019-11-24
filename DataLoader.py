import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import torch
from matplotlib import pyplot
from Configuration import pixelfulldict,dry_wet_cloth
from torchvision.transforms import Normalize,RandomHorizontalFlip,RandomRotation,ToTensor
from PIL import Image
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
            #This Part is to upload Depth Maps

            DepthDate = self.FindDapthVisDate(self.list_dates_depth, date)
            DepthPath= glob.glob(self.DATA_DIR + DepthDate + '**/*.raw')
            # A=np.fromfile(DepthPath[0], dtype='int8', sep="")
            # A = np.reshape(A, (1280, 1024, 2))

            data=np.transpose(pyplot.imread(file[0]))
            norm_dry= np.median(data[dry_wet_cloth['dry'][0][0]:dry_wet_cloth['dry'][0][1],dry_wet_cloth['dry'][1][0]:dry_wet_cloth['dry'][1][1]])
            norm_wet= np.median(data[dry_wet_cloth['wet'][0][0]:dry_wet_cloth['wet'][0][1],dry_wet_cloth['wet'][1][0]:dry_wet_cloth['wet'][1][1]])
            for key in self.pixeldict.keys():
                img = data[self.pixeldict[key][0][0]:self.pixeldict[key][0][1],self.pixeldict[key][1][0]:self.pixeldict[key][1][1]]

                #normalization
                range= np.max(img) - np.min(img)
                img= (img-np.min(img))/range

                range2= norm_dry-norm_wet
                img= img*range2 + norm_wet


                xsize,ysize = img.shape

                padx = self.padsize - xsize ; pady = self.padsize - ysize
                Padded_im = np.pad(img,((padx//2,padx//2),(pady//2,pady//2)),constant_values=0,mode='constant')

                image_tensor = Image.fromarray(Padded_im)
                flipped= self.Random_flip(image_tensor)
                rotated = self.Random_Rotation(flipped)
                norm= self.Normalize_img(rotated)


                ListOfCroppedImages.append(norm.squeeze(dim=0))

            CatTensor = torch.stack([img for img in ListOfCroppedImages])

        return CatTensor

    @staticmethod
    def Random_Rotation(img):
        transform= RandomRotation(degrees=(0,180))
        rotated_img= transform.__call__(img)
        to_tensor= ToTensor()
        return to_tensor(rotated_img)

    @staticmethod
    def Random_flip(img):
        transform = RandomHorizontalFlip()
        flipped_img= transform.__call__(img)
        return flipped_img

    @staticmethod
    def Normalize_img(img):
        # transform = Normalize(mean=(0,0,0),std=(1,1,1))
        # norm_img = transform.__call__(img)
        return img
    @staticmethod

    def FindDapthVisDate(list_dates_depth, date):
        i=0
        while True:
            date_depth= list_dates_depth[i] ;
            depth_object = datetime(int(date_depth.split('_')[0]), int(date_depth.split('_')[1]), int(date_depth.split('_')[2]),
                         int(date_depth.split('_')[3]), int(date_depth.split('_')[4]))

            date_obect= datetime(int(date.split('_')[0]),int(date.split('_')[1]),int(date.split('_')[2]),
                                  int(date.split('_')[3]),int(date.split('_')[4]))
            if depth_object < date_obect:
                break
            else:
                i+=1
        return date_depth


class DataLoader(ImageLoader):
    def __init__(self,DATA_DIR:str,LABEL_DIR:str,device:str,padsize:int,**kwargs):
        list_dates_depth = self.GetDepthDates(DATA_DIR)
        super().__init__(DATA_DIR=DATA_DIR,LABEL_DIR=LABEL_DIR,device=device,padsize=padsize,list_dates_depth=list_dates_depth)

    def LoadData(self):
        list_dates_depth= self.GetDepthDates(self.DATA_DIR)  # this is to retrieve depth maps and images_no_filter
        labeldict = self.GetDateTimeLabel(self.LABEL_DIR,self.pixeldict)
        with tqdm(total=len(labeldict['lys1']),desc='Loading Data') as pbar:
            label_list,data_list = [],[]
            for i in range(len(labeldict['lys1'])):
                labels = [torch.Tensor(np.asarray(labeldict[key][i][0])) for key in labeldict.keys()] ; date = labeldict['lys1'][i][1]
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
    def GetDepthDates(DATA_DIR):
        list_dates_depth= []
        Depth_names= glob.glob(os.path.join(DATA_DIR,'**Depth_day_night*/'))
        for name in sorted(Depth_names):
            list_dates_depth.append(name.split('/')[-2][:19])

        return list_dates_depth


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



import argparse
from os.path import expanduser
import os
from datetime import datetime

CD= os.getcwd()
now= datetime.now()
SAVE_DIR = os.path.join(CD, 'Exp %s_%s_%s_%s_%s_%s' % (now.strftime("%m"), now.strftime("%d"), now.strftime("%Y"),
                                                       now.strftime("%H"), now.strftime("%M"), now.strftime("%S")))

pixelfulldict = {'1':((498,558),(86,192)),'2':((510,638),(256,378)),'3':((395,499),(46,198)),'4':((375,493),(246,384)),
             '5':((269,373),(86,194)),'6':((252,372),(232,380)),'7':((157,267),(39,191)),'8':((137,247),(235,375)),
             '9':((38,150),(117,221)),'10':((10,124),(258,386))}

dry_wet_cloth= {'dry':((306,324),(394,408)),'wet':((584,594),(210,220))}


DATA_DIR = "//Users/roiklein/Dropbox/Msc Project/Deep Learning Project/Exp1000_Full/"
LABEL_DIR = '/Users/roiklein/Dropbox/Msc Project/Deep Learning Project/lys_prc.csv'
# SAVE_DIR = '/Users/roiklein/Dropbox/Msc Project/Deep Learning Project/TimeSeriesAnalysis'
#
# DATA_DIR = expanduser('~/Exp1000/')
# LABEL_DIR = expanduser('~/PNN-AI/lys_prc.csv')
# SAVE_DIR = expanduser('~/PNN-AI/')
#
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',type=int,default=16,help='Number of samples in each batch')
parser.add_argument('--momentum',type=float,default=0.9,help='Momentum constant')
parser.add_argument('--padsize',type=int,default=300,help='Pad array of images with zeros in "padsize" size')
parser.add_argument('--Epochs',type=int,default=500,help='Number of Epochs')
parser.add_argument('--NumPlants',type=int,default=10,help='Number of Plants in the Expirament')

def parse_args(is_training=True):
    if is_training:
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate decay')
        parser.add_argument('--num_levels',type=int,default=10,help='Number of TCN levels')
        parser.add_argument('--num_hidden',type=int,default=1000,help='Number of TCN hidden feature maps')
        parser.add_argument('--embedding_size',type=int,default=2048,help='TCN Output layer dimension')
        parser.add_argument('--kernel_size',type=int,default=3,help='TCN feature map size (nxn)')
        parser.add_argument('--dropout',type=float,default=0.3,help='')
        parser.add_argument('--weight_decay',type=float,default=0,help='')
    return parser.parse_args()
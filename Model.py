import torch
from torch import nn
from torchvision.models import inception_v3
from TCN.tcn import TemporalConvNet

def greyscale_to_RGB(image: torch.Tensor, add_channels_dim=False) -> torch.Tensor:
    if add_channels_dim:
        image = image.unsqueeze(-3)

    dims = [-1] * len(image.shape)
    dims[-3] = 3
    return image.expand(*dims)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor,self).__init__()

        self.inception = inception_v3(pretrained=True, transform_input=True, aux_logits=True)
        self.inception.fc = Identity()
        self.inception.eval()

        for p in self.inception.parameters():
            p.requires_grad = False

    # make sure that the inception model stays on eval
    def train(self, mode=True):
        return self

    def forward(self, x: torch.Tensor):
        """

        :param x: a batch of image sequences of either size (NxTxCxHxW) or the squeezed size (NxTxHxW)
        :return: a batch of feature vectors for each image of size (NxTx2048)
        """
        # if the image is greyscale convert it to RGB
        if len(x.shape) < 5 or len(x.shape) >= 5 and x.shape[-3] == 1:
            x = greyscale_to_RGB(x, add_channels_dim=len(x.shape) < 5)

        # if we got a batch of sequences we have to calculate each sequence separately
        N, T = x.shape[:2]
        # return self.inception(x.view(-1, *x.shape[2:])).view(N, T, -1)
        return self.inception(x.reshape(N*T,*x.shape[2:])).view(N, T, -1)

    def _prepare_for_gpu(self):
        # if torch.cuda.device_count() > 1:
        #     print("{0} GPUs are detected.".format(torch.cuda.device_count()))
        #     self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.inception.cuda()


class TCN(TemporalConvNet):
    def __init__(self, num_levels: int = 3, num_hidden: int = 600, embedding_size: int = 128, kernel_size=2,
                 dropout=0.2):
        """

        :param num_levels: the number of TCN layers
        :param num_hidden: number of Feature Maps used in the hidden layers
        :param embedding_size: size of final feature vector
        :param kernel_size: kernel size, make sure that it matches the feature vector size
        :param dropout: dropout probability
        :return: a TemporalConvNet matching the inputted params
        """
        num_channels = [num_hidden] * (num_levels - 1) + [embedding_size]
        super().__init__(2048, num_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor):
        """

        :param x: input tensor of size (NxTx2048),
                where N is the batch size, T is the sequence length and 2048 is the input embedding dim
        :return: tensor of size (N x T x embedding_size) where out[:,t,:] is the output given all values up to time t
        """

        # transpose each sequence so that we get the correct size for the TemporalConvNet
        x = torch.stack([m.t() for m in x])

        out = super().forward(x)


        # undo the previous transpose
        return torch.stack([m.t() for m in out])

class TemporalSpatialModel(nn.Module):
    def __init__(self,num_levels: int, num_hidden: int , embedding_size: int, kernel_size,
                 dropout):
        super().__init__()
        self.FeatureVectore= ImageFeatureExtractor()
        self.TCN= TCN(num_levels=num_levels,num_hidden=num_hidden,embedding_size=embedding_size,kernel_size=kernel_size,
                      dropout=dropout)
        # self.FinalFC = nn.Sequential(nn.Linear(embedding_size,embedding_size//2), nn.LeakyReLU(negative_slope=0.1)
        #                              ,nn.Linear(embedding_size//2,embedding_size//4),nn.LeakyReLU(negative_slope=0.1)
        #                              ,nn.Linear(embedding_size//4,64),nn.LeakyReLU(negative_slope=0.1)
        #                              ,nn.Linear(64,1),nn.ReLU())
        self.FinalFC= nn.Sequential(
            nn.Linear(embedding_size*10*8,embedding_size*10,bias=True),nn.ReLU()
            ,nn.Linear(embedding_size*10,embedding_size,bias=True),nn.ReLU()
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        FeatureVectore= self.FeatureVectore(x)
        outputs= self.TCN(FeatureVectore)
        return self.FinalFC(outputs.view(-1)).view(*outputs.shape[:2])
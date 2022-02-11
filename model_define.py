import torch
import torch.nn as nn
import mmcv
from mmcv.ops import DeformConv2dPack as DCN
from tifffile import imread as tiff_read
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from numpy import array


class FeatureExtraction(nn.Module): # return feature_lv1,feature_lv2,feature_lv3
    def __init__(self,origin_channel,internal_channel=16,) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(origin_channel,internal_channel,3,padding=1)
        self.conv2 = nn.Conv2d(internal_channel,internal_channel,3)
        self.conv3 = nn.Conv2d(internal_channel,internal_channel,3)
        self.relu = nn.ReLU(True)
    
    def forward(self,origin_image):
        feature_lv1 = self.relu(self.conv1(origin_image))
        feature_lv2 = self.relu(self.conv2(feature_lv1))
        feature_lv3 = self.relu(self.conv3(feature_lv2))
        return [feature_lv1,feature_lv2,feature_lv3]
class Data(object):
        def __init__(self,image=None,level=3) -> None:#lv: level of feature pyramid
            self.image = image
            self.feature = [] #should contain torch.Tensor
            self.level =level
            


class Algin(nn.Module):
    def __init__(self,internal_channel = 16,groups=8,lv=3) -> None:
        super().__init__()
        self.internal_channel = internal_channel
        self.groups=groups
        self.relu = nn.ReLU(True)
        self.level=lv
        self.feature_extraction = FeatureExtraction(internal_channel) #feature -> [tensor(N,C,H,W),]
        #offset conv is used on concatted feature
        self.offset_conv = []
        for i in range(self.level):
            self.offset_conv.append(nn.Conv2d(internal_channel*2,internal_channel,3,padding=1))
        self.dcn = DCN(internal_channel,internal_channel,3,padding=1,deform_groups=groups)
    
    def forward(self,ref_image,unreg_image):
        data_ref = Data(ref_image,level=3)
        data_unreg = Data(unreg_image,level=3)
        data_aligned = Data()
        data_aligned.offset = []
        data_ref.feature =  self.feature_extraction(ref_image,) 
        data_unreg.feature = self.feature_extraction(unreg_image)
        for i in range(data_ref.level-1,-1,-1):
            data_aligned.feature[i]=torch.cat(data_ref.feature[i],data_unreg.feature[i])
            #todo: add step that uses down level offset to generate up one
            data_aligned.offset[i]=self.relu(self.offset_conv[i](data_aligned.feature))
            data_aligned.feature[i] = DCN(data_aligned.feature[i].data_aligned.offset[i])
            
class DenoisedDataset(torch.utils.data.Dataset):
    def __init__(self,img_dir='data/',) -> None:
        super().__init__()
        self.img_dir = img_dir

    def __len__():
        return None

    def __getitem__(self,idx):
        imgpath = self.imgdir + f'{idx}.tif'
    
    


if __name__ == "__main__":
    print('main')
    a =Algin()
    ima = tiff_read('data/16.tif')
    convert_tensor =ToTensor()
    image = convert_tensor(ima)
    image = nn.functional.normalize(image)
    print(torch.std_mean(image))
from cv2 import magnitude
import torch
import torch.nn as nn
import torchfields
from mmcv.ops import DeformConv2dPack as DCN
from tifffile import imread as tiff_read
from tifffile import imwrite as tiff_write
from matplotlib import pyplot as plt
import torchvision.transforms
from motion_generate import MotionGenerator
import numpy

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
    print('model_define as main')
    ima = torch.from_numpy(tiff_read('data/ground_truth/1.tif'))
    max = torch.max(ima,0)[0]
    max = torch.max(max,0)[0]
    ima/=max
    g = MotionGenerator(10,magnitude=1)
    displace_field = torch.Field.identity(1, 2, 512 , 512)
    displace_field = g(displace_field)
    torch.save(displace_field,'data/df1.pt') 
    warpped_img = displace_field(ima)
    warpped_img*=max
    tiff_write('data/wrapped_image/1_wrapped.tif',warpped_img.numpy(),imagej=True)
    
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchfields
from mmcv.ops import DeformConv2d

from matplotlib import pyplot as plt
import torchvision.transforms
from motion_generate import MotionGenerator
import numpy

#TODO: Residual connection may required
class FeatureExtraction(nn.Module): # return feature_lv1,feature_lv2,feature_lv3
    def __init__(self,origin_channel,internal_channel=16,lv=3) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.level = lv
        self.feature_layer = nn.ModuleList()
        self.feature = []
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.origin_channel = origin_channel
        for i in range(self.level):
            input_channel = origin_channel if i==0 else internal_channel
            feature_block = nn.Sequential(
                                nn.Conv2d(input_channel,internal_channel,3,padding=1),
                                self.relu,
                                nn.Conv2d(internal_channel,internal_channel,3,padding=1),
                                self.relu
                            )
            self.feature_layer.append(feature_block)
            self.feature.append('python list sucks')
    
    def forward(self,origin_image):
        for i in range(self.level):
            if i==0:
                self.feature[i]=self.feature_layer[i](origin_image)
            else:
                self.feature[i]=self.max_pool(self.feature_layer[i](self.feature[i-1]))
        return self.feature
            
class Data(object):
        def __init__(self,image = torch.zeros(1,1,1,1)) -> None:
            self.image = image
            self.feature = [] 

class Algin(nn.Module):
    def __init__(self,original_channel,internal_channel = 16,groups=8,lv=3) -> None:
        super().__init__()
        self.internal_channel = internal_channel
        self.groups=groups
        self.relu = nn.ReLU()
        self.level=lv
        self.feature_extraction = FeatureExtraction(original_channel,internal_channel) #feature -> [tensor(N,C,H,W),]
        
        self.deformable_conv = nn.ModuleList()
        '''
        input: unreg image feature: tensor(N,C,H/2**lv,W/2**lv),  
               offset: tensor(N,3*3*2,H,W)
        out: internal feature: tensor(N,C,H/2**lv,W/2**lv)
        '''
        self.feature_conv = nn.ModuleList()
        '''
        input : concat(internal feature,upsambled lower level aligned feature): tensor(N,2C,H/2**lv,W/2**lv)
                or if LOWEST level:
                    internal feature:tensor(N,C,H/2**lv,W/2**lv)
        out : aligned feature: tensor(N,C,H/2**lv,W/2**lv)
        '''
        self.displace_field_predict_offset_bottleneck = nn.Conv2d(2*internal_channel,internal_channel,3,padding=1)
        self.displace_field_predict_offset_generator = nn.Conv2d(internal_channel+18,36,3,padding=1)
        self.displace_field_predict_deform_conv = DeformConv2d(internal_channel,2,3,padding=1,deform_groups=2)
        for i in range(self.level):
            input_channel_offset = internal_channel if i==self.level-1 else internal_channel+18
            input_channel_fea = internal_channel if i==self.level-1 else 2*internal_channel
            self.offset_bottleneck.append(nn.Conv2d(2*internal_channel,internal_channel,3,padding=1))
            self.offset_generator.append(nn.Conv2d(input_channel_offset,18,3,padding=1))
            self.deformable_conv.append(DeformConv2d(internal_channel,internal_channel,3,padding=1))
            self.feature_conv.append(nn.Conv2d(input_channel_fea,internal_channel,3,padding=1))
    
    def forward(self,ref_image,unreg_image):
        ref_data = Data(ref_image)
        unreg_data = Data(unreg_image)
        aligned_data = Data()
        aligned_data.offset = []
        ref_data.feature =  self.feature_extraction(ref_data.image) 
        unreg_data.feature = self.feature_extraction(unreg_data.image)
        print(f'extract:{torch.cuda.memory_allocated()}')
        #init list
        for i in range(self.level):
            aligned_data.offset.append(torch.cat((ref_data.feature[i],unreg_data.feature[i]),dim=1))
            aligned_data.feature.append('python list sucks')
        #upsamble
        print(f'list:{torch.cuda.memory_allocated()}')
        for i in range(self.level-1,-1,-1):
            #deform conv
            aligned_data.offset[i]=self.relu(self.offset_bottleneck[i](aligned_data.offset[i]))
            if i==self.level-1:
                internal_offset_feature = aligned_data.offset[i]
            else:
                upsambled_lower_level_offset = functional.interpolate(aligned_data.offset[i+1],scale_factor=2,mode='bilinear') 
                internal_offset_feature = torch.cat((aligned_data.offset[i],upsambled_lower_level_offset),dim=1)
            aligned_data.offset[i] =self.relu(self.offset_generator[i](internal_offset_feature))
            aligned_data.feature[i]= self.relu(self.deformable_conv[i](unreg_data.feature[i],aligned_data.offset[i]))
            #fusion level
            if i==self.level-1:
                internal_feature = aligned_data.feature[i]
            else:
                upsambled_lower_level_feature = functional.interpolate(aligned_data.feature[i+1],scale_factor=2,mode='bilinear') 
                internal_feature = torch.cat((aligned_data.feature[i],upsambled_lower_level_feature),dim=1)
            aligned_data.feature[i] = self.relu(self.feature_conv[i](internal_feature))
        #final field
        print(f'conv:{torch.cuda.memory_allocated()}')
        displace_field_offset = self.displace_field_predict_offset_bottleneck(
                        torch.concat((ref_data.feature[0],aligned_data.feature[0]),dim=1))
        displace_field_offset = self.displace_field_predict_offset_generator(
                                torch.cat((displace_field_offset,aligned_data.offset[0]),dim=1))
        displace_field = self.displace_field_predict_deform_conv(aligned_data.feature[0],offset =displace_field_offset)
        return displace_field
          

    
if __name__ == "__main__":
    print('model_define as main')
    #test FeatureExtraction block
    f = FeatureExtraction(1)
    t = torch.rand(2,1,64,64)
    t2 = torch.rand(2,1,64,64)
    print(t)
    y = f(t)
    for fe in y:
        print(fe.shape)
    g = Algin(1)
    u = g(t,t2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(g.parameters(),lr=1e-3)
    predict_field = u.field()
    print(f'predict_field{predict_field.size()}')
    x_predict = predict_field(t)
    print(f'x_predict:{x_predict.size()}')
    loss = loss_fn(x_predict,t2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


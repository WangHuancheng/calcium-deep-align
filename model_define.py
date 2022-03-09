import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchfields

class Conv2d_Bn_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        padding_mode: str = 'zeros',device=None,dtype=None):
        #args: same as torch.nn.Conv2d
        super().__init__()
        self.in_channels:int = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias,
                      padding_mode=padding_mode,device=device,dtype=dtype)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out_conv = self.conv(x)
        out_bn = self.bn(out_conv)
        out = self.relu(out_bn)
        return out 
class Conv2d_Bn_Tanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, dilation=1, groups=1, bias=True,
        padding_mode: str = 'zeros',device=None,dtype=None):
        #args: same as torch.nn.Conv2d
        super().__init__()
        self.in_channels:int = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias,
                      padding_mode=padding_mode,device=device,dtype=dtype)
        self.bn = nn.BatchNorm2d(out_channels)
        self.Tanh= nn.Tanh()
    def forward(self,x):
        out_conv = self.conv(x)
        out_bn = self.bn(out_conv)
        out = self.Tanh(out_bn)
        return out 

class FeatureExtraction(nn.Module): # return feature_lv1,feature_lv2,feature_lv3
    def __init__(self,origin_channel,internal_channel=16) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.level = 4
        self.feature =[0,0,0]
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.origin_channel = origin_channel
        self.feature_layer_0 = nn.Sequential(Conv2d_Bn_Relu(origin_channel,internal_channel,3,padding=1),
                                            Conv2d_Bn_Relu(internal_channel,internal_channel,3,padding=1))
        self.feature_layer_1 = nn.Sequential(Conv2d_Bn_Relu(internal_channel,internal_channel,3,padding=1),
                                            Conv2d_Bn_Relu(internal_channel,internal_channel,3,padding=1))
        self.feature_layer_2 = nn.Sequential(Conv2d_Bn_Relu(internal_channel,internal_channel,3,padding=1),
                                            Conv2d_Bn_Relu(internal_channel,internal_channel,3,padding=1))

    
    def forward(self,origin_image):
        self.feature[0] = self.feature_layer_0(origin_image)
        downsampled_feature_0 = self.max_pool(self.feature[0])
        self.feature[1] = self.feature_layer_1(downsampled_feature_0)
        downsampled_feature_1 = self.max_pool(self.feature[1])
        self.feature[2] = self.feature_layer_2(downsampled_feature_1)
        
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
        self.field_predict_layer_0 = nn.Sequential(Conv2d_Bn_Relu(internal_channel*2+2,internal_channel,3,padding=1),
                                            Conv2d_Bn_Tanh(internal_channel,2,3,padding=1))
        self.field_predict_layer_1 = nn.Sequential(Conv2d_Bn_Relu(internal_channel*2+2,internal_channel,3,padding=1),
                                            Conv2d_Bn_Tanh(internal_channel,2,3,padding=1))
        self.field_predict_layer_2 = nn.Sequential(Conv2d_Bn_Relu(internal_channel*2,internal_channel,3,padding=1),
                                            Conv2d_Bn_Tanh(internal_channel,2,3,padding=1))
        self.field = [0,0,0]
        
    def forward(self,ref_image,unreg_image):
        ref_data = Data(ref_image)
        unreg_data = Data(unreg_image)
        ref_data.feature =  self.feature_extraction(ref_data.image) 
        unreg_data.feature = self.feature_extraction(unreg_data.image)
        self.field[2] = self.field_predict_layer_2(
                    torch.cat((unreg_data.feature[2],ref_data.feature[2]),dim=1)).field()
        upsambled_feature_2 = functional.interpolate(self.field[2],scale_factor=2,mode='bilinear')
        self.field[1] = upsambled_feature_2+self.field_predict_layer_1(
                    torch.cat((unreg_data.feature[1],ref_data.feature[1],upsambled_feature_2),dim=1))
        upsambled_feature_1 = functional.interpolate(self.field[1],scale_factor=2,mode='bilinear')
        self.field[0] = upsambled_feature_1+self.field_predict_layer_0(
                    torch.cat((unreg_data.feature[0],ref_data.feature[0],upsambled_feature_1),dim=1)).field()
                    
        return self.field[0]
          

    
if __name__ == "__main__":
    print('model_define as main')
    #test FeatureExtraction block
    f = FeatureExtraction(1)
    t = torch.rand(2,1,64,64)
    t2 = torch.rand(2,1,64,64)
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
    loss.backward(retain_graph=True)
    optimizer.step()


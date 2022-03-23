from turtle import forward
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as functional


class NCC(nn.Module):
    def __init__(self,reduction=True) -> None:
        super().__init__()
        self.reduction = reduction
    
    def forward(self,image1:Tensor,image2:Tensor) ->Tensor :
        alert = 'input size do not match, expect the same (N,C,H,W) for both input' 
        assert image1.size() == image2.size(), alert
        m1 = image1.mean()
        m2 = image2.mean()
        batchsize = image1.size(dim=0)
        result = (batchsize/(image1.numel()*torch.var(image1)*torch.var(image2)) 
                    * functional.conv2d(image1.transpose(0,1)-m1,image2-m2,groups=batchsize))
        if self.reduction==False:
            return result.transpose(0,1)
            
        else:
            return result.mean()

    

    
if __name__ == "__main__":
    t1 = torch.randn((32,1,512,512))
    t2 = torch.randn((32,1,512,512))
    ncc = NCC(False)
    r1 = ncc(t1,t2)
    print(r1.size())

import time
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tifffile import imread as tiff_read
from tifffile import imwrite as tiff_write
from tqdm import trange

def MyDataLoad():
    for i in trange(4800,6000):
        input=torch.from_numpy(tiff_read(f'data/dataset1/test/gd/{i+1}_gd.tif')).reshape(1,1,512,512)
        if(i==4800):
            tensor=input
        else:
            tensor = torch.cat((tensor,input))
    torch.save(tensor,'tensor.pt')

if __name__ == "__main__":
    MyDataLoad()
    
    


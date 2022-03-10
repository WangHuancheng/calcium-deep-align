import torch
import torch.nn as nn
import torch.nn.functional as functional
from tifffile import imread as tiff_read
from tifffile import imwrite as tiff_write
from model_define import Algin
from torch.utils.data import DataLoader
import torchfields
from torchinfo import summary
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print('trainning align model')
    gd = torch.load('dataset1/train_gd.pt')
    gd/=2455 #maxinum of grey value
    wrapped = torch.load('dataset1/train_wrapped.pt')
    wrapped /=2455
    print('data loaded')
    dataset=torch.utils.data.TensorDataset(wrapped,gd)
    train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,num_workers=20)
    for train_features, train_labels in train_dataloader:
        print(train_features.max())
        print(train_features.mean())
        print(train_labels.mean())
        print(train_labels.max())
        break
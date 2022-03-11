
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
    model = Algin(1)
    model = model.cuda(1)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    result = []
    for i in trange(1000):
        for train_features, train_labels in train_dataloader:
            train_features = train_features.cuda(1)
            train_labels = train_labels.cuda(1)
            predict_field = model(train_labels,train_features).field()
            x_predict = predict_field(train_features)
            loss = loss_fn(x_predict,train_labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        current_result = loss.item()
        result.append(current_result)
        if i==0 or current_result<best_result:
            best_result = current_result
            best_it = i+1
            torch.save(model,f'best_model.pth')
            tiff_write(f'dataset1/best_input.tif',train_features[0,0].cpu().detach().numpy()*2455,imagej=True)
            tiff_write(f'dataset1/best_output.tif',x_predict[0,0].cpu().detach().numpy()*2455,imagej=True)
            tiff_write(f'dataset1/best_gd.tif',train_labels[0,0].cpu().detach().numpy()*2455,imagej=True)
        if (i+1) % 100 == 0:
            torch.save(model,f'{i+1}_model.pth')
            tiff_write(f'dataset1/{i+1}_input.tif',train_features[0,0].cpu().detach().numpy()*2455,imagej=True)
            tiff_write(f'dataset1/{i+1}_output.tif',x_predict[0,0].cpu().detach().numpy()*2455,imagej=True)
            tiff_write(f'dataset1/{i+1}_gd.tif',train_labels[0,0].cpu().detach().numpy()*2455,imagej=True)
    x_axis = [i for i in range(1000)]
    plt.plot(x_axis,result)
    plt.savefig(f'loss_{best_it}is_the_best.png')

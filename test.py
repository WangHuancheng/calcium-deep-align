import torch
import torch.nn as nn
import torch.nn.functional as functional
from tifffile import imwrite as tiff_write
from model_define import Algin
from torch.utils.data import DataLoader
import torchfields



if __name__ == "__main__":

    with torch.no_grad():
        model = torch.load('best_model.pth',map_location=torch.device('cpu') )
        model = model.eval()
        print('model loaded')
        loss_fn = nn.MSELoss(reduction='sum')
        loss = 0
        loss_input = 0

        '''
        test_gd = torch.load('dataset1/test_gd.pt')
        test_gd/=2455 #maxinum of grey value
        test_wrapped = torch.load('dataset1/test_wrapped.pt')
        test_wrapped /=2455
        print('test data loaded')
        dataset=torch.utils.data.TensorDataset(test_wrapped,test_gd)
        test_dataloader = DataLoader(dataset,batch_size=32,num_workers=16)
       
        for train_features, train_labels in test_dataloader:
            predict_field = model(train_labels,train_features).field()
            x_predict = predict_field(train_features)
            loss += loss_fn(x_predict,train_labels)
            loss_input+= loss_fn(train_features,train_labels)
        
        loss/= 1200
        loss_input/=1200
        print(f'test mean loss:{loss}')
        print(f'test mean loss of input:{loss_input}')
        tiff_write(f'dataset1/test_input.tif',train_features[0,0].cpu().detach().numpy()*2455,imagej=True)
        tiff_write(f'dataset1/test_output.tif',x_predict[0,0].cpu().detach().numpy()*2455,imagej=True)
        tiff_write(f'dataset1/test_gd.tif',train_labels[0,0].cpu().detach().numpy()*2455,imagej=True)
        '''
        train_gd = torch.load('dataset1/train_gd.pt')
        train_gd/=2455 #maxinum of grey value
        train_wrapped = torch.load('dataset1/train_wrapped.pt')
        train_wrapped /=2455
        dataset=torch.utils.data.TensorDataset(train_wrapped,train_gd)
        train_dataloader = DataLoader(dataset,batch_size=32,num_workers=16)
        print('train data loaded')
        loss = 0
        loss_input=0
        for train_features, train_labels in train_dataloader:
            predict_field = model(train_labels,train_features).field()
            x_predict = predict_field(train_features)
            loss += loss_fn(x_predict,train_labels)
            loss_input+= loss_fn(train_features,train_labels)
        loss/=4800
        loss_input/=4800
        print(f'train mean loss:{loss}')
        print(f'train mean loss of input:{loss_input}')
    
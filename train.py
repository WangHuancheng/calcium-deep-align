
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tifffile import imread as tiff_read
from tifffile import imwrite as tiff_write
from model_define import Algin
from torch.utils.data import DataLoader
import torchfields

class DataSet1Train(torch.utils.data.Dataset):
    def __init__(self,img_dir='data',) -> None:
        super().__init__()
        self.img_dir = img_dir

    def __len__(self):
        return 4800

    def __getitem__(self,idx):
        groud_truth_path = f'{self.img_dir }/gd/{idx}_gd.tif'
        unreg_path = f'{self.img_dir}/wrapped/{idx}_wrapped.tif'
        gd = torch.from_numpy(tiff_read(groud_truth_path))
        max = torch.max(gd,0)[0]
        max = torch.max(max,0)[0]
        gd/=max
        unreg_image = torch.from_numpy(tiff_read(unreg_path))
        max = torch.max(unreg_image,0)[0]
        max = torch.max(max,0)[0]
        unreg_image/=max
        
        return unreg_image.unsqueeze(0), gd.unsqueeze(0)
def fuckfield():
    pass
if __name__ == "__main__":

    print('trainning align model')
    train_data = DataSet1Train('data/dataset1/train')
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f'train_features:{train_features.size()}')
    print(f'train_labels:{train_labels.size()}')
    model = Algin(1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    predict_field = model(train_labels,train_features).field()
    print(f'predict_field{predict_field.size()}')
    x_predict = predict_field(train_features)
    print(f'x_predict:{x_predict.size()}')
    loss = loss_fn(x_predict,train_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("done")
    
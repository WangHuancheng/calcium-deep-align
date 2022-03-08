
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
        groud_truth_path = f'{self.img_dir }/gd/{idx+1}_gd.tif'
        unreg_path = f'{self.img_dir}/wrapped/{idx+1}_wrapped.tif'
        gd = torch.from_numpy(tiff_read(groud_truth_path))
        m1 = torch.max(gd,0)[0]
        m1 = torch.max(m1,0)[0]
        gd/=m1
        unreg_image = torch.from_numpy(tiff_read(unreg_path))
        m2 = torch.max(unreg_image,0)[0]
        m2 = torch.max(m2,0)[0]
        unreg_image/=m2
        
        return unreg_image.unsqueeze(0), gd.unsqueeze(0),m1,m2
def fuckfield():
    pass
if __name__ == "__main__":

    print('trainning align model')
    train_data = DataSet1Train('dataset1/train')
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True,pin_memory=True, num_workers=10)
    model = Algin(1)
    model = model.cuda()
    print(f'model:{torch.cuda.memory_allocated()}')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    for train_features, train_labels,m1,m2 in train_dataloader:
        train_features = train_features.cuda()
        train_labels = train_labels.cuda()
        m1 = m1.cuda()
        print(f'data:{torch.cuda.memory_allocated()}')
        predict_field = model(train_labels,train_features).field()
        #print(f'predict_field{predict_field.size()}')
        x_predict = predict_field(train_features)
        #print(f'x_predict:{x_predict.size()}')
        loss = loss_fn(x_predict,train_labels)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    print("done")

    result = []
    result.append(loss.mean().item())
    torch.save(model,'model.pth')
    x_predict*=m1
    tiff_write(f'dataset1/visual.tif',x_predict[0,0].cpu().numpy(),imagej=True)


import torch
from torch import Tensor
import torchfields
import torchvision.transforms as T
from PIL import Image
from tifffile import imread as tiff_read
from tifffile import imwrite as tiff_write
from tqdm import trange
class MotionGenerator(torch.nn.Module):
    def __init__(self, num_init_vector:int,width=512,lenth=512,magnitude=1) -> None:
        """Generate random motion for 2D map,from sevral random vector,using Gaussian smooth voitng 
        to full the whole map 
            Arg:
                num_init_vector:
                    how many initial vectors are used to generate displacement field
        """
        super().__init__()
        self.num_init_vector=num_init_vector
        self.magnitude = magnitude
        self.width = width
        self.lenth = lenth
        self.magnitude = magnitude
    def __gauuse_voting(self,x,y,sigma =32) -> torch.Tensor:
        result = 0
        distance_x = x - self.x_coord
        distance_y = y - self.y_coord
        distance =  (distance_x*distance_x+distance_y*distance_y)/(2*sigma*sigma)
        result = (-distance).exp()
        return result

    def forward(self) -> torch.field:
        #TODO: regular magnitude of init vector
        
        self.displace_field = torch.zeros(1,2,self.width,self.lenth)
        self.init_vectors = torch.randn(2,self.num_init_vector)/ ((self.lenth+self.width)/32) * self.magnitude
        self.x_coord= torch.randint(0,self.width,(self.num_init_vector,))
        self.y_coord= torch.randint(0,self.lenth,(self.num_init_vector,))
        #ordinary field is zeros(no warp)
        
        #since the field in 2 direction have the exactly since shape(euqual to the image)
        #we only need to iterate through one dim to get job done
        for field_u in self.displace_field[:,0]:
            for x, field_u_at_x in enumerate(field_u):
                for y in range(field_u_at_x.size(dim=0)):
                    modulate = self.__gauuse_voting(x,y)
                    #dim 0,x direction
                    self.displace_field[0,0,x,y] = (modulate * self.init_vectors[0]).sum()
                    #dim 1,y direction
                    self.displace_field[0,1,x,y] = (modulate * self.init_vectors[1]).sum()
        return self.displace_field.field()

if __name__ == "__main__":
    #generate grid to visualize displacement field
    '''
    width = 512
    lenth = 512
    grid = torch.ones((width,lenth))
    grid[0:width:16] = 0
    grid[:, 0:lenth:16] = 0
    g = MotionGenerator(10)
    displace_field = g()
    print('here1')
    result = displace_field(grid)
    print('here2')
    #transform 0-1 float tensor to uint8 PIL image
    result *=255
    result=result.to(torch.uint8)
    result_img = T.ToPILImage(mode='L')(result)
    '''
    g = MotionGenerator(10,magnitude=1)
    for i in trange(1,4801):
        ima = tiff_read(f'data/ground_truth/{i}.tif')
        tiff_write(f'data/dataset1/train/gd/{i}_gd.tif',ima,imagej=True)
        ima = torch.from_numpy(ima)
        max = torch.max(ima,0)[0]
        max = torch.max(max,0)[0]
        ima/=max
        displace_field = g()
        warpped_img = displace_field(ima)
        warpped_img*=max
        tiff_write(f'data/dataset1/train/wrapped/{i}_wrapped.tif',warpped_img.numpy(),imagej=True)
    for i in trange(4801,6001):
        ima = tiff_read(f'data/ground_truth/{i}.tif')
        tiff_write(f'data/dataset1/test/gd/{i}_gd.tif',ima,imagej=True)
        ima = torch.from_numpy(ima)
        max = torch.max(ima,0)[0]
        max = torch.max(max,0)[0]
        ima/=max
        displace_field = g()
        warpped_img = displace_field(ima)
        warpped_img*=max
        tiff_write(f'data/dataset1/test/wrapped/{i}_wrapped.tif',warpped_img.numpy(),imagej=True)
from math import exp
import torch
from torch import Tensor, tensor
import torchfields
import torchvision.transforms as T
from PIL import Image
from line_profiler import LineProfiler
import random
class MotionGenerator(torch.nn.Module):
    def __init__(self, num_init_vector:int) -> None:
        """Generate random motion for 2D map,from sevral random vector,using Gaussian smooth voitng 
        to full the whole map 
            Arg:
            num_init_vector:
                how many initial vectors are used to generate displacement field
        """
        super().__init__()
        self.num_init_vector=num_init_vector
    @profile
    def __gauuse_voting(self,x,y,sigma =32) -> torch.tensor:
        result = 0
        distance_x = x - self.x_coord
        distance_y = y - self.y_coord
        distance =  (distance_x*distance_x+distance_y*distance_y)/(2*sigma*sigma)
        result = (-distance).exp()
        return result

    def forward(self, width=512,lenth=512) -> torch.field:
        #TODO: regular magnitude of init vector
        self.init_vectors = torch.randn(2,self.num_init_vector)/32
        self.x_coord= torch.randint(0,width,(self.num_init_vector,))
        self.y_coord= torch.randint(0,lenth,(self.num_init_vector,))
        #ordinary field is zeros(no warp)
        self.displace_field = torch.zeros(1,2,width,lenth)
        #TODO: Gauuse voting
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
    result_img.show()
    

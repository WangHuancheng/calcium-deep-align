from turtle import forward
import torch
import torchfields

class MotionGenerator(torch.nn.Module):
    def __init__(self,num_init_vector:int,map:torch.tensor) -> None:#num_vector:how many initial vectors are used to generate displacement field
        super().__init__()
        self.num_init_vector=num_init_vector

    def forward():
        torch.randn()
        pass
import torch 
from model.mgdip import MultiLevelGDIP
from model.yolov3 import Yolov3

class Yolov3MGatedDIP(torch.nn.Module):
    def __init__(self):
        super(Yolov3MGatedDIP,self).__init__()
        self.mgdip = MultiLevelGDIP(256,7)
        self.yolov3 = Yolov3()
    
    def forward(self,x):
        out_x,_,gates_list = self.mgdip(x)
        p,p_d = self.yolov3(out_x)
        return out_x,gates_list,p,p_d

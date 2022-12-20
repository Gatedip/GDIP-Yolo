import torch 
from model.multilevel_gdip import MultiLevelGDIP
from model.yolov3 import Yolov3

class Yolov3MultiGDIP(torch.nn.Module):
    def __init__(self, cfg):
        super(Yolov3MultiGDIP,self).__init__()
        print("Multi-level Yolov3GatedDIP with custom Encoder!!")
        self.multi_gdip = MultiLevelGDIP(256,reversed=True)
        self.yolov3 = Yolov3(cfg)
    
    def forward(self,x):
        out_x, gates = self.multi_gdip(x)
        p,p_d = self.yolov3(out_x)
        return out_x, gates, p, p_d

import sys
from turtle import forward
sys.path.append("..")

import torch.nn as nn
import torch
from model.backbones.darknet53 import Darknet53
from model.necks.yolo_fpn import FPN_YOLOV3
from model.head.yolo_head import Yolo_head
from model.layers.conv_module import Convolutional
# import config.yolov3_config_voc as cfg
import numpy as np
from utils.tools import *
import torchvision
import math

class GatedDIP(torch.nn.Module):
    def __init__(self,encoder_output_dim=256,num_of_gates=7):
        super(GatedDIP,self).__init__()
        self.gaussian_blur = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 5.0))

        ##### GDIP
        # Gating Module
        self.gate_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,num_of_gates,bias=True))
        # White-Balance Module
        self.wb_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,3,bias=True))
        # Gamma Module
        self.gamma_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # Sharpning Module
        self.sharpning_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # De-Fogging Module
        self.defogging_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # Contrast Module
        self.contrast_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # Contrast Module
        self.tone_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,8,bias=True))

    def tanh01(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        return torch.tanh(x)*0.5+0.5

    def tanh_range(self,x : torch.tensor,left : float,right : float):
        """_summary_

        Args:
            x (torch.tensor): _description_
            left (float): _description_
            right (float): _description_

        Returns:
            _type_: _description_
        """
        return self.tanh01(x)*(right-left)+ left
    
    def rgb2lum(self,img: torch.tensor):
        """_summary_

        Args:
            img (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :,:] + 0.06 * img[:, 2, :, :]
        return img 
    
    def lerp(self ,a : int , b : int , l : torch.tensor):
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b


    def dark_channel(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        z = x.min(dim=1)[0].unsqueeze(1)
        return z 
    
    def atmospheric_light(self,x : torch.tensor,dark : torch.tensor ,top_k : int=1000):
        """_summary_

        Args:
            x (torch.tensor): _description_
            top_k (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        h,w = x.shape[2],x.shape[3]
        imsz = h * w 
        numpx = int(max(math.floor(imsz/top_k),1))
        darkvec = dark.reshape(x.shape[0],imsz,1)
        imvec = x.reshape(x.shape[0],3,imsz).transpose(1,2)
        indices = darkvec.argsort(1)
        indices = indices[:,imsz-numpx:imsz]
        atmsum = torch.zeros([x.shape[0],1,3]).to(x.device)
        for b in range(x.shape[0]):
            for ind in range(1,numpx):
                atmsum[b,:,:] = atmsum[b,:,:] + imvec[b,indices[b,ind],:]
        a = atmsum/numpx
        a = a.squeeze(1).unsqueeze(2).unsqueeze(3)
        return a
    
    def blur(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        return self.gaussian_blur(x)


    def defog(self,x:torch.tensor ,latent_out : torch.tensor ,fog_gate : torch.tensor):
        """Defogging module is used for removing the fog from the image using ASM 
        (Atmospheric Scattering Model).
        I(X) = (1-T(X)) * J(X) + T(X) * A(X)
        I(X) => image containing the fog.
        T(X) => Transmission map of the image.
        J(X) => True image Radiance.
        A(X) => Atmospheric scattering factor.

        Args:
            x (torch.tensor): Input image I(X)
            latent_out (torch.tensor): Feature representation from DIP Module.
            fog_gate (torch.tensor): Gate value raning from (0. - 1.) which enables defog module.

        Returns:
            torch.tensor : Returns defogged image with true image radiance.
        """
        omega = self.defogging_module(latent_out).unsqueeze(2).unsqueeze(3)
        omega = self.tanh_range(omega,torch.tensor(0.1),torch.tensor(1.))
        dark_i = self.dark_channel(x) 
        a = self.atmospheric_light(x,dark_i)
        i = x/a 
        i = self.dark_channel(i)
        t = 1. - (omega*i)
        j = ((x-a)/(torch.maximum(t,torch.tensor(0.01))))+a
        j = (j - j.min())/(j.max()-j.min())
        j = j* fog_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return j
    
    def white_balance(self,x : torch.tensor,latent_out : torch.tensor ,wb_gate: torch.tensor):
        """ White balance of the image is predicted using latent output of an encoder.

        Args:
            x (torch.tensor): Input RGB image.
            latent_out (torch.tensor): Output from the last layer of an encoder.
            wb_gate (torch.tensor): White-balance gate used to change the influence of color scaled image.

        Returns:
            torch.tensor: returns White-Balanced image. 
        """
        log_wb_range = 0.5
        wb = self.wb_module(latent_out)
        wb = torch.exp(self.tanh_range(wb,-log_wb_range,log_wb_range))
        
        color_scaling = 1./(1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] +
        0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1)*wb
        wb_out = wb.unsqueeze(2).unsqueeze(3)*x
        wb_out = (wb_out-wb_out.min())/(wb_out.max()-wb_out.min())
        wb_out = wb_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)*wb_out
        return wb_out
    
    def gamma_balance(self,x : torch.tensor,latent_out : torch.tensor,gamma_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            gamma_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        log_gamma = torch.log(torch.tensor(2.5))
        gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(self.tanh_range(gamma,-log_gamma,log_gamma))
        g = torch.pow(torch.maximum(x,torch.tensor(1e-4)),gamma)
        g = (g-g.min())/(g.max()-g.min())
        g = g*gamma_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return g

    def sharpning(self,x : torch.tensor,latent_out: torch.tensor,sharpning_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            sharpning_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        out_x = self.blur(x)
        y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        y = self.tanh_range(y,torch.tensor(0.1),torch.tensor(1.))
        s = x + (y*(x-out_x))
        s = (s-s.min())/(s.max()-s.min())
        s = s * (sharpning_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        return s
    
    def identity(self,x : torch.tensor,identity_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            identity_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        x = x*identity_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return x

    def contrast(self,x : torch.tensor,latent_out : torch.tensor,contrast_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            contrast_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        alpha = torch.tanh(self.contrast_module(latent_out))
        luminance = torch.minimum(torch.maximum(self.rgb2lum(x), torch.tensor(0.0)), torch.tensor(1.0)).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha) 
        contrast_image = (contrast_image-contrast_image.min())/(contrast_image.max()-contrast_image.min())
        contrast_image = contrast_image * contrast_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return contrast_image

    def tone(self,x : torch.tensor,latent_out : torch.tensor,tone_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            tone_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        curve_steps = 8
        tone_curve = self.tone_module(latent_out).reshape(-1,1,curve_steps)
        tone_curve = self.tanh_range(tone_curve,0.5, 2)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        total_image = x * 0
        for i in range(curve_steps):
            total_image += torch.clamp(x - 1.0 * i /curve_steps, 0, 1.0 /curve_steps) \
                            * tone_curve[:,:,i].unsqueeze(2).unsqueeze(3)
        total_image *= curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        total_image = (total_image-total_image.min())/(total_image.max()-total_image.min())
        total_image = total_image * tone_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return total_image

    def forward(self,x,linear_proj):
        gate = self.tanh_range(self.gate_module(linear_proj),0.01,1.0)
        wb_out = self.white_balance(x,linear_proj,gate[:,0])
        gamma_out = self.gamma_balance(x,linear_proj,gate[:,1])
        identity_out = self.identity(x,gate[:,2])
        sharpning_out = self.sharpning(x,linear_proj,gate[:,3])
        fog_out = self.defog(x,linear_proj,gate[:,4])
        contrast_out = self.contrast(x,linear_proj,gate[:,5])
        tone_out = self.tone(x,linear_proj,gate[:,6])
        x = wb_out + gamma_out   + fog_out + sharpning_out + contrast_out + tone_out + identity_out
        x = (x-x.min())/(x.max()-x.min())
        return x,gate


class MultiLevelGDIP(nn.Module):
    def __init__(self):
        super(MultiLevelGDIP,self).__init__()
        self.adp_pool_s = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_s = torch.nn.Sequential(torch.nn.Linear(256,256),
                                                torch.nn.ReLU(True))
        self.gdip_s = GatedDIP(256,7)

        self.adp_pool_m = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_m = torch.nn.Sequential(torch.nn.Linear(512,256),
                                                torch.nn.ReLU(True))
        self.gdip_m = GatedDIP(256,7)

        self.adp_pool_l = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_l = torch.nn.Sequential(torch.nn.Linear(1024,256),
                                                torch.nn.ReLU(True))
        self.gdip_l = GatedDIP(256,7)
    
    def forward(self,x,x_s,x_m,x_l):
        img_list = list()
        gates_list = list()
        adp_s = self.adp_pool_s(x_s)
        linear_proj_s = self.linear_proj_s(adp_s.view(adp_s.shape[0],-1))
        x, gate_s = self.gdip_s(x,linear_proj_s)
        img_list.append(x)
        gates_list.append(gate_s)

        adp_m = self.adp_pool_m(x_m)
        linear_proj_m = self.linear_proj_m(adp_m.view(adp_m.shape[0],-1))
        x, gate_m = self.gdip_m(x,linear_proj_m)
        img_list.append(x)
        gates_list.append(gate_m)

        adp_l = self.adp_pool_l(x_l)
        linear_proj_l = self.linear_proj_l(adp_l.view(adp_l.shape[0],-1))
        x, gate_l = self.gdip_l(x,linear_proj_l)
        img_list.append(x)
        gates_list.append(gate_l)

        return x,img_list,gates_list





class Yolov3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, cfg, init_weights=True):
        super(Yolov3, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backnone = Darknet53()
        self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        # small
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        
        # medium
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])

        # large
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        # multilevel gdip
        self.__multilevel_gdip = MultiLevelGDIP()


        if init_weights:
            self.__init_weights()


    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.__backnone(x)
        out_x,img_list,gates_list = self.__multilevel_gdip(x,x_s,x_m,x_l)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return out_x,gates_list[-1],p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return out_x,gates_list[-1],p, torch.cat(p_d, 0)


    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))


if __name__ == '__main__':
    net = Yolov3()
    print(net)

    in_img = torch.randn(4, 3, 448, 448)
    x,gates_list,p, p_d = net(in_img)
    print('out_img shape:',x.shape)
    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)

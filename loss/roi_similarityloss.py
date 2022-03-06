import torch
import torch.nn.functional as F
import torch.nn as nn
import face_alignment 
import numpy as np
from torchvision import models
from pathlib import Path
class VGGFeat(torch.nn.Module):
    """
    Input: (B, C, H, W), RGB, [-1, 1]
    """
    def __init__(self, model_dir):
        super().__init__()
        self.model = models.vgg19(pretrained=False)
        self.build_vgg_layers()
        weight_path = Path(model_dir,'vgg19.pth')
        self.model.load_state_dict(torch.load(weight_path))

        self.register_parameter("RGB_mean", nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)))
        self.register_parameter("RGB_std", nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))
        
        # self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        self.features = []
        # feature_layers = [0, 3, 8, 17, 26, 35]
        feature_layers = [0, 8, 17, 26, 35]
        for i in range(len(feature_layers)-1): 
            module_layers = torch.nn.Sequential() 
            for j in range(feature_layers[i], feature_layers[i+1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        x = (x + 1) / 2
        x = (x - self.RGB_mean) / self.RGB_std
        if x.shape[3] < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        features = []
        for m in self.features:
            # print(m)
            x = m(x)
            features.append(x)
        return features 

class ROI_Similarityloss(torch.nn.Module):
    
    def __init__(self, device):
        super(ROI_Similarityloss, self).__init__() 

        self.fl_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device=device, flip_input=False)

        self.GetVggFeats = VGGFeat()
    
    def get_part_location(self,Landmarks):

        '''
            borrow code from https://github.com/csxmli2016/DFDNet/blob/whole/test_FaceDict.py
        '''
        
        Landmarks = []

        Landmarks = np.array(Landmarks) 
        Map_LE = list(np.hstack((range(17,22), range(36,42))))
        Map_RE = list(np.hstack((range(22,27), range(42,48))))
        Map_NO = list(range(29,36))
        Map_MO = list(range(48,68))
        try:
            #left eye
            Mean_LE = np.mean(Landmarks[Map_LE],0)
            L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
            Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
            #right eye
            Mean_RE = np.mean(Landmarks[Map_RE],0)
            L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
            Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
            #nose
            Mean_NO = np.mean(Landmarks[Map_NO],0)
            L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
            Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
            #mouth
            Mean_MO = np.mean(Landmarks[Map_MO],0)
            L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))
            Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
        except:
            #if wrong for landmark detection, it means the generated face not good
            return 0
        return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)
    
    def forward(self, gen_im, ref_im):
        self.fl_detector(gen_im)
        
        ref_landmarks = self.fl_detector(ref_im)

        Gen_VggFeatures = self.GetVggFeats(gen_im)
        Ref_VggFeatures = self.GetVggFeats(ref_im)
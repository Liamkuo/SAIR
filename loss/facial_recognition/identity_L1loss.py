import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from loss.facial_recognition.iresnet_encoder import IResNet, IBasicBlock

class ID_L1loss(torch.nn.Module):
    
    def __init__(self, model_dir):
        super(ID_L1loss, self).__init__()      
        
        f = Path(model_dir,'irestnet50.pth')
        #iresnet50
        self.face_encoder = IResNet(IBasicBlock, [3, 4, 14, 3])
        for param in self.face_encoder.features.parameters():
            param.requires_grad = False

        self.face_encoder.load_state_dict(torch.load(str(f),map_location='cpu'))
        self.face_encoder.cuda()
        self.face_encoder.eval()
        self.cos_loss = torch.nn.CosineSimilarity(dim=1)
        for param in self.face_encoder.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).cuda()
        self.std= torch.tensor([0.229, 0.224, 0.225], requires_grad=False).cuda()
        torch.autograd.set_detect_anomaly(True)
        
    def tensor_normalize(self,img):
        norm_img= (img - self.mean[None, :, None, None])/self.std[None, :, None, None]
        return norm_img

    def forward(self, gen_im, ref_im,tensor_norm=False):

        gen_input_norm = self.tensor_normalize(gen_im)
        id_input_norm = self.tensor_normalize(ref_im)

        gen_input = gen_input_norm[:,:,100:900,100:900]
        id_input = id_input_norm[:,:,100:900,100:900]

        gen_input = F.interpolate(gen_input, size=(112,112), mode='bicubic',align_corners=True)
        id_input = F.interpolate(id_input, size=(112,112), mode='bicubic',align_corners=True)

        gen_face_encoder = self.face_encoder(gen_input)
        ref_face_encoder = self.face_encoder(id_input)
        return gen_face_encoder, ref_face_encoder

def crop(variable,tw,th):
    w, h = variable.size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[x1:x1+tw,y1:y1+th]
    



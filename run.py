import argparse
import math
import os,sys,cv2
from numpy.core.fromnumeric import repeat
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
sys.path.append('.')
sys.path.append('models')
sys.path.append('loss')
import torch.nn.functional as F 
import torch,torchvision
from torch import optim
from tqdm import tqdm
import numpy as np
from PIL import Image
from models.stylegan2.model import Generator
from loss.facial_recognition.identity_L1loss import ID_L1loss
from loss.Bicubic import BicubicDownSample
from loss.histogram import HistogramLoss
from loss.DiffJPEG import DiffJPEG
import face_alignment
fa_handle = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device='cuda', flip_input=False)


def get_decay_lr(_t, _initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - _t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, _t / rampup)

    return _initial_lr * lr_ramp

def cross_loss(latent):      
    X = latent.view(-1, 1, 18, 512)
    Y = latent.view(-1, 18, 1, 512)
    A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
    B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
    _loss = 2*torch.atan2(A, B)
    _loss = ((_loss.pow(2)*512).mean((1, 2))/8.).sum()
    return _loss

def check_checkpoint_exists(model_weights_filename):
    d_dict={"pretrained_models/stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1CW6jrPkE3kAp9ovQ0Lz6_-h1K6jV72t5",
            "pretrained_models/irestnet50.pth": "https://drive.google.com/uc?id=10ygGBl9PBqff1VVasXHdxcKzBcAyS3Yq"}

    if not os.path.isfile(model_weights_filename) and model_weights_filename in d_dict:
        
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )
        try:
            from gdown import download as drive_download

            drive_download(d_dict[model_weights_filename], model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or manually download the checkpoint file:",
                d_dict[model_weights_filename]
            )

def main(args):

    results_dir = args.results_dir
    lr_im_path = args.input_img_path  
    id_im_path = args.guide_img_path 
    stylegan_size = args.stylegan_size
    model_path = args.model_dir
    e_type = args.emotion
    up_scale = args.ds_size
    

    os.makedirs(results_dir, exist_ok=True)


    emotion_norm = { 'disgust':torch.from_numpy(np.load('pretrained_models/latent_direction/emotion_disgust.npy')).float().view(-1, 18, 512).cuda().detach(),
                        'sad':torch.from_numpy(np.load('pretrained_models/latent_direction/emotion_sad.npy')).float().view(-1, 18, 512).cuda().detach(),
                        'angry':torch.from_numpy(np.load('pretrained_models/latent_direction/emotion_angry.npy')).float().view(-1, 18, 512).cuda().detach(),
                        'neutral':torch.from_numpy(np.load('pretrained_models/latent_direction/emotion_neutral.npy')).float().view(-1, 18, 512).cuda().detach(),
                        'surprise':torch.from_numpy(np.load('pretrained_models/latent_direction/emotion_surprise.npy')).float().view(-1, 18, 512).cuda().detach(),
                        'happy':torch.from_numpy(np.load('pretrained_models/latent_direction/emotion_surprise.npy')).float().view(-1, 18, 512).cuda().detach(),
                        'fear':torch.from_numpy(np.load('pretrained_models/latent_direction/emotion_fear.npy')).float().view(-1, 18, 512).cuda().detach()}
    
    emotion_gama = {'disgust':2.5,
                 'sad':8,
                 'angry':5,
                 'neutral':3,
                 'surprise':3, 
                 'happy':3,
                 'fear':3}


    G_w = Generator(stylegan_size, 512, 8)
    g_path = os.path.join(model_path,"stylegan2-ffhq-config-f.pt")
    check_checkpoint_exists(g_path)
    f_path = os.path.join(model_path,"irestnet50.pth")
    check_checkpoint_exists(f_path)
    
    G_w.load_state_dict(torch.load(g_path)["g_ema"], strict=False)
    G_w.eval()
    G_w = G_w.cuda()
    mean_latent = G_w.mean_latent(4096)
 
    if args.guide_latent_path:
        latent_code_init = torch.load(args.guide_latent_path).cuda()
        latent_code_init = latent_code_init[0].unsqueeze(0) #  choose the first inverse latent code as the default guide latent code
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    latent_code_init = latent_code_init.detach()
    latent_code_init.requires_grad=False

    latent = latent_code_init.clone()
    latent.requires_grad = True


    optimizer = optim.Adam([latent], lr=args.lr)

    
    transform_totensor = torchvision.transforms.ToTensor()

    id_im = transform_totensor(Image.open(id_im_path)).unsqueeze(0).cuda().detach()
    lr_im = transform_totensor(Image.open(lr_im_path)).unsqueeze(0).cuda().detach()


    D_bicubic = BicubicDownSample(factor=up_scale)

    D_JPEG = DiffJPEG(height=1024//up_scale, width=1024//up_scale, differentiable=True, quality=10).cuda()
    

    id_fea_extract = ID_L1loss(model_path)
    id_loss = torch.nn.CosineSimilarity(dim=1).cuda()
    emotion_loss = torch.nn.CosineSimilarity(dim=2).cuda()

    histogram_loss = HistogramLoss( loss_fn='emd', num_bins=256, yuvgrad=False)
    histogram_loss.histlayer = histogram_loss.histlayer.cuda()


    tqbar = tqdm(range(args.step))

    hist_weight = 0

    for i in tqbar:



        t_decay = i / args.step
        lr = get_decay_lr(t_decay, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen_tmp, _ = G_w([latent], input_is_latent=True, randomize_noise=True)

        
        img_gen_tmp = torch.clamp(img_gen_tmp,-1,1)
        _min = img_gen_tmp.min().detach()
        _max = img_gen_tmp.max().detach()
        img_gen  = (img_gen_tmp - _min)/(_max-_min)

        l2_loss = ((latent_code_init - latent) ** 2).sum()
        
        l2_ds_loss = (D_JPEG(D_bicubic(img_gen)) - lr_im).pow(2).mean((1, 2, 3)).clamp(min=0.0001).sum() \
                    if args.jpeg_degrade else (D_bicubic(img_gen) - lr_im).pow(2).mean((1, 2, 3)).clamp(min=0.0001).sum()

        gen_fea, id_fea = id_fea_extract.forward(img_gen,id_im.detach())
        cos_loss = 1-id_loss(gen_fea,id_fea)

        # cross loss
        c_loss = cross_loss(latent)

        
        emt_loss = 1-torch.abs(emotion_loss(latent,emotion_norm[e_type])).mean()

        loss = 120*l2_ds_loss + 0.05*c_loss  + 0.4*cos_loss + 0.001*l2_loss 
        
        if args.enable_emotion:
           loss = loss + 0.1*emt_loss 
        


        if args.enable_hist:
            if i >200:
                hist_weight = 0.05 #0.05
                ## face mask 
                img_convexhull = img_gen.squeeze().cpu().detach().numpy()*255.0
                img_batch_input = torch.Tensor(np.stack([img_convexhull])).cuda()
                img_landmarks = fa_handle.get_landmarks_from_batch(img_batch_input)
                
                hull_pt_dst = cv2.convexHull(np.float32(img_landmarks[0]))
                face_mask = np.zeros((1024,1024), dtype=np.float32)
                cv2.fillConvexPoly(face_mask, np.array(hull_pt_dst,dtype=np.int), 255)/255.0
                face_mask = cv2.resize(face_mask,(256,256))
                face_mask = torch.Tensor(face_mask)          
                face_mask = face_mask.view([1,1,256,256]).repeat(1,3,1,1).cuda().detach()

                emd_loss, _ = histogram_loss( face_mask* F.interpolate(img_gen,scale_factor=1/4,mode='bicubic'),face_mask*F.interpolate(lr_im,scale_factor=256/(1024/up_scale),mode='bicubic'))

                hist_loss = hist_weight*emd_loss/(256*256)
                loss = loss + hist_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermidate_iter > 0 and i % args.save_intermidate_iter == 0:
            with torch.no_grad():
                latent_emotion = latent+ emotion_norm[e_type]*emotion_gama[e_type] if args.enable_emotion else latent 
                img_gen_tmp, _ = G_w([latent_emotion], input_is_latent=True, randomize_noise=True)      
                
                img_gen_tmp = torch.clamp(img_gen_tmp,-1,1)
                _min = img_gen_tmp.min().detach()
                _max = img_gen_tmp.max().detach()
                img_gen  = (img_gen_tmp - _min)/(_max-_min)
            torchvision.utils.save_image(img_gen, f"{results_dir}/{str(i).zfill(5)}.png", normalize=True, range=(0, 1))
            final_result = img_gen

    return final_result,torch.cat([F.interpolate(lr_im,scale_factor=up_scale,mode='bicubic'),final_result],dim=3)


#plate hair
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-e","--emotion", type=str, default="happy",choices=["happy","neutral","surprise","angry", "sad","disgust","fear"], help="set facial expression")
    parser.add_argument("-ee","--enable_emotion", action='store_true',  help="enable the facial expression control") 
    parser.add_argument("-eh","--enable_hist", action='store_true',  help="enable the histgram loss")
    parser.add_argument("-j","--jpeg_degrade", action='store_true',  help="enable the jpeg degradation model")
    parser.add_argument("-i","--input_img_path", type=str, default=None, help="the input image path") 
    parser.add_argument("-gl","--guide_latent_path", type=str, default=None, help="the guide latent path")
    parser.add_argument("-gi","--guide_img_path", type=str, default=None, help="the guide image path")
    parser.add_argument("-s","--save_intermidate_iter", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("-o","--results_dir", type=str, default= "results", help="output path of the results")
    parser.add_argument("--model_dir", type=str, default="pretrained_models", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--ds_size", type=int, default=32, help="upsample scale size")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=400, help="number of optimization steps")
    parser.add_argument("--lambda", type=str, default="120,0.05,0.4,0.001,0.1,0.05", help="trade-off parameters in loss function")

    parser.add_argument("--gpu_id", type=str, default="0",help="select gpu")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join('%s' % id for id in args.gpu_id)

    for i in os.environ["CUDA_VISIBLE_DEVICES"]:
        torch.cuda.set_device(int(i))

    print("Current selected GPU is %s" % (torch.cuda.current_device()))

    result_image,concat_img = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final.jpg"), normalize=True, scale_each=True, range=(0, 1))
    torchvision.utils.save_image(concat_img.detach().cpu(), os.path.join(args.results_dir, "final_cmp.jpg"), normalize=True, scale_each=True, range=(0, 1))



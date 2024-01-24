# SAIR


  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

> For image restoration, methods leveraging priors from generative models have been proposed and demonstrated a promising capacity to robustly restore photorealistic and high-quality results. However, these methods are susceptible to semantic ambiguity, particularly with images that have obviously correct semantics such as facial images. In this paper, we propose a semantic-aware latent space exploration method for image restoration (SAIR). By explicitly modeling semantics information from a given reference image, SAIR is able to reliably restore severely degraded images not only to high-resolution and highly realistic looks but also to correct semantics. Quantitative and qualitative experiments collectively demonstrate the superior performance of the proposed SAIR.
<p align="center">
<img src="figure/sample1.jpg" width="800px"/>
</p>

<p align="center">
<img src="figure/sample2.jpg" width="800px"/>
</p>


## Description

This is the code for the paper "Self-Supervised Face Image Restoration with One-shot  Reference" (SAIR)

To avoid the torturous environment configuration, we recommend you use [Anaconda](https://www.anaconda.com/products/individual#Downloads) to finish the next configuration. 

## Configuration

1. `conda create -n SAIR python=3.6`  
3. `source activate SAIR && conda install cudatoolkit=10.1`
4. `pip3 install -r requirement.txt -f https://download.pytorch.org/whl/torch_stable.html`
5. Download "stylegan2-ffhq-config-f.pt" by [Google Drvier](https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT) and "irestnet50.pth" by [Google Driver](https://drive.google.com/file/d/1ivbPmsqTFXB1PFi5C7jNr8yH7QffVQUq/view?usp=drive_link). Put these two model files in the directory "pretrained_models/"
6. Put the guide image and the corresponding inverse latent code in "guide_info/your_dir/". We solve for the inverse latent code by [e4e](https://github.com/omertov/encoder4editing) . In "guide_info", we provide two examples. 
7. All done. 

## Inference

`python run.py -i test_img/obama.png -gl guide_info/obama/latents.pt  -gi guide_info/obama/ref.png -e disgust -ee -eh`

## Citation
```
@inproceedings{guo2023selfsupervised,
      title={Self-Supervised Face Image Restoration with a One-Shot Reference}, 
      author={Yanhui Guo and Fangzhou Luo and Shaoyuan Xu},
      journal={The 49th IEEE International Conference on Acoustics, Speech, & Signal Processing (ICASSP)},
      year={2024}
}
```


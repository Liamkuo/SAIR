# SAIR


  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

> For image restoration, most existing deep learning based methods tend to overfit the training data leading to bad results when encountering unseen degradations out of the assumptions for training. To improve the robustness, generative adversarial network (GAN) prior based methods have been proposed, revealing a promising capability to restore photo-realistic and high-quality results. But these methods suffer from semantic confusion, especially on semantically significant images such as face images. In this paper, we propose a semantic-aware latent space exploration method for image restoration (SAIR). By explicitly modeling referenced semantics information, SAIR can consistently restore severely degraded images not only to high-resolution highly-realistic looks but also to correct semantics. Quantitative and qualitative experiments collectively demonstrate the effectiveness of the proposed SAIR.

<p align="center">
<img src="figure/sample1.jpg" width="800px"/>
</p>

<p align="center">
<img src="figure/sample2.jpg" width="800px"/>
</p>


## Description

This is the code for the paper "Semantic-Aware Latent Space Exploration for Image Restoration" (SAIR)

To avoid the torturous environment configuration, we recommend you use [Anaconda](https://www.anaconda.com/products/individual#Downloads) to finish the next configuration. 

## Configuration

1. `conda create -n SAIR python=3.6`  
3. `source activate SAIR && conda install cudatoolkit=10.1`
4. `pip3 install -r requirement.txt -f https://download.pytorch.org/whl/torch_stable.html`
5. Download "stylegan2-ffhq-config-f.pt" by [Google Drvier](https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT) and "irestnet50.pth" by [Google Driver](https://drive.google.com/uc?id=10ygGBl9PBqff1VVasXHdxcKzBcAyS3Yq). Put these two model files in the directory "pretrained_models/"
6. Put the guide image and the corresponding inverse latent code in "guide_info/your_dir/". We solve for the inverse latent code by [e4e](https://github.com/omertov/encoder4editing) . In "guide_info", we provide two examples. 
7. All done. 

## Inference

`python run.py -i test_img/obama.png -gl guide_info/obama/latents.pt  -gi guide_info/obama/ref.png -e disgust -ee -eh`


## Citation

Please cite the following paper if you feel SAIR useful to your research

```
@article{guo2022semantic,
  title={Semantic-Aware Latent Space Exploration for Face Image Restoration},
  author={Guo, Yanhui and Luo, Fangzhou and Wu, Xiaolin},
  journal={arXiv preprint arXiv:2203.03005},
  year={2022}
}
```

# **GDIP: Gated Differentiable Image Processing for Object-Detection in Adverse Conditions**  [[Website](https://gatedip.github.io/)][[Paper](https://arxiv.org/pdf/2209.14922.pdf)]
![GDIP-Yolo](docs/images/architecture_gdip.png)

## **Abstract** 
Detecting objects under adverse weather and
lighting conditions is crucial for the safe and continuous
operation of an autonomous vehicle, and remains an unsolved
problem. We present a Gated Differentiable Image Processing
(GDIP) block, a domain-agnostic network architecture, which
can be plugged into existing object detection networks (e.g.,
Yolo) and trained end-to-end with adverse condition images
such as those captured under fog and low lighting. Our proposed GDIP block learns to enhance images directly through the
downstream object detection loss. This is achieved by learning
parameters of multiple image pre-processing (IP) techniques
that operate concurrently, with their outputs combined using
weights learned through a novel gating mechanism. We further
improve GDIP through a multi-stage guidance procedure for
progressive image enhancement. Finally, trading off accuracy
for speed, we propose a variant of GDIP that can be used as
a regularizer for training Yolo, which eliminates the need for
GDIP-based image enhancement during inference, resulting in
higher throughput and plausible real-world deployment. We
demonstrate significant improvement in detection performance
over several state-of-the-art methods through quantitative and
qualitative studies on synthetic datasets such as PascalVOC, and
real-world foggy (RTTS) and low-lighting (ExDark) datasets.

## **Datasets** ##
|Dataset Name|Link|
|----|----|
|PascalVOC2007|[link](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)|
|PascalVOC2012|[link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)|
|PascalVOC(test)|[link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sanket_kalwar_research_iiit_ac_in/EapxTscsKVtBos1dcZgKCQ8BpttDvrEslW2nqrjdg-_ZaQ?e=KTPKjF)|
|ExDark|[link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sanket_kalwar_research_iiit_ac_in/ER-luZLMSldBlg_YG7hkTWwBICaSUrBD0jh3Y_w0FGVyTQ?e=uUKYRG)|
|RTTS|[link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sanket_kalwar_research_iiit_ac_in/ER2WrtnuU8JJjmhqgZ1Y-eEBUjxLmFyFD6waidovlpQWmg?e=fX7xqz)|

## **Weights File**  ##
|Model|RTTS|ExDark|
|---|---|---|
|GDIP-Yolo|[link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sanket_kalwar_research_iiit_ac_in/ElCY3EnsKtNFqTMXU8aYrpUB6XgdvACuXo4EMNQmRTbxaQ?e=yZTnj1)|[link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sanket_kalwar_research_iiit_ac_in/ErOQ2LhleKVIpEYYAXg3oKMBDNlfjodl2XSj2DtbH9Ml8g?e=2Xc55M)|
|MGDIP-Yolo|[link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sanket_kalwar_research_iiit_ac_in/EiIOIJTd70VNv9hFaFx6NQ8BX23zVtan5TthzM-wui3kkA?e=fj3auq)|[link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sanket_kalwar_research_iiit_ac_in/EpjhL9slOqxNiVCKAboN0ycBRW6GxoAeEcdkBKurTg6ZSg?e=jOneHD)|
|GDIP-regularizer|[link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sanket_kalwar_research_iiit_ac_in/EhnRsxF8AUxNmiaHejavZB0B_BM10AHd7yO9MX60MCETJA?e=h5Pf4h)|[link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sanket_kalwar_research_iiit_ac_in/EvWkGr9GzE1HqMCkLkbnD0oBjLPTs5nXeVAm1UftU245jQ?e=W1NIRA)|

# Setting up the environment: #
```bash
# Install PyTorch 1.12.0+cu116
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

```

# **Steps for evaluation:** #
```bash
# Evaluate GDIP-Yolo on ExDark images
> python3 test_GDIP_ExDark.py --weights /path/to/best.pt

# Evaluate GDIP-Yolo on RTTS images
> python3 test_GDIP_RTTS.py --weights /path/to/best.pt

# Evaluate MGDIP-Yolo on ExDark images
> python3 test_MGDIP_ExDark.py --weights /path/to/best.pt

# Evaluate MGDIP-Yolo on RTTS images
> python3 test_MGDIP_RTTS.py --weights /path/to/best.pt

# Evaluate GDIP-REG on ExDark images
> python3 test_GDIP_REG_ExDark.py --weights /path/to/best.pt

# Evaluate GDIP-REG on RTTS images
> python3 test_GDIP_REG_RTTS.py --weights /path/to/best.pt
```
# **Running Inference:** #
```bash
# Infer GDIP-Yolo(trained on ExDark) on custom images
> python3 infer_GDIP_ExDark.py --weights /path/to/best.pt --visiual /path/to/images

# Infer GDIP-Yolo(trained on RTTS) on custom images
> python3 infer_GDIP_RTTS.py --weights /path/to/best.pt --visiual /path/to/images

# Infer MGDIP-Yolo(trained on ExDark) on custom images
> python3 infer_MGDIP_ExDark.py --weights /path/to/best.pt --visiual /path/to/images

# Infer MGDIP-Yolo(trained on RTTS) on custom images
> python3 infer_MGDIP_RTTS.py --weights /path/to/best.pt --visiual /path/to/images

# Infer GDIP-REG(trained on ExDark) on custom images
> python3 infer_GDIP_REG_ExDark.py --weights /path/to/best.pt --visiual /path/to/images

# Infer GDIP-REG(trained on RTTS) on custom images
> python3 infer_GDIP_REG_RTTS.py --weights /path/to/best.pt --visiual /path/to/images
```

# Training Script:
This work is a funded project, so we have only released evaluation and inference scripts for now. We shall release the training script in the future. However, for now, you can follow the implementation details for training from the paper. As we have provided the class for GDIP-Yolo, importing it along with the hyperparameters mentioned in the paper should work.

# Who to Contact:
* [Sanket H Kalwar](sanket.kalwar@reseach.iiit.ac.in)
* [Dhruv Patel](dhruv.patel@research.iiit.ac.in)

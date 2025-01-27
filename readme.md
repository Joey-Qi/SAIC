<p align="center">

  <h2 align="center">Style-Aligned Image Composition for Robust Detection of Abnormal Cells in Cytopathology</h2>
  <p align="center">
    <a href="https://xavierchen34.github.io/"><strong>Xi Chen</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=JYVCn3AAAAAJ&hl=en"><strong>Lianghua Huang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=8zksQb4AAAAJ&hl=zh-CN"><strong>Yu Liu</strong></a>
    ·
    <a href="https://shenyujun.github.io/"><strong>Yujun Shen</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=7LhjCn0AAAAJ&hl=en"><strong>Deli Zhao</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Hengshuang Zhao</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2307.09481"><img src='https://img.shields.io/badge/arXiv-AnyDoor-red' alt='Paper PDF'></a>
        <a href='https://ali-vilab.github.io/AnyDoor-Page/'><img src='https://img.shields.io/badge/Project_Page-AnyDoor-green' alt='Project Page'></a>
        <a href='https://modelscope.cn/studios/damo/AnyDoor-online/summary'><img src='https://img.shields.io/badge/ModelScope-AnyDoor-yellow'></a>
        <a href='https://huggingface.co/spaces/xichenhku/AnyDoor-online'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
        <a href='https://replicate.com/lucataco/anydoor'><img src='https://replicate.com/lucataco/anydoor/badge'></a>
    <br>
    <b>The University of Hong Kong &nbsp; | &nbsp;  Alibaba Group  | &nbsp;  Ant Group </b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/Teaser.png">
    </td>
    </tr>
  </table>

## News
* **[2023.12.17]** Release train & inference & demo code, and pretrained checkpoint.
* **[2023.12.24]** 🔥 Support online demo on [ModelScope](https://modelscope.cn/studios/damo/AnyDoor-online/summary) and 
 [HuggingFace](https://huggingface.co/spaces/xichenhku/AnyDoor-online).
* **[Soon]** Release the new version paper.
* **[On-going]** Scale-up the training data and release stronger models as the foundaition model for downstream region-to-region generation tasks.
* **[On-going]** Release specific-designed models for downstream tasks like virtual tryon, face swap, text and logo transfer, etc.


## Installation
Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate anydoor
```
or `pip`:
```bash
pip install -r requirements.txt
```
Additionally, for training, you need to install panopticapi, pycocotools, and lvis-api.
```bash
pip install git+https://github.com/cocodataset/panopticapi.git

pip install pycocotools -i https://pypi.douban.com/simple

pip install lvis
```
## Download Checkpoints
Download AnyDoor checkpoint: 
* [ModelScope](https://modelscope.cn/models/damo/AnyDoor/files)
* [HuggingFace](https://huggingface.co/spaces/xichenhku/AnyDoor/tree/main)

**Note:** We include all the optimizer params for Adam, so the checkpoint is big. You could only keep the "state_dict" to make it much smaller.


Download DINOv2 checkpoint and revise `/configs/anydoor.yaml` for the path (line 83)
* URL: https://github.com/facebookresearch/dinov2?tab=readme-ov-file

Download Stable Diffusion V2.1 if you want to train from scratch.
* URL: https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main


## Inference
We provide inference code in `run_inference.py` (from Line 222 - ) for both inference single image and inference a dataset (VITON-HD Test). You should modify the data path and run the following code. The generated results are provided in `examples/TestDreamBooth/GEN` for single image, and `VITONGEN` for VITON-HD Test.

```bash
python run_inference.py
```
The inferenced results on VITON-Test would be like [garment, ground truth, generation].

*Noticing that AnyDoor does not contain any specific design/tuning for tryon, we think it would be helpful to add skeleton infos or warped garment, and tune on tryon data to make it better :)*
  <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/tryon.png">
    </td>
    </tr>
  </table>


Our evaluation data for DreamBooth an COCOEE coud be downloaded at Google Drive:
* URL: [to be released]


## Acknowledgements
This project is developped on the codebase of [ControlNet](https://github.com/lllyasviel/ControlNet). We  appreciate this great work! 

<p align="center">

  <h2 align="center">Style-Aligned Image Composition for Robust Detection of Abnormal Cells in Cytopathology</h2>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/framework.png">
    </td>
    </tr>
  </table>


## Installation
Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate SAIC
```
or `pip`:
```bash
pip install -r requirements.txt
```

## Download Checkpoints
Download SAIC checkpoint and revise `/configs/inference.yaml` for the path (line 1) 
* [ModelScope](https://modelscope.cn/models/JoeyQi/SAIC/files)

**Note:** We include all the optimizer params for Adam, so the checkpoint is big. You could only keep the "state_dict" to make it much smaller.


Download DINOv2 checkpoint and revise `/configs/SAIC.yaml` for the path (line 83)
* URL: https://github.com/facebookresearch/dinov2?tab=readme-ov-file

Download dinov2-giant and revise `/run_inference_FullFull.py` for the path (line 42-43)

```bash
# Change source
export HF_ENDPOINT=https://hf-mirror.com

# Download model from Huggingface 
huggingface-cli download facebook/dinov2-giant --resume-download --local-dir /mnt/data/dinov2
```

## Inference
We provide inference code in `run_inference_FullFull.py` (from Line 493 - ) for cytopathological images synthesis. You should modify the data path (Organize like `examples`) and run the following code.

```bash
python run_inference_FullFull.py
```

*Some examples are shown as follows:*
  <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/demonstration.png">
    </td>
    </tr>
  </table>

## Acknowledgements
This project is developped on the codebase of [ControlNet](https://github.com/lllyasviel/ControlNet). We appreciate this great work! 

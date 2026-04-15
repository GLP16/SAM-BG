<h2 align="center">A visual transformer and multi-task learning framework for building generalization from raster maps </h2>
This repository is the code implementation of the paper "A visual transformer and multi-task learning framework for building generalization from raster maps"

<h3 align="left"><b>Step-by-step guides</b></h3>

### I. Install dependencies:
```bash
pip install -r requirements.txt
```
### II. Dataset Structure
The authors do not have permission to share the OSM-Stuttgar data used in this study. The Swisstopo-Zurich dataset is publicly available from the Swiss Federal Office of Topography at https://www.swisstopo.admin.ch/de. The data structure is as follows:
```
dataset
├── train
|    ├── img
|    ├── label
├── val
|    ├── img
|    ├── label
├── test
     ├── img
     ├── label
```
### III. Weights Download 
The download address for SAM's pre-trained ViT-B weights is:
https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth
### Ⅳ. Training
```bash
python train.py
```

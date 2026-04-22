<h2 align="center">A visual transformer and multi-task learning framework for building generalization from raster maps </h2>
This repository is the code implementation of the paper "A visual transformer and multi-task learning framework for building generalization from raster maps"

<h3 align="left"><b>Step-by-step guides</b></h3>

### I. Install dependencies:
```bash
pip install -r requirements.txt
```
### II. Dataset Structure
Due to licensing restrictions on the synthetic dataset used in this study, the full dataset cannot be shared. The Swiss datasets are publicly available from the Swiss Federal Office of Topography (swisstopo). The 1:10,000 dataset can be accessed at
https://www.swisstopo.admin.ch/fr/carte-nationale-swiss-map-vector-10,
and the 1:25,000 dataset can be accessed at
https://www.swisstopo.admin.ch/fr/carte-nationale-swiss-map-vector-25. 

After rasterizing the vector data, run the following scripts in the data_process folder sequentially:
```bash
python segmentation.py
```
```bash
python filter.py
```
```bash
python indivation.py
```
The final dataset structure is as follows:
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
The download address for SAM's pre-trained ViT-B weights is: https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth
. Place the weights in the project root directory for subsequent training.
### Ⅳ. Training
To train the model, simply run `train.py` directly.
```bash
python train.py
```

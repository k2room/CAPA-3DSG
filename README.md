# CAPA-3DSG
**Context-Aware Part-Affordance 3D Scene Graph Generation with Open Vocabulary**

We introduce CAPA-3DSG, a novel framework for Context-Aware Part-Affordance 3D Scene Graph Generation with Open Vocabulary. Our approach enables fine-grained reasoning over object parts and their affordances while generalizing to unseen categories through open-vocabulary learning, bridging the gap between part-level semantics and contextual scene understanding.

# Setting
## Create conda environment
```
conda create -n capa python=3.10

conda activate capa

conda install -y -c pytorch -c nvidia pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 

conda install -y -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

conda install -y -c nvidia cuda-toolkit=11.8 cuda-nvcc=11.8

conda install -y https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2

# some packages change the numpy version, or confuse conda env and pip. use constraints.txt.
# (e.g.) export PIP_CONSTRAINT="/home/main/workspace/k2room2/CAPA-3DSG/constraints.txt"
export PIP_CONSTRAINT="path/CAPA-3DSG/constraints.txt"

pip install --upgrade-strategy only-if-needed tyro timm==1.0.17 open_clip_torch wandb h5py openai hydra-core distinctipy ultralytics dill supervision==0.21.0 open3d imageio natsort kornia rerun-sdk pyliblzfse pypng git+https://github.com/ultralytics/CLIP.git 

# for RAM
pip install --upgrade-strategy only-if-needed "transformers==4.35.2" "tokenizers==0.14.1" "huggingface-hub==0.17.3" "accelerate==0.24.1" "safetensors==0.4.2" 

# check path by 'conda env list'
# (e.g.) export CUDA_HOME=/home/k2room/.conda/envs/capa
export CUDA_HOME=/path/to/anaconda3/envs/capa 

cd CAPA-3DSG && pip install -e .
```

## Install the third party repository
```
# Build flags
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
# CUDA_HOME is already set above

# 1) Segment Anything
python -m pip install -e src/thirdparty/groundedsam/segment_anything

# 2) GroundingDINO
python -m pip install --no-build-isolation -e src/thirdparty/groundedsam/GroundingDINO

# 3) RAM 
python -m pip install -e src/thirdparty/groundedsam/recognize-anything

# 4) ConceptGraph
python -m pip install -e src/thirdparty/conceptgraph

# 5) VLPart
# Download detectron2 (Reference: https://github.com/facebookresearch/detectron2/discussions/5200)
# Find the proper version in here: https://miropsota.github.io/torch_packages_builder/detectron2/ 
# We use 'detectron2-0.6+2a420edpt2.0.1cu118-cp310-cp310-linux_x86_64.whl'
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder 'detectron2==0.6+2a420edpt2.0.1cu118'

python -m pip install -e src/thirdparty/vlpart --no-deps
```
## Check the third-party repository
### Check the path
```
python scripts/import_test.py
```
### Run the demo codes of the third-party repository
```
cd src/thirdparty/groundedsam

python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /home/main/workspace/k2room2/CO3DSG/checkpoints/groundingdino_swint_ogc.pth \
  --sam_checkpoint /home/main/workspace/k2room2/CO3DSG/checkpoints/sam_vit_h_4b8939.pth \
  --input_image "/home/main/workspace/k2room2/CAPA-3DSG/assets/test_imgs/6.png" \
  --output_dir "/home/main/workspace/k2room2/CAPA-3DSG/assets/test_outputs" \
  --box_threshold 0.2 \
  --text_threshold 0.2 \
  --text_prompt "cabinet, counter top, dish washer, exhaust hood, floor, hardwood, hardwood floor, kitchen, microwave, oven, stove, switch, button, dial, handle" \
  --device "cuda"
```
```
cd src/thirdparty/vlpart

python demo/demo.py --config-file configs/joint/swinbase_cascade_lvis_paco.yaml \
  --input /home/main/workspace/k2room2/CAPA-3DSG/assets/test_imgs/6.png \
  --output /home/main/workspace/k2room2/CAPA-3DSG/outputs/output_image \
  --vocabulary paco \
  --confidence-threshold 0.3 \
  --opts MODEL.WEIGHTS /home/main/workspace/k2room2/CAPA-3DSG/checkpoints/swinbase_cascade_lvis_paco.pth VIS.BOX False
```

### Git Subtree History
We used subtree because we take several repositories and use them by modifying some codes. The process below is just for recording and **does not need to be performed.**

```
# Concept Graphs
git remote add conceptgraph https://github.com/concept-graphs/concept-graphs.git
git subtree add --prefix=src/thirdparty/conceptgraph conceptgraph ali-dev --squash

# Grounded Segment Anything
git remote add groundedsam https://github.com/IDEA-Research/Grounded-Segment-Anything.git
git subtree add --prefix=src/thirdparty/groundedsam groundedsam main --squash

# GroundingDINO for Grounded Segment Anything (already exists)
git remote add groundingdino https://github.com/IDEA-Research/GroundingDINO.git
git subtree add --prefix=src/thirdparty/groundedsam/GroundingDINO groundingdino main --squash

# SAM for Grounded Segment Anything (already exists)
git remote add segmentanything https://github.com/facebookresearch/segment-anything.git
git subtree add --prefix=src/thirdparty/groundedsam/segment_anything segmentanything main --squash

# RAM for Grounded Segment Anything
git remote add recognizeanything https://github.com/xinyu1205/recognize-anything.git
git subtree add --prefix=src/thirdparty/groundedsam/recognize-anything recognizeanything main --squash

# VLPart
git remote add vlpart https://github.com/facebookresearch/VLPart.git
git subtree add --prefix=src/thirdparty/vlpart vlpart main --squash

```


# Data

## Symbolic links for Datasets
```
sudo ln -s /home/main/storage/gpuserver00_storage/s3dis /home/main/workspace/k2room2/CAPA-3DSG/dataset/S3DIS

sudo ln -s /home/main/storage/gpuserver00_storage/replica/Replica /home/main/workspace/k2room2/CAPA-3DSG/dataset/Replica

sudo ln -s /home/main/workspace/k2room2/gpuserver00_storage/SceneFun3D /home/main/workspace/k2room2/CAPA-3DSG/dataset/SceneFun3D

sudo ln -s /home/main/workspace/k2room2/gpuserver00_storage/FunGraph3D /home/main/workspace/k2room2/CAPA-3DSG/dataset/FunGraph3D

```

## Checkpoints
```
cd checkpoints

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth

# wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth

wget https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco.pth
```
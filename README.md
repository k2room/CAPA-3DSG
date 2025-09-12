# CAPA-3DSG
**Context-Aware Part-Affordance 3D Scene Graph Generation with Open Vocabulary**

We introduce CAPA-3DSG, a novel framework for Context-Aware Part-Affordance 3D Scene Graph Generation with Open Vocabulary. Our approach enables fine-grained reasoning over object parts and their affordances while generalizing to unseen categories through open-vocabulary learning, bridging the gap between part-level semantics and contextual scene understanding.

# Setting
## Creat conda environment
```
conda create -n capa python=3.10

conda activate capa

conda install -y -c pytorch -c nvidia pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 

conda install -y -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

conda install -y -c nvidia cuda-toolkit=11.8

conda install -y https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2

# some packages change the numpy version, or confuse conda env and pip. use constraints.txt.
# (e.g.) export PIP_CONSTRAINT="/home/main/workspace/k2room2/CAPA-3DSG/constraints.txt"
export PIP_CONSTRAINT="path/CAPA-3DSG/constraints.txt"

pip install --upgrade-strategy only-if-needed tyro open_clip_torch wandb h5py openai hydra-core distinctipy ultralytics dill supervision==0.21.0 open3d imageio natsort kornia rerun-sdk pyliblzfse pypng git+https://github.com/ultralytics/CLIP.git

# check path by 'conda env list'
# (e.g.) export CUDA_HOME=/home/k2room/.conda/envs/capa
export CUDA_HOME=/path/to/anaconda3/envs/capa 

cd CAPA-3DSG && pip install -e .
```

우리는 여러 repository를 가져와 일부 코드를 수정하여 사용하기 때문에, subtree를 사용하였다. 아래 과정은 기록을 위한 것이며 수행하기 않아도 된다. 

```
# Concept Graphs
git remote add conceptgraph https://github.com/concept-graphs/concept-graphs.git
git subtree add --prefix=src/thirdparty/conceptgraph conceptgraph ali-dev --squash

# Grounded Segment Anything
git remote add groundedsam https://github.com/IDEA-Research/Grounded-Segment-Anything.git
git subtree add --prefix=src/thirdparty/groundedsam groundedsam main --squash

# GroundingDINO for Grounded Segment Anything
git remote add groundingdino https://github.com/IDEA-Research/GroundingDINO.git
git subtree add --prefix=src/thirdparty/groundedsam/GroundingDINO groundingdino main --squash

# SAM for Grounded Segment Anything
git remote add segmentanything https://github.com/facebookresearch/segment-anything.git
git subtree add --prefix=src/thirdparty/groundedsam/segment_anything segmentanything main --squash

# RAM for Grounded Segment Anything
git remote add recognizeanything https://github.com/xinyu1205/recognize-anything.git
git subtree add --prefix=src/thirdparty/groundedsam/recognize-anything recognizeanything main --squash

```
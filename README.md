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

pip install --upgrade-strategy only-if-needed tyro open_clip_torch wandb h5py openai hydra-core distinctipy ultralytics dill supervision==0.21.0 open3d imageio natsort kornia rerun-sdk pyliblzfse pypng git+https://github.com/ultralytics/CLIP.git 

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
```
## Check the third-party repository
```
python scripts/import_test.py
```
## Run the demo codes of the third-party repository
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
### troubleshooting
- Permission denied issue
```
# 누가 소유/권한을 갖는지 확인
ls -ld src/thirdparty/groundedsam/segment_anything

# 내 계정으로 소유권 되돌리기
sudo chown -R "$USER":"$(id -gn)" src/thirdparty/groundedsam/segment_anything

# 이전에 잘못 만들어진 egg-info가 있으면 정리
sudo rm -rf src/thirdparty/groundedsam/segment_anything/*.egg-info 2>/dev/null || true

# 다시 설치
python -m pip install -e src/thirdparty/groundedsam/segment_anything

```
- python -m pip install --no-build-isolation -e src/thirdparty/groundedsam/GroundingDINO
```
conda install -y -c conda-forge gxx_linux-64=11.* gcc_linux-64=11.* sysroot_linux-64=2.17
conda install -y -c conda-forge cmake ninja make

# Thrust 헤더가 있는지 확인
ls "$CONDA_PREFIX/targets/x86_64-linux/include/thrust/complex.h" || \
ls "$CONDA_PREFIX/include/thrust/complex.h" || \
echo ">> Thrust not found"

# CUDA 경로(11.8)로 고정
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Thrust 포함된 include 경로를 컴파일러가 보게 함
export CUDATK_INC="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include"
export CPATH="$CUDATK_INC:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDATK_INC:${CPLUS_INCLUDE_PATH:-}"

# Conda 툴체인 사용 (g++ 11 계열 권장)
which x86_64-conda-linux-gnu-g++ >/dev/null 2>&1 || \
  conda install -y -c conda-forge gxx_linux-64=11.* gcc_linux-64=11.* sysroot_linux-64=2.17

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# NVCC의 host compiler 중복 지정 경고 제거
unset CUDAHOSTCXX

rm -rf ~/.cache/torch_extensions/*
find src/thirdparty/groundedsam/GroundingDINO -maxdepth 2 -name build -type d -exec rm -rf {} +

python -m pip install --no-build-isolation -v -e src/thirdparty/groundedsam/GroundingDINO
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

```


# Dataset

## Symbolic links
```
sudo ln -s /home/main/storage/gpuserver00_storage/s3dis /home/main/workspace/k2room2/CAPA-3DSG/dataset/S3DIS

sudo ln -s /home/main/storage/gpuserver00_storage/replica/Replica /home/main/workspace/k2room2/CAPA-3DSG/dataset/Replica

sudo ln -s /home/main/workspace/k2room2/gpuserver00_storage/SceneFun3D /home/main/workspace/k2room2/CAPA-3DSG/dataset/SceneFun3D

sudo ln -s /home/main/workspace/k2room2/gpuserver00_storage/FunGraph3D /home/main/workspace/k2room2/CAPA-3DSG/dataset/FunGraph3D

```

# Checkpoints
```
cd checkpoints

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth

# wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
```
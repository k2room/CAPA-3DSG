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
- $ python -m pip install -e src/thirdparty/vlpart --no-deps
```
# src/thirdparty/vlpart/pyproject.toml 생성
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vlpart"                       # wheel/설치명 (필요시 "vlpart-acsl" 등으로 변경)
version = "0.0.0+vendored"
description = "Vendored VLPart for CAPA-3DSG"
requires-python = ">=3.9"

# 의존성은 우리 constraints로 관리하므로 명시하지 않음(=pip가 환경을 뒤집지 않게)
# dependencies = []

[tool.setuptools]
include-package-data = true           # (wheel 빌드시 비파이썬 파일 포함)

[tool.setuptools.packages.find]
include = ["vlpart*"]                 # ← 실제 패키지명에 맞게 수정
exclude = ["configs*", "datasets*", "demo*", "tools*", "docs*", "assets*"]
```
```
# src/thirdparty/vlpart/MANIFEST.in 생성
recursive-include configs *.yaml *.py
recursive-include datasets *.json *.md *.txt
recursive-include assets *.jpg *.png *.json
prune .git
```
```
# 권한 오류인지 확인
sudo chmod +777 /home/main/workspace/k2room2/CAPA-3DSG/src/thirdparty/vlpart/
```
FROM continuumio/miniconda3

RUN conda install \
    python=3.11 \
    pytorch torchvision torchaudio pytorch-cuda=11.8  \
    lightning pandas piq astropy timm matplotlib python-dotenv pydantic loguru tqdm\
    -c pytorch -c conda-forge -c photosynthesis-team -c nvidia

RUN pip install wandb einops
# Install onnxruntime for CUDA 11.8
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

FROM samsweere/xmm-epicpn-simulator

USER 0

RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install lightning pandas piq astropy timm matplotlib python-dotenv pydantic loguru tqdm tensorboard einops \
    uv cache clean

USER heasoft

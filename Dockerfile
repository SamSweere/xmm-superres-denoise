FROM continuumio/miniconda3

RUN conda install \
    python=3.11 \
    pytorch torchvision torchaudio cpuonly  \
    lightning pandas piq astropy timm matplotlib python-dotenv pydantic loguru tqdm\
    -c pytorch -c conda-forge -c photosynthesis-team

RUN pip install wandb

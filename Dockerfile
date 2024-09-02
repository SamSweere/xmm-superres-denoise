FROM continuumio/miniconda3

RUN conda install \
    pytorch torchvision torchaudio cpuonly lightning pandas piq astropy timm matplotlib python-dotenv pydantic \
    -c pytorch -c conda-forge -c photosynthesis-team

RUN pip install wandb
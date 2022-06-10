# Deep Learning-Based Super-Resolution and De-Noising for XMM-Newton Images
The results of this research are described in the paper <em>``Deep Learning-Based Super-Resolution and De-Noising for XMM-Newton Images"</em>. Pre-print available at: https://arxiv.org/pdf/2205.01152.pdf

More implementation details are described in the master's thesis included in this repository.


## Requirements
- Gpu: This model was developed using a Nvidia gtx 2080ti with 12 gb of vram. To run and train the models on a gpu, a gpu with at least 12 gb of vram is needed. 
Nvidia cuda also has to be installed on the computer to use the gpu.
- Cpu: The model can be run and trained on the cpu, training on a cpu is not recommended since it will take a long time. The pc needs at least 16 gb of ram to run and train the model.

## Setup
 - Clone the repository: `https://github.com/SamSweere/xmm_superres.git`
 - Enter the repository, i.e: `cd xmm_superres`
 - Create a virtual environment: `python3 -m venv xmm_superres_venv`
 - Activate the environment: `source xmm_superres_venv/bin/activate`
 - Install PyTorch 1.9.0 of your required compute platform. Note that the requirements.txt does not always get this correctly:
https://pytorch.org/ <br>
In my case I used:  <br>
`pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
 - Install the requirements: `pip3 install -r requirements.txt`

## Training
Setup the config file `run_config.yaml` for your environment:
 - When using gpu: check which gpu is free by running: `nvidia-smi` <br>
Set the gpu number in the `run_config.yaml` file under `gpus: [{your gpu num}]` 
 - Set the dataset_dir

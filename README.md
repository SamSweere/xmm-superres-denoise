# Deep Learning-Based Super-Resolution and De-Noising for XMM-Newton Images
This repository contains the implementation for our research study titled _Deep Learning-Based Super-Resolution and De-Noising for XMM-Newton Image_, [2022, MNRAS, 517, 4054](https://doi.org/10.1093/mnras/stac2437). Pre-print is available in [arXiv](https://arxiv.org/pdf/2205.01152.pdf).

More implementation details are described in [Sam Sweere's master thesis](<../Sam Sweere - Master Thesis - Deep Learning-Based Super-Resolution and De-Noising for XMM-Newton EPIC-pn.pdf>) included in this repository.

## Prerequisites
The hardware requirements to run and train the models:

- **GPU**: This model was developed using an Nvidia GTX 2080 Ti with 12 GB of VRAM. For similar performance, a GPU with at least 12 GB of VRAM is recommended. Additionally, Nvidia CUDA must be installed on your system to utilize the GPU capabilities.

- **CPU**: While the model can be run and trained on the CPU, this is not recommended due to potentially long training times. To run and train the model, your system needs at least 16 GB of RAM.

## Setup
This guide provides the simplest method to run the code from this project. If you anticipate modifying or further developing the code, it's advisable to follow the setup instructions specified in the 'Development' section below.

The project is built using Python 3.10. Make sure it's installed on your system. If Python 3.10 is not installed, you can find the installation instructions further below in the develoment section of the readme. We recommend using venv and pip for setting up your Python environment:

1. **Clone the repository:** If it's not already done, you can clone the repository using the following command:

    ```
    git clone https://github.com/SamSweere/xmm-superres-denoise.git
    ```

2. **Enter the repository:** Navigate to the cloned repository:

    ```
    cd xmm_superres_denoise
    ```

3. **Create a virtual environment with venv:** We will create a new virtual environment for this project. This isolates our project and avoids conflicts between different versions of packages. Make sure Python 3.10 is the active version before you do this.

    ```
    python3 -m venv xmm_superres_venv
    ```

4. **Activate the environment:** To activate the created environment, use the following command:

    ```
    source xmm_superres_venv/bin/activate
    ```

5. **Install the requirements:** Within the activated environment, install the necessary packages listed in the `requirements.txt` file:

    ```
    pip install -r requirements.txt
    ```

Now, you're all set up and ready to run the code!



## Inference
The notebook `inference_example.ipynb` and source code `xmm_superres_denoise/inference.py` demonstrate how to generate super-resolution and de-noised images using example data.

Please note that if you wish to run the models on your own data, the model config files will need to be updated to match your data configurations.

## Training
To set up your environment for training, follow the steps outlined below:

- For GPU utilization, determine the available GPU by running the `nvidia-smi` command. Next, update the `xmm_superres_denoise/run_config.yaml` file under `gpus: [{your gpu num}]` with the corresponding GPU number.

- Update `dataset_dir` in the `xmm_superres_denoise/run_config.yaml` file with your dataset directory.

We look forward to seeing your improvements and applications using our models. For any queries, please open an issue in this repository.

# Development

## Setup
This project uses Python 3.10. For development we recommend using pyenv and poetry to setup your development enviroment:

1. **Install Pyenv:** Pyenv is a Python version management tool. If it is not already installed, you can install it using the following commands:

    On Ubuntu:

    ```
    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
    curl https://pyenv.run | bash
    ```

    On macOS (using Homebrew):

    ```
    brew update
    brew install pyenv
    ```

    Remember to add `pyenv` to your shell so that it is available every time a new terminal session starts.

    ```
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
    ```
    Replace `.bash_profile` with `.zshrc` if you are using Zsh.

2. **Install Python 3.10.11 with Pyenv:**

    ```
    pyenv install 3.10.11
    ```

3. **Create a virtual environment with Pyenv:** We will create a new virtual environment for this project. This isolates our project and avoids conflicts between different versions of packages.

    ```
    pyenv virtualenv 3.10.11 xmm_superres_denoise
    pyenv local xmm_superres_denoise
    ```

4. **Install Poetry:** Poetry is a tool for dependency management in Python. If not already installed, you can install it with:

    ```
    curl -sSL https://install.python-poetry.org | python -
    ```

5. **Install dependencies:** Navigate to the project directory and run the following command to install the dependencies for this project:

    ```
    poetry install
    ```

6. **Install pre-commit:** Activate the installed environemnt and install the pre-commit packages:
    ```
    poetry shell
    pre-commit install
    ```

Now, you're all set up and ready to run or modify the code!

## Using Poetry for Dependency Management
In this project, we use Poetry for managing our dependencies. The benefit of using Poetry is that it ensures all the packages used in the project are compatible with each other.

Poetry locks the dependencies in a `poetry.lock` file. This means that when we install the dependencies in a different location, we will get exactly the same versions, ensuring consistency across different development and deployment environments.

Here are some useful Poetry commands that you might find helpful during development:

- **Adding Packages:** To add a new package to your project, use the following command:

    ```
    poetry add <package_name>
    ```

    Replace `<package_name>` with the name of the package you want to add.

- **Updating Packages:** To update your packages to their latest versions that still comply with the version constraints defined in your `pyproject.toml` file, use:

    ```
    poetry update
    ```

- **Removing Packages:** If you want to remove a package from your project, use:

    ```
    poetry remove <package_name>
    ```

    Replace `<package_name>` with the name of the package you want to remove.

These commands make it easy to manage your project's dependencies and ensure that the project environment is reproducible across different systems.

# Acknowledgements
Many thanks to [Bojan Todorkov](https://github.com/bojobo) for his code improvements and bug fixes to the codebase!

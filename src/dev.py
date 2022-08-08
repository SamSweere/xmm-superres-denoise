import torch
from torchinfo import summary
from models.srflow.srflow_config import config as _config

from models.srflow.srflow_arch import SRFlowNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srflow = SRFlowNet(config=_config).to(device)
summary(srflow, (3, 160, 160))

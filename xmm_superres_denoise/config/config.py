from pydantic import BaseModel, PositiveInt

from typing import Literal
from pathlib import Path


class Dataset(BaseModel):
    batch_size: PositiveInt
    check_files: bool
    crop_mode: Literal["center", "random", "boresight"]
    det_mask: bool
    directory: Path
    mode: Literal["img", "agn"]
    name: str
    scaling: Literal["linear", "sqrt", "asinh", "log"] | None
    type: Literal["sim", "real"]
from pydantic import computed_field, BaseModel, NonNegativeFloat, NonNegativeInt, PositiveInt
from typing import Literal
from pathlib import Path


class HrDatasetCfg(BaseModel):
    agn: bool
    bkg: bool
    exp: PositiveInt
    clamp_max: NonNegativeFloat
    res: PositiveInt


class LrDatasetCfg(BaseModel):
    agn: bool | NonNegativeInt
    bkg: bool | NonNegativeInt
    exps: list[PositiveInt]
    clamp_max: NonNegativeFloat
    res: PositiveInt


class DatasetCfg(BaseModel):
    batch_size: PositiveInt
    check_files: bool
    crop_mode: Literal["center", "random", "boresight"]
    det_mask: bool
    directory: Path
    mode: Literal["img", "agn"]
    name: str
    scaling: Literal["linear", "sqrt", "asinh", "log"] | None
    type: Literal["sim", "real"]
    lr: LrDatasetCfg
    hr: HrDatasetCfg

    @computed_field
    @property
    def res_mult(self) -> int:
        return self.hr.res // self.lr.res


class OptimizerCfg(BaseModel):
    learning_rate: NonNegativeFloat
    betas: tuple[NonNegativeFloat]

class ModelCfg(BaseModel):

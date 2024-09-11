from pydantic import computed_field, SecretStr, AfterValidator, BeforeValidator, BaseModel, NonNegativeFloat, NonNegativeInt, PositiveInt
from typing import Annotated, Literal
from pathlib import Path
from enum import StrEnum


class ConfigError(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)


class DatasetType(StrEnum):
    SIM = "sim"
    REAL = "real"
    BORING = "boring"

class ImageType(StrEnum):
    IMG = "img"
    AGN = "agn"
    BKG = "bkg"


def _check_path_before(value: str) -> Path | None:
    if value != "":
        return Path(value)


def _check_path_after(value: Path) -> Path | None:
    if value is not None:
        if not value.exists():
            raise FileNotFoundError(f"Detector mask does not exist at '{value}!'")

        if value.is_dir():
            raise FileExistsError(f"Path to detector mask is a directory! Given path: '{value}'")

        return value


class HrDatasetCfg(BaseModel):
    # TODO If dataset type is real and exp == 0 the initialisation of this class can be skipped
    det_mask: Annotated[Path | None, BeforeValidator(_check_path_before), AfterValidator(_check_path_after)]
    agn: bool
    exp: NonNegativeInt
    clamp_max: NonNegativeFloat
    res: PositiveInt


class LrDatasetCfg(BaseModel):
    bkg: bool | NonNegativeInt
    det_mask: Annotated[Path | None, BeforeValidator(_check_path_before), AfterValidator(_check_path_after)]
    exps: list[PositiveInt]
    clamp_max: NonNegativeFloat
    res: PositiveInt


class DatasetCfg(BaseModel):
    agn: bool | NonNegativeInt
    batch_size: PositiveInt
    check_files: bool
    debug: bool
    crop_mode: Literal["center", "random", "boresight"]
    directory: Path
    mode: Literal["img", "agn"]
    name: str
    scaling: Literal["linear", "sqrt", "asinh", "log"]
    # TODO What about display type?
    type: DatasetType
    lr: LrDatasetCfg
    hr: HrDatasetCfg

    @computed_field
    @property
    def res_mult(self) -> int:
        if self.type is DatasetType.REAL:
            return 1
        else:
            return self.hr.res // self.lr.res

    @computed_field
    @property
    def img_dir(self) -> Path:
        return self._mode_dir(ImageType.IMG)


    @computed_field
    @property
    def agn_dir(self) -> Path:
        return self._mode_dir(ImageType.AGN)


    @computed_field
    @property
    def bkg_dir(self) -> Path:
        return self._mode_dir(ImageType.BKG)


    # --- Helper Functions --- #
    def _mode_dir(self, mode: ImageType) -> Path:
        res: Path | None = None
        if self.type is DatasetType.SIM:
            return self.directory / self.name / mode

        if mode == ImageType.IMG:
            if self.type is DatasetType.REAL:
                return self.directory / self.name

        msg = f"Something went wrong while setting {mode.upper()} directory for type '{self.type}': "
        if res is None:
            msg = f"{msg}\tPath to {mode.upper()} dir has not been set!"

        if not res.exists():
            msg = f"{msg}\t{res} does not exist!"

        if not res.is_dir():
            msg = f"{msg}\t{res} is not a directory!"

        raise ConfigError(msg)


class OptimizerCfg(BaseModel):
    learning_rate: NonNegativeFloat
    betas: tuple[NonNegativeFloat]

class ModelCfg(BaseModel):
    name: Literal["esr_gen"]
    memory_efficient: bool

class WandbCfg(BaseModel):
    api_key: SecretStr
    project: str
    online: bool
    run_id: str
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)
from typing_extensions import Self


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


class BaseModels(StrEnum):
    ESR_GEN = "esr_gen"
    RRDB_DENOISE = "rrdb_denoise"
    SWINFIR = "swinfir"
    DRCT = "drct"
    HAT = "hat"
    RESTORMER = "restormer"


class TrainerStrategy(StrEnum):
    AUTO = "auto"
    DDP = "ddp"
    FSDP = "fsdp"


class TrainerAccelerator(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


def _check_path_before(value: str) -> Path | None:
    if value != "":
        return Path(value)

    return None


def _check_path_after(value: Path) -> Path | None:
    if value is not None:
        if not value.exists():
            raise FileNotFoundError(f"Detector mask does not exist at '{value}!'")

        if value.is_dir():
            raise FileExistsError(
                f"Path to detector mask is a directory! Given path: '{value}'"
            )

        return value


class HrDatasetCfg(BaseModel):
    # TODO If dataset type is real and exp == 0 the initialisation of this class can be skipped
    det_mask: Annotated[
        Path | None,
        BeforeValidator(_check_path_before),
        AfterValidator(_check_path_after),
    ]
    agn: bool
    exp: NonNegativeInt
    clamp_max: NonNegativeFloat
    res: PositiveInt


class LrDatasetCfg(BaseModel):
    bkg: bool | NonNegativeInt
    det_mask: Annotated[
        Path | None,
        BeforeValidator(_check_path_before),
        AfterValidator(_check_path_after),
    ]
    exps: list[PositiveInt]
    clamp_max: NonNegativeFloat
    res: PositiveInt


class DatasetCfg(BaseModel):
    agn: bool | NonNegativeInt
    batch_size: PositiveInt
    check_files: bool
    debug: bool
    comb_hr: bool
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
    betas: tuple[NonNegativeFloat, NonNegativeFloat]


class RrdbCfg(BaseModel):
    base_model: Literal["esr_gen", "rrdb_denoise"]
    in_channels: PositiveInt
    out_channels: PositiveInt
    filters: PositiveInt
    residual_blocks: PositiveInt


class TransformerCfg(BaseModel):
    base_model: Literal["swinfir", "drct", "hat"]
    patch_size: PositiveInt
    img_size: PositiveInt
    window_size: PositiveInt
    embed_dim: PositiveInt
    upsampler: Literal["pixelshuffle", "pixelshuffledirect", "nearest+conv", ""]
    in_channels: PositiveInt
    num_heads: list[PositiveInt]
    depths: list[PositiveInt]


class RestormerCfg(BaseModel):
    base_model: Literal["restormer"]
    in_channels: PositiveInt
    out_channels: PositiveInt
    dim: PositiveInt


class ModelCfg(BaseModel):
    name: BaseModels
    memory_efficient: bool
    batch_size: PositiveInt
    model: RrdbCfg | TransformerCfg | RestormerCfg = Field(
        ..., discriminator="base_model"
    )
    optimizer: OptimizerCfg


class WandbCfg(BaseModel):
    api_key: str
    project: str
    online: bool
    run_id: str
    log_model: bool


class TrainerCfg(BaseModel):
    accelerator: TrainerAccelerator
    strategy: TrainerStrategy
    checkpoint_path: Annotated[
        Path | None,
        BeforeValidator(_check_path_before),
    ]
    checkpoint_root: Annotated[
        Path | None,
        BeforeValidator(_check_path_before),
    ]
    devices: PositiveInt | Literal["auto"]
    epochs: PositiveInt
    log_images_every_n_epochs: NonNegativeInt


class LossCfg(BaseModel):
    l1: float = Field(ge=0, le=1)
    poisson: float = Field(ge=0, le=1)
    psnr: float = Field(ge=0, le=1)
    ssim: float = Field(ge=0, le=1)
    ms_ssim: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_sum(self) -> Self:
        p_sum = self.l1 + self.poisson + self.psnr + self.ssim + self.ms_ssim
        if 0 < p_sum <= 1:
            return self

        raise ConfigError(
            f"Sum of relative percentages has to be between 0 and 1, got {p_sum}!"
        )

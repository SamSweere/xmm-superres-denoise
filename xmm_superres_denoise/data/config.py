from enum import StrEnum
from pathlib import Path
from typing import Self

from astropy.io import fits
from loguru import logger
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)


class DatasetType(StrEnum):
    SIM = "sim"
    REAL = "real"


class ImageType(StrEnum):
    IMG = "img"
    AGN = "agn"
    BKG = "bkg"


class DatasetVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"


class StretchingFunction(StrEnum):
    LINEAR = "linear"
    SQRT = "sqrt"
    ASINH = "asinh"
    LOG = "log"


class DatasetCfgError(Exception):
    def __init__(self, message: str = ""):
        super().__init__(message)


class DatasetSubCfg(BaseModel):
    res: PositiveInt
    clamp_max: NonNegativeFloat
    exp: NonNegativeInt
    det_mask: Path | None

    @field_validator("det_mask", mode="before")
    @classmethod
    def check_path_before(cls, value: str) -> Path | None:
        if value == "":
            logger.info(f"No detector mask provided for {cls.__name__}.")
            return None

        return Path(value)

    @field_validator("det_mask", mode="after")
    @classmethod
    def check_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None

        if not value.exists():
            raise DatasetCfgError(f"Detector mask does not exist at '{value}!'")

        if not value.is_file():
            raise DatasetCfgError(
                f"Path to detector mask is not a file! Given path: '{value}'"
            )

        try:
            with fits.open(value) as hdul:
                if len(hdul) != 1:
                    raise DatasetCfgError(
                        f"Detector mask should only contain one HDU. Found {len(hdul)}."
                    )
                if hdul[0].data is None:
                    raise DatasetCfgError(
                        f"Detector mask does not contain any data! Given path: '{value}'"
                    )
                if hdul[0].data.ndim != 2:
                    raise DatasetCfgError(
                        f"Detector mask should be a 2D image. Found {hdul[0].data.ndim}D."
                    )
        except Exception as e:
            raise DatasetCfgError(
                f"Error while opening detector mask: {e}. Given path: '{value}'"
            )

        return value


class DatasetCfg(BaseModel):
    agn: bool
    bkg: bool
    batch_size: PositiveInt
    debug: bool
    comb_hr: bool
    directory: Path
    name: str
    scaling: StretchingFunction
    # TODO What about display type?
    type: DatasetType
    lq: DatasetSubCfg
    hq: DatasetSubCfg
    version: DatasetVersion

    @computed_field
    @property
    def res_mult(self) -> int:
        res_mult = self.hq.res // self.lq.res

        if res_mult == 1:
            return res_mult

        if res_mult % 2 != 0:
            raise DatasetCfgError(
                f"res_mult is not a multiple of two but {res_mult}, "
                f"based on {self.hq.res=} and {self.lq.res=}"
            )

        return res_mult

    @computed_field
    @property
    def num_workers(self) -> int:
        if self.debug:
            return 0

        return 12

    @computed_field
    @property
    def pin_memory(self) -> bool:
        return not self.debug

    @computed_field
    @property
    def persistent_workers(self) -> bool:
        return not self.debug

    @field_validator("directory", mode="before")
    @classmethod
    def check_path_before(cls, value: str) -> Path:
        if value == "":
            raise DatasetCfgError(
                "Dataset directory is empty! Please provide a valid path."
            )

        return Path(value)

    @field_validator("directory", mode="after")
    @classmethod
    def check_path(cls, value: Path) -> Path:
        if not value.exists():
            raise DatasetCfgError(f"Dataset directory does not exist at '{value}!'")

        return value

    @model_validator(mode="after")
    def check_exposures(self) -> Self:
        if self.hq.exp == self.lq.exp:
            raise DatasetCfgError(
                f"High quality exposure {self.hq.exp} is equal to low quality exposure {self.lq.exps}!"
            )

        return self

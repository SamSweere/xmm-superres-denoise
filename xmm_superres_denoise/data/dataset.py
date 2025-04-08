import tarfile
from io import BytesIO
from pathlib import Path
from random import choice

import torch
from astropy.io import fits
from data.config import DatasetCfg, DatasetType, ImageType
from data.normalize import normalize_image
from data.tools import load_fits, reshape_img_to_res
from data.transform import upsample_image
from loguru import logger
from torch.utils.data import Dataset


def load_sample(sample: tuple[tarfile.TarFile, str]) -> torch.Tensor:
    # TODO Add caching method
    tar, name = sample
    with fits.open(BytesIO(tar.extractfile(name).read()), lazy_load_hdus=False) as hdul:
        # Choose a random HDU (first one is not an image) and get its data
        data = choice(hdul[1:]).data

    return torch.from_numpy(data).unsqueeze(dim=0)


def load_and_combine(
    res: int,
    samples: list[tuple[tarfile.TarFile, str]],
    upsample: bool,
    det_mask: Path | None = None,
    scale_factor: int = 1,
):
    sample = load_sample(samples[0])

    for s in samples[1:]:
        sample.add_(load_sample(s))

    if det_mask is not None:
        # TODO check
        sample.mul_(load_fits(det_mask))

    if upsample and scale_factor > 1:
        upsample_image(sample, scale_factor, True)

    sample = reshape_img_to_res(res=res, img=sample)

    return sample


def find_files(
    parent: Path, image_type: ImageType, exp: int, suffix: str
) -> list[tuple[tarfile.TarFile, str]]:
    res: dict[tarfile.TarFile, list[str]] = {}
    paths = list((parent / image_type).glob(f"{exp}ks-*.tgz"))
    logger.info(
        f"({image_type.upper()}) Found {len(paths)} tgz files for exposure {exp}ks."
    )
    paths.sort()
    for path in paths:
        logger.info(f"Retrieving files from {path}.")
        tar = tarfile.open(path, "r")
        new_members = filter(
            lambda member: member.name.endswith(suffix), tar.getmembers()
        )
        members = res.get(tar, [])
        members.extend(member.name for member in new_members)
        res[tar] = members

    for tar in res:
        logger.info(f"Sorting members of {tar.name}")
        res[tar].sort()

    if len(res) == 0:
        raise FileNotFoundError(
            f"No {image_type.upper()} files found for exposure {exp}ks."
        )

    logger.info(
        f"({image_type.upper()}) Found {sum(len(res[tar]) for tar in res)} files for exposure {exp}ks."
    )

    return res


def check_samples(
    lq: dict[tarfile.TarFile, list[str]],
    hq: dict[tarfile.TarFile, list[str]],
    lq_suffix: str,
    hq_suffix: str,
    image_type: ImageType,
) -> None:
    logger.info(
        f"Checking that lq_{image_type} and hq_{image_type} contain the same members in identical order"
    )
    for lq_names, hq_names in zip(lq.values(), hq.values(), strict=True):
        for lq_name, hq_name in zip(lq_names, hq_names, strict=True):
            match image_type:
                case ImageType.IMG:
                    # Check that the parent dir is equal
                    lq_parent = lq_name[: lq_name.rfind("/")]
                    hq_parent = hq_name[: hq_name.rfind("/")]
                    if lq_parent != hq_parent:
                        raise ValueError(
                            f"Parents do not match!\n\t{lq_parent} != {hq_parent}"
                        )

                    # Check that the projection is equal
                    lq_proj = lq_name[lq_name.find("axis") :].removesuffix(lq_suffix)
                    hq_proj = hq_name[hq_name.find("axis") :].removesuffix(hq_suffix)
                    if lq_proj != hq_proj:
                        raise ValueError(
                            f"Projections do not match!\n\t{lq_proj} != {hq_proj}"
                        )
                case ImageType.AGN:
                    # Check that the name is equal
                    lq_name = lq_name.removesuffix(lq_suffix)
                    hq_name = hq_name.removesuffix(hq_suffix)
                    if lq_name != hq_name:
                        raise ValueError(
                            f"Names do not match!\n\t{lq_name} != {hq_name}"
                        )


class XmmDataset(Dataset):
    """XMM-Newton simulated dataset"""

    def __init__(self, config: DatasetCfg):
        self.config = config
        # url = f"https://huggingface.co/datasets/bojanto/xmm-superres-denoise/tree/main/{version}/{dataset}"
        # token = get_token()
        # TODO Add possibility to download the dataset
        self.base_path = (
            Path(self.config.directory) / self.config.version / self.config.name
        )

        match self.config.type:
            case DatasetType.REAL:
                hq_suffix = ".fits"
            case DatasetType.SIM:
                hq_suffix = ".comb" if self.config.comb_hr else ".2x"

        self.lq_img: dict[tarfile.TarFile, list[str]] = find_files(
            parent=self.base_path,
            image_type=ImageType.IMG,
            exp=self.config.lq.exp,
            suffix=".1x",
        )

        # TODO This doesn't make sense for the real dataset
        self.hq_img: dict[tarfile.TarFile, list[str]] = find_files(
            parent=self.base_path,
            image_type=ImageType.IMG,
            exp=self.config.hq.exp,
            suffix=hq_suffix,
        )

        self.length = sum(len(self.lq_img[tar]) for tar in self.lq_img)

        check_samples(self.lq_img, self.hq_img, ".1x", hq_suffix, ImageType.IMG)

        self.upsample = (
            self.config.res_mult > 1 and self.config.type is DatasetType.REAL
        )

        # --- AGN images --- #
        self.lq_agn: dict[tarfile.TarFile, list[str]] | None = None
        self.hq_agn: dict[tarfile.TarFile, list[str]] | None = None
        if not self.config.type is DatasetType.REAL:
            if self.config.agn:
                self.lq_agn = find_files(
                    parent=self.base_path,
                    image_type=ImageType.AGN,
                    exp=self.config.lq.exp,
                    suffix=".1x",
                )

                self.hq_agn = find_files(
                    parent=self.base_path,
                    image_type=ImageType.AGN,
                    exp=self.config.hq.exp,
                    suffix=hq_suffix,
                )

                check_samples(self.lq_agn, self.hq_agn, ".1x", hq_suffix, ImageType.AGN)

        # --- BKG images --- #
        self.bkg: dict[tarfile.TarFile, list[str]] | None = None
        if self.config.bkg and not self.config.type is DatasetType.REAL:
            self.bkg = find_files(
                parent=self.base_path,
                image_type=ImageType.BKG,
                exp=self.config.lq.exp,
                suffix=".fits",
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        lq_samples = []
        hq_samples = []

        for lq_tar, hq_tar in zip(self.lq_img, self.hq_img):
            if idx < len(self.lq_img[lq_tar]):
                lq_samples.append((lq_tar, self.lq_img[lq_tar][idx]))
                hq_samples.append((hq_tar, self.hq_img[hq_tar][idx]))
                break
            idx = idx - len(self.lq_img[lq_tar])

        if self.lq_agn is not None:
            # Not the prettiest solution, but it works
            lq_agn_tars = list(self.lq_agn.keys())
            hq_agn_tars = list(self.hq_agn.keys())
            agn_tar = choice(range(len(lq_agn_tars)))
            lq_agn_tar = lq_agn_tars[agn_tar]
            hq_agn_tar = hq_agn_tars[agn_tar]
            lq_samples.append((lq_agn_tar, choice(self.lq_agn[lq_agn_tar])))
            hq_samples.append((hq_agn_tar, choice(self.hq_agn[hq_agn_tar])))

        if self.bkg is not None:
            bkg_tar = choice(list(self.bkg))
            bkg_member = choice(self.bkg[bkg_tar])
            lq_samples.append((bkg_tar, bkg_member))

        lq = load_and_combine(
            res=self.config.lq.res,
            samples=lq_samples,
            det_mask=self.config.lq.det_mask,
            upsample=False,
        )

        lq = lq / self.config.lq.exp

        lq = normalize_image(
            stretch_mode=self.config.scaling,
            image=lq,
            max_val=self.config.lq.clamp_max,
            inplace=True,
        )

        # --- HQ --- #
        hq = load_and_combine(
            res=self.config.hq.res,
            samples=hq_samples,
            det_mask=self.config.hq.det_mask,
            upsample=self.upsample,
            scale_factor=self.config.res_mult,
        )

        hq = hq / self.config.hq.exp

        hq = normalize_image(
            stretch_mode=self.config.scaling,
            image=hq,
            max_val=self.config.hq.clamp_max,
            inplace=True,
        )

        return lq, hq

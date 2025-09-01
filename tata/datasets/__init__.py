"""Dataset loading utilities for TATA."""

from tata.datasets.loaders import (
    load_bot_iot,
    load_cic_ddos2019,
    load_cic_unsw,
    load_cifar10,
    load_csv_directory,
    load_csv_file,
    load_ctu13,
    load_dataset,
    load_ids2017,
    load_ids2018,
    load_iscx_ids2012,
    load_iscx_tor,
    load_mnist,
    load_nsl_kdd,
    load_ton_iot,
    load_unsw_nb15,
    load_vpn_nonvpn,
)
from tata.datasets.splits import generate_multiple_splits, stratified_split

__all__ = [
    "load_csv_directory",
    "load_csv_file",
    "load_dataset",
    "load_ids2017",
    "load_ids2018",
    "load_nsl_kdd",
    "load_unsw_nb15",
    "load_bot_iot",
    "load_ton_iot",
    "load_cic_ddos2019",
    "load_ctu13",
    "load_iscx_ids2012",
    "load_iscx_tor",
    "load_vpn_nonvpn",
    "load_cic_unsw",
    "load_mnist",
    "load_cifar10",
    "stratified_split",
    "generate_multiple_splits",
]

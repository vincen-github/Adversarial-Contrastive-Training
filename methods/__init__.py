from .act import ACT
from .w_act import WhiteningACT
from .contrastive import Contrastive
from .byol import BYOL
from .w_mse import WMSE
from .haochen22 import Haochen22
from .vicreg import Vicreg
from .barlow_twins import BarlowTwins


METHOD_LIST = ["act", "wact", "byol", "contrastive", "w_mse", "haochen22", "vicreg", "barlow_twins"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "act":
        return ACT
    elif name == "w_act":
        return WhiteningACT
    elif name == "contrastive":
        return Contrastive
    elif name == "byol":
        return BYOL
    elif name == "w_mse":
        return WMSE
    elif name == "haochen22":
        return Haochen22
    elif name == "vicreg":
        return Vicreg
    elif name == "barlow_twins":
        return BarlowTwins
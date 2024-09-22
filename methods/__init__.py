from .act import ACT, WhiteningACT
from .contrastive import Contrastive
from .byol import BYOL
from .w_mse import WMSE


METHOD_LIST = ["act", "w_act", "byol", "contrastive", "w_mse"]


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

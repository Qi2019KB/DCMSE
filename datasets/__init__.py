from .dataset import CommDataset as DS
from .dataset_mt import MTDataset as DS_mt
from .dataset_multi import MultiDataset as DS_mul

__all__ = ("DS", "DS_mt", "DS_mul")
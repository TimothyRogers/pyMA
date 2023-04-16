"""
Stochastic Subspace OMA Methods
"""

from . import OMA
from pyma.utils import ModalType
from pyma.postprocess import ss_to_modal

import numpy as np
from typing import Any, Union, List, Tuple
from collections.abc import Iterable


class FDD(OMA):
    def __init__(self, opts={}) -> None:
        """Setup FDD

        No setup required really, only thing is to update opts as in OMA

        Args:
            opts (dict, optional): User options. Defaults to {}, i.e. use defaults.
        """
        super().__init__(opts)

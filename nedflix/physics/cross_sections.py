from abc import ABC
from typing import Tuple, Callable

from .interactions import InteractionType

class CrossSection(ABC):

    def __init__(
        self,
        nc_f: Callable,
        cc_f: Callable,
        diff_nc_f: Callable,
        diff_cc_f: Callable
    ):
        self._tot_dict = {
            InteractionType.ChargedCurrent: cc_f,
            InteractionType.NeutralCurrent: nc_f,
        }
        self._diff_dict = {
            InteractionType.ChargedCurrent: diff_cc_f,
            InteractionType.NeutralCurrent: diff_nc_f,
        }


    def __getitem__(self, interaction: InteractionType) -> Tuple[Callable, Callable]:
        return (self._tot_dict[interaction], self._diff_dict[interaction])

class DummyCrossSection(CrossSection):

    def __init__(self):
        nc_f = lambda _: 1
        cc_f = lambda _: 3
        def diff_nc_f(ein, eout):
            raise NotImplementedError()
        def diff_cc_f(ein, eout):
            raise NotImplementedError()
        super(DummyCrossSection, self).__init__(nc_f, cc_f, diff_nc_f, diff_cc_f)


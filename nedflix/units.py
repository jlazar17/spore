from dataclasses import dataclass
from collections import namedtuple
from typing import Union

class NotUnitfulError(Exception):
    pass

class DimensionMisMatchError(Exception):
    pass

def has_same_dimensionality(x, y):
    return x._l==y._l and x._e==y._e and x._t==y._t and x._q==y._q


def check_same_dimensionality(x, y):
    if not has_same_dimensionality(x, y):
        raise DimensionMisMatchError

def check_is_unitful(x):
    if not isinstance(x, UnitfulQuantity):
        raise NotUnitfulError

@dataclass
class UnitfulQuantity:

    # def __init__(self, val: float, l: float, e: float, t: float, q: float):
    _val: float
    _l: int
    _e: int
    _t: int
    _q: int

    def __eq__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return self._val==other._val

    def __le__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return self._val<=other._val

    def __ge__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return self._val>=other._val

    def __lt__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return self._val < other._val

    def __gt__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return self._val > other._val

    def __add__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return UnitfulQuantity(self._val + other._val, self._l, self._e, self._t, self._q)

    def __sub__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return UnitfulQuantity(self._val - other._val, self._l, self._e, self._t, self._q)

    def __mul__(self, other):
        if isinstance(other, UnitfulQuantity):
            return UnitfulQuantity(self._val * other._val, self._l+other._l, self._e+other._e, self._t+other._t, self._q+other._q)
        return UnitfulQuantity(self._val * other, self._l, self._e, self._t, self._q)

    def __rmul__(self, other):
        return UnitfulQuantity(self._val * other, self._l, self._e, self._t, self._q)

    def __truediv__(self, other):
        if isinstance(other, UnitfulQuantity):
            return UnitfulQuantity(self._val / other._val, self._l-other._l, self._e-other._e, self._t-other._t, self._q-other._q)
        return UnitfulQuantity(self._val / other, self._l, self._e, self._t, self._q)

    def __rtruediv__(self, other):
        return UnitfulQuantity(self._val / other, self._l, self._e, self._t, self._q)

    def __mod__(self, other):
        check_is_unitful(other)
        check_same_dimensionality(self, other)
        return self._val / other._val

# Base units are 

unit_names = "m GeV s coulomb sr cm ns us ms eV keV MeV TeV PeV joule c".split()
Units = namedtuple("Units", unit_names)

# Base quantities
_m = UnitfulQuantity(1.0, 1.0, 0.0, 0.0, 0.0)
_GeV = UnitfulQuantity(1.0, 0.0, 1.0, 0.0, 0.0)
_s = UnitfulQuantity(1.0, 0.0, 0.0, 1.0, 0.0)
_coulomb = UnitfulQuantity(1.0, 0.0, 0.0, 0.0, 1.0)
_sr = UnitfulQuantity(1.0, 0.0, 0.0, 0.0, 0.0)
# Derived quantities
_cm = _m / 100
_ns = _s / 1e9
_us = _s / 1e6
_ms = _s / 1e3
_eV = _GeV / 1e9
_keV = _GeV / 1e6
_MeV = _GeV / 1e3
_TeV = _GeV * 1e3
_PeV = _GeV * 1e6
_joule = 6241506479.9632 *_GeV
_c = 299_792_458 * _m / _s

units = Units(
    _m, _GeV, _s, _coulomb, _sr, _cm, _ns, _us, _ms, _eV, _keV, _MeV, _TeV, _PeV, _joule, _c
)

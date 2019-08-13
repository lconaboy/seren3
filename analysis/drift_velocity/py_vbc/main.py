import numpy as np

from py_vbc.spectra import *
from py_vbc.constants import *
from py_vbc.derivatives import calc_derivs

"""
A Python version of the vbc_transfer module written by Matt
McQuinn & Ryan O'Leary (astro-ph/1204.1344). Almost identical save for
the removal of some obsolete features and Python optimisation.

TODO 
- figure out how to read in parameters and pass them to the
relevant functions 

- figure out motivation for initial values of
delta_b, delta_c, delta_b_dot, delta_c_dot

- should costh really be a constant input at runtime? or more
physically motivated, perhaps even random 

- derive temperature fluctuation equations, do they agree with Ahn?
where does the v_bc.k dependence come in?  

- rewrite tests for the new module format

- problem with vbc != 0
"""

def run_pyvbc(vbc, zstart, zend, dz, kmin=0.1, kmax=10000, n=64):
    k = np.logspace(np.log10(kmin), np.log10(kmax), num=n)

    g = calc_derivs(k, vbc, zstart, zend, dz)

    p_c, p_b = calc_power_spec(k, g, zstart)

    d_c = calc_delta(k, p_c)
    d_b = calc_delta(k, p_b)

    return k, d_c, d_b




def run_tests():
    from py_vbc.tests.itf_test import itf_test
    from py_vbc.tests.irf_test import irf_test
    from py_vbc.tests.g_test import g_test
    from py_vbc.tests.p_test import p_test

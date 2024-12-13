"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-13T19:23:08.248476
 * python version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
 * albert version: 0.0.0
 * caller: /home/ollie/git/albert/albert/code/einsum.py
 * node: ollie-desktop
 * system: Linux
 * processor: x86_64
 * release: 6.8.0-49-generic
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, dirsum, Namespace


def update_amps_external(f=None, t3=None, t4=None, v=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        f: 
        t3: 
        t4: 
        v: 

    Returns:
        t1new: 
        t2new: 
    """

    t1new = Namespace()
    t1new.OV = einsum(v.OOVV, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5)) * 0.25
    t2new = Namespace()
    t2new.OOVV = einsum(t3, (0, 1, 2, 3, 4, 5), f.OV, (2, 5), (0, 1, 3, 4))
    t2new.OOVV += einsum(t4, (0, 1, 2, 3, 4, 5, 6, 7), v.OOVV, (2, 3, 6, 7), (0, 1, 4, 5)) * 0.25
    tmp0 = einsum(v.oVOO, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 1), (4, 5, 6, 0)) * -1
    t2new.OoVV = np.copy(np.transpose(tmp0, (0, 3, 2, 1))) * -0.5
    t2new.oOVV = np.copy(np.transpose(tmp0, (3, 0, 2, 1))) * 0.5
    del tmp0
    tmp1 = einsum(v.OvVV, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 3), (4, 5, 6, 1))
    t2new.OOvV = np.copy(np.transpose(tmp1, (0, 1, 3, 2))) * 0.5
    t2new.OOVv = np.copy(tmp1) * -0.5
    del tmp1

    return {"t1new": t1new, "t2new": t2new}

def update_amps_mixed(t1=None, t3=None, v=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        t1: 
        t3: 
        v: 

    Returns:
        t2new: 
    """

    tmp0 = einsum(t1, (0, 1), v.oOvV, (0, 2, 1, 3), (2, 3))
    t2new = Namespace()
    t2new.OOVV = einsum(t3, (0, 1, 2, 3, 4, 5), tmp0, (2, 5), (0, 1, 3, 4))
    del tmp0
    tmp1 = einsum(t1, (0, 1), v.OOvV, (2, 3, 1, 4), (2, 3, 4, 0)) * -1
    tmp2 = einsum(tmp1, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 2), (4, 5, 6, 3)) * -1
    del tmp1
    t2new.OoVV = np.copy(np.transpose(tmp2, (0, 3, 2, 1))) * -0.5
    t2new.oOVV = np.copy(np.transpose(tmp2, (3, 0, 2, 1))) * 0.5
    del tmp2
    tmp3 = einsum(v.oOVV, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 3), (4, 5, 6, 0)) * -1
    tmp4 = einsum(t1, (0, 1), tmp3, (2, 3, 4, 0), (2, 3, 4, 1))
    del tmp3
    t2new.OOvV = np.copy(np.transpose(tmp4, (0, 1, 3, 2))) * -0.5
    t2new.OOVv = np.copy(tmp4) * 0.5
    del tmp4

    return {"t2new": t2new}

def convert_c1_to_t1(c1=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        c1: 

    Returns:
        t1: 
    """

    t1 = np.copy(c1)

    return {"t1": t1}

def convert_c2_to_t2(c2=None, t1=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        c2: 
        t1: 

    Returns:
        t2: 
    """

    t2 = np.copy(c2)
    t2 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3))
    t2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * -1

    return {"t2": t2}

def convert_c3_to_t3(c3=None, t1=None, t2=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        c3: 
        t1: 
        t2: 

    Returns:
        t3: 
    """

    t3 = np.copy(c3)
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1)) * -1
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5))
    t3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 3, 5))
    t3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 2, 3)) * -1
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1
    tmp0 = np.copy(t2)
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t3 += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 5, 2))
    t3 += einsum(t1, (0, 1), tmp0, (2, 3, 4, 5), (3, 2, 0, 5, 1, 4))
    t3 += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (1, 4, 0, 3, 5, 2)) * -1
    del tmp0

    return {"t3": t3}

def convert_c4_to_t4(c4=None, t1=None, t2=None, t3=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        c4: 
        t1: 
        t2: 
        t3: 

    Returns:
        t4: 
    """

    t4 = np.copy(c4)
    t4 += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 1, 2, 6, 7, 3, 4, 5))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 6, 7)) * -1
    t4 += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 1, 2, 6, 3, 4, 7, 5))
    t4 += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 1, 2, 6, 3, 4, 5, 7)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 2, 3, 6, 7)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 3, 7)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 6, 2, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 6, 7, 2, 3)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 6, 7, 2, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 2, 6, 3, 7)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 2, 6, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 6, 2, 7, 3)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 3, 6, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 7, 2, 3)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 2, 6, 3, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 7, 3)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 3, 7)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 6, 2, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 6, 7, 2, 3)) * -1
    tmp1 = np.copy(t3)
    tmp1 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1
    tmp1 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 5, 3))
    tmp1 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 3, 5)) * -1
    tmp1 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 2, 3))
    tmp1 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5)) * -1
    tmp1 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    tmp0 = np.copy(t2)
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    tmp1 += einsum(t1, (0, 1), tmp0, (2, 3, 4, 5), (0, 3, 2, 1, 5, 4))
    tmp1 += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 2, 5))
    tmp1 += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 5, 2)) * -1
    del tmp0
    t4 += einsum(tmp1, (0, 1, 2, 3, 4, 5), t1, (6, 7), (6, 1, 2, 0, 4, 7, 5, 3))
    t4 += einsum(tmp1, (0, 1, 2, 3, 4, 5), t1, (6, 7), (6, 1, 2, 0, 4, 5, 3, 7))
    t4 += einsum(t1, (0, 1), tmp1, (2, 3, 4, 5, 6, 7), (0, 3, 4, 2, 1, 6, 7, 5)) * -1
    t4 += einsum(t1, (0, 1), tmp1, (2, 3, 4, 5, 6, 7), (0, 3, 4, 2, 6, 7, 1, 5)) * -1
    del tmp1
    tmp2 = np.copy(t3)
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5))
    tmp2 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (4, 0, 1, 2, 5, 3)) * -1
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1))
    tmp2 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 2, 3))
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5)) * -1
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t4 += einsum(t1, (0, 1), tmp2, (2, 3, 4, 5, 6, 7), (3, 0, 4, 2, 1, 6, 7, 5))
    t4 += einsum(t1, (0, 1), tmp2, (2, 3, 4, 5, 6, 7), (3, 0, 4, 2, 6, 7, 1, 5))
    t4 += einsum(tmp2, (0, 1, 2, 3, 4, 5), t1, (6, 7), (1, 6, 2, 0, 4, 7, 5, 3)) * -1
    t4 += einsum(t1, (0, 1), tmp2, (2, 3, 4, 5, 6, 7), (3, 0, 4, 2, 6, 7, 5, 1)) * -1
    del tmp2
    tmp3 = np.copy(t3)
    tmp3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5))
    tmp3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (4, 0, 1, 2, 5, 3)) * -1
    tmp3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1))
    t4 += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 4, 0, 2, 6, 1, 7, 5))
    t4 += einsum(tmp3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (1, 2, 6, 0, 4, 5, 3, 7))
    t4 += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 4, 0, 2, 1, 6, 7, 5)) * -1
    t4 += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 4, 0, 2, 6, 7, 1, 5)) * -1
    del tmp3

    return {"t4": t4}

def convert_t1_to_c1(t1=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        t1: 

    Returns:
        t1new: 
    """

    t1new = Namespace()
    t1new.ov = np.copy(t1)

    return {"t1new": t1new}

def convert_t2_to_c2(t1=None, t2=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        t1: 
        t2: 

    Returns:
        t2new: 
    """

    t2new = Namespace()
    t2new.oovv = np.copy(t2)
    t2new.oovv += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    t2new.oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))

    return {"t2new": t2new}

def convert_t3_to_c3(t1=None, t2=None, t3=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        t1: 
        t2: 
        t3: 

    Returns:
        t3new: 
    """

    t3new = np.copy(t3)
    t3new += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1
    t3new += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 5, 3))
    t3new += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 3, 5)) * -1
    t3new += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 2, 3))
    t3new += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5)) * -1
    t3new += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    tmp0 = np.copy(t2)
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t3new += einsum(t1, (0, 1), tmp0, (2, 3, 4, 5), (0, 3, 2, 1, 5, 4))
    t3new += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 2, 5))
    t3new += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 5, 2)) * -1
    del tmp0

    return {"t3new": t3new}

def convert_t4_to_c4(t1=None, t2=None, t3=None, t4=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        t1: 
        t2: 
        t3: 
        t4: 

    Returns:
        t4new: 
    """

    t4new = np.copy(t4)
    t4new += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 1, 2, 6, 7, 3, 4, 5)) * -1
    t4new += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 6, 7))
    t4new += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 1, 2, 6, 3, 4, 7, 5)) * -1
    t4new += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 1, 2, 6, 3, 4, 5, 7))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 2, 3, 6, 7))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 7, 3)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 3, 7))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 6, 2, 7, 3)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 6, 7, 2, 3))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 6, 7, 2, 3)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 2, 6, 3, 7))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 7, 3)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 2, 6, 7, 3)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 6, 2, 7, 3))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 3, 6, 7)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 7, 2, 3))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 2, 6, 3, 7)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 7, 3))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 3, 7))
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 6, 2, 7, 3)) * -1
    t4new += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 6, 7, 2, 3))
    tmp1 = np.copy(t3)
    tmp1 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1
    tmp1 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 5, 3))
    tmp1 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 3, 5)) * -1
    tmp1 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 2, 3))
    tmp1 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5)) * -1
    tmp1 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    tmp0 = np.copy(t2)
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    tmp1 += einsum(t1, (0, 1), tmp0, (2, 3, 4, 5), (0, 3, 2, 1, 5, 4))
    tmp1 += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 2, 5))
    tmp1 += einsum(tmp0, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 5, 2)) * -1
    del tmp0
    t4new += einsum(t1, (0, 1), tmp1, (2, 3, 4, 5, 6, 7), (0, 3, 4, 2, 1, 6, 7, 5))
    t4new += einsum(t1, (0, 1), tmp1, (2, 3, 4, 5, 6, 7), (0, 3, 4, 2, 6, 7, 1, 5))
    t4new += einsum(tmp1, (0, 1, 2, 3, 4, 5), t1, (6, 7), (6, 1, 2, 0, 4, 7, 5, 3)) * -1
    t4new += einsum(tmp1, (0, 1, 2, 3, 4, 5), t1, (6, 7), (6, 1, 2, 0, 4, 5, 3, 7)) * -1
    del tmp1
    tmp2 = np.copy(t3)
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5))
    tmp2 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (4, 0, 1, 2, 5, 3)) * -1
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1))
    tmp2 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 2, 3))
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5)) * -1
    tmp2 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t4new += einsum(tmp2, (0, 1, 2, 3, 4, 5), t1, (6, 7), (1, 6, 2, 0, 4, 7, 5, 3))
    t4new += einsum(t1, (0, 1), tmp2, (2, 3, 4, 5, 6, 7), (3, 0, 4, 2, 6, 7, 5, 1))
    t4new += einsum(t1, (0, 1), tmp2, (2, 3, 4, 5, 6, 7), (3, 0, 4, 2, 1, 6, 7, 5)) * -1
    t4new += einsum(t1, (0, 1), tmp2, (2, 3, 4, 5, 6, 7), (3, 0, 4, 2, 6, 7, 1, 5)) * -1
    del tmp2
    tmp3 = np.copy(t3)
    tmp3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5))
    tmp3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (4, 0, 1, 2, 5, 3)) * -1
    tmp3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1))
    t4new += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 4, 0, 2, 1, 6, 7, 5))
    t4new += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 4, 0, 2, 6, 7, 1, 5))
    t4new += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 4, 0, 2, 6, 1, 7, 5)) * -1
    t4new += einsum(tmp3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (1, 2, 6, 0, 4, 5, 3, 7)) * -1
    del tmp3

    return {"t4new": t4new}


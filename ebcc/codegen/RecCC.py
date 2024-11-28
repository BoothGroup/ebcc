"""Code generated by `albert` version 0.0.0.

 * date: 2024-11-28T11:00:25.944104
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


def update_amps_external(f=None, t3=None, t4=None, t4a=None, v=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        f: 
        t3: 
        t4: 
        t4a: 
        v: 

    Returns:
        t1new: 
        t2new: 
    """

    tmp4 = einsum(t3, (0, 1, 2, 3, 4, 5), v.oOOV, (6, 0, 2, 5), (1, 3, 4, 6)) * -1
    t2new = Namespace()
    t2new.OoVV = np.copy(np.transpose(tmp4, (0, 3, 2, 1)))
    tmp5 = einsum(t3, (0, 1, 2, 3, 4, 5), v.oOOV, (6, 1, 2, 5), (0, 3, 4, 6))
    t2new.OoVV += np.transpose(tmp5, (0, 3, 1, 2)) * -1
    t2new.oOVV = np.copy(np.transpose(tmp4, (3, 0, 1, 2)))
    del tmp4
    t2new.oOVV += np.transpose(tmp5, (3, 0, 2, 1)) * -1
    del tmp5
    tmp9 = einsum(v.OVvV, (0, 1, 2, 3), t3, (4, 5, 0, 1, 6, 3), (4, 5, 6, 2))
    t2new.OOvV = np.copy(np.transpose(tmp9, (0, 1, 3, 2))) * -1
    tmp10 = einsum(v.OVvV, (0, 1, 2, 3), t3, (4, 5, 0, 6, 3, 1), (4, 5, 6, 2))
    t2new.OOvV += np.transpose(tmp10, (1, 0, 3, 2))
    t2new.OOVv = np.copy(tmp10)
    del tmp10
    t2new.OOVv += np.transpose(tmp9, (1, 0, 2, 3)) * -1
    del tmp9
    tmp6 = einsum(f.OV, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    t2new.OOVV = np.copy(tmp6)
    tmp7 = einsum(t4a, (0, 1, 2, 3, 4, 5, 6, 7), v.OVOV, (1, 7, 3, 5), (0, 2, 4, 6)) * -1
    t2new.OOVV += np.transpose(tmp7, (0, 1, 3, 2)) * -1
    del tmp7
    tmp8 = einsum(t4, (0, 1, 2, 3, 4, 5, 6, 7), v.OVOV, (2, 6, 3, 7), (0, 1, 4, 5))
    t2new.OOVV += tmp8
    del tmp8
    t2new.OOVV += np.transpose(tmp6, (1, 0, 3, 2))
    del tmp6
    tmp0 = einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 0, 2, 5, 3, 1), (4, 5))
    t1new = Namespace()
    t1new.OV = np.copy(tmp0) * -0.5
    del tmp0
    tmp1 = einsum(t3, (0, 1, 2, 3, 4, 5), v.OVOV, (1, 4, 2, 5), (0, 3))
    t1new.OV += tmp1 * 1.5
    del tmp1
    tmp2 = einsum(t3, (0, 1, 2, 3, 4, 5), v.OVOV, (1, 5, 2, 3), (0, 4))
    t1new.OV += tmp2 * 0.5
    del tmp2
    tmp3 = einsum(t3, (0, 1, 2, 3, 4, 5), v.OVOV, (0, 5, 2, 3), (1, 4)) * -1
    t1new.OV += tmp3 * 0.5
    del tmp3

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

    tmp8 = einsum(t1, (0, 1), v.OVOv, (2, 3, 4, 1), (2, 4, 3, 0))
    tmp9 = einsum(t3, (0, 1, 2, 3, 4, 5), tmp8, (2, 0, 5, 6), (1, 3, 4, 6)) * -1
    t2new = Namespace()
    t2new.OoVV = np.copy(np.transpose(tmp9, (0, 3, 2, 1)))
    tmp10 = einsum(t3, (0, 1, 2, 3, 4, 5), tmp8, (2, 1, 5, 6), (0, 3, 4, 6))
    del tmp8
    t2new.OoVV += np.transpose(tmp10, (0, 3, 1, 2)) * -1
    t2new.oOVV = np.copy(np.transpose(tmp9, (3, 0, 1, 2)))
    del tmp9
    t2new.oOVV += np.transpose(tmp10, (3, 0, 2, 1)) * -1
    del tmp10
    tmp0 = einsum(v.oVOv, (0, 1, 2, 3), t1, (0, 3), (2, 1))
    tmp1 = einsum(tmp0, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    del tmp0
    t2new.OOVV = np.copy(tmp1) * -1
    tmp2 = einsum(v.ovOV, (0, 1, 2, 3), t1, (0, 1), (2, 3))
    tmp3 = einsum(t3, (0, 1, 2, 3, 4, 5), tmp2, (2, 5), (0, 1, 3, 4))
    del tmp2
    t2new.OOVV += tmp3 * 2
    t2new.OOVV += np.transpose(tmp1, (1, 0, 3, 2)) * -1
    del tmp1
    t2new.OOVV += np.transpose(tmp3, (1, 0, 3, 2)) * 2
    del tmp3
    tmp4 = einsum(t3, (0, 1, 2, 3, 4, 5), v.oVOV, (6, 5, 2, 3), (0, 1, 4, 6))
    tmp5 = einsum(tmp4, (0, 1, 2, 3), t1, (3, 4), (0, 1, 2, 4))
    del tmp4
    t2new.OOvV = np.copy(np.transpose(tmp5, (0, 1, 3, 2)))
    tmp6 = einsum(t3, (0, 1, 2, 3, 4, 5), v.oVOV, (6, 4, 2, 5), (0, 1, 3, 6))
    tmp7 = einsum(t1, (0, 1), tmp6, (2, 3, 4, 0), (2, 3, 4, 1))
    del tmp6
    t2new.OOvV += np.transpose(tmp7, (1, 0, 3, 2)) * -1
    t2new.OOVv = np.copy(tmp7) * -1
    del tmp7
    t2new.OOVv += np.transpose(tmp5, (1, 0, 2, 3))
    del tmp5

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

    t2 = einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1)) * -1
    t2 += c2

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
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 5, 4, 1))
    t3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 3, 2))
    t3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 2, 3, 5)) * -1
    tmp0 = np.copy(t2)
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    tmp1 = np.copy(np.transpose(tmp0, (0, 1, 3, 2))) * -1
    tmp1 += tmp0
    del tmp0
    t3 += einsum(tmp1, (0, 1, 2, 3), t1, (4, 5), (1, 4, 0, 3, 5, 2)) * -1
    del tmp1

    return {"t3": t3}

def convert_c4_to_t4(c4=None, c4a=None, t1=None, t2=None, t3=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        c4: 
        c4a: 
        t1: 
        t2: 
        t3: 

    Returns:
        t4: 
        t4a: 
    """

    t4 = np.copy(c4)
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 5, 1, 7, 6)) * -1
    t4 += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 6, 2, 1, 3, 4, 5, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 7, 6))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 7, 6, 2, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 6, 2, 3, 7)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 7, 3, 2, 6)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 2, 3, 7, 6))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 2, 3, 6, 7)) * -1
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 6, 3, 2, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 3, 2, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 6, 7, 2, 3)) * -1
    tmp0 = np.copy(t2)
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    tmp2 = np.copy(np.transpose(tmp0, (0, 1, 3, 2)))
    tmp2 += tmp0 * -1
    tmp1 = np.copy(np.transpose(tmp0, (0, 1, 3, 2))) * -1
    tmp1 += tmp0
    del tmp0
    t4 += einsum(tmp1, (0, 1, 2, 3), tmp2, (4, 5, 6, 7), (1, 5, 0, 4, 3, 6, 2, 7)) * -1
    tmp3 = np.copy(np.transpose(t3, (0, 2, 1, 3, 5, 4)))
    tmp3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5))
    tmp3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (4, 0, 1, 2, 5, 3)) * -1
    tmp3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1
    tmp3 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 5, 3))
    t4 += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (0, 3, 4, 2, 7, 6, 1, 5))
    t4 += einsum(tmp3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (2, 1, 6, 0, 7, 4, 5, 3))
    t4 += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (0, 3, 4, 2, 1, 6, 7, 5)) * -1
    t4 += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (4, 3, 0, 2, 7, 6, 1, 5)) * -1
    t4a = np.copy(c4a)
    t4a += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (6, 0, 1, 2, 7, 3, 4, 5)) * -1
    t4a += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (6, 0, 1, 2, 3, 5, 4, 7)) * -1
    t4a += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4a += einsum(t3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (0, 6, 1, 2, 3, 5, 4, 7))
    t4a += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 4, 3, 0, 1, 5, 6, 7)) * -1
    t4a += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 4, 3, 0, 5, 7, 6, 1)) * -1
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 2, 6, 3, 7))
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 1, 5, 2, 7, 3, 6)) * -1
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 2, 3, 7, 6))
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 3, 2, 7, 6)) * -1
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 2, 3, 7, 6)) * -1
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 3, 2, 7, 6))
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 6, 2, 7, 3)) * -1
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 1, 6, 3, 7, 2))
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 3, 6, 2, 7)) * -1
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 3, 7, 2, 6))
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 6, 7, 2, 3)) * -1
    t4a += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 1, 7, 6, 2, 3))
    tmp4 = np.copy(np.transpose(t3, (0, 2, 1, 3, 5, 4)))
    tmp4 += np.transpose(t3, (0, 2, 1, 3, 4, 5)) * -1
    tmp4 += np.transpose(t3, (0, 2, 1, 4, 3, 5))
    tmp4 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 5, 4))
    tmp4 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1
    tmp4 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 5, 1, 4)) * -1
    tmp4 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 4, 1, 2, 5, 3))
    tmp4 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 5, 4, 1))
    tmp4 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 4, 5, 1)) * -1
    tmp4 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 3, 2)) * -1
    tmp4 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 5, 2, 3))
    tmp4 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 3, 5, 2))
    tmp4 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5)) * -1
    tmp4 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 3, 2, 5)) * -1
    tmp4 += einsum(t2, (0, 1, 2, 3), t1, (4, 5), (0, 1, 4, 2, 3, 5))
    tmp4 += einsum(t1, (0, 1), tmp2, (2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1
    tmp4 += einsum(tmp1, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 5, 2)) * -1
    tmp4 += einsum(tmp1, (0, 1, 2, 3), t1, (4, 5), (4, 1, 0, 3, 2, 5))
    t4a += einsum(t1, (0, 1), tmp4, (2, 3, 4, 5, 6, 7), (3, 4, 0, 2, 7, 6, 1, 5))
    del tmp4
    t4a += einsum(tmp3, (0, 1, 2, 3, 4, 5), t1, (6, 7), (6, 1, 2, 0, 4, 7, 5, 3))
    t4a += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 2, 4, 0, 6, 1, 7, 5))
    t4a += einsum(t1, (0, 1), tmp3, (2, 3, 4, 5, 6, 7), (3, 0, 4, 2, 6, 1, 7, 5)) * -1
    del tmp3
    t4a += einsum(tmp1, (0, 1, 2, 3), t2, (4, 5, 6, 7), (1, 0, 4, 5, 2, 7, 6, 3)) * -1
    t4a += einsum(tmp2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 1, 5, 0, 3, 6, 7, 2)) * -1
    del tmp2
    t4a += einsum(tmp1, (0, 1, 2, 3), t2, (4, 5, 6, 7), (1, 4, 5, 0, 3, 6, 7, 2)) * -1
    del tmp1

    return {"t4": t4, "t4a": t4a}


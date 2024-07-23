"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-07-18T20:30:30.323303
  * python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /home/ollie/git/albert/albert/codegen/einsum.py
  * node: ollie-desktop
  * system: Linux
  * processor: x86_64
  * release: 6.5.0-44-generic
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:30:30.670233.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    e_cc : float
        Coupled cluster energy.
    """

    tmp0 = t2.transpose((1, 0, 2, 3)).copy() * -1
    tmp0 += t2.transpose((1, 0, 3, 2)) * 2
    tmp1 = einsum(v.xov, (0, 1, 2), tmp0, (1, 3, 2, 4), (3, 4, 0), optimize=True) * 0.5
    del tmp0
    e_cc = einsum(tmp1, (0, 1, 2), v.xov, (2, 0, 1), (), optimize=True) * 2
    del tmp1

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:30:34.113854.

    Parameters
    ----------
    f : array
        Fock matrix.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    t2new : array
        Updated T2 residuals.
    """

    tmp0 = einsum(t2, (0, 1, 2, 3), v.xov, (4, 1, 2), (0, 3, 4), optimize=True)
    t2new = einsum(tmp0, (0, 1, 2), tmp0, (3, 4, 2), (3, 0, 4, 1), optimize=True)
    tmp11 = t2.transpose((1, 0, 2, 3)).copy() * -1
    tmp11 += t2.transpose((1, 0, 3, 2)) * 2
    tmp1 = einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (3, 1, 4, 2), optimize=True)
    t2new += tmp1.transpose((1, 0, 3, 2))
    tmp6 = v.xov.transpose((1, 2, 0)).copy()
    tmp6 += tmp0 * -1
    tmp12 = einsum(tmp11, (0, 1, 2, 3), v.xov, (4, 0, 2), (1, 3, 4), optimize=True)
    del tmp11
    tmp5 = einsum(v.xvv, (0, 1, 2), v.xoo, (0, 3, 4), (3, 4, 1, 2), optimize=True)
    tmp2 = einsum(tmp1, (0, 1, 2, 3), t2, (4, 0, 3, 5), (4, 1, 5, 2), optimize=True)
    t2new += einsum(t2, (0, 1, 2, 3), tmp2, (4, 1, 5, 2), (4, 0, 3, 5), optimize=True)
    tmp7 = einsum(tmp6, (0, 1, 2), v.xov, (2, 3, 4), (3, 0, 4, 1), optimize=True)
    del tmp6
    tmp13 = einsum(v.xov, (0, 1, 2), tmp12, (1, 3, 0), (2, 3), optimize=True) * 0.5
    tmp15 = einsum(tmp12, (0, 1, 2), v.xov, (2, 3, 1), (3, 0), optimize=True)
    del tmp12
    tmp8 = tmp5.transpose((1, 0, 3, 2)).copy()
    tmp8 += tmp2 * -1
    del tmp2
    tmp8 += tmp7.transpose((1, 0, 3, 2)) * -2
    del tmp7
    tmp14 = einsum(tmp13, (0, 1), t2, (2, 3, 0, 4), (2, 3, 4, 1), optimize=True) * 2
    del tmp13
    tmp16 = einsum(t2, (0, 1, 2, 3), tmp15, (0, 4), (1, 4, 2, 3), optimize=True)
    del tmp15
    tmp4 = einsum(v.xov, (0, 1, 2), tmp0, (3, 4, 0), (3, 1, 4, 2), optimize=True)
    del tmp0
    tmp9 = einsum(tmp8, (0, 1, 2, 3), t2, (1, 4, 3, 5), (4, 0, 5, 2), optimize=True)
    del tmp8
    tmp23 = tmp1.transpose((1, 0, 2, 3)).copy() * 2
    tmp23 += tmp1.transpose((1, 0, 3, 2)) * -1
    tmp19 = einsum(tmp5, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2), optimize=True)
    del tmp5
    tmp18 = einsum(t2, (0, 1, 2, 3), f.vv, (4, 2), (1, 0, 4, 3), optimize=True)
    tmp17 = tmp14.transpose((1, 0, 2, 3)).copy()
    del tmp14
    tmp17 += tmp16.transpose((0, 1, 3, 2))
    del tmp16
    t2new += tmp17.transpose((1, 0, 3, 2)) * -1
    t2new += tmp17 * -1
    del tmp17
    tmp10 = tmp4.copy()
    del tmp4
    tmp10 += tmp9
    del tmp9
    t2new += tmp10.transpose((1, 0, 3, 2)) * -1
    t2new += tmp10 * -1
    del tmp10
    tmp24 = einsum(tmp23, (0, 1, 2, 3), t2, (0, 4, 3, 5), (4, 1, 5, 2), optimize=True)
    del tmp23
    t2new += einsum(tmp24, (0, 1, 2, 3), t2, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True) * 2
    del tmp24
    tmp3 = einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (3, 2, 1, 4), optimize=True)
    t2new += einsum(tmp3, (0, 1, 2, 3), t2, (4, 5, 3, 1), (4, 5, 0, 2), optimize=True)
    del tmp3
    tmp22 = einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (4, 1, 2, 3), optimize=True)
    tmp22 += einsum(t2, (0, 1, 2, 3), tmp1, (4, 5, 2, 3), (4, 1, 5, 0), optimize=True)
    del tmp1
    t2new += einsum(tmp22, (0, 1, 2, 3), t2, (0, 2, 4, 5), (1, 3, 5, 4), optimize=True)
    del tmp22
    tmp20 = tmp18.copy()
    del tmp18
    tmp20 += tmp19 * -1
    del tmp19
    t2new += tmp20.transpose((1, 0, 2, 3))
    t2new += tmp20.transpose((0, 1, 3, 2))
    del tmp20
    tmp21 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3), optimize=True)
    t2new += tmp21.transpose((1, 0, 2, 3)) * -1
    t2new += tmp21.transpose((0, 1, 3, 2)) * -1
    del tmp21

    return {f"t2new": t2new}

def update_lams(f=None, l2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:30:39.075372.

    Parameters
    ----------
    f : array
        Fock matrix.
    l2 : array
        L2 amplitudes.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    l2new : array
        Updated L2 residuals.
    """

    tmp12 = t2.transpose((1, 0, 2, 3)).copy() * -1
    tmp12 += t2.transpose((1, 0, 3, 2)) * 2
    tmp24 = t2.transpose((1, 0, 2, 3)).copy() * -0.5
    tmp24 += t2.transpose((1, 0, 3, 2))
    tmp32 = t2.transpose((1, 0, 2, 3)).copy() * -1
    tmp32 += t2.transpose((1, 0, 3, 2)) * 2
    tmp13 = einsum(tmp12, (0, 1, 2, 3), v.xov, (4, 0, 2), (1, 3, 4), optimize=True)
    del tmp12
    tmp8 = l2.transpose((3, 2, 0, 1)).copy() * 2
    tmp8 += l2.transpose((3, 2, 1, 0)) * -1
    tmp1 = einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (3, 1, 4, 2), optimize=True)
    l2new = tmp1.transpose((3, 2, 1, 0)).copy()
    tmp25 = einsum(l2, (0, 1, 2, 3), tmp24, (2, 4, 0, 1), (3, 4), optimize=True) * 2
    tmp28 = einsum(l2, (0, 1, 2, 3), tmp24, (2, 3, 4, 1), (0, 4), optimize=True) * 2
    del tmp24
    tmp33 = einsum(tmp32, (0, 1, 2, 3), v.xov, (4, 0, 2), (1, 3, 4), optimize=True)
    del tmp32
    tmp14 = v.xov.transpose((1, 2, 0)).copy()
    tmp14 += tmp13
    del tmp13
    tmp15 = l2.transpose((3, 2, 0, 1)).copy() * -1
    tmp15 += l2.transpose((3, 2, 1, 0)) * 2
    tmp9 = einsum(tmp8, (0, 1, 2, 3), t2, (0, 4, 3, 5), (4, 1, 5, 2), optimize=True)
    del tmp8
    tmp7 = einsum(t2, (0, 1, 2, 3), l2, (4, 3, 5, 0), (5, 1, 4, 2), optimize=True)
    tmp20 = einsum(tmp1, (0, 1, 2, 3), t2, (4, 1, 2, 5), (4, 0, 5, 3), optimize=True)
    tmp5 = einsum(v.xvv, (0, 1, 2), v.xoo, (0, 3, 4), (3, 4, 1, 2), optimize=True)
    tmp26 = einsum(tmp25, (0, 1), v.xov, (2, 1, 3), (0, 3, 2), optimize=True) * 0.5
    del tmp25
    tmp29 = einsum(tmp28, (0, 1), v.xov, (2, 3, 1), (3, 0, 2), optimize=True)
    del tmp28
    tmp34 = einsum(v.xov, (0, 1, 2), tmp33, (1, 3, 0), (2, 3), optimize=True) * 0.5
    tmp36 = einsum(tmp33, (0, 1, 2), v.xov, (2, 3, 1), (3, 0), optimize=True)
    del tmp33
    tmp16 = einsum(tmp14, (0, 1, 2), tmp15, (0, 3, 1, 4), (3, 4, 2), optimize=True)
    del tmp14, tmp15
    tmp10 = tmp7.copy()
    del tmp7
    tmp10 += tmp9.transpose((1, 0, 3, 2)) * -1
    del tmp9
    tmp21 = tmp5.transpose((1, 0, 3, 2)).copy() * -1
    tmp21 += tmp20
    del tmp20
    tmp27 = einsum(tmp26, (0, 1, 2), v.xov, (2, 3, 4), (3, 0, 4, 1), optimize=True) * 2
    del tmp26
    tmp30 = einsum(tmp29, (0, 1, 2), v.xov, (2, 3, 4), (3, 0, 4, 1), optimize=True)
    del tmp29
    tmp35 = einsum(tmp34, (0, 1), l2, (1, 2, 3, 4), (3, 4, 2, 0), optimize=True) * 2
    del tmp34
    tmp37 = einsum(l2, (0, 1, 2, 3), tmp36, (4, 2), (4, 3, 0, 1), optimize=True)
    del tmp36
    tmp17 = einsum(tmp16, (0, 1, 2), v.xov, (2, 3, 4), (3, 0, 4, 1), optimize=True)
    del tmp16
    tmp6 = einsum(tmp5, (0, 1, 2, 3), l2, (4, 3, 5, 1), (5, 0, 4, 2), optimize=True)
    del tmp5
    tmp11 = einsum(tmp1, (0, 1, 2, 3), tmp10, (4, 0, 5, 3), (1, 4, 2, 5), optimize=True)
    del tmp10
    tmp19 = einsum(l2, (0, 1, 2, 3), f.vv, (4, 0), (3, 2, 4, 1), optimize=True)
    tmp22 = einsum(l2, (0, 1, 2, 3), tmp21, (2, 4, 1, 5), (3, 4, 0, 5), optimize=True)
    del tmp21
    tmp31 = tmp27.transpose((1, 0, 2, 3)).copy()
    del tmp27
    tmp31 += tmp30.transpose((0, 1, 3, 2))
    del tmp30
    l2new += tmp31.transpose((2, 3, 1, 0)) * -1
    l2new += tmp31.transpose((3, 2, 0, 1)) * -1
    del tmp31
    tmp38 = tmp35.transpose((1, 0, 2, 3)).copy()
    del tmp35
    tmp38 += tmp37.transpose((1, 0, 3, 2))
    del tmp37
    l2new += tmp38.transpose((3, 2, 1, 0)) * -1
    l2new += tmp38.transpose((2, 3, 0, 1)) * -1
    del tmp38
    tmp3 = einsum(tmp1, (0, 1, 2, 3), t2, (4, 5, 3, 2), (5, 4, 0, 1), optimize=True)
    l2new += einsum(tmp3, (0, 1, 2, 3), l2, (4, 5, 1, 0), (5, 4, 2, 3), optimize=True)
    del tmp3
    tmp2 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5), optimize=True)
    l2new += einsum(tmp2, (0, 1, 2, 3), tmp1, (2, 3, 4, 5), (5, 4, 1, 0), optimize=True)
    del tmp1, tmp2
    tmp39 = einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3), optimize=True)
    l2new += tmp39.transpose((2, 3, 1, 0)) * -1
    l2new += tmp39.transpose((3, 2, 0, 1)) * -1
    del tmp39
    tmp4 = einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (3, 2, 1, 4), optimize=True)
    l2new += einsum(l2, (0, 1, 2, 3), tmp4, (4, 1, 5, 0), (5, 4, 3, 2), optimize=True)
    del tmp4
    tmp18 = tmp6.copy()
    del tmp6
    tmp18 += tmp11.transpose((1, 0, 3, 2)) * -1
    del tmp11
    tmp18 += tmp17.transpose((1, 0, 3, 2)) * -1
    del tmp17
    l2new += tmp18.transpose((3, 2, 1, 0)) * -1
    l2new += tmp18.transpose((2, 3, 0, 1)) * -1
    del tmp18
    tmp23 = tmp19.copy()
    del tmp19
    tmp23 += tmp22
    del tmp22
    l2new += tmp23.transpose((2, 3, 1, 0))
    l2new += tmp23.transpose((3, 2, 0, 1))
    del tmp23
    tmp0 = einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 4, 3, 2), optimize=True)
    l2new += einsum(tmp0, (0, 1, 2, 3), l2, (4, 5, 3, 1), (4, 5, 0, 2), optimize=True)
    del tmp0

    return {f"l2new": l2new}

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:30:39.392217.

    Parameters
    ----------
    l2 : array
        L2 amplitudes.
    t2 : array
        T2 amplitudes.

    Returns
    -------
    rdm1 : array
        One-particle reduced density matrix.
    """

    rdm1 = Namespace()
    delta = Namespace(
        oo=np.eye(t2.shape[0]),
        vv=np.eye(t2.shape[-1]),
    )
    tmp1 = t2.transpose((1, 0, 2, 3)).copy()
    tmp1 += t2.transpose((1, 0, 3, 2)) * -0.5
    rdm1.vv = einsum(tmp1, (0, 1, 2, 3), l2, (3, 4, 0, 1), (4, 2), optimize=True) * 4
    del tmp1
    tmp0 = t2.transpose((1, 0, 2, 3)).copy() * -0.5
    tmp0 += t2.transpose((1, 0, 3, 2))
    rdm1.oo = einsum(l2, (0, 1, 2, 3), tmp0, (2, 4, 0, 1), (4, 3), optimize=True) * -4
    del tmp0
    rdm1.oo += delta.oo.transpose((1, 0)) * 2
    del delta
    rdm1.ov = np.zeros((t2.shape[0], t2.shape[-1]))
    rdm1.vo = np.zeros((t2.shape[-1], t2.shape[0]))
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:30:50.605518.

    Parameters
    ----------
    l2 : array
        L2 amplitudes.
    t2 : array
        T2 amplitudes.

    Returns
    -------
    rdm2 : array
        Two-particle reduced density matrix.
    """

    rdm2 = Namespace()
    delta = Namespace(
        oo=np.eye(t2.shape[0]),
        vv=np.eye(t2.shape[-1]),
    )
    tmp3 = l2.transpose((3, 2, 0, 1)).copy() * 2
    tmp3 += l2.transpose((3, 2, 1, 0)) * -1
    tmp6 = l2.transpose((3, 2, 0, 1)).copy()
    tmp6 += l2.transpose((3, 2, 1, 0)) * -1
    tmp12 = l2.transpose((3, 2, 0, 1)).copy() * 2
    tmp12 += l2.transpose((3, 2, 1, 0)) * -1
    tmp0 = t2.transpose((1, 0, 2, 3)).copy()
    tmp0 += t2.transpose((1, 0, 3, 2)) * -0.5
    tmp4 = einsum(tmp3, (0, 1, 2, 3), t2, (0, 4, 3, 5), (4, 1, 5, 2), optimize=True)
    del tmp3
    rdm2.voov = tmp4.transpose((3, 0, 1, 2)).copy()
    rdm2.voov += tmp4.transpose((3, 0, 1, 2))
    rdm2.ovvo = tmp4.transpose((0, 3, 2, 1)).copy()
    rdm2.ovvo += tmp4.transpose((0, 3, 2, 1))
    tmp7 = einsum(t2, (0, 1, 2, 3), tmp6, (0, 4, 5, 3), (1, 4, 2, 5), optimize=True)
    rdm2.ovvo += tmp7.transpose((0, 3, 2, 1)) * -1
    rdm2.ovvo += tmp7.transpose((0, 3, 2, 1)) * -1
    tmp13 = einsum(t2, (0, 1, 2, 3), tmp12, (0, 1, 3, 4), (2, 4), optimize=True) * 0.5
    rdm2.ovvo += einsum(delta.oo, (0, 1), tmp13, (2, 3), (1, 3, 2, 0), optimize=True) * -2
    rdm2.ovvo += einsum(delta.oo, (0, 1), tmp13, (2, 3), (1, 3, 2, 0), optimize=True) * -2
    rdm2.ovov = einsum(delta.oo, (0, 1), tmp13, (2, 3), (1, 3, 0, 2), optimize=True) * 2
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp13, (2, 3), (1, 3, 0, 2), optimize=True) * 2
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp13, (2, 3), (1, 3, 0, 2), optimize=True) * 2
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp13, (2, 3), (1, 3, 0, 2), optimize=True) * 2
    tmp1 = einsum(l2, (0, 1, 2, 3), tmp0, (2, 4, 1, 0), (3, 4), optimize=True) * 2
    del tmp0
    rdm2.oooo = einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 3, 0), optimize=True) * -1
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 3, 2, 0), optimize=True)
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3), optimize=True)
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2), optimize=True) * -1
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 3, 0), optimize=True) * -1
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2), optimize=True) * -1
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 3, 0), optimize=True) * -1
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2), optimize=True) * -1
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 3, 0), optimize=True) * -1
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 3, 2, 0), optimize=True)
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3), optimize=True)
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2), optimize=True) * -1
    tmp25 = l2.transpose((3, 2, 0, 1)).copy() * -1
    tmp25 += l2.transpose((3, 2, 1, 0)) * 2
    tmp23 = t2.transpose((1, 0, 2, 3)).copy() * 2
    tmp23 += t2.transpose((1, 0, 3, 2)) * -1
    tmp21 = t2.transpose((1, 0, 2, 3)).copy()
    tmp21 += t2.transpose((1, 0, 3, 2)) * -1
    tmp5 = einsum(tmp4, (0, 1, 2, 3), t2, (1, 4, 3, 5), (4, 0, 5, 2), optimize=True)
    rdm2.oovv = tmp5.copy() * 2
    rdm2.oovv += tmp5 * 2
    tmp8 = einsum(t2, (0, 1, 2, 3), tmp7, (4, 0, 5, 3), (1, 4, 2, 5), optimize=True)
    del tmp7
    tmp14 = einsum(tmp13, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0), optimize=True) * 2
    del tmp13
    tmp15 = einsum(tmp1, (0, 1), t2, (0, 2, 3, 4), (2, 1, 3, 4), optimize=True)
    del tmp1
    tmp19 = einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5), optimize=True)
    rdm2.vovo = tmp19.transpose((2, 1, 3, 0)).copy() * -1
    rdm2.vovo += tmp19.transpose((2, 1, 3, 0)) * -1
    rdm2.ovov += tmp19.transpose((1, 2, 0, 3)) * -1
    rdm2.ovov += tmp19.transpose((1, 2, 0, 3)) * -1
    tmp17 = einsum(t2, (0, 1, 2, 3), l2, (4, 2, 5, 1), (5, 0, 4, 3), optimize=True)
    rdm2.voov += tmp17.transpose((2, 1, 0, 3)) * -1
    rdm2.voov += tmp17.transpose((2, 1, 0, 3)) * -1
    rdm2.ovvo += tmp17.transpose((1, 2, 3, 0)) * -1
    rdm2.ovvo += tmp17.transpose((1, 2, 3, 0)) * -1
    tmp2 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 1, 0), (2, 3, 5, 4), optimize=True)
    rdm2.oooo += tmp2.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp2.transpose((2, 3, 1, 0)) * -1
    rdm2.oooo += tmp2.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp2.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp2.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp2.transpose((2, 3, 1, 0)) * -1
    tmp29 = einsum(t2, (0, 1, 2, 3), l2, (4, 5, 0, 1), (4, 5, 2, 3), optimize=True)
    rdm2.vvvv = tmp29.transpose((1, 0, 3, 2)).copy()
    rdm2.vvvv += tmp29.transpose((1, 0, 2, 3)) * -1
    rdm2.vvvv += tmp29.transpose((1, 0, 3, 2))
    rdm2.vvvv += tmp29.transpose((1, 0, 3, 2))
    rdm2.vvvv += tmp29.transpose((1, 0, 3, 2))
    rdm2.vvvv += tmp29.transpose((1, 0, 2, 3)) * -1
    del tmp29
    tmp27 = einsum(t2, (0, 1, 2, 3), tmp12, (0, 1, 3, 4), (2, 4), optimize=True)
    del tmp12
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp27, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp27, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp27, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp27, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.voov += einsum(delta.oo, (0, 1), tmp27, (2, 3), (3, 1, 0, 2), optimize=True) * -1
    rdm2.voov += einsum(delta.oo, (0, 1), tmp27, (2, 3), (3, 1, 0, 2), optimize=True) * -1
    del tmp27
    tmp26 = einsum(tmp25, (0, 1, 2, 3), t2, (0, 4, 2, 5), (4, 1, 5, 3), optimize=True)
    del tmp25
    rdm2.vovo += tmp26.transpose((3, 0, 2, 1)) * -1
    rdm2.vovo += tmp26.transpose((3, 0, 2, 1)) * -1
    rdm2.ovvo += tmp26.transpose((0, 3, 2, 1))
    rdm2.ovvo += tmp26.transpose((0, 3, 2, 1))
    del tmp26
    tmp28 = einsum(t2, (0, 1, 2, 3), tmp6, (0, 4, 3, 5), (1, 4, 2, 5), optimize=True)
    del tmp6
    rdm2.vovo += tmp28.transpose((3, 0, 2, 1)) * -1
    rdm2.vovo += tmp28.transpose((3, 0, 2, 1)) * -1
    del tmp28
    tmp24 = einsum(l2, (0, 1, 2, 3), tmp23, (2, 4, 5, 0), (3, 4, 1, 5), optimize=True)
    del tmp23
    rdm2.voov += tmp24.transpose((2, 1, 0, 3))
    rdm2.voov += tmp24.transpose((2, 1, 0, 3))
    rdm2.ovov += tmp24.transpose((1, 2, 0, 3)) * -1
    rdm2.ovov += tmp24.transpose((1, 2, 0, 3)) * -1
    del tmp24
    tmp22 = einsum(l2, (0, 1, 2, 3), tmp21, (2, 4, 5, 1), (3, 4, 0, 5), optimize=True)
    del tmp21
    rdm2.voov += tmp22.transpose((2, 1, 0, 3)) * -1
    rdm2.voov += tmp22.transpose((2, 1, 0, 3)) * -1
    rdm2.ovov += tmp22.transpose((1, 2, 0, 3))
    rdm2.ovov += tmp22.transpose((1, 2, 0, 3))
    del tmp22
    tmp9 = tmp5.copy() * 2
    del tmp5
    tmp9 += tmp8.transpose((1, 0, 3, 2))
    del tmp8
    rdm2.oovv += tmp9.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp9
    rdm2.oovv += tmp9.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp9
    del tmp9
    tmp10 = einsum(t2, (0, 1, 2, 3), tmp4, (4, 0, 5, 3), (1, 4, 2, 5), optimize=True)
    del tmp4
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp10.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp10.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    del tmp10
    tmp16 = tmp14.transpose((1, 0, 2, 3)).copy()
    del tmp14
    tmp16 += tmp15.transpose((0, 1, 3, 2))
    del tmp15
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp16.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp16 * -1
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16 * -1
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16 * -1
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp16.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp16 * -1
    del tmp16
    tmp20 = einsum(tmp19, (0, 1, 2, 3), t2, (4, 0, 2, 5), (4, 1, 5, 3), optimize=True)
    del tmp19
    rdm2.oovv += tmp20.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp20.transpose((0, 1, 3, 2))
    del tmp20
    tmp18 = einsum(tmp17, (0, 1, 2, 3), t2, (4, 0, 2, 5), (4, 1, 5, 3), optimize=True)
    del tmp17
    rdm2.oovv += tmp18
    rdm2.oovv += tmp18
    del tmp18
    tmp11 = einsum(t2, (0, 1, 2, 3), tmp2, (0, 1, 4, 5), (4, 5, 2, 3), optimize=True)
    del tmp2
    rdm2.oovv += tmp11.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp11
    rdm2.oovv += tmp11
    rdm2.oovv += tmp11
    rdm2.oovv += tmp11.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp11
    del tmp11
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    del delta
    rdm2.oovv += t2.transpose((1, 0, 2, 3)) * -1
    rdm2.oovv += t2.transpose((1, 0, 3, 2))
    rdm2.oovv += t2.transpose((1, 0, 3, 2))
    rdm2.oovv += t2.transpose((1, 0, 3, 2))
    rdm2.oovv += t2.transpose((1, 0, 2, 3)) * -1
    rdm2.oovv += t2.transpose((1, 0, 3, 2))
    rdm2.vvoo = l2.transpose((0, 1, 3, 2)).copy() * -1
    rdm2.vvoo += l2.transpose((1, 0, 3, 2))
    rdm2.vvoo += l2.transpose((1, 0, 3, 2))
    rdm2.vvoo += l2.transpose((1, 0, 3, 2))
    rdm2.vvoo += l2.transpose((0, 1, 3, 2)) * -1
    rdm2.vvoo += l2.transpose((1, 0, 3, 2))
    rdm2.ooov = np.zeros((t2.shape[0], t2.shape[0], t2.shape[0], t2.shape[-1]))
    rdm2.oovo = np.zeros((t2.shape[0], t2.shape[0], t2.shape[-1], t2.shape[0]))
    rdm2.ovoo = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[0], t2.shape[0]))
    rdm2.vooo = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[0], t2.shape[0]))
    rdm2.ovvv = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[-1], t2.shape[-1]))
    rdm2.vovv = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[-1], t2.shape[-1]))
    rdm2.vvov = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[0], t2.shape[-1]))
    rdm2.vvvo = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[-1], t2.shape[0]))
    rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv)
    rdm2 = rdm2.swapaxes(1, 2)

    return rdm2

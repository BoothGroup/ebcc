"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-16T23:17:15.781080
  * python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /home/ollie/git/albert/albert/codegen/einsum.py
  * node: ollie-desktop
  * system: Linux
  * processor: x86_64
  * release: 6.8.0-40-generic
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:16.095003.

    Parameters
    ----------
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    e_mp : array
    """

    e_mp = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 2), ()) * -1
    e_mp += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_mp += einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (0, 2, 1, 3), ())

    return e_mp

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:16.597503.

    Parameters
    ----------
    l2 : Namespace of arrays
        L2 amplitudes.
    t2 : Namespace of arrays
        T2 amplitudes.

    Returns
    -------
    rdm1 : Namespace of arrays
        One-particle reduced density matrix.
    """

    rdm1 = Namespace()
    rdm1.aa = Namespace()
    rdm1.bb = Namespace()
    delta = Namespace(
        aa=Namespace(oo=np.eye(t2.aaaa.shape[0]), vv=np.eye(t2.aaaa.shape[-1])),
        bb=Namespace(oo=np.eye(t2.bbbb.shape[0]), vv=np.eye(t2.bbbb.shape[-1])),
    )
    rdm1.bb.vv = einsum(t2.bbbb, (0, 1, 2, 3), l2.bbbb, (4, 3, 0, 1), (4, 2)) * 2
    rdm1.bb.vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    rdm1.aa.vv = einsum(t2.aaaa, (0, 1, 2, 3), l2.aaaa, (4, 3, 0, 1), (4, 2)) * 2
    rdm1.aa.vv += einsum(t2.abab, (0, 1, 2, 3), l2.abab, (4, 3, 0, 1), (4, 2))
    rdm1.bb.oo = einsum(t2.bbbb, (0, 1, 2, 3), l2.bbbb, (2, 3, 4, 1), (0, 4)) * -2
    rdm1.bb.oo += delta.bb.oo
    rdm1.bb.oo += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (4, 3)) * -1
    rdm1.aa.oo = einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (4, 2)) * -2
    rdm1.aa.oo += einsum(t2.abab, (0, 1, 2, 3), l2.abab, (2, 3, 4, 1), (0, 4)) * -1
    rdm1.aa.oo += delta.aa.oo
    del delta
    rdm1.aa.ov = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[-1]))
    rdm1.aa.vo = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[0]))
    rdm1.bb.ov = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[-1]))
    rdm1.bb.vo = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[0]))
    rdm1.aa = np.block([[rdm1.aa.oo, rdm1.aa.ov], [rdm1.aa.vo, rdm1.aa.vv]])
    rdm1.bb = np.block([[rdm1.bb.oo, rdm1.bb.ov], [rdm1.bb.vo, rdm1.bb.vv]])

    return rdm1

def make_rdm2_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:16.891730.

    Parameters
    ----------
    l2 : Namespace of arrays
        L2 amplitudes.
    t2 : Namespace of arrays
        T2 amplitudes.

    Returns
    -------
    rdm2 : Namespace of arrays
        Two-particle reduced density matrix.
    """

    rdm2 = Namespace()
    rdm2.aaaa = Namespace()
    rdm2.abab = Namespace()
    rdm2.bbbb = Namespace()
    delta = Namespace(
        aa=Namespace(oo=np.eye(t2.aaaa.shape[0]), vv=np.eye(t2.aaaa.shape[-1])),
        bb=Namespace(oo=np.eye(t2.bbbb.shape[0]), vv=np.eye(t2.bbbb.shape[-1])),
    )
    rdm2.bbbb.vvoo = l2.bbbb.transpose((2, 3, 0, 1)).copy() * 0.25
    rdm2.bbbb.vvoo += l2.bbbb.transpose((2, 3, 0, 1)) * 0.25
    rdm2.abab.vvoo = l2.abab.transpose((2, 3, 0, 1)).copy() * 0.25
    rdm2.aaaa.vvoo = l2.aaaa.transpose((2, 3, 0, 1)).copy() * 0.25
    rdm2.aaaa.vvoo += l2.aaaa.transpose((2, 3, 0, 1)) * 0.25
    rdm2.bbbb.oovv = t2.bbbb.copy() * 0.25
    rdm2.bbbb.oovv += t2.bbbb * 0.25
    rdm2.abab.oovv = t2.abab.copy() * 0.25
    rdm2.aaaa.oovv = t2.aaaa.copy() * 0.25
    rdm2.aaaa.oovv += t2.aaaa * 0.25
    rdm2.bbbb.oooo = einsum(delta.bb.oo, (0, 1), delta.bb.oo, (2, 3), (0, 2, 1, 3))
    rdm2.bbbb.oooo += einsum(delta.bb.oo, (0, 1), delta.bb.oo, (2, 3), (0, 2, 3, 1)) * -1
    rdm2.abab.oooo = einsum(delta.aa.oo, (0, 1), delta.bb.oo, (2, 3), (0, 2, 1, 3))
    rdm2.aaaa.oooo = einsum(delta.aa.oo, (0, 1), delta.aa.oo, (2, 3), (0, 2, 1, 3))
    rdm2.aaaa.oooo += einsum(delta.aa.oo, (0, 1), delta.aa.oo, (2, 3), (0, 2, 3, 1)) * -1
    del delta
    rdm2.aaaa.oooo = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[0]))
    rdm2.aaaa.oovv = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[-1]))
    rdm2.aaaa.vvoo = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[0]))
    rdm2.abab.oooo = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[0], t2.aaaa.shape[0], t2.bbbb.shape[0]))
    rdm2.abab.oovv = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[0], t2.aaaa.shape[-1], t2.bbbb.shape[-1]))
    rdm2.abab.vvoo = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[-1], t2.aaaa.shape[0], t2.bbbb.shape[0]))
    rdm2.bbbb.oooo = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[0]))
    rdm2.bbbb.oovv = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[-1]))
    rdm2.bbbb.vvoo = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[0]))
    rdm2.aaaa = pack_2e(rdm2.aaaa.oooo, rdm2.aaaa.ooov, rdm2.aaaa.oovo, rdm2.aaaa.ovoo, rdm2.aaaa.vooo, rdm2.aaaa.oovv, rdm2.aaaa.ovov, rdm2.aaaa.ovvo, rdm2.aaaa.voov, rdm2.aaaa.vovo, rdm2.aaaa.vvoo, rdm2.aaaa.ovvv, rdm2.aaaa.vovv, rdm2.aaaa.vvov, rdm2.aaaa.vvvo, rdm2.aaaa.vvvv)
    rdm2.abab = pack_2e(rdm2.abab.oooo, rdm2.abab.ooov, rdm2.abab.oovo, rdm2.abab.ovoo, rdm2.abab.vooo, rdm2.abab.oovv, rdm2.abab.ovov, rdm2.abab.ovvo, rdm2.abab.voov, rdm2.abab.vovo, rdm2.abab.vvoo, rdm2.abab.ovvv, rdm2.abab.vovv, rdm2.abab.vvov, rdm2.abab.vvvo, rdm2.abab.vvvv)
    rdm2.bbbb = pack_2e(rdm2.bbbb.oooo, rdm2.bbbb.ooov, rdm2.bbbb.oovo, rdm2.bbbb.ovoo, rdm2.bbbb.vooo, rdm2.bbbb.oovv, rdm2.bbbb.ovov, rdm2.bbbb.ovvo, rdm2.bbbb.voov, rdm2.bbbb.vovo, rdm2.bbbb.vvoo, rdm2.bbbb.ovvv, rdm2.bbbb.vovv, rdm2.bbbb.vvov, rdm2.bbbb.vvvo, rdm2.bbbb.vvvv)
    rdm2 = Namespace(
        aaaa=rdm2.aaaa.swapaxes(1, 2),
        aabb=rdm2.abab.swapaxes(1, 2),
        bbbb=rdm2.bbbb.swapaxes(1, 2),
    )

    return rdm2

def hbar_matvec_ip_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:20.037413.

    Parameters
    ----------
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    tmp10 : array
    tmp12 : array
    tmp2 : array
    tmp4 : array
    """

    tmp12 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    tmp10 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1
    tmp4 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 1, 3), (4, 0))
    tmp2 = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 1, 3), (0, 4))

    return {f"tmp10": tmp10, f"tmp12": tmp12, f"tmp2": tmp2, f"tmp4": tmp4}

def hbar_matvec_ip(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:20.042814.

    Parameters
    ----------
    f : Namespace of arrays
        Fock matrix.
    r1 : Namespace of arrays
        R1 amplitudes.
    r2 : Namespace of arrays
        R2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    r1new : Namespace of arrays
        Updated R1 residuals.
    r2new : Namespace of arrays
        Updated R2 residuals.
    """

    ints = kwargs["ints"]
    r1new = Namespace()
    r2new = Namespace()
    tmp2 = ints.tmp10.copy() * 2
    del ints.tmp10
    tmp2 += ints.tmp12
    del ints.tmp12
    tmp0 = ints.tmp2.copy() * 2
    del ints.tmp2
    tmp0 += ints.tmp4
    del ints.tmp4
    tmp7 = einsum(f.bb.oo, (0, 1), r2.bbb, (2, 1, 3), (0, 2, 3))
    tmp6 = einsum(r1.b, (0,), v.bbbb.ooov, (1, 0, 2, 3), (1, 2, 3))
    tmp4 = einsum(v.aaaa.ooov, (0, 1, 2, 3), r1.a, (1,), (0, 2, 3))
    tmp5 = einsum(r2.aaa, (0, 1, 2), f.aa.oo, (3, 1), (3, 0, 2))
    tmp3 = f.bb.oo.copy() * 2
    tmp3 += tmp2
    tmp3 += tmp2.transpose((1, 0))
    del tmp2
    tmp1 = f.aa.oo.copy() * 2
    tmp1 += tmp0
    tmp1 += tmp0.transpose((1, 0))
    del tmp0
    r2new.bbb = einsum(r2.bbb, (0, 1, 2), f.bb.vv, (3, 2), (0, 1, 3)) * 2
    r2new.bbb += tmp6 * -1
    r2new.bbb += tmp6.transpose((1, 0, 2))
    del tmp6
    r2new.bbb += tmp7 * 2
    r2new.bbb += tmp7.transpose((1, 0, 2)) * -2
    del tmp7
    r2new.bab = einsum(v.aabb.ooov, (0, 1, 2, 3), r1.a, (1,), (2, 0, 3))
    r2new.bab += einsum(f.aa.oo, (0, 1), r2.bab, (2, 1, 3), (2, 0, 3)) * -1
    r2new.bab += einsum(r2.bab, (0, 1, 2), f.bb.oo, (3, 0), (3, 1, 2)) * -1
    r2new.bab += einsum(r2.bab, (0, 1, 2), f.bb.vv, (3, 2), (0, 1, 3))
    r2new.aba = einsum(f.aa.oo, (0, 1), r2.aba, (1, 2, 3), (0, 2, 3)) * -1
    r2new.aba += einsum(r2.aba, (0, 1, 2), f.aa.vv, (3, 2), (0, 1, 3))
    r2new.aba += einsum(v.aabb.ovoo, (0, 1, 2, 3), r1.b, (3,), (0, 2, 1))
    r2new.aba += einsum(r2.aba, (0, 1, 2), f.bb.oo, (3, 1), (0, 3, 2)) * -1
    r2new.aaa = einsum(r2.aaa, (0, 1, 2), f.aa.vv, (3, 2), (0, 1, 3)) * 2
    r2new.aaa += tmp4 * -1
    r2new.aaa += tmp4.transpose((1, 0, 2))
    del tmp4
    r2new.aaa += tmp5 * 2
    r2new.aaa += tmp5.transpose((1, 0, 2)) * -2
    del tmp5
    r1new.b = einsum(r2.aba, (0, 1, 2), v.aabb.ovoo, (0, 2, 3, 1), (3,))
    r1new.b += einsum(r2.bbb, (0, 1, 2), v.bbbb.ooov, (3, 0, 1, 2), (3,)) * -2
    r1new.b += einsum(tmp3, (0, 1), r1.b, (0,), (1,)) * -0.5
    del tmp3
    r1new.a = einsum(v.aabb.ooov, (0, 1, 2, 3), r2.bab, (2, 1, 3), (0,))
    r1new.a += einsum(r2.aaa, (0, 1, 2), v.aaaa.ooov, (3, 1, 0, 2), (3,)) * 2
    r1new.a += einsum(r1.a, (0,), tmp1, (0, 1), (1,)) * -0.5
    del tmp1

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ea_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:23.275435.

    Parameters
    ----------
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    tmp10 : array
    tmp12 : array
    tmp2 : array
    tmp4 : array
    """

    tmp12 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    tmp10 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1
    tmp4 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    tmp2 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 4, 1), (4, 3)) * -1

    return {f"tmp10": tmp10, f"tmp12": tmp12, f"tmp2": tmp2, f"tmp4": tmp4}

def hbar_matvec_ea(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:23.281024.

    Parameters
    ----------
    f : Namespace of arrays
        Fock matrix.
    r1 : Namespace of arrays
        R1 amplitudes.
    r2 : Namespace of arrays
        R2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    r1new : Namespace of arrays
        Updated R1 residuals.
    r2new : Namespace of arrays
        Updated R2 residuals.
    """

    ints = kwargs["ints"]
    r1new = Namespace()
    r2new = Namespace()
    tmp7 = einsum(f.bb.vv, (0, 1), r2.bbb, (2, 1, 3), (3, 0, 2))
    tmp8 = einsum(r1.b, (0,), v.bbbb.ovvv, (1, 2, 3, 0), (1, 2, 3))
    tmp4 = einsum(f.aa.vv, (0, 1), r2.aaa, (2, 1, 3), (3, 0, 2))
    tmp5 = einsum(v.aaaa.ovvv, (0, 1, 2, 3), r1.a, (3,), (0, 1, 2))
    tmp2 = ints.tmp10.copy()
    del ints.tmp10
    tmp2 += ints.tmp12 * 0.5
    del ints.tmp12
    tmp0 = ints.tmp2.copy()
    del ints.tmp2
    tmp0 += ints.tmp4 * 0.5
    del ints.tmp4
    tmp9 = tmp7.copy() * 2
    del tmp7
    tmp9 += tmp8
    del tmp8
    tmp6 = tmp4.copy() * 2
    del tmp4
    tmp6 += tmp5
    del tmp5
    tmp3 = f.bb.vv.copy() * -1
    tmp3 += tmp2
    tmp3 += tmp2.transpose((1, 0))
    del tmp2
    tmp1 = f.aa.vv.copy() * -1
    tmp1 += tmp0
    tmp1 += tmp0.transpose((1, 0))
    del tmp0
    r2new.bbb = einsum(r2.bbb, (0, 1, 2), f.bb.oo, (3, 2), (0, 1, 3)) * -2
    r2new.bbb += tmp9.transpose((1, 2, 0)) * -1
    r2new.bbb += tmp9.transpose((2, 1, 0))
    del tmp9
    r2new.bab = einsum(f.aa.vv, (0, 1), r2.bab, (2, 1, 3), (2, 0, 3))
    r2new.bab += einsum(r1.a, (0,), v.aabb.vvov, (1, 0, 2, 3), (3, 1, 2)) * -1
    r2new.bab += einsum(f.bb.oo, (0, 1), r2.bab, (2, 3, 1), (2, 3, 0)) * -1
    r2new.bab += einsum(r2.bab, (0, 1, 2), f.bb.vv, (3, 0), (3, 1, 2))
    r2new.aba = einsum(f.aa.oo, (0, 1), r2.aba, (2, 3, 1), (2, 3, 0)) * -1
    r2new.aba += einsum(r2.aba, (0, 1, 2), f.aa.vv, (3, 0), (3, 1, 2))
    r2new.aba += einsum(r2.aba, (0, 1, 2), f.bb.vv, (3, 1), (0, 3, 2))
    r2new.aba += einsum(r1.b, (0,), v.aabb.ovvv, (1, 2, 3, 0), (2, 3, 1)) * -1
    r2new.aaa = einsum(f.aa.oo, (0, 1), r2.aaa, (2, 3, 1), (2, 3, 0)) * -2
    r2new.aaa += tmp6.transpose((1, 2, 0)) * -1
    r2new.aaa += tmp6.transpose((2, 1, 0))
    del tmp6
    r1new.b = einsum(v.bbbb.ovvv, (0, 1, 2, 3), r2.bbb, (3, 1, 0), (2,)) * 2
    r1new.b += einsum(r2.aba, (0, 1, 2), v.aabb.ovvv, (2, 0, 3, 1), (3,)) * -1
    r1new.b += einsum(r1.b, (0,), tmp3, (0, 1), (1,)) * -1
    del tmp3
    r1new.a = einsum(v.aabb.vvov, (0, 1, 2, 3), r2.bab, (3, 1, 2), (0,)) * -1
    r1new.a += einsum(r2.aaa, (0, 1, 2), v.aaaa.ovvv, (2, 0, 3, 1), (3,)) * -2
    r1new.a += einsum(r1.a, (0,), tmp1, (0, 1), (1,)) * -1
    del tmp1

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ee_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:30.742041.

    Parameters
    ----------
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    tmp17 : array
    tmp19 : array
    tmp2 : array
    tmp23 : array
    tmp25 : array
    tmp32 : array
    tmp34 : array
    tmp4 : array
    """

    tmp34 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    tmp32 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4))
    tmp25 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    tmp23 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    tmp19 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 1, 3), (4, 0))
    tmp17 = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1
    tmp4 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    tmp2 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 4, 3), (4, 1))

    return {f"tmp17": tmp17, f"tmp19": tmp19, f"tmp2": tmp2, f"tmp23": tmp23, f"tmp25": tmp25, f"tmp32": tmp32, f"tmp34": tmp34, f"tmp4": tmp4}

def hbar_matvec_ee(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-16T23:17:30.752782.

    Parameters
    ----------
    f : Namespace of arrays
        Fock matrix.
    r1 : Namespace of arrays
        R1 amplitudes.
    r2 : Namespace of arrays
        R2 amplitudes.
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    r1new : Namespace of arrays
        Updated R1 residuals.
    r2new : Namespace of arrays
        Updated R2 residuals.
    """

    ints = kwargs["ints"]
    r1new = Namespace()
    r2new = Namespace()
    tmp5 = v.aaaa.ovov.transpose((0, 2, 3, 1)).copy()
    tmp5 += v.aaaa.ovov.transpose((0, 2, 1, 3)) * -1
    tmp2 = v.bbbb.ovov.transpose((0, 2, 3, 1)).copy()
    tmp2 += v.bbbb.ovov.transpose((0, 2, 1, 3)) * -1
    tmp19 = einsum(v.bbbb.ooov, (0, 1, 2, 3), r1.bb, (1, 4), (0, 2, 4, 3))
    tmp20 = einsum(r1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp15 = einsum(r1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp14 = einsum(v.aaaa.ooov, (0, 1, 2, 3), r1.aa, (1, 4), (0, 2, 4, 3))
    tmp6 = einsum(r1.aa, (0, 1), tmp5, (0, 2, 1, 3), (2, 3))
    del tmp5
    tmp0 = einsum(r1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    tmp3 = einsum(tmp2, (0, 1, 2, 3), r1.bb, (0, 2), (1, 3))
    del tmp2
    tmp1 = einsum(v.aabb.ovov, (0, 1, 2, 3), r1.aa, (0, 1), (2, 3))
    tmp21 = tmp19.copy()
    del tmp19
    tmp21 += tmp20
    del tmp20
    tmp23 = einsum(r2.bbbb, (0, 1, 2, 3), f.bb.vv, (4, 3), (0, 1, 4, 2))
    tmp22 = einsum(f.bb.oo, (0, 1), r2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp17 = einsum(r2.aaaa, (0, 1, 2, 3), f.aa.oo, (4, 1), (4, 0, 2, 3))
    tmp18 = einsum(f.aa.vv, (0, 1), r2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp16 = tmp14.copy()
    del tmp14
    tmp16 += tmp15
    del tmp15
    tmp12 = ints.tmp23.transpose((1, 0)).copy() * 2
    del ints.tmp23
    tmp12 += ints.tmp25.transpose((1, 0))
    del ints.tmp25
    tmp12 += f.bb.vv * -1
    tmp13 = f.bb.oo.copy()
    tmp13 += ints.tmp32.transpose((1, 0)) * 2
    del ints.tmp32
    tmp13 += ints.tmp34.transpose((1, 0))
    del ints.tmp34
    tmp11 = v.bbbb.ovov.transpose((0, 2, 1, 3)).copy()
    tmp11 += v.bbbb.oovv * -1
    tmp7 = tmp0.copy()
    tmp7 += tmp6 * -1
    del tmp6
    tmp4 = tmp1.copy()
    tmp4 += tmp3 * -1
    del tmp3
    tmp9 = ints.tmp2.transpose((1, 0)).copy() * 2
    del ints.tmp2
    tmp9 += ints.tmp4.transpose((1, 0))
    del ints.tmp4
    tmp9 += f.aa.vv * -1
    tmp10 = f.aa.oo.copy()
    tmp10 += ints.tmp17.transpose((1, 0)) * 2
    del ints.tmp17
    tmp10 += ints.tmp19.transpose((1, 0))
    del ints.tmp19
    tmp8 = v.aaaa.ovov.transpose((0, 2, 1, 3)).copy()
    tmp8 += v.aaaa.oovv * -1
    r2new.bbbb = tmp21.copy() * -1
    r2new.bbbb += tmp21.transpose((0, 1, 3, 2))
    r2new.bbbb += tmp21.transpose((1, 0, 2, 3))
    r2new.bbbb += tmp21.transpose((1, 0, 3, 2)) * -1
    del tmp21
    r2new.bbbb += tmp22.transpose((0, 1, 3, 2)) * -2
    r2new.bbbb += tmp22.transpose((1, 0, 3, 2)) * 2
    del tmp22
    r2new.bbbb += tmp23.transpose((1, 0, 2, 3)) * 2
    r2new.bbbb += tmp23.transpose((1, 0, 3, 2)) * -2
    del tmp23
    r2new.abab = einsum(f.aa.oo, (0, 1), r2.abab, (1, 2, 3, 4), (0, 2, 3, 4)) * -1
    r2new.abab += einsum(v.aabb.ooov, (0, 1, 2, 3), r1.aa, (1, 4), (0, 2, 4, 3)) * -1
    r2new.abab += einsum(v.aabb.vvov, (0, 1, 2, 3), r1.aa, (4, 1), (4, 2, 0, 3))
    r2new.abab += einsum(r2.abab, (0, 1, 2, 3), f.aa.vv, (4, 2), (0, 1, 4, 3))
    r2new.abab += einsum(f.bb.oo, (0, 1), r2.abab, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    r2new.abab += einsum(v.aabb.ovoo, (0, 1, 2, 3), r1.bb, (3, 4), (0, 2, 1, 4)) * -1
    r2new.abab += einsum(r2.abab, (0, 1, 2, 3), f.bb.vv, (4, 3), (0, 1, 2, 4))
    r2new.abab += einsum(r1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    r2new.aaaa = tmp16.copy() * -1
    r2new.aaaa += tmp16.transpose((0, 1, 3, 2))
    r2new.aaaa += tmp16.transpose((1, 0, 2, 3))
    r2new.aaaa += tmp16.transpose((1, 0, 3, 2)) * -1
    del tmp16
    r2new.aaaa += tmp17.transpose((0, 1, 3, 2)) * -2
    r2new.aaaa += tmp17.transpose((1, 0, 3, 2)) * 2
    del tmp17
    r2new.aaaa += tmp18.transpose((1, 0, 2, 3)) * 2
    r2new.aaaa += tmp18.transpose((1, 0, 3, 2)) * -2
    del tmp18
    r1new.bb = einsum(r2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 0, 1, 3), (4, 2)) * -2
    r1new.bb += einsum(v.aabb.ovoo, (0, 1, 2, 3), r2.abab, (0, 3, 1, 4), (2, 4)) * -1
    r1new.bb += einsum(r2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (1, 2, 4, 3), (0, 4)) * -2
    r1new.bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), r2.abab, (0, 4, 1, 3), (4, 2))
    r1new.bb += tmp1
    del tmp1
    r1new.bb += einsum(tmp4, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2
    r1new.bb += einsum(t2.abab, (0, 1, 2, 3), tmp7, (0, 2), (1, 3))
    r1new.bb += einsum(tmp11, (0, 1, 2, 3), r1.bb, (0, 2), (1, 3))
    del tmp11
    r1new.bb += einsum(tmp12, (0, 1), r1.bb, (2, 0), (2, 1)) * -1
    del tmp12
    r1new.bb += einsum(tmp13, (0, 1), r1.bb, (0, 2), (1, 2)) * -1
    del tmp13
    r1new.aa = einsum(r2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 3), (0, 4))
    r1new.aa += tmp0
    del tmp0
    r1new.aa += einsum(v.aaaa.ooov, (0, 1, 2, 3), r2.aaaa, (1, 2, 4, 3), (0, 4)) * -2
    r1new.aa += einsum(v.aabb.ooov, (0, 1, 2, 3), r2.abab, (1, 2, 4, 3), (0, 4)) * -1
    r1new.aa += einsum(r2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (1, 3, 4, 2), (0, 4)) * 2
    r1new.aa += einsum(t2.abab, (0, 1, 2, 3), tmp4, (1, 3), (0, 2))
    del tmp4
    r1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp7, (1, 3), (0, 2)) * 2
    del tmp7
    r1new.aa += einsum(r1.aa, (0, 1), tmp8, (0, 2, 1, 3), (2, 3))
    del tmp8
    r1new.aa += einsum(tmp9, (0, 1), r1.aa, (2, 0), (2, 1)) * -1
    del tmp9
    r1new.aa += einsum(tmp10, (0, 1), r1.aa, (0, 2), (1, 2)) * -1
    del tmp10
    r2new.baba = r2new.abab.transpose(1, 0, 3, 2)

    return {f"r1new": r1new, f"r2new": r2new}


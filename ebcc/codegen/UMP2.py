"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-09-28T12:18:22.343441
  * python version: 3.10.15 (main, Sep  9 2024, 03:02:45) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/albert/codegen/einsum.py
  * node: fv-az1272-977
  * system: Linux
  * processor: x86_64
  * release: 6.8.0-1014-azure
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, dirsum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:22.859062.

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
    e_mp += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1
    e_mp += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 1, 3), ())

    return e_mp

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:23.721758.

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
    rdm1.bb.vv = einsum(t2.abab, (0, 1, 2, 3), l2.abab, (2, 4, 0, 1), (4, 3))
    rdm1.bb.vv += einsum(t2.bbbb, (0, 1, 2, 3), l2.bbbb, (4, 3, 0, 1), (4, 2)) * 2
    rdm1.aa.vv = einsum(t2.abab, (0, 1, 2, 3), l2.abab, (4, 3, 0, 1), (4, 2))
    rdm1.aa.vv += einsum(t2.aaaa, (0, 1, 2, 3), l2.aaaa, (4, 3, 0, 1), (4, 2)) * 2
    rdm1.bb.oo = einsum(t2.abab, (0, 1, 2, 3), l2.abab, (2, 3, 0, 4), (1, 4)) * -1
    rdm1.bb.oo += delta.bb.oo
    rdm1.bb.oo += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (4, 2)) * -2
    rdm1.aa.oo = einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (4, 2)) * -2
    rdm1.aa.oo += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (4, 2)) * -1
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
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:23.852266.

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
    rdm2.bbbb.vvoo = np.copy(l2.bbbb) * 2
    rdm2.abab.vvoo = np.copy(l2.abab)
    rdm2.aaaa.vvoo = np.copy(l2.aaaa) * 2
    rdm2.bbbb.oovv = np.copy(t2.bbbb) * 2
    rdm2.abab.oovv = np.copy(t2.abab)
    rdm2.aaaa.oovv = np.copy(t2.aaaa) * 2
    rdm2.aaaa.oooo = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[0]))
    rdm2.aaaa.ooov = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[-1]))
    rdm2.aaaa.oovo = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[0]))
    rdm2.aaaa.ovoo = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[0]))
    rdm2.aaaa.vooo = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[0]))
    rdm2.aaaa.ovov = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[-1]))
    rdm2.aaaa.ovvo = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[0]))
    rdm2.aaaa.voov = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[0], t2.aaaa.shape[-1]))
    rdm2.aaaa.vovo = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[0]))
    rdm2.aaaa.ovvv = np.zeros((t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[-1]))
    rdm2.aaaa.vovv = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[-1], t2.aaaa.shape[-1]))
    rdm2.aaaa.vvov = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[0], t2.aaaa.shape[-1]))
    rdm2.aaaa.vvvo = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[0]))
    rdm2.aaaa.vvvv = np.zeros((t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[-1], t2.aaaa.shape[-1]))
    rdm2.abab.oooo = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[0], t2.aaaa.shape[0], t2.bbbb.shape[0]))
    rdm2.abab.ooov = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[0], t2.aaaa.shape[0], t2.bbbb.shape[-1]))
    rdm2.abab.oovo = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[0], t2.aaaa.shape[-1], t2.bbbb.shape[0]))
    rdm2.abab.ovoo = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[-1], t2.aaaa.shape[0], t2.bbbb.shape[0]))
    rdm2.abab.vooo = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[0], t2.aaaa.shape[0], t2.bbbb.shape[0]))
    rdm2.abab.ovov = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[-1], t2.aaaa.shape[0], t2.bbbb.shape[-1]))
    rdm2.abab.ovvo = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[-1], t2.aaaa.shape[-1], t2.bbbb.shape[0]))
    rdm2.abab.voov = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[0], t2.aaaa.shape[0], t2.bbbb.shape[-1]))
    rdm2.abab.vovo = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[0], t2.aaaa.shape[-1], t2.bbbb.shape[0]))
    rdm2.abab.ovvv = np.zeros((t2.aaaa.shape[0], t2.bbbb.shape[-1], t2.aaaa.shape[-1], t2.bbbb.shape[-1]))
    rdm2.abab.vovv = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[0], t2.aaaa.shape[-1], t2.bbbb.shape[-1]))
    rdm2.abab.vvov = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[-1], t2.aaaa.shape[0], t2.bbbb.shape[-1]))
    rdm2.abab.vvvo = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[-1], t2.aaaa.shape[-1], t2.bbbb.shape[0]))
    rdm2.abab.vvvv = np.zeros((t2.aaaa.shape[-1], t2.bbbb.shape[-1], t2.aaaa.shape[-1], t2.bbbb.shape[-1]))
    rdm2.bbbb.oooo = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[0]))
    rdm2.bbbb.ooov = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[-1]))
    rdm2.bbbb.oovo = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[0]))
    rdm2.bbbb.ovoo = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[0]))
    rdm2.bbbb.vooo = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[0]))
    rdm2.bbbb.ovov = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[-1]))
    rdm2.bbbb.ovvo = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[0]))
    rdm2.bbbb.voov = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[0], t2.bbbb.shape[-1]))
    rdm2.bbbb.vovo = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[0]))
    rdm2.bbbb.ovvv = np.zeros((t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[-1]))
    rdm2.bbbb.vovv = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[-1], t2.bbbb.shape[-1]))
    rdm2.bbbb.vvov = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[0], t2.bbbb.shape[-1]))
    rdm2.bbbb.vvvo = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[0]))
    rdm2.bbbb.vvvv = np.zeros((t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[-1], t2.bbbb.shape[-1]))
    rdm2.aaaa = pack_2e(rdm2.aaaa.oooo, rdm2.aaaa.ooov, rdm2.aaaa.oovo, rdm2.aaaa.ovoo, rdm2.aaaa.vooo, rdm2.aaaa.oovv, rdm2.aaaa.ovov, rdm2.aaaa.ovvo, rdm2.aaaa.voov, rdm2.aaaa.vovo, rdm2.aaaa.vvoo, rdm2.aaaa.ovvv, rdm2.aaaa.vovv, rdm2.aaaa.vvov, rdm2.aaaa.vvvo, rdm2.aaaa.vvvv)
    rdm2.abab = pack_2e(rdm2.abab.oooo, rdm2.abab.ooov, rdm2.abab.oovo, rdm2.abab.ovoo, rdm2.abab.vooo, rdm2.abab.oovv, rdm2.abab.ovov, rdm2.abab.ovvo, rdm2.abab.voov, rdm2.abab.vovo, rdm2.abab.vvoo, rdm2.abab.ovvv, rdm2.abab.vovv, rdm2.abab.vvov, rdm2.abab.vvvo, rdm2.abab.vvvv)
    rdm2.bbbb = pack_2e(rdm2.bbbb.oooo, rdm2.bbbb.ooov, rdm2.bbbb.oovo, rdm2.bbbb.ovoo, rdm2.bbbb.vooo, rdm2.bbbb.oovv, rdm2.bbbb.ovov, rdm2.bbbb.ovvo, rdm2.bbbb.voov, rdm2.bbbb.vovo, rdm2.bbbb.vvoo, rdm2.bbbb.ovvv, rdm2.bbbb.vovv, rdm2.bbbb.vvov, rdm2.bbbb.vvvo, rdm2.bbbb.vvvv)
    rdm2 = Namespace(
        aaaa=np.transpose(rdm2.aaaa, (0, 2, 1, 3)),
        aabb=np.transpose(rdm2.abab, (0, 2, 1, 3)),
        bbbb=np.transpose(rdm2.bbbb, (0, 2, 1, 3)),
    )
    rdm1 = make_rdm1_f(t2=t2, l2=l2)
    nocc = Namespace(a=t2.aaaa.shape[0], b=t2.bbbb.shape[0])
    rdm1.aa[np.diag_indices(nocc.a)] -= 1
    rdm1.bb[np.diag_indices(nocc.b)] -= 1
    for i in range(nocc.a):
        rdm2.aaaa[i, i, :, :] += rdm1.aa.T
        rdm2.aaaa[:, :, i, i] += rdm1.aa.T
        rdm2.aaaa[:, i, i, :] -= rdm1.aa.T
        rdm2.aaaa[i, :, :, i] -= rdm1.aa
        rdm2.aabb[i, i, :, :] += rdm1.bb.T
    for i in range(nocc.b):
        rdm2.bbbb[i, i, :, :] += rdm1.bb.T
        rdm2.bbbb[:, :, i, i] += rdm1.bb.T
        rdm2.bbbb[:, i, i, :] -= rdm1.bb.T
        rdm2.bbbb[i, :, :, i] -= rdm1.bb
        rdm2.aabb[:, :, i, i] += rdm1.aa.T
    for i in range(nocc.a):
        for j in range(nocc.a):
            rdm2.aaaa[i, i, j, j] += 1
            rdm2.aaaa[i, j, j, i] -= 1
    for i in range(nocc.b):
        for j in range(nocc.b):
            rdm2.bbbb[i, i, j, j] += 1
            rdm2.bbbb[i, j, j, i] -= 1
    for i in range(nocc.a):
        for j in range(nocc.b):
            rdm2.aabb[i, i, j, j] += 1

    return rdm2

def hbar_matvec_ip_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:29.058429.

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

    tmp12 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 4, 1, 3), (4, 2))
    tmp10 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4))
    tmp4 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 1, 3), (4, 0))
    tmp2 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (4, 2, 3, 1), (4, 0)) * -1

    return {f"tmp10": tmp10, f"tmp12": tmp12, f"tmp2": tmp2, f"tmp4": tmp4}

def hbar_matvec_ip(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:29.069055.

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
    tmp2 = np.copy(ints.tmp10) * 2
    del ints.tmp10
    tmp2 += ints.tmp12
    del ints.tmp12
    tmp0 = np.copy(ints.tmp2) * 2
    del ints.tmp2
    tmp0 += ints.tmp4
    del ints.tmp4
    tmp7 = einsum(r2.bbb, (0, 1, 2), f.bb.oo, (3, 1), (3, 0, 2))
    tmp6 = einsum(r1.b, (0,), v.bbbb.ooov, (1, 0, 2, 3), (1, 2, 3))
    tmp5 = einsum(r2.aaa, (0, 1, 2), f.aa.oo, (3, 1), (3, 0, 2))
    tmp4 = einsum(r1.a, (0,), v.aaaa.ooov, (1, 0, 2, 3), (1, 2, 3))
    tmp3 = np.copy(f.bb.oo) * 2
    tmp3 += tmp2
    tmp3 += np.transpose(tmp2, (1, 0))
    del tmp2
    tmp1 = np.copy(f.aa.oo) * 2
    tmp1 += tmp0
    tmp1 += np.transpose(tmp0, (1, 0))
    del tmp0
    r2new.bbb = einsum(f.bb.vv, (0, 1), r2.bbb, (2, 3, 1), (2, 3, 0)) * 2
    r2new.bbb += tmp6 * -1
    r2new.bbb += np.transpose(tmp6, (1, 0, 2))
    del tmp6
    r2new.bbb += tmp7 * 2
    r2new.bbb += np.transpose(tmp7, (1, 0, 2)) * -2
    del tmp7
    r2new.bab = einsum(v.aabb.ooov, (0, 1, 2, 3), r1.a, (1,), (2, 0, 3))
    r2new.bab += einsum(f.aa.oo, (0, 1), r2.bab, (2, 1, 3), (2, 0, 3)) * -1
    r2new.bab += einsum(r2.bab, (0, 1, 2), f.bb.vv, (3, 2), (0, 1, 3))
    r2new.bab += einsum(f.bb.oo, (0, 1), r2.bab, (1, 2, 3), (0, 2, 3)) * -1
    r2new.aba = einsum(f.aa.oo, (0, 1), r2.aba, (1, 2, 3), (0, 2, 3)) * -1
    r2new.aba += einsum(r2.aba, (0, 1, 2), f.aa.vv, (3, 2), (0, 1, 3))
    r2new.aba += einsum(r1.b, (0,), v.aabb.ovoo, (1, 2, 3, 0), (1, 3, 2))
    r2new.aba += einsum(f.bb.oo, (0, 1), r2.aba, (2, 1, 3), (2, 0, 3)) * -1
    r2new.aaa = einsum(r2.aaa, (0, 1, 2), f.aa.vv, (3, 2), (0, 1, 3)) * 2
    r2new.aaa += tmp4 * -1
    r2new.aaa += np.transpose(tmp4, (1, 0, 2))
    del tmp4
    r2new.aaa += tmp5 * 2
    r2new.aaa += np.transpose(tmp5, (1, 0, 2)) * -2
    del tmp5
    r1new.b = einsum(v.aabb.ovoo, (0, 1, 2, 3), r2.aba, (0, 3, 1), (2,))
    r1new.b += einsum(v.bbbb.ooov, (0, 1, 2, 3), r2.bbb, (1, 2, 3), (0,)) * -2
    r1new.b += einsum(r1.b, (0,), tmp3, (0, 1), (1,)) * -0.5
    del tmp3
    r1new.a = einsum(v.aabb.ooov, (0, 1, 2, 3), r2.bab, (2, 1, 3), (0,))
    r1new.a += einsum(v.aaaa.ooov, (0, 1, 2, 3), r2.aaa, (2, 1, 3), (0,)) * 2
    r1new.a += einsum(tmp1, (0, 1), r1.a, (0,), (1,)) * -0.5
    del tmp1

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ea_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:34.279262.

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

    tmp12 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 1, 4), (4, 3))
    tmp10 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (0, 2, 4, 1), (4, 3)) * -1
    tmp4 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 4, 3), (4, 1))
    tmp2 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 4, 3), (4, 1))

    return {f"tmp10": tmp10, f"tmp12": tmp12, f"tmp2": tmp2, f"tmp4": tmp4}

def hbar_matvec_ea(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:34.289700.

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
    tmp7 = einsum(r2.bbb, (0, 1, 2), f.bb.vv, (3, 1), (2, 3, 0))
    tmp8 = einsum(r1.b, (0,), v.bbbb.ovvv, (1, 2, 3, 0), (1, 2, 3))
    tmp5 = einsum(v.aaaa.ovvv, (0, 1, 2, 3), r1.a, (3,), (0, 1, 2))
    tmp4 = einsum(r2.aaa, (0, 1, 2), f.aa.vv, (3, 1), (2, 3, 0))
    tmp2 = np.copy(ints.tmp10)
    del ints.tmp10
    tmp2 += ints.tmp12 * 0.5
    del ints.tmp12
    tmp0 = np.copy(ints.tmp2)
    del ints.tmp2
    tmp0 += ints.tmp4 * 0.5
    del ints.tmp4
    tmp9 = np.copy(tmp7) * 2
    del tmp7
    tmp9 += tmp8
    del tmp8
    tmp6 = np.copy(tmp4) * 2
    del tmp4
    tmp6 += tmp5
    del tmp5
    tmp3 = np.copy(f.bb.vv) * -1
    tmp3 += tmp2
    tmp3 += np.transpose(tmp2, (1, 0))
    del tmp2
    tmp1 = np.copy(f.aa.vv) * -1
    tmp1 += tmp0
    tmp1 += np.transpose(tmp0, (1, 0))
    del tmp0
    r2new.bbb = einsum(r2.bbb, (0, 1, 2), f.bb.oo, (3, 2), (0, 1, 3)) * -2
    r2new.bbb += np.transpose(tmp9, (1, 2, 0)) * -1
    r2new.bbb += np.transpose(tmp9, (2, 1, 0))
    del tmp9
    r2new.bab = einsum(r2.bab, (0, 1, 2), f.bb.oo, (3, 2), (0, 1, 3)) * -1
    r2new.bab += einsum(r2.bab, (0, 1, 2), f.aa.vv, (3, 1), (0, 3, 2))
    r2new.bab += einsum(r1.a, (0,), v.aabb.vvov, (1, 0, 2, 3), (3, 1, 2)) * -1
    r2new.bab += einsum(r2.bab, (0, 1, 2), f.bb.vv, (3, 0), (3, 1, 2))
    r2new.aba = einsum(f.aa.oo, (0, 1), r2.aba, (2, 3, 1), (2, 3, 0)) * -1
    r2new.aba += einsum(r2.aba, (0, 1, 2), f.aa.vv, (3, 0), (3, 1, 2))
    r2new.aba += einsum(r2.aba, (0, 1, 2), f.bb.vv, (3, 1), (0, 3, 2))
    r2new.aba += einsum(r1.b, (0,), v.aabb.ovvv, (1, 2, 3, 0), (2, 3, 1)) * -1
    r2new.aaa = einsum(f.aa.oo, (0, 1), r2.aaa, (2, 3, 1), (2, 3, 0)) * -2
    r2new.aaa += np.transpose(tmp6, (1, 2, 0)) * -1
    r2new.aaa += np.transpose(tmp6, (2, 1, 0))
    del tmp6
    r1new.b = einsum(v.bbbb.ovvv, (0, 1, 2, 3), r2.bbb, (3, 1, 0), (2,)) * 2
    r1new.b += einsum(r2.aba, (0, 1, 2), v.aabb.ovvv, (2, 0, 3, 1), (3,)) * -1
    r1new.b += einsum(tmp3, (0, 1), r1.b, (0,), (1,)) * -1
    del tmp3
    r1new.a = einsum(r2.aaa, (0, 1, 2), v.aaaa.ovvv, (2, 1, 3, 0), (3,)) * 2
    r1new.a += einsum(v.aabb.vvov, (0, 1, 2, 3), r2.bab, (3, 1, 2), (0,)) * -1
    r1new.a += einsum(r1.a, (0,), tmp1, (0, 1), (1,)) * -1
    del tmp1

    return {f"r1new": r1new, f"r2new": r2new}


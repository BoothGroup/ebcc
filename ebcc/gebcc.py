"""General electron-boson coupled cluster.
"""

import functools
import itertools
import numpy as np
from typing import Tuple
from types import SimpleNamespace
from pyscf import lib, ao2mo
from ebcc.rebcc import REBCC


class GEBCC(REBCC):
    @staticmethod
    def _convert_mf(mf):
        return mf.to_ghf()

    def init_amps(self, eris=None):
        """Initialise amplitudes.
        """

        if eris is None:
            eris = self.get_eris()

        amplitudes = dict()
        e_ia = lib.direct_sum("i-a->ia", self.eo, self.ev)

        # Build T amplitudes:
        for n in self.rank_numeric[0]:
            if n == 1:
                amplitudes["t%d" % n] = self.fock.vo.T / e_ia
            elif n == 2:
                e_ijab = lib.direct_sum("ia,jb->ijab", e_ia, e_ia)
                amplitudes["t%d" % n] = eris.oovv / e_ijab
            else:
                amplitudes["t%d" % n] = np.zeros((self.nocc,) * n + (self.nvir,) * n)

        if not (self.rank[1] == self.rank[2] == ""):
            # Only true for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for n in self.rank_numeric[1]:
            if n == 1:
                amplitudes["s%d" % n] = -H / self.omega
            else:
                amplitudes["s%d" % n] = np.zeros((self.nbos,) * n)

        # Build U amplitudes:
        for n in self.rank_numeric[2]:
            if n == 1:
                e_xia = lib.direct_sum("ia-x->xia", e_ia, self.omega)
                amplitudes["u1%d" % n] = h.bov / e_xia
            else:
                amplitudes["u1%d" % n] = np.zeros((self.nbos,) * n + (self.nocc, self.nvir))

        return amplitudes

    def get_mean_field_G(self):
        val = lib.einsum("Ipp->I", self.g.boo)
        val -= self.xi * self.omega

        if self.bare_G is not None:
            val += self.bare_G

        return val

    def get_eris(self):
        o = self.mf.mo_occ > 0
        v = self.mf.mo_occ == 0
        a = slice(None, self.mf.mol.nao)
        b = slice(self.mf.mol.nao, None)
        slices = {"o": o, "v": v, "a": slice(None)}

        mo_a = self.mf.mo_coeff[a]
        mo_b = self.mf.mo_coeff[b]

        eri  = ao2mo.kernel(self.mf._eri, mo_a)
        eri += ao2mo.kernel(self.mf._eri, mo_b)
        eri1 = ao2mo.kernel(self.mf._eri, (mo_a, mo_a, mo_b, mo_b))
        eri += eri1
        eri += eri1.T

        eri = ao2mo.addons.restore(1, eri, self.nmo)
        eri = eri.reshape((self.nmo,) * 4)
        eri = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)

        class two_e_blocks:
            def __getattr__(blocks, key):
                i, j, k, l = (slices[k] for k in key)
                return eri[:, :, :, l][:, :, k][:, j][i]

        return two_e_blocks()

    @property
    def xi(self):
        if self.shift:
            xi = lib.einsum("Iii->I", self.g.boo)
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros_like(self.omega)
        return xi

    @property
    def name(self):
        return "GCC" + "-".join(self.rank)



if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()

    ccsd = GEBCC(mf)
    ccsd.kernel()

"""Unrestricted electron-boson coupled cluster.
"""

import functools
import itertools
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, lib

from ebcc import rebcc, util


class Amplitudes(rebcc.Amplitudes):
    """Amplitude container class. Consists of a dictionary with keys
    that are strings of the name of each amplitude. Values are
    namespaces with keys indicating whether each fermionic dimension
    is alpha (`"a"`) or beta (`"b"`) spin, and values are arrays whose
    dimension depends on the particular amplitude. For purely bosonic
    amplitudes the values of `Amplitudes` are simply arrays, with no
    fermionic spins to index.
    """

    pass


class ERIs(SimpleNamespace):
    """Electronic repulsion integral container class. Consists of a
    namespace with keys that are length-4 string of `"a"` or `"b"`
    signifying whether the corresponding dimension is alpha or beta
    spin, and values are of type `rebcc.ERIs`.
    """

    def __init__(self, ebcc):
        self.mf = ebcc.mf
        o = [slice(None, n) for n in ebcc.nocc]
        v = [slice(n, None) for n in ebcc.nocc]
        slices = [{"o": o1, "v": v1} for o1, v1 in zip(o, v)]
        mo_coeff = self.mf.mo_coeff

        self.aaaa = rebcc.ERIs(
            ebcc,
            slices=[slices[i] for i in (0, 0, 0, 0)],
            mo_coeff=[mo_coeff[i] for i in (0, 0, 0, 0)],
        )
        self.aabb = rebcc.ERIs(
            ebcc,
            slices=[slices[i] for i in (0, 0, 1, 1)],
            mo_coeff=[mo_coeff[i] for i in (0, 0, 1, 1)],
        )
        self.bbaa = rebcc.ERIs(
            ebcc,
            slices=[slices[i] for i in (1, 1, 0, 0)],
            mo_coeff=[mo_coeff[i] for i in (1, 1, 0, 0)],
        )
        self.bbbb = rebcc.ERIs(
            ebcc,
            slices=[slices[i] for i in (1, 1, 1, 1)],
            mo_coeff=[mo_coeff[i] for i in (1, 1, 1, 1)],
        )


@util.inherit_docstrings
class UEBCC(rebcc.REBCC):
    Amplitudes = Amplitudes
    ERIs = ERIs

    @staticmethod
    def _convert_mf(mf):
        return mf.to_uhf()

    @classmethod
    def from_rebcc(cls, rcc):
        """Initialise an UEBCC object from an REBCC object."""

        ucc = cls(
            rcc.mf,
            log=rcc.log,
            fermion_excitations=rcc.fermion_excitations,
            boson_excitations=rcc.boson_excitations,
            fermion_coupling_rank=rcc.fermion_coupling_rank,
            boson_coupling_rank=rcc.boson_coupling_rank,
            omega=rcc.omega,
            g=rcc.bare_g,
            G=rcc.bare_G,
            options=rcc.options,
        )

        ucc.e_corr = rcc.e_corr
        ucc.converged = rcc.converged
        ucc.converged_lambda = rcc.converged_lambda

        has_amps = rcc.amplitudes is not None
        has_lams = rcc.lambdas is not None

        if has_amps:
            amplitudes = cls.Amplitudes()

            for n in rcc.rank_numeric[0]:
                amplitudes["t%d" % n] = SimpleNamespace()
                for comb in util.generate_spin_combinations(n):
                    subscript = comb[:n] + comb[n:].upper()
                    tn = rcc.amplitudes["t%d" % n]
                    tn = util.symmetrise(subscript, tn, symmetry="-"*2*n)
                    setattr(amplitudes["t%d" % n], comb, tn)

            for n in rcc.rank_numeric[1]:
                amplitudes["s%d" % n] = rcc.amplitudes["s%d" % n].copy()

            for nf in rcc.rank_numeric[2]:
                for nb in rcc.rank_numeric[3]:
                    amplitudes["u%d%d" % (nf, nb)] = SimpleNamespace()
                    for comb in util.generate_spin_combinations(nf):
                        tn = rcc.amplitudes["u%d%d" % (nf, nb)]
                        setattr(amplitudes["u%d%d" % (nf, nb)], comb, tn)

            ucc.amplitudes = amplitudes

        if has_lams:
            lambdas = cls.Amplitudes()

            for n in rcc.rank_numeric[0]:
                lambdas["l%d" % n] = SimpleNamespace()
                for comb in util.generate_spin_combinations(n):
                    tn = rcc.lambdas["l%d" % n]
                    tn = util.symmetrise(subscript, tn, symmetry="-"*2*n)
                    setattr(lambdas["l%d" % n], comb, tn)

            for n in rcc.rank_numeric[1]:
                lambdas["ls%d" % n] = rcc.lambdas["ls%d" % n].copy()

            for nf in rcc.rank_numeric[2]:
                for nb in rcc.rank_numeric[3]:
                    lambdas["lu%d%d" % (nf, nb)] = SimpleNamespace()
                    for comb in util.generate_spin_combinations(n):
                        tn = rcc.lambdas["lu%d%d" % (nf, nb)]
                        setattr(lambdas["lu%d%d" % (nf, nb)], comb, tn)

            ucc.lambdas = lambdas

        return ucc

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.get_eris()

        amplitudes = self.Amplitudes()
        e_ia = SimpleNamespace(
            aa=lib.direct_sum("i-a->ia", self.eo.a, self.ev.a),
            bb=lib.direct_sum("i-a->ia", self.eo.b, self.ev.b),
        )

        # Build T amplitudes
        for n in self.rank_numeric[0]:
            if n == 1:
                tn = SimpleNamespace(
                    aa=self.fock.aa.vo.T / e_ia.aa,
                    bb=self.fock.bb.vo.T / e_ia.bb,
                )
                amplitudes["t%d" % n] = tn
            elif n == 2:
                e_ijab = SimpleNamespace(
                    aaaa=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.aa),
                    abab=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.bb),
                    baba=lib.direct_sum("ia,jb->ijab", e_ia.bb, e_ia.aa),
                    bbbb=lib.direct_sum("ia,jb->ijab", e_ia.bb, e_ia.bb),
                )
                tn = SimpleNamespace(
                    aaaa=eris.aaaa.ovov.swapaxes(1, 2) / e_ijab.aaaa,
                    abab=eris.aabb.ovov.swapaxes(1, 2) / e_ijab.abab,
                    baba=eris.bbaa.ovov.swapaxes(1, 2) / e_ijab.baba,
                    bbbb=eris.bbbb.ovov.swapaxes(1, 2) / e_ijab.bbbb,
                )
                # TODO generalise:
                tn.aaaa = tn.aaaa - tn.aaaa.swapaxes(0, 1)
                tn.bbbb = tn.bbbb - tn.bbbb.swapaxes(0, 1)
                amplitudes["t%d" % n] = tn
            else:
                raise NotImplementedError  # TODO

        if not (self.rank[1] == self.rank[2] == ""):
            # Only tue for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for n in self.rank_numeric[1]:
            if n == 1:
                amplitudes["s%d" % n] = -H / self.omega
            else:
                amplitudes["s%d" % n] = np.zeros((self.nbos,) * n)

        # Build U amplitudes:
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                if nb == 1:
                    e_xia = SimpleNamespace(
                        aa=lib.direct_sum("ia-x->xia", e_ia.aa, self.omega),
                        bb=lib.direct_sum("ia-x->xia", e_ia.bb, self.omega),
                    )
                    u1n = SimpleNamespace(
                        aa=h.aa.bov / e_xia.aa,
                        bb=h.bb.bov / e_xia.bb,
                    )
                    amplitudes["u%d%d" % (nf, nb)] = u1n
                else:
                    u1n = SimpleNamespace(
                        aa=np.zeros((self.nbos,) * nb + (self.nocc[0], self.nvir[0])),
                        bb=np.zeros((self.nbos,) * nb + (self.nocc[1], self.nvir[1])),
                    )
                    amplitudes["u%d%d" % (nf, nb)] = u1n

        return amplitudes

    def init_lams(self, amplitudes=None):
        if amplitudes is None:
            amplitudes = self.amplitudes

        lambdas = self.Amplitudes()

        # Build L amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(n, 2 * n)) + list(range(n))
            lambdas["l%d" % n] = SimpleNamespace()
            for key in amplitudes["t%d" % n].__dict__.keys():
                ln = getattr(amplitudes["t%d" % n], key).transpose(perm)
                setattr(lambdas["l%d" % n], key, ln)

        # Build LS amplitudes:
        for n in self.rank_numeric[1]:
            lambdas["ls%d" % n] = amplitudes["s%d" % n]

        # Build LU amplitudes:
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                perm = list(range(nb)) + [nb + 1, nb]
                lambdas["lu%d%d" % (nf, nb)] = SimpleNamespace()
                for key in amplitudes["u%d%d" % (nf, nb)].__dict__.keys():
                    lu1n = getattr(amplitudes["u%d%d" % (nf, nb)], key).transpose(perm)
                    setattr(lambdas["lu%d%d" % (nf, nb)], key, lu1n)

        return lambdas

    def update_amps(self, eris=None, amplitudes=None):
        func, kwargs = self._load_function(
            "update_amps",
            eris=eris,
            amplitudes=amplitudes,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        e_ia = SimpleNamespace(
            aa=lib.direct_sum("i-a->ia", self.eo.a, self.ev.a),
            bb=lib.direct_sum("i-a->ia", self.eo.b, self.ev.b),
        )

        # Divide T amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
            for comb in util.generate_spin_combinations(n):
                subscript = comb[:n] + comb[n:].upper()
                es = [getattr(e_ia, comb[i] + comb[i + n]) for i in range(n)]
                d = functools.reduce(np.add.outer, es)
                d = d.transpose(perm)
                tn = getattr(res["t%d" % n], comb)
                tn /= d
                tn += getattr(amplitudes["t%d" % n], comb)
                tn = util.symmetrise(subscript, tn, symmetry="-"*(2*n))
                setattr(res["t%d" % n], comb, tn)

        # Divide S amplitudes:
        for n in self.rank_numeric[1]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n))
            res["s%d" % n] /= d
            res["s%d" % n] += amplitudes["s%d" % n]

        # Divide U amplitudes:
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ia.aa] * nf))
                tn = res["u%d%d" % (nf, nb)].aa
                tn /= d
                tn += amplitudes["u%d%d" % (nf, nb)].aa
                d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ia.bb] * nf))
                res["u%d%d" % (nf, nb)].aa = tn
                tn = res["u%d%d" % (nf, nb)].bb
                tn /= d
                tn += amplitudes["u%d%d" % (nf, nb)].bb
                res["u%d%d" % (nf, nb)].bb = tn

        return res

    def update_lams(self, eris=None, amplitudes=None, lambdas=None):
        func, kwargs = self._load_function(
            "update_lams",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        e_ai = SimpleNamespace(
            aa=lib.direct_sum("i-a->ai", self.eo.a, self.ev.a),
            bb=lib.direct_sum("i-a->ai", self.eo.b, self.ev.b),
        )

        # Divide T amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
            for comb in util.generate_spin_combinations(n):
                subscript = comb[:n] + comb[n:].upper()
                es = [getattr(e_ai, comb[i] + comb[i + n]) for i in range(n)]
                d = functools.reduce(np.add.outer, es)
                d = d.transpose(perm)
                tn = getattr(res["l%d" % n], comb)
                tn /= d
                tn += getattr(lambdas["l%d" % n], comb)
                tn = util.symmetrise(subscript, tn, symmetry="-"*(2*n))
                setattr(res["l%d" % n], comb, tn)

        # Divide S amplitudes:
        for n in self.rank_numeric[1]:
            d = functools.reduce(np.add.outer, [-self.omega] * n)
            res["ls%d" % n] /= d
            res["ls%d" % n] += lambdas["ls%d" % n]

        # Divide U amplitudes:
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ai.aa] * nf))
                tn = res["lu%d%d" % (nf, nb)].aa
                tn /= d
                tn += lambdas["lu%d%d" % (nf, nb)].aa
                d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ai.bb] * nf))
                res["lu%d%d" % (nf, nb)].aa = tn
                tn = res["lu%d%d" % (nf, nb)].bb
                tn /= d
                tn += lambdas["lu%d%d" % (nf, nb)].bb
                res["lu%d%d" % (nf, nb)].bb = tn

        return res

    def make_rdm1_f(self, eris=None, amplitudes=None, lambdas=None, hermitise=True):
        func, kwargs = self._load_function(
            "make_rdm1_f",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:
            dm.aa = 0.5 * (dm.aa + dm.aa.T)
            dm.bb = 0.5 * (dm.bb + dm.bb.T)

        return dm

    def make_rdm2_f(self, eris=None, amplitudes=None, lambdas=None, hermitise=True):
        func, kwargs = self._load_function(
            "make_rdm2_f",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:

            def transpose1(dm):
                dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2, 3, 0, 1))
                return dm

            def transpose2(dm):
                dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(1, 0, 3, 2))
                return dm

            dm.aaaa = transpose2(transpose1(dm.aaaa))
            dm.aabb = transpose2(dm.aabb)
            dm.bbaa = transpose2(dm.bbaa)
            dm.bbbb = transpose2(transpose1(dm.bbbb))

        return dm

    def make_eb_coup_rdm(
        self, eris=None, amplitudes=None, lambdas=None, unshifted=True, hermitise=True
    ):
        func, kwargs = self._load_function(
            "make_eb_coup_rdm",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm_eb = func(**kwargs)

        if hermitise:
            dm_eb.aa[0] = 0.5 * (dm_eb.aa[0] + dm_eb.aa[1].transpose(0, 2, 1))
            dm_eb.bb[0] = 0.5 * (dm_eb.bb[0] + dm_eb.bb[1].transpose(0, 2, 1))
            dm_eb.aa[1] = dm_eb.aa[0].transpose(0, 2, 1).copy()
            dm_eb.bb[1] = dm_eb.bb[0].transpose(0, 2, 1).copy()

        if unshifted and self.options.shift:
            rdm1_f = self.make_rdm1_f(hermitise=hermitise)
            shift = lib.einsum("x,ij->xij", self.xi, rdm1_f.aa)
            dm_eb.aa -= shift[None]
            shift = lib.einsum("x,ij->xij", self.xi, rdm1_f.bb)
            dm_eb.bb -= shift[None]

        return dm_eb

    def get_mean_field_G(self):
        val = lib.einsum("Ipp->I", self.g.aa.boo)
        val += lib.einsum("Ipp->I", self.g.bb.boo)
        val -= self.xi * self.omega

        if self.bare_G is not None:
            # Require bare_G to have a spin index for now:
            assert np.shape(self.bare_G) == val.shape
            val += self.bare_G

        return val

    def get_g(self, g):
        if np.array(g).ndim != 4:
            g = np.array([g, g])

        gs = SimpleNamespace()

        boo = g[0][:, : self.nocc[0], : self.nocc[0]]
        bov = g[0][:, : self.nocc[0], self.nocc[0] :]
        bvo = g[0][:, self.nocc[0] :, : self.nocc[0]]
        bvv = g[0][:, self.nocc[0] :, self.nocc[0] :]
        gs.aa = SimpleNamespace(boo=boo, bov=bov, bvo=bvo, bvv=bvv)

        boo = g[1][:, : self.nocc[1], : self.nocc[1]]
        bov = g[1][:, : self.nocc[1], self.nocc[1] :]
        bvo = g[1][:, self.nocc[1] :, : self.nocc[1]]
        bvv = g[1][:, self.nocc[1] :, self.nocc[1] :]
        gs.bb = SimpleNamespace(boo=boo, bov=bov, bvo=bvo, bvv=bvv)

        return gs

    @property
    def bare_fock(self):
        fock = lib.einsum(
            "npq,npi,nqj->nij", self.mf.get_fock(), self.mf.mo_coeff, self.mf.mo_coeff
        )
        fock = SimpleNamespace(aa=fock[0], bb=fock[1])
        return fock

    @property
    def xi(self):
        if self.options.shift:
            xi = lib.einsum("Iii->I", self.g.aa.boo)
            xi += lib.einsum("Iii->I", self.g.bb.boo)
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros_like(self.omega)

        return xi

    def get_fock(self):
        fock = self.bare_fock
        if self.options.shift:
            xi = self.xi

        f = SimpleNamespace()

        oo = fock.aa[: self.nocc[0], : self.nocc[0]]
        ov = fock.aa[: self.nocc[0], self.nocc[0] :]
        vo = fock.aa[self.nocc[0] :, : self.nocc[0]]
        vv = fock.aa[self.nocc[0] :, self.nocc[0] :]

        if self.options.shift:
            g = self.g
            oo -= lib.einsum("I,Iij->ij", xi, g.aa.boo + g.aa.boo.transpose(0, 2, 1))
            ov -= lib.einsum("I,Iia->ia", xi, g.aa.bov + g.aa.bvo.transpose(0, 2, 1))
            vo -= lib.einsum("I,Iai->ai", xi, g.aa.bvo + g.aa.bov.transpose(0, 2, 1))
            vv -= lib.einsum("I,Iab->ab", xi, g.aa.bvv + g.aa.bvv.transpose(0, 2, 1))

        f.aa = SimpleNamespace(oo=oo, ov=ov, vo=vo, vv=vv)

        oo = fock.bb[: self.nocc[1], : self.nocc[1]]
        ov = fock.bb[: self.nocc[1], self.nocc[1] :]
        vo = fock.bb[self.nocc[1] :, : self.nocc[1]]
        vv = fock.bb[self.nocc[1] :, self.nocc[1] :]

        if self.options.shift:
            g = self.g
            oo -= lib.einsum("I,Iij->ij", xi, g.bb.boo + g.bb.boo.transpose(0, 2, 1))
            ov -= lib.einsum("I,Iia->ia", xi, g.bb.bov + g.bb.bvo.transpose(0, 2, 1))
            vo -= lib.einsum("I,Iai->ai", xi, g.bb.bvo + g.bb.bov.transpose(0, 2, 1))
            vv -= lib.einsum("I,Iab->ab", xi, g.bb.bvv + g.bb.bvv.transpose(0, 2, 1))

        f.bb = SimpleNamespace(oo=oo, ov=ov, vo=vo, vv=vv)

        return f

    def get_eris(self):
        return self.ERIs(self)

    def amplitudes_to_vector(self, amplitudes):
        vectors = []

        for n in self.rank_numeric[0]:
            for spin in util.generate_spin_combinations(n):
                tn = getattr(amplitudes["t%d" % n], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for n in self.rank_numeric[1]:
            vectors.append(amplitudes["s%d" % n].ravel())

        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                vectors.append(amplitudes["u%d%d" % (nf, nb)].aa.ravel())
                vectors.append(amplitudes["u%d%d" % (nf, nb)].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        amplitudes = self.Amplitudes()
        i0 = 0

        for n in self.rank_numeric[0]:
            amplitudes["t%d" % n] = SimpleNamespace()
            for spin in util.generate_spin_combinations(n):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(subscript, a=self.nocc[0], b=self.nocc[1], A=self.nvir[0], B=self.nvir[1])
                shape = tuple([
                    *[self.nocc["ab".index(s)] for s in spin[:n]],
                    *[self.nvir["ab".index(s)] for s in spin[n:]],
                ])
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                setattr(amplitudes["t%d" % n], spin, tn)
                i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes["s%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                amplitudes["u%d%d" % (nf, nb)] = SimpleNamespace()
                shape = (self.nbos,) * nb + (self.nocc[0], self.nvir[0]) * nf
                size = np.prod(shape)
                amplitudes["u%d%d" % (nf, nb)].aa = vector[i0 : i0 + size].reshape(shape)
                i0 += size
                shape = (self.nbos,) * nb + (self.nocc[1], self.nvir[1]) * nf
                size = np.prod(shape)
                amplitudes["u%d%d" % (nf, nb)].bb = vector[i0 : i0 + size].reshape(shape)
                i0 += size

        assert i0 == len(vector)

        return amplitudes

    def lambdas_to_vector(self, lambdas):
        vectors = []

        for n in self.rank_numeric[0]:
            for spin in util.generate_spin_combinations(n):
                tn = getattr(lambdas["l%d" % n], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for n in self.rank_numeric[1]:
            vectors.append(lambdas["ls%d" % n].ravel())

        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                vectors.append(lambdas["lu%d%d" % (nf, nb)].aa.ravel())
                vectors.append(lambdas["lu%d%d" % (nf, nb)].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        lambdas = self.Amplitudes()
        i0 = 0
        spin_indices = {"a": 0, "b": 1}

        for n in self.rank_numeric[0]:
            lambdas["l%d" % n] = SimpleNamespace()
            for spin in util.generate_spin_combinations(n):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(subscript, a=self.nvir[0], b=self.nvir[1], A=self.nocc[0], B=self.nocc[1])
                shape = tuple([
                    *[self.nvir["ab".index(s)] for s in spin[:n]],
                    *[self.nocc["ab".index(s)] for s in spin[n:]],
                ])
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                setattr(lambdas["l%d" % n], spin, tn)
                i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["ls%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                lambdas["lu%d%d" % (nf, nb)] = SimpleNamespace()
                shape = (self.nbos,) * nb + (self.nvir[0], self.nocc[0]) * nf
                size = np.prod(shape)
                lambdas["lu%d%d" % (nf, nb)].aa = vector[i0 : i0 + size].reshape(shape)
                i0 += size
                shape = (self.nbos,) * nb + (self.nvir[1], self.nocc[1]) * nf
                size = np.prod(shape)
                lambdas["lu%d%d" % (nf, nb)].bb = vector[i0 : i0 + size].reshape(shape)
                i0 += size

        assert i0 == len(vector)

        return lambdas

    def excitations_to_vector_ip(self, *excitations):
        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            for spin in util.generate_spin_combinations(n, excited=True):
                vn = getattr(excitations[m], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return np.concatenate(vectors)

    def excitations_to_vector_ee(self, *excitations):
        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            for spin in util.generate_spin_combinations(n):
                vn = getattr(excitations[m], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return np.concatenate(vectors)

    def vector_to_excitations_ip(self, vector):
        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            amp = SimpleNamespace()
            for spin in util.generate_spin_combinations(n, excited=True):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(subscript, a=self.nocc[0], b=self.nocc[1], A=self.nvir[0], B=self.nvir[1])
                shape = tuple([
                    *[self.nocc["ab".index(s)] for s in spin[:n]],
                    *[self.nvir["ab".index(s)] for s in spin[n:]],
                ])
                vn_tril = vector[i0 : i0 + size]
                vn = util.decompress_axes(subscript, vn_tril, shape=shape)
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        assert i0 == len(vector)

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            amp = SimpleNamespace()
            for spin in util.generate_spin_combinations(n, excited=True):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(subscript, a=self.nvir[0], b=self.nvir[1], A=self.nocc[0], B=self.nocc[1])
                shape = tuple([
                    *[self.nvir["ab".index(s)] for s in spin[:n]],
                    *[self.nocc["ab".index(s)] for s in spin[n:]],
                ])
                vn_tril = vector[i0 : i0 + size]
                factor = max(spin[:n].count(s) for s in set(spin[:n]))  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        assert i0 == len(vector)

        return tuple(excitations)

    def vector_to_excitations_ee(self, vector):
        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            amp = SimpleNamespace()
            for spin in util.generate_spin_combinations(n):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(subscript, a=self.nocc[0], b=self.nocc[1], A=self.nvir[0], B=self.nvir[1])
                shape = tuple([
                    *[self.nocc["ab".index(s)] for s in spin[:n]],
                    *[self.nvir["ab".index(s)] for s in spin[n:]],
                ])
                vn_tril = vector[i0 : i0 + size]
                vn = util.decompress_axes(subscript, vn_tril, shape=shape)
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        assert i0 == len(vector)

        return tuple(excitations)

    @property
    def name(self):
        return super().name.replace("R", "U", 1)

    @property
    def nmo(self):
        assert self.mf.mo_occ[0].size == self.mf.mo_occ[1].size
        return self.mf.mo_occ[0].size

    @property
    def nocc(self):
        return tuple(np.sum(mo_occ > 0) for mo_occ in self.mf.mo_occ)

    @property
    def nvir(self):
        return tuple(self.nmo - nocc for nocc in self.nocc)

    @property
    def eo(self):
        eo = SimpleNamespace(
            a=np.diag(self.fock.aa.oo),
            b=np.diag(self.fock.bb.oo),
        )
        return eo

    @property
    def ev(self):
        ev = SimpleNamespace(
            a=np.diag(self.fock.aa.vv),
            b=np.diag(self.fock.bb.vv),
        )
        return ev


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = "H 0 0 0; F 0 0 1.1"
    mol.basis = "6-31g"
    mol.verbose = 5
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    nbos = 5
    np.random.seed(1)
    g = np.random.random((nbos, mol.nao, mol.nao)) * 0.03
    g = g + g.transpose(0, 2, 1)
    omega = np.random.random((nbos)) * 0.5

    cc = UEBCC(mf, rank=("SD", "SD", "S"), omega=omega, g=g, shift=False)
    cc.kernel()
    cc.solve_lambda()

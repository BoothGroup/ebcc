"""Unrestricted electron-boson coupled cluster.
"""

import functools
import itertools
import types
from typing import Sequence

import numpy as np
from pyscf import ao2mo, lib

from ebcc import rebcc, ueom, util
from ebcc.brueckner import BruecknerUEBCC
from ebcc.eris import UERIs
from ebcc.fock import UFock
from ebcc.space import Space


class Amplitudes(util.Namespace):
    """Amplitude container class. Consists of a dictionary with keys
    that are strings of the name of each amplitude. Values are
    namespaces with keys indicating whether each fermionic dimension
    is alpha (`"a"`) or beta (`"b"`) spin, and values are arrays whose
    dimension depends on the particular amplitude. For purely bosonic
    amplitudes the values of `Amplitudes` are simply arrays, with no
    fermionic spins to index.
    """

    pass


@util.inherit_docstrings
class UEBCC(rebcc.REBCC):
    Amplitudes = Amplitudes
    ERIs = UERIs
    Brueckner = BruecknerUEBCC

    @staticmethod
    def _convert_mf(mf):
        return mf.to_uhf()

    @classmethod
    def from_rebcc(cls, rcc):
        """Initialise an UEBCC object from an REBCC object."""

        ucc = cls(
            rcc.mf,
            log=rcc.log,
            ansatz=rcc.ansatz,
            space=(rcc.space, rcc.space),
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

            for n in rcc.ansatz.correlated_cluster_ranks[0]:
                amplitudes["t%d" % n] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript = comb[:n] + comb[n:].upper()
                    tn = rcc.amplitudes["t%d" % n]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    setattr(amplitudes["t%d" % n], comb, tn)

            for n in rcc.ansatz.correlated_cluster_ranks[1]:
                amplitudes["s%d" % n] = rcc.amplitudes["s%d" % n].copy()

            for nf in rcc.ansatz.correlated_cluster_ranks[2]:
                for nb in rcc.ansatz.correlated_cluster_ranks[3]:
                    amplitudes["u%d%d" % (nf, nb)] = util.Namespace()
                    for comb in util.generate_spin_combinations(nf, unique=True):
                        tn = rcc.amplitudes["u%d%d" % (nf, nb)]
                        setattr(amplitudes["u%d%d" % (nf, nb)], comb, tn)

            ucc.amplitudes = amplitudes

        if has_lams:
            lambdas = cls.Amplitudes()

            for n in rcc.ansatz.correlated_cluster_ranks[0]:
                lambdas["l%d" % n] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript = comb[:n] + comb[n:].upper()
                    tn = rcc.lambdas["l%d" % n]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    setattr(lambdas["l%d" % n], comb, tn)

            for n in rcc.ansatz.correlated_cluster_ranks[1]:
                lambdas["ls%d" % n] = rcc.lambdas["ls%d" % n].copy()

            for nf in rcc.ansatz.correlated_cluster_ranks[2]:
                for nb in rcc.ansatz.correlated_cluster_ranks[3]:
                    lambdas["lu%d%d" % (nf, nb)] = util.Namespace()
                    for comb in util.generate_spin_combinations(nf, unique=True):
                        tn = rcc.lambdas["lu%d%d" % (nf, nb)]
                        setattr(lambdas["lu%d%d" % (nf, nb)], comb, tn)

            ucc.lambdas = lambdas

        return ucc

    def _pack_codegen_kwargs(self, *extra_kwargs, eris=None):
        eris = self.get_eris(eris)

        omega = np.diag(self.omega) if self.omega is not None else None

        kwargs = dict(
            f=self.fock,
            v=eris,
            g=self.g,
            G=self.G,
            w=omega,
            nocc=(self.space[0].ncocc, self.space[1].ncocc),  # FIXME rename?
            nvir=(self.space[0].ncvir, self.space[1].ncvir),  # FIXME rename?
            nbos=self.nbos,
        )
        for kw in extra_kwargs:
            if kw is not None:
                kwargs.update(kw)

        return kwargs

    def init_space(self):
        space = (
            Space(
                self.mo_occ[0] > 0,
                np.zeros_like(self.mo_occ[0], dtype=bool),
                np.zeros_like(self.mo_occ[0], dtype=bool),
            ),
            Space(
                self.mo_occ[1] > 0,
                np.zeros_like(self.mo_occ[1], dtype=bool),
                np.zeros_like(self.mo_occ[1], dtype=bool),
            ),
        )

        return space

    def init_amps(self, eris=None):
        eris = self.get_eris(eris)

        amplitudes = self.Amplitudes()
        e_ia = util.Namespace(
            aa=lib.direct_sum("i-a->ia", self.eo.a, self.ev.a),
            bb=lib.direct_sum("i-a->ia", self.eo.b, self.ev.b),
        )

        # Build T amplitudes
        for n in self.ansatz.correlated_cluster_ranks[0]:
            if n == 1:
                tn = util.Namespace(
                    aa=self.fock.aa.vo.T / e_ia.aa,
                    bb=self.fock.bb.vo.T / e_ia.bb,
                )
                amplitudes["t%d" % n] = tn
            elif n == 2:
                e_ijab = util.Namespace(
                    aaaa=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.aa),
                    abab=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.bb),
                    bbbb=lib.direct_sum("ia,jb->ijab", e_ia.bb, e_ia.bb),
                )
                tn = util.Namespace(
                    aaaa=eris.aaaa.ovov.swapaxes(1, 2) / e_ijab.aaaa,
                    abab=eris.aabb.ovov.swapaxes(1, 2) / e_ijab.abab,
                    bbbb=eris.bbbb.ovov.swapaxes(1, 2) / e_ijab.bbbb,
                )
                # TODO generalise:
                tn.aaaa = 0.5 * (tn.aaaa - tn.aaaa.swapaxes(0, 1))
                tn.bbbb = 0.5 * (tn.bbbb - tn.bbbb.swapaxes(0, 1))
                amplitudes["t%d" % n] = tn
            else:
                tn = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    shape = tuple(self.space["ab".index(s)].ncocc for s in comb[:n])
                    shape += tuple(self.space["ab".index(s)].ncvir for s in comb[n:])
                    amp = np.zeros(shape)
                    setattr(tn, comb, amp)
                amplitudes["t%d" % n] = tn

        if self.boson_ansatz:
            # Only tue for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for n in self.ansatz.correlated_cluster_ranks[1]:
            if n == 1:
                amplitudes["s%d" % n] = -H / self.omega
            else:
                amplitudes["s%d" % n] = np.zeros((self.nbos,) * n)

        # Build U amplitudes:
        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                if nb == 1:
                    e_xia = util.Namespace(
                        aa=lib.direct_sum("ia-x->xia", e_ia.aa, self.omega),
                        bb=lib.direct_sum("ia-x->xia", e_ia.bb, self.omega),
                    )
                    u1n = util.Namespace(
                        aa=h.aa.bov / e_xia.aa,
                        bb=h.bb.bov / e_xia.bb,
                    )
                    amplitudes["u%d%d" % (nf, nb)] = u1n
                else:
                    u1n = util.Namespace(
                        aa=np.zeros((self.nbos,) * nb + (self.space[0].ncocc, self.space[0].ncvir)),
                        bb=np.zeros((self.nbos,) * nb + (self.space[1].ncocc, self.space[1].ncvir)),
                    )
                    amplitudes["u%d%d" % (nf, nb)] = u1n

        return amplitudes

    def init_lams(self, amplitudes=None):
        if amplitudes is None:
            amplitudes = self.amplitudes

        lambdas = self.Amplitudes()

        # Build L amplitudes:
        for n in self.ansatz.correlated_cluster_ranks[0]:
            perm = list(range(n, 2 * n)) + list(range(n))
            lambdas["l%d" % n] = util.Namespace()
            for key in dict(amplitudes["t%d" % n]).keys():
                ln = getattr(amplitudes["t%d" % n], key).transpose(perm)
                setattr(lambdas["l%d" % n], key, ln)

        # Build LS amplitudes:
        for n in self.ansatz.correlated_cluster_ranks[1]:
            lambdas["ls%d" % n] = amplitudes["s%d" % n]

        # Build LU amplitudes:
        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                perm = list(range(nb)) + [nb + 1, nb]
                lambdas["lu%d%d" % (nf, nb)] = util.Namespace()
                for key in dict(amplitudes["u%d%d" % (nf, nb)]).keys():
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

        e_ia = util.Namespace(
            aa=lib.direct_sum("i-a->ia", self.eo.a, self.ev.a),
            bb=lib.direct_sum("i-a->ia", self.eo.b, self.ev.b),
        )

        # Divide T amplitudes:
        for n in self.ansatz.correlated_cluster_ranks[0]:
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript = comb[:n] + comb[n:].upper()
                es = [getattr(e_ia, comb[i] + comb[i + n]) for i in range(n)]
                d = functools.reduce(np.add.outer, es)
                d = d.transpose(perm)
                tn = getattr(res["t%d" % n], comb)
                tn /= d
                tn += getattr(amplitudes["t%d" % n], comb)
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                setattr(res["t%d" % n], comb, tn)

        # Divide S amplitudes:
        for n in self.ansatz.correlated_cluster_ranks[1]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n))
            res["s%d" % n] /= d
            res["s%d" % n] += amplitudes["s%d" % n]

        # Divide U amplitudes:
        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
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

    def update_lams(self, eris=None, amplitudes=None, lambdas=None, lambdas_pert=None):
        func, kwargs = self._load_function(
            "update_lams",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
            lambdas_pert=lambdas_pert,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        e_ai = util.Namespace(
            aa=lib.direct_sum("i-a->ai", self.eo.a, self.ev.a),
            bb=lib.direct_sum("i-a->ai", self.eo.b, self.ev.b),
        )

        # Divide T amplitudes:
        for n in self.ansatz.correlated_cluster_ranks[0]:
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript = comb[:n] + comb[n:].upper()
                es = [getattr(e_ai, comb[i] + comb[i + n]) for i in range(n)]
                d = functools.reduce(np.add.outer, es)
                d = d.transpose(perm)
                tn = getattr(res["l%d" % n], comb)
                tn /= d
                tn += getattr(lambdas["l%d" % n], comb)
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                setattr(res["l%d" % n], comb, tn)

        # Divide S amplitudes:
        for n in self.ansatz.correlated_cluster_ranks[1]:
            d = functools.reduce(np.add.outer, [-self.omega] * n)
            res["ls%d" % n] /= d
            res["ls%d" % n] += lambdas["ls%d" % n]

        # Divide U amplitudes:
        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
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

        slices = [
            {
                "x": space.correlated,
                "o": space.correlated_occupied,
                "v": space.correlated_virtual,
                "O": space.active_occupied,
                "V": space.active_virtual,
                "i": space.inactive_occupied,
                "a": space.inactive_virtual,
            }
            for space in self.space
        ]

        def constructor(s):
            class Blocks:
                def __getattr__(selffer, key):
                    assert key[0] == "b"
                    i = slices[s][key[1]]
                    j = slices[s][key[2]]
                    return g[s][:, i, j].copy()

            return Blocks()

        gs = util.Namespace()
        gs.aa = constructor(0)
        gs.bb = constructor(1)

        return gs

    @property
    def bare_fock(self):
        fock = lib.einsum("npq,npi,nqj->nij", self.mf.get_fock(), self.mo_coeff, self.mo_coeff)
        fock = util.Namespace(aa=fock[0], bb=fock[1])
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
        return UFock(self, array=(self.bare_fock.aa, self.bare_fock.bb))

    def get_eris(self, eris=None):
        """Get blocks of the ERIs.

        Parameters
        ----------
        eris : tuple of np.ndarray or ERIs, optional.
            Electronic repulsion integrals, either in the form of a
            dense array for each spin channel or an ERIs object.
            Default value is `None`.

        Returns
        -------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.ERIs()`.
        """
        if (eris is None) or isinstance(eris, tuple):
            return self.ERIs(self, array=eris)
        else:
            return eris

    def ip_eom(self, options=None, **kwargs):
        return ueom.IP_UEOM(self, options=options, **kwargs)

    def ea_eom(self, options=None, **kwargs):
        return ueom.EA_UEOM(self, options=options, **kwargs)

    def ee_eom(self, options=None, **kwargs):
        return ueom.EE_UEOM(self, options=options, **kwargs)

    def amplitudes_to_vector(self, amplitudes):
        vectors = []

        for n in self.ansatz.correlated_cluster_ranks[0]:
            for spin in util.generate_spin_combinations(n, unique=True):
                tn = getattr(amplitudes["t%d" % n], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for n in self.ansatz.correlated_cluster_ranks[1]:
            vectors.append(amplitudes["s%d" % n].ravel())

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                vectors.append(amplitudes["u%d%d" % (nf, nb)].aa.ravel())
                vectors.append(amplitudes["u%d%d" % (nf, nb)].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        amplitudes = self.Amplitudes()
        i0 = 0

        for n in self.ansatz.correlated_cluster_ranks[0]:
            amplitudes["t%d" % n] = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncocc,
                    b=self.space[1].ncocc,
                    A=self.space[0].ncvir,
                    B=self.space[1].ncvir,
                )
                shape = tuple(
                    [
                        *[self.space["ab".index(s)].ncocc for s in spin[:n]],
                        *[self.space["ab".index(s)].ncvir for s in spin[n:]],
                    ]
                )
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                setattr(amplitudes["t%d" % n], spin, tn)
                i0 += size

        for n in self.ansatz.correlated_cluster_ranks[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes["s%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                amplitudes["u%d%d" % (nf, nb)] = util.Namespace()
                shape = (self.nbos,) * nb + (self.space[0].ncocc, self.space[0].ncvir) * nf
                size = np.prod(shape)
                amplitudes["u%d%d" % (nf, nb)].aa = vector[i0 : i0 + size].reshape(shape)
                i0 += size
                shape = (self.nbos,) * nb + (self.space[1].ncocc, self.space[1].ncvir) * nf
                size = np.prod(shape)
                amplitudes["u%d%d" % (nf, nb)].bb = vector[i0 : i0 + size].reshape(shape)
                i0 += size

        assert i0 == len(vector)

        return amplitudes

    def lambdas_to_vector(self, lambdas):
        vectors = []

        for n in self.ansatz.correlated_cluster_ranks[0]:
            for spin in util.generate_spin_combinations(n, unique=True):
                tn = getattr(lambdas["l%d" % n], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for n in self.ansatz.correlated_cluster_ranks[1]:
            vectors.append(lambdas["ls%d" % n].ravel())

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                vectors.append(lambdas["lu%d%d" % (nf, nb)].aa.ravel())
                vectors.append(lambdas["lu%d%d" % (nf, nb)].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        lambdas = self.Amplitudes()
        i0 = 0
        spin_indices = {"a": 0, "b": 1}

        for n in self.ansatz.correlated_cluster_ranks[0]:
            lambdas["l%d" % n] = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncvir,
                    b=self.space[1].ncvir,
                    A=self.space[0].ncocc,
                    B=self.space[1].ncocc,
                )
                shape = tuple(
                    [
                        *[self.space["ab".index(s)].ncvir for s in spin[:n]],
                        *[self.space["ab".index(s)].ncocc for s in spin[n:]],
                    ]
                )
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                setattr(lambdas["l%d" % n], spin, tn)
                i0 += size

        for n in self.ansatz.correlated_cluster_ranks[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["ls%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            if nf != 1:
                raise util.ModelNotImplemented
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                lambdas["lu%d%d" % (nf, nb)] = util.Namespace()
                shape = (self.nbos,) * nb + (self.space[0].ncvir, self.space[0].ncocc) * nf
                size = np.prod(shape)
                lambdas["lu%d%d" % (nf, nb)].aa = vector[i0 : i0 + size].reshape(shape)
                i0 += size
                shape = (self.nbos,) * nb + (self.space[1].ncvir, self.space[1].ncocc) * nf
                size = np.prod(shape)
                lambdas["lu%d%d" % (nf, nb)].bb = vector[i0 : i0 + size].reshape(shape)
                i0 += size

        assert i0 == len(vector)

        return lambdas

    def excitations_to_vector_ip(self, *excitations):
        vectors = []
        m = 0

        for n in self.ansatz.correlated_cluster_ranks[0]:
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                vn = getattr(excitations[m], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def excitations_to_vector_ee(self, *excitations):
        vectors = []
        m = 0

        for n in self.ansatz.correlated_cluster_ranks[0]:
            for spin in util.generate_spin_combinations(n):
                vn = getattr(excitations[m], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def vector_to_excitations_ip(self, vector):
        excitations = []
        i0 = 0

        for n in self.ansatz.correlated_cluster_ranks[0]:
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncocc,
                    b=self.space[1].ncocc,
                    A=self.space[0].ncvir,
                    B=self.space[1].ncvir,
                )
                shape = tuple(
                    [
                        *[self.space["ab".index(s)].ncocc for s in spin[:n]],
                        *[self.space["ab".index(s)].ncvir for s in spin[n:]],
                    ]
                )
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        excitations = []
        i0 = 0

        for n in self.ansatz.correlated_cluster_ranks[0]:
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncvir,
                    b=self.space[1].ncvir,
                    A=self.space[0].ncocc,
                    B=self.space[1].ncocc,
                )
                shape = tuple(
                    [
                        *[self.space["ab".index(s)].ncvir for s in spin[:n]],
                        *[self.space["ab".index(s)].ncocc for s in spin[n:]],
                    ]
                )
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    def vector_to_excitations_ee(self, vector):
        excitations = []
        i0 = 0

        for n in self.ansatz.correlated_cluster_ranks[0]:
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n):
                subscript = spin[:n] + spin[n:].upper()
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncocc,
                    b=self.space[1].ncocc,
                    A=self.space[0].ncvir,
                    B=self.space[1].ncvir,
                )
                shape = tuple(
                    [
                        *[self.space["ab".index(s)].ncocc for s in spin[:n]],
                        *[self.space["ab".index(s)].ncvir for s in spin[n:]],
                    ]
                )
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        for nf in self.ansatz.correlated_cluster_ranks[2]:
            for nb in self.ansatz.correlated_cluster_ranks[3]:
                raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    @property
    def spin_type(self):
        return "U"

    @property
    def nmo(self):
        assert self.mo_occ[0].size == self.mo_occ[1].size
        return self.mo_occ[0].size

    @property
    def nocc(self):
        return tuple(np.sum(mo_occ > 0) for mo_occ in self.mo_occ)

    @property
    def nvir(self):
        return tuple(self.nmo - nocc for nocc in self.nocc)

    @property
    def eo(self):
        eo = util.Namespace(
            a=np.diag(self.fock.aa.oo),
            b=np.diag(self.fock.bb.oo),
        )
        return eo

    @property
    def ev(self):
        ev = util.Namespace(
            a=np.diag(self.fock.aa.vv),
            b=np.diag(self.fock.bb.vv),
        )
        return ev

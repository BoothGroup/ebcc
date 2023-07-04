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

        # FIXME won't work with active spaces
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

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript = comb[:n] + comb[n:].upper()
                    tn = rcc.amplitudes[name]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    setattr(amplitudes[name], comb, tn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = rcc.amplitudes[name].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = util.Namespace()
                for comb in util.generate_spin_combinations(nf, unique=True):
                    tn = rcc.amplitudes[name]
                    setattr(amplitudes[name], comb, tn)

            ucc.amplitudes = amplitudes

        if has_lams:
            lambdas = cls.Amplitudes()

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                lname = name.replace("t", "l")
                lambdas[lname] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript = comb[:n] + comb[n:].upper()
                    tn = rcc.lambdas[lname]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    setattr(lambdas[lname], comb, tn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = rcc.lambdas[lname].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = util.Namespace()
                for comb in util.generate_spin_combinations(nf, unique=True):
                    tn = rcc.lambdas[lname]
                    setattr(lambdas[lname], comb, tn)

            ucc.lambdas = lambdas

        return ucc

    def _pack_codegen_kwargs(self, *extra_kwargs, eris=None):
        eris = self.get_eris(eris)

        omega = np.diag(self.omega) if self.omega is not None else None

        debug_space = lambda: None
        debug_space.naocc = (self.space[0].naocc, self.space[1].naocc)
        debug_space.navir = (self.space[0].navir, self.space[1].navir)
        debug_space.niocc = (self.space[0].niocc, self.space[1].niocc)
        debug_space.nivir = (self.space[0].nivir, self.space[1].nivir)

        kwargs = dict(
            f=self.fock,
            v=eris,
            g=self.g,
            G=self.G,
            w=omega,
            space=debug_space,
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

        # Build T amplitudes
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                ei = getattr(self, "e" + key[0])
                ea = getattr(self, "e" + key[1])
                e_ia = util.Namespace(
                    aa=lib.direct_sum("i-a->ia", ei.a, ea.a),
                    bb=lib.direct_sum("i-a->ia", ei.b, ea.b),
                )
                tn = util.Namespace(
                    aa=getattr(self.fock.aa, key) / e_ia.aa,
                    bb=getattr(self.fock.bb, key) / e_ia.bb,
                )
                amplitudes[name] = tn
            elif n == 2:
                ei = getattr(self, "e" + key[0])
                ej = getattr(self, "e" + key[1])
                ea = getattr(self, "e" + key[2])
                eb = getattr(self, "e" + key[3])
                e_ia = util.Namespace(
                    aa=lib.direct_sum("i-a->ia", ei.a, ea.a),
                    bb=lib.direct_sum("i-a->ia", ei.b, ea.b),
                )
                e_ijab = util.Namespace(
                    aaaa=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.aa),
                    abab=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.bb),
                    bbbb=lib.direct_sum("ia,jb->ijab", e_ia.bb, e_ia.bb),
                )
                key_t = key[0] + key[2] + key[1] + key[3]
                tn = util.Namespace(
                    aaaa=getattr(eris.aaaa, key_t).swapaxes(1, 2) / e_ijab.aaaa,
                    abab=getattr(eris.aabb, key_t).swapaxes(1, 2) / e_ijab.abab,
                    bbbb=getattr(eris.bbbb, key_t).swapaxes(1, 2) / e_ijab.bbbb,
                )
                # TODO generalise:
                tn.aaaa = 0.5 * (tn.aaaa - tn.aaaa.swapaxes(0, 1))
                tn.bbbb = 0.5 * (tn.bbbb - tn.bbbb.swapaxes(0, 1))
                amplitudes[name] = tn
            else:
                tn = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(comb, key))
                    amp = np.zeros(shape)
                    setattr(tn, comb, amp)
                amplitudes[name] = tn

        if self.boson_ansatz:
            # Only tue for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                amplitudes[name] = -H / self.omega
            else:
                shape = (self.nbos,) * n
                amplitudes[name] = np.zeros(shape)

        # Build U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            if nb == 1:
                ei = getattr(self, "e" + key[1])
                ea = getattr(self, "e" + key[2])
                e_xia = util.Namespace(
                    aa=lib.direct_sum("i-a-x->xia", ei.a, ea.a, self.omega),
                    bb=lib.direct_sum("i-a-x->xia", ei.b, ea.b, self.omega),
                )
                u1n = util.Namespace(
                    aa=getattr(h.aa, key) / e_xia.aa,
                    bb=getattr(h.bb, key) / e_xia.bb,
                )
                amplitudes[name] = u1n
            else:
                u1n = util.Namespace(
                    aa=np.zeros((self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:])),
                    bb=np.zeros((self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:])),
                )
                amplitudes[name] = u1n

        return amplitudes

    def init_lams(self, amplitudes=None):
        if amplitudes is None:
            amplitudes = self.amplitudes

        lambdas = self.Amplitudes()

        # Build L amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            perm = list(range(n, 2 * n)) + list(range(n))
            lambdas[lname] = util.Namespace()
            for key in dict(amplitudes[name]).keys():
                ln = getattr(amplitudes[name], key).transpose(perm)
                setattr(lambdas[lname], key, ln)

        # Build LS amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            lambdas["l" + name] = amplitudes[name]

        # Build LU amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            perm = list(range(nb)) + [nb + 1, nb]
            lambdas["l" + name] = util.Namespace()
            for key in dict(amplitudes[name]).keys():
                lu1n = getattr(amplitudes[name], key).transpose(perm)
                setattr(lambdas["l"+name], key, lu1n)

        return lambdas

    def update_amps(self, eris=None, amplitudes=None):
        func, kwargs = self._load_function(
            "update_amps",
            eris=eris,
            amplitudes=amplitudes,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript = comb[:n] + comb[n:].upper()
                e_ia_list = [
                    lib.direct_sum(
                        "i-a->ia",
                        getattr(getattr(self, "e" + o), so),
                        getattr(getattr(self, "e" + v), sv),
                    )
                    for (o, v), (so, sv) in zip(zip(key[:n], key[n:]), zip(comb[:n], comb[n:]))
                ]
                d = functools.reduce(np.add.outer, e_ia_list)
                d = d.transpose(perm)
                tn = getattr(res[name], comb)
                tn /= d
                tn += getattr(amplitudes[name], comb)
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                setattr(res[name], comb, tn)

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            d = functools.reduce(np.add.outer, ([-self.omega] * n))
            res[name] /= d
            res[name] += amplitudes[name]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            e_ia = util.Namespace(
                aa=lib.direct_sum(
                    "i-a->ia",
                    getattr(self, "e" + key[nb]).a,
                    getattr(self, "e" + key[nb+1]).a,
                ),
                bb=lib.direct_sum(
                    "i-a->ia",
                    getattr(self, "e" + key[nb]).b,
                    getattr(self, "e" + key[nb+1]).b,
                ),
            )
            d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ia.aa] * nf))
            tn = res[name].aa
            tn /= d
            tn += amplitudes[name].aa
            d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ia.bb] * nf))
            res[name].aa = tn
            tn = res[name].bb
            tn /= d
            tn += amplitudes[name].bb
            res[name].bb = tn

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

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript = comb[:n] + comb[n:].upper()
                e_ai_list = [
                    lib.direct_sum(
                        "i-a->ai",
                        getattr(getattr(self, "e" + o), so),
                        getattr(getattr(self, "e" + v), sv),
                    )
                    for (o, v), (so, sv) in zip(zip(key[:n], key[n:]), zip(comb[:n], comb[n:]))
                ]
                d = functools.reduce(np.add.outer, e_ai_list)
                d = d.transpose(perm)
                tn = getattr(res[lname], comb)
                tn /= d
                tn += getattr(lambdas[lname], comb)
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                setattr(res[lname], comb, tn)

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            lname = "l" + name
            d = functools.reduce(np.add.outer, [-self.omega] * n)
            res[lname] /= d
            res[lname] += lambdas[lname]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            lname = "l" + name
            e_ai = util.Namespace(
                aa=lib.direct_sum(
                    "i-a->ai",
                    getattr(self, "e" + key[nb]).a,
                    getattr(self, "e" + key[nb+1]).a,
                ),
                bb=lib.direct_sum(
                    "i-a->ai",
                    getattr(self, "e" + key[nb]).b,
                    getattr(self, "e" + key[nb+1]).b,
                ),
            )
            d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ai.aa] * nf))
            tn = res[lname].aa
            tn /= d
            tn += lambdas[lname].aa
            d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ai.bb] * nf))
            res[lname].aa = tn
            tn = res[lname].bb
            tn /= d
            tn += lambdas[lname].bb
            res[lname].bb = tn

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
                    return g[s][:, i][:, :, j].copy()

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for spin in util.generate_spin_combinations(n, unique=True):
                tn = getattr(amplitudes[name], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(amplitudes[name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            vectors.append(amplitudes[name].aa.ravel())
            vectors.append(amplitudes[name].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        amplitudes = self.Amplitudes()
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            amplitudes[name] = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                # FIXME this will break for active space methods
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncocc,
                    b=self.space[1].ncocc,
                    A=self.space[0].ncvir,
                    B=self.space[1].ncvir,
                )
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                setattr(amplitudes[name], spin, tn)
                i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes[name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            amplitudes[name] = util.Namespace()
            shape = (self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:])
            size = np.prod(shape)
            amplitudes[name].aa = vector[i0 : i0 + size].reshape(shape)
            i0 += size
            shape = (self.nbos,) * nb + tuple(self.space[1].size(k) for k in key[nb:])
            size = np.prod(shape)
            amplitudes[name].bb = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        assert i0 == len(vector)

        return amplitudes

    def lambdas_to_vector(self, lambdas):
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            for spin in util.generate_spin_combinations(n, unique=True):
                tn = getattr(lambdas[lname], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(lambdas["l" + name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            vectors.append(lambdas["l" + name].aa.ravel())
            vectors.append(lambdas["l" + name].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        lambdas = self.Amplitudes()
        i0 = 0
        spin_indices = {"a": 0, "b": 1}

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            key = key[n:] + key[:n]
            lambdas[lname] = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                # FIXME this will break for active space methods
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncvir,
                    b=self.space[1].ncvir,
                    A=self.space[0].ncocc,
                    B=self.space[1].ncocc,
                )
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                setattr(lambdas[lname], spin, tn)
                i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["l" + name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            lname = "l" + name
            key = key[:nb] + key[nb+nf:] + key[nb:nb+nf]
            lambdas[lname] = util.Namespace()
            shape = (self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:])
            size = np.prod(shape)
            lambdas[lname].aa = vector[i0 : i0 + size].reshape(shape)
            i0 += size
            shape = (self.nbos,) * nb + tuple(self.space[1].size(k) for k in key[nb:])
            size = np.prod(shape)
            lambdas[lname].bb = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        assert i0 == len(vector)

        return lambdas

    def excitations_to_vector_ip(self, *excitations):
        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                vn = getattr(excitations[m], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def excitations_to_vector_ee(self, *excitations):
        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for spin in util.generate_spin_combinations(n):
                vn = getattr(excitations[m], spin)
                subscript = spin[:n] + spin[n:].upper()
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def vector_to_excitations_ip(self, vector):
        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                # FIXME this will break for active space methods
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncocc,
                    b=self.space[1].ncocc,
                    A=self.space[0].ncvir,
                    B=self.space[1].ncvir,
                )
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[:n-1]
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript = spin[:n] + spin[n:].upper()
                # FIXME this will break for active space methods
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncvir,
                    b=self.space[1].ncvir,
                    A=self.space[0].ncocc,
                    B=self.space[1].ncocc,
                )
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    def vector_to_excitations_ee(self, vector):
        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n):
                subscript = spin[:n] + spin[n:].upper()
                # FIXME this will break for active space methods
                size = util.get_compressed_size(
                    subscript,
                    a=self.space[0].ncocc,
                    b=self.space[1].ncocc,
                    A=self.space[0].ncvir,
                    B=self.space[1].ncvir,
                )
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                setattr(amp, spin, vn)
                i0 += size

            excitations.append(amp)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
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

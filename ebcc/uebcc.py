"""Unrestricted electron-boson coupled cluster."""

from pyscf import lib

from ebcc import numpy as np
from ebcc import rebcc, ueom, util
from ebcc.brueckner import BruecknerUEBCC
from ebcc.cderis import UCDERIs
from ebcc.eris import UERIs
from ebcc.fock import UFock
from ebcc.precision import types
from ebcc.space import Space


@util.has_docstring
class UEBCC(rebcc.REBCC, metaclass=util.InheritDocstrings):
    ERIs = UERIs
    Fock = UFock
    CDERIs = UCDERIs
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
            amplitudes = util.Namespace()

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript = util.combine_subscripts(key, comb)
                    tn = rcc.amplitudes[name]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    amplitudes[name][comb] = tn

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = rcc.amplitudes[name].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = util.Namespace()
                for comb in util.generate_spin_combinations(nf, unique=True):
                    tn = rcc.amplitudes[name]
                    amplitudes[name][comb] = tn

            ucc.amplitudes = amplitudes

        if has_lams:
            lambdas = util.Namespace()

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                lname = name.replace("t", "l")
                lambdas[lname] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript = util.combine_subscripts(key, comb)
                    tn = rcc.lambdas[lname]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    lambdas[lname][comb] = tn

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = rcc.lambdas[lname].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = util.Namespace()
                for comb in util.generate_spin_combinations(nf, unique=True):
                    tn = rcc.lambdas[lname]
                    lambdas[lname][comb] = tn

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
            space=self.space,
            nocc=(self.space[0].ncocc, self.space[1].ncocc),  # FIXME rename?
            nvir=(self.space[0].ncvir, self.space[1].ncvir),  # FIXME rename?
            nbos=self.nbos,
        )
        if isinstance(eris, self.CDERIs):
            kwargs["naux"] = self.mf.with_df.get_naoaux()
        for kw in extra_kwargs:
            if kw is not None:
                kwargs.update(kw)

        return kwargs

    @util.has_docstring
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

    @util.has_docstring
    def init_amps(self, eris=None):
        eris = self.get_eris(eris)
        amplitudes = util.Namespace()

        # Build T amplitudes
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            tn = util.Namespace()
            for comb in util.generate_spin_combinations(n, unique=True):
                if n == 1:
                    tn[comb] = self.fock[comb][key] / self.energy_sum(key, comb)
                elif n == 2:
                    comb_t = comb[0] + comb[2] + comb[1] + comb[3]
                    key_t = key[0] + key[2] + key[1] + key[3]
                    tn[comb] = eris[comb_t][key_t].swapaxes(1, 2) / self.energy_sum(key, comb)
                    if comb in ("aaaa", "bbbb"):
                        # TODO generalise:
                        tn[comb] = 0.5 * (tn[comb] - tn[comb].swapaxes(0, 1))
                else:
                    shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(comb, key))
                    tn[comb] = np.zeros(shape, dtype=types[float])
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
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        # Build U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            if nb == 1:
                tn = util.Namespace(
                    aa=h.aa[key] / self.energy_sum(key, "_aa"),
                    bb=h.bb[key] / self.energy_sum(key, "_aa"),
                )
                amplitudes[name] = tn
            else:
                tn = util.Namespace(
                    aa=np.zeros(
                        (self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:]),
                        dtype=types[float],
                    ),
                    bb=np.zeros(
                        (self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:]),
                        dtype=types[float],
                    ),
                )
                amplitudes[name] = tn

        return amplitudes

    @util.has_docstring
    def init_lams(self, amplitudes=None):
        if amplitudes is None:
            amplitudes = self.amplitudes

        lambdas = util.Namespace()

        # Build L amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            perm = list(range(n, 2 * n)) + list(range(n))
            lambdas[lname] = util.Namespace()
            for key in dict(amplitudes[name]).keys():
                ln = amplitudes[name][key].transpose(perm)
                lambdas[lname][key] = ln

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
                ln = amplitudes[name][key].transpose(perm)
                lambdas["l" + name][key] = ln

        return lambdas

    @util.has_docstring
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
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript = util.combine_subscripts(key, comb)
                tn = res[name][comb]
                tn /= self.energy_sum(key, comb)
                tn += amplitudes[name][comb]
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                res[name][comb] = tn

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            res[name] /= self.energy_sum(key, "_" * n)
            res[name] += amplitudes[name]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            tn = res[name].aa
            tn /= self.energy_sum(key, "_" * nb + "aa")
            tn += amplitudes[name].aa
            res[name].aa = tn
            tn = res[name].bb
            tn /= self.energy_sum(key, "_" * nb + "bb")
            tn += amplitudes[name].bb
            res[name].bb = tn

        return res

    @util.has_docstring
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
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript = util.combine_subscripts(key, comb)
                tn = res[lname][comb]
                tn /= self.energy_sum(key[n:] + key[:n], comb)
                tn += lambdas[lname][comb]
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                res[lname][comb] = tn

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            lname = "l" + name
            res[lname] /= self.energy_sum(key, "_" * n)
            res[lname] += lambdas[lname]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            lname = "l" + name
            key = key[:nb] + key[nb + nf :] + key[nb : nb + nf]
            tn = res[lname].aa
            tn /= self.energy_sum(key, "_" * nb + "aa")
            tn += lambdas[lname].aa
            res[lname].aa = tn
            tn = res[lname].bb
            tn /= self.energy_sum(key, "_" * nb + "bb")
            tn += lambdas[lname].bb
            res[lname].bb = tn

        return res

    @util.has_docstring
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

    @util.has_docstring
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

    @util.has_docstring
    def make_eb_coup_rdm(
        self,
        eris=None,
        amplitudes=None,
        lambdas=None,
        unshifted=True,
        hermitise=True,
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

    @util.has_docstring
    def get_mean_field_G(self):
        val = lib.einsum("Ipp->I", self.g.aa.boo)
        val += lib.einsum("Ipp->I", self.g.bb.boo)
        val -= self.xi * self.omega

        if self.bare_G is not None:
            # Require bare_G to have a spin index for now:
            assert np.shape(self.bare_G) == val.shape
            val += self.bare_G

        return val

    @util.has_docstring
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
            class Blocks(util.Namespace):
                def __getitem__(selffer, key):
                    assert key[0] == "b"
                    i = slices[s][key[1]]
                    j = slices[s][key[2]]
                    return g[s][:, i][:, :, j].copy()

                __getattr__ = __getitem__

            return Blocks()

        gs = util.Namespace()
        gs.aa = constructor(0)
        gs.bb = constructor(1)

        return gs

    @property
    @util.has_docstring
    def bare_fock(self):
        fock = lib.einsum(
            "npq,npi,nqj->nij",
            self.mf.get_fock().astype(types[float]),
            self.mo_coeff,
            self.mo_coeff,
        )
        fock = util.Namespace(aa=fock[0], bb=fock[1])
        return fock

    @property
    @util.has_docstring
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

    @util.has_docstring
    def get_fock(self):
        return self.Fock(self, array=(self.bare_fock.aa, self.bare_fock.bb))

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
            if (
                isinstance(eris, tuple) and isinstance(eris[0], np.ndarray) and eris[0].ndim == 3
            ) or getattr(self.mf, "with_df", None):
                return self.CDERIs(self, array=eris)
            else:
                return self.ERIs(self, array=eris)
        else:
            return eris

    @util.has_docstring
    def ip_eom(self, options=None, **kwargs):
        return ueom.IP_UEOM(self, options=options, **kwargs)

    @util.has_docstring
    def ea_eom(self, options=None, **kwargs):
        return ueom.EA_UEOM(self, options=options, **kwargs)

    @util.has_docstring
    def ee_eom(self, options=None, **kwargs):
        return ueom.EE_UEOM(self, options=options, **kwargs)

    @util.has_docstring
    def amplitudes_to_vector(self, amplitudes):
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for spin in util.generate_spin_combinations(n, unique=True):
                tn = amplitudes[name][spin]
                subscript = util.combine_subscripts(key, spin)
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(amplitudes[name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            vectors.append(amplitudes[name].aa.ravel())
            vectors.append(amplitudes[name].bb.ravel())

        return np.concatenate(vectors)

    @util.has_docstring
    def vector_to_amplitudes(self, vector):
        amplitudes = util.Namespace()
        i0 = 0
        sizes = {(o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")}

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            amplitudes[name] = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                amplitudes[name][spin] = tn
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

    @util.has_docstring
    def lambdas_to_vector(self, lambdas):
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            key = key[n:] + key[:n]
            for spin in util.generate_spin_combinations(n, unique=True):
                tn = lambdas[lname][spin]
                subscript = util.combine_subscripts(key, spin)
                vectors.append(util.compress_axes(subscript, tn).ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(lambdas["l" + name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            vectors.append(lambdas["l" + name].aa.ravel())
            vectors.append(lambdas["l" + name].bb.ravel())

        return np.concatenate(vectors)

    @util.has_docstring
    def vector_to_lambdas(self, vector):
        lambdas = util.Namespace()
        i0 = 0
        sizes = {(o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")}

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            key = key[n:] + key[:n]
            lambdas[lname] = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                lambdas[lname][spin] = tn
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
            key = key[:nb] + key[nb + nf :] + key[nb : nb + nf]
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

    @util.has_docstring
    def excitations_to_vector_ip(self, *excitations):
        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                vn = excitations[m][spin]
                subscript = util.combine_subscripts(key[:-1], spin)
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    @util.has_docstring
    def excitations_to_vector_ea(self, *excitations):
        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[:n]
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                vn = excitations[m][spin]
                subscript = util.combine_subscripts(key[:-1], spin)
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    @util.has_docstring
    def excitations_to_vector_ee(self, *excitations):
        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for spin in util.generate_spin_combinations(n):
                vn = excitations[m][spin]
                subscript = util.combine_subscripts(key, spin)
                vectors.append(util.compress_axes(subscript, vn).ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    @util.has_docstring
    def vector_to_excitations_ip(self, vector):
        excitations = []
        i0 = 0
        sizes = {(o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")}

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                amp[spin] = vn
                i0 += size

            excitations.append(amp)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    @util.has_docstring
    def vector_to_excitations_ea(self, vector):
        excitations = []
        i0 = 0
        sizes = {(o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")}

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                amp[spin] = vn
                i0 += size

            excitations.append(amp)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    @util.has_docstring
    def vector_to_excitations_ee(self, vector):
        excitations = []
        i0 = 0
        sizes = {(o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")}

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            amp = util.Namespace()
            for spin in util.generate_spin_combinations(n):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                amp[spin] = vn
                i0 += size

            excitations.append(amp)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return tuple(excitations)

    @property
    @util.has_docstring
    def spin_type(self):
        return "U"

    @property
    @util.has_docstring
    def nmo(self):
        assert self.mo_occ[0].size == self.mo_occ[1].size
        return self.mo_occ[0].size

    @property
    @util.has_docstring
    def nocc(self):
        return tuple(np.sum(mo_occ > 0) for mo_occ in self.mo_occ)

    @property
    @util.has_docstring
    def nvir(self):
        return tuple(self.nmo - nocc for nocc in self.nocc)

    def energy_sum(self, subscript, spins, signs_dict=None):
        """
        Get a direct sum of energies.

        Parameters
        ----------
        subscript : str
            The direct sum subscript, where each character indicates the
            sector for each energy. For the default slice characters, see
            `Space`. Occupied degrees of freedom are assumed to be
            positive, virtual and bosonic negative (the signs can be
            changed via the `signs_dict` keyword argument).
        spins : str
            String of spins, length must be the same as `subscript` with
            each character being one of `"a"` or `"b"`.
        signs_dict : dict, optional
            Dictionary defining custom signs for each sector. If `None`,
            initialised such that `["o", "O", "i"]` are positive, and
            `["v", "V", "a", "b"]` negative. Default value is `None`.

        Returns
        -------
        energy_sum : numpy.ndarray
            Array of energy sums.
        """

        n = 0

        def next_char():
            nonlocal n
            if n < 26:
                char = chr(ord("a") + n)
            else:
                char = chr(ord("A") + n)
            n += 1
            return char

        if signs_dict is None:
            signs_dict = {}
        for k, s in zip("vVaoOib", "---+++-"):
            if k not in signs_dict:
                signs_dict[k] = s

        energies = []
        for key, spin in zip(subscript, spins):
            if key == "b":
                energies.append(self.omega)
            else:
                energies.append(np.diag(self.fock[spin + spin][key + key]))

        subscript = "".join([signs_dict[k] + next_char() for k in subscript])
        energy_sum = lib.direct_sum(subscript, *energies)

        return energy_sum

"""Unrestricted electron-boson coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import lib

from ebcc import numpy as np
from ebcc import util
from ebcc.cc.base import BaseEBCC
from ebcc.cderis import UCDERIs
from ebcc.eom import EA_UEOM, EE_UEOM, IP_UEOM
from ebcc.eris import UERIs
from ebcc.fock import UFock
from ebcc.opt.ubrueckner import BruecknerUEBCC
from ebcc.precision import types
from ebcc.space import Space

if TYPE_CHECKING:
    from ebcc.cc.rebcc import REBCC
    from ebcc.numpy.typing import NDArray


class UEBCC(BaseEBCC):
    """Unrestricted electron-boson coupled cluster.

    Attributes:
        mf: PySCF mean-field object.
        log: Log to write output to.
        options: Options for the EBCC calculation.
        e_corr: Correlation energy.
        amplitudes: Cluster amplitudes.
        converged: Convergence flag.
        lambdas: Cluster lambda amplitudes.
        converged_lambda: Lambda convergence flag.
        name: Name of the method.
    """

    # Types
    ERIs = UERIs
    Fock = UFock
    CDERIs = UCDERIs
    Brueckner = BruecknerUEBCC

    @property
    def spin_type(self):
        """Get a string representation of the spin type."""
        return "U"

    def ip_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> IP_UEOM:
        """Get the IP-EOM object.

        Args:
            options: Options for the IP-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            IP-EOM object.
        """
        return IP_UEOM(self, options=options, **kwargs)

    def ea_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> EA_UEOM:
        """Get the EA-EOM object.

        Args:
            options: Options for the EA-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EA-EOM object.
        """
        return EA_UEOM(self, options=options, **kwargs)

    def ee_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> EE_UEOM:
        """Get the EE-EOM object.

        Args:
            options: Options for the EE-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EE-EOM object.
        """
        return EE_UEOM(self, options=options, **kwargs)

    @staticmethod
    def _convert_mf(mf: SCF) -> UHF:
        """Convert the mean-field object to the appropriate type."""
        return mf.to_uhf()

    @classmethod
    def from_rebcc(cls, rcc: REBCC) -> UEBCC:
        """Initialise an `UEBCC` object from an `REBCC` object.

        Args:
            rcc: Restricted electron-boson coupled cluster object.

        Returns:
            UEBCC object.
        """
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

        has_amps = bool(rcc.amplitudes)
        has_lams = bool(rcc.lambdas)

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

    def init_space(self) -> Namespace[Space]:
        """Initialise the fermionic space.

        Returns:
            Fermionic space. All fermionic degrees of freedom are assumed to be correlated.
        """
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

    def _pack_codegen_kwargs(
        self, *extra_kwargs: dict[str, Any], eris: Optional[ERIsInputType] = None
    ) -> dict[str, Any]:
        """Pack all the keyword arguments for the generated code."""
        kwargs = dict(
            f=self.fock,
            v=self.get_eris(eris),
            g=self.g,
            G=self.G,
            w=np.diag(self.omega) if self.omega is not None else None,
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

    def init_amps(self, eris: Optional[ERIsInputType] = None) -> Namespace[AmplitudeType]:
        """Initialise the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.

        Returns:
            Initial cluster amplitudes.
        """
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

    def init_lams(
        self, amplitudes: Optional[Namespace[AmplitudeType]] = None
    ) -> Namespace[AmplitudeType]:
        """Initialise the cluster lambda amplitudes.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Initial cluster lambda amplitudes.
        """
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

    def update_amps(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
    ) -> Namespace[AmplitudeType]:
        """Update the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Updated cluster amplitudes.
        """
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

    def update_lams(
        self,
        eris: ERIsInputType = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        lambdas_pert: Optional[Namespace[AmplitudeType]] = None,
        perturbative: bool = False,
    ) -> Namespace[AmplitudeType]:
        """Update the cluster lambda amplitudes.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            lambdas_pert: Perturbative cluster lambda amplitudes.
            perturbative: Flag to include perturbative correction.

        Returns:
            Updated cluster lambda amplitudes.
        """
        # TODO active
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

    def make_rdm1_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> Any:
        r"""Make the one-particle fermionic reduced density matrix :math:`\langle i^+ j \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            hermitise: Hermitise the density matrix.

        Returns:
            One-particle fermion reduced density matrix.
        """
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

    def make_rdm2_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> Any:
        r"""Make the two-particle fermionic reduced density matrix :math:`\langle i^+j^+lk \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            hermitise: Hermitise the density matrix.

        Returns:
            Two-particle fermion reduced density matrix.
        """
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
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        unshifted: bool = True,
        hermitise: bool = True,
    ) -> Any:
        r"""Make the electron-boson coupling reduced density matrix :math:`\langle b^+ c^+ c b \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            unshifted: If `self.options.shift` is `True`, return the unshifted density matrix. Has
                no effect if `self.options.shift` is `False`.
            hermitise: Hermitise the density matrix.

        Returns:
            Electron-boson coupling reduced density matrix.
        """
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

    def energy_sum(
        self, subscript: str, spins: str, signs_dict: dict[str, int] = None
    ) -> NDArray[float]:
        """Get a direct sum of energies.

        Args:
            subscript: Subscript for the direct sum.
            spins: Spins for energies.
            signs_dict: Signs of the energies in the sum. Default sets `("o", "O", "i")` to be
                positive, and `("v", "V", "a", "b")` to be negative.

        Returns:
            Sum of energies.
        """
        n = 0

        def next_char() -> str:
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

    def amplitudes_to_vector(self, amplitudes: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
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

    def vector_to_amplitudes(self, vector: NDArray[float]) -> Namespace[AmplitudeType]:
        """Construct a namespace of amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
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

    def lambdas_to_vector(self, lambdas: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the lambda amplitudes used in the given ansatz.

        Args:
            lambdas: Cluster lambda amplitudes.

        Returns:
            Cluster lambda amplitudes as a vector.
        """
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

    def vector_to_lambdas(self, vector: NDArray[float]) -> Namespace[AmplitudeType]:
        """Construct a namespace of lambda amplitudes from a vector.

        Args:
            vector: Cluster lambda amplitudes as a vector.

        Returns:
            Cluster lambda amplitudes.
        """
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

    def excitations_to_vector_ip(self, *excitations: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the IP-EOM excitations.

        Args:
            excitations: IP-EOM excitations.

        Returns:
            IP-EOM excitations as a vector.
        """
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

    def excitations_to_vector_ea(self, *excitations: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the EA-EOM excitations.

        Args:
            excitations: EA-EOM excitations.

        Returns:
            EA-EOM excitations as a vector.
        """
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

    def excitations_to_vector_ee(self, *excitations: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the EE-EOM excitations.

        Args:
            excitations: EE-EOM excitations.

        Returns:
            EE-EOM excitations as a vector.
        """
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

    def vector_to_excitations_ip(
        self, vector: NDArray[float]
    ) -> tuple[Namespace[AmplitudeType], ...]:
        """Construct a namespace of IP-EOM excitations from a vector.

        Args:
            vector: IP-EOM excitations as a vector.

        Returns:
            IP-EOM excitations.
        """
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

    def vector_to_excitations_ea(
        self, vector: NDArray[float]
    ) -> tuple[Namespace[AmplitudeType], ...]:
        """Construct a namespace of EA-EOM excitations from a vector.

        Args:
            vector: EA-EOM excitations as a vector.

        Returns:
            EA-EOM excitations.
        """
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

    def vector_to_excitations_ee(
        self, vector: NDArray[float]
    ) -> tuple[Namespace[AmplitudeType], ...]:
        """Construct a namespace of EE-EOM excitations from a vector.

        Args:
            vector: EE-EOM excitations as a vector.

        Returns:
            EE-EOM excitations.
        """
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

    def get_mean_field_G(self) -> NDArray[float]:
        """Get the mean-field boson non-conserving term.

        Returns:
            Mean-field boson non-conserving term.
        """
        # FIXME should this also sum in frozen orbitals?
        val = lib.einsum("Ipp->I", self.g.aa.boo)
        val += lib.einsum("Ipp->I", self.g.bb.boo)
        val -= self.xi * self.omega
        if self.bare_G is not None:
            # Require bare_G to have a spin index for now:
            assert np.shape(self.bare_G) == val.shape
            val += self.bare_G
        return val

    def get_g(self, g: NDArray[float]) -> Namespace[Namespace[NDArray[float]]]:
        """Get the blocks of the electron-boson coupling matrix.

        This matrix corresponds to the bosonic annihilation operator.

        Args:
            g: Electron-boson coupling matrix.

        Returns:
            Blocks of the electron-boson coupling matrix.
        """
        # TODO make a proper class for this
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
    def bare_fock(self) -> Namespace[NDArray[float]]:
        """Get the mean-field Fock matrix in the MO basis, including frozen parts.

        Returns an array and not a `BaseFock` object.

        Returns:
            Mean-field Fock matrix.
        """
        fock = lib.einsum(
            "npq,npi,nqj->nij",
            self.mf.get_fock().astype(types[float]),
            self.mo_coeff,
            self.mo_coeff,
        )
        fock = util.Namespace(aa=fock[0], bb=fock[1])
        return fock

    @property
    def xi(self) -> NDArray[float]:
        """Get the shift in the bosonic operators to diagonalise the photon Hamiltonian.

        Returns:
            Shift in the bosonic operators.
        """
        if self.options.shift:
            xi = lib.einsum("Iii->I", self.g.aa.boo)
            xi += lib.einsum("Iii->I", self.g.bb.boo)
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros_like(self.omega)
        return xi

    def get_fock(self) -> UFock:
        """Get the Fock matrix.

        Returns:
            Fock matrix.
        """
        return self.Fock(self, array=(self.bare_fock.aa, self.bare_fock.bb))

    def get_eris(self, eris: Optional[ERIsInputType] = None) -> Union[UERIs, UCDERIs]:
        """Get the electron repulsion integrals.

        Args:
            eris: Input electron repulsion integrals.

        Returns:
            Electron repulsion integrals.
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

    @property
    def nmo(self) -> int:
        """Get the number of molecular orbitals.

        Returns:
            Number of molecular orbitals.
        """
        assert self.mo_occ[0].size == self.mo_occ[1].size
        return self.mo_occ[0].size

    @property
    def nocc(self) -> tuple[int, int]:
        """Get the number of occupied molecular orbitals.

        Returns:
            Number of occupied molecular orbitals for each spin.
        """
        return tuple(np.sum(mo_occ > 0) for mo_occ in self.mo_occ)

    @property
    def nvir(self) -> tuple[int, int]:
        """Get the number of virtual molecular orbitals.

        Returns:
            Number of virtual molecular orbitals for each spin.
        """
        return tuple(self.nmo - nocc for nocc in self.nocc)

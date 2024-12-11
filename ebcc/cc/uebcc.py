"""Unrestricted electron-boson coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ebcc import numpy as np
from ebcc import util
from ebcc.cc.base import BaseEBCC
from ebcc.core.precision import types
from ebcc.eom import EA_UEOM, EE_UEOM, IP_UEOM
from ebcc.ext.eccc import ExternalCorrectionUEBCC
from ebcc.ext.tcc import TailorUEBCC
from ebcc.ham.cderis import UCDERIs
from ebcc.ham.elbos import UElectronBoson
from ebcc.ham.eris import UERIs
from ebcc.ham.fock import UFock
from ebcc.ham.space import Space
from ebcc.opt.uopt import BruecknerUEBCC, OptimisedUEBCC

if TYPE_CHECKING:
    from typing import Any, Optional, TypeAlias, Union

    from numpy import floating
    from numpy.typing import NDArray
    from pyscf.scf.hf import SCF
    from pyscf.scf.uhf import UHF

    from ebcc.cc.rebcc import REBCC
    from ebcc.util import Namespace

    T = floating

    ERIsInputType: TypeAlias = Union[UERIs, UCDERIs, tuple[NDArray[T], ...]]
    SpinArrayType: TypeAlias = Namespace[NDArray[T]]
    SpaceType: TypeAlias = tuple[Space, Space]


class UEBCC(BaseEBCC):
    """Unrestricted electron-boson coupled cluster."""

    # Types
    ERIs = UERIs
    Fock = UFock
    CDERIs = UCDERIs
    ElectronBoson = UElectronBoson
    Brueckner = BruecknerUEBCC
    Optimised = OptimisedUEBCC
    ExternalCorrection = ExternalCorrectionUEBCC
    Tailor = TailorUEBCC

    # Attributes
    space: SpaceType
    amplitudes: Namespace[SpinArrayType]
    lambdas: Namespace[SpinArrayType]
    fock: UFock

    @property
    def spin_type(self) -> str:
        """Get a string representation of the spin type."""
        return "U"

    def ip_eom(self, **kwargs: Any) -> IP_UEOM:
        """Get the IP-EOM object.

        Args:
            options: Options for the IP-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            IP-EOM object.
        """
        return IP_UEOM(self, **kwargs)

    def ea_eom(self, **kwargs: Any) -> EA_UEOM:
        """Get the EA-EOM object.

        Args:
            options: Options for the EA-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EA-EOM object.
        """
        return EA_UEOM(self, **kwargs)

    def ee_eom(self, **kwargs: Any) -> EE_UEOM:
        """Get the EE-EOM object.

        Args:
            options: Options for the EE-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EE-EOM object.
        """
        return EE_UEOM(self, **kwargs)

    @staticmethod
    def _convert_mf(mf: SCF) -> UHF:
        """Convert the mean-field object to the appropriate type."""
        return mf.to_uhf()
        hf = mf.to_uhf()
        if hasattr(mf, "xc"):
            hf.e_tot = hf.energy_tot()
        return hf

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
            amplitudes: Namespace[SpinArrayType] = util.Namespace()

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript, _ = util.combine_subscripts(key, comb)
                    tn = rcc.amplitudes[name]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    amplitudes[name][comb] = tn

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = np.copy(rcc.amplitudes[name])  # type: ignore

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = util.Namespace()
                for comb in util.generate_spin_combinations(nf, unique=True):
                    tn = rcc.amplitudes[name]
                    amplitudes[name][comb] = tn

            ucc.amplitudes = amplitudes

        if has_lams:
            lambdas: Namespace[SpinArrayType] = util.Namespace()

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                lname = name.replace("t", "l")
                lambdas[lname] = util.Namespace()
                for comb in util.generate_spin_combinations(n, unique=True):
                    subscript, _ = util.combine_subscripts(key, comb)
                    tn = rcc.lambdas[lname]
                    tn = util.symmetrise(subscript, tn, symmetry="-" * 2 * n)
                    lambdas[lname][comb] = tn

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = np.copy(rcc.lambdas[lname])  # type: ignore

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = util.Namespace()
                for comb in util.generate_spin_combinations(nf, unique=True):
                    tn = rcc.lambdas[lname]
                    lambdas[lname][comb] = tn

            ucc.lambdas = lambdas

        return ucc

    def init_space(self) -> SpaceType:
        """Initialise the fermionic space.

        Returns:
            Fermionic space. All fermionic degrees of freedom are assumed to be correlated.
        """
        space = (
            Space(
                self.mo_occ[0] > 0,
                np.zeros(self.mo_occ[0].shape, dtype=np.bool_),
                np.zeros(self.mo_occ[0].shape, dtype=np.bool_),
            ),
            Space(
                self.mo_occ[1] > 0,
                np.zeros(self.mo_occ[1].shape, dtype=np.bool_),
                np.zeros(self.mo_occ[1].shape, dtype=np.bool_),
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

    def init_amps(self, eris: Optional[ERIsInputType] = None) -> Namespace[SpinArrayType]:
        """Initialise the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.

        Returns:
            Initial cluster amplitudes.
        """
        eris = self.get_eris(eris)
        amplitudes: Namespace[SpinArrayType] = util.Namespace()

        # Build T amplitudes
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            tn: SpinArrayType = util.Namespace()
            for comb in util.generate_spin_combinations(n, unique=True):
                if n == 1:
                    tn[comb] = self.fock[comb][key] / self.energy_sum(key, comb)
                elif n == 2:
                    comb_t = comb[0] + comb[2] + comb[1] + comb[3]
                    key_t = key[0] + key[2] + key[1] + key[3]
                    tn[comb] = np.transpose(eris[comb_t][key_t], (0, 2, 1, 3)) / self.energy_sum(
                        key, comb
                    )
                    if comb in ("aaaa", "bbbb"):
                        # TODO generalise:
                        tn[comb] = (tn[comb] - np.transpose(tn[comb], (1, 0, 2, 3))) * 0.5
                else:
                    shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(comb, key))
                    tn[comb] = np.zeros(shape, dtype=types[float])
                amplitudes[name] = tn

        # Build S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            if self.omega is None or self.G is None:
                raise ValueError("Bosonic parameters not set.")
            if n == 1:
                amplitudes[name] = -self.G / self.omega  # type: ignore
            else:
                shape = (self.nbos,) * n
                amplitudes[name] = np.zeros(shape, dtype=types[float])  # type: ignore

        # Build U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if self.omega is None or self.g is None:
                raise ValueError("Bosonic parameters not set.")
            if nf != 1:
                raise util.ModelNotImplemented
            if nb == 1:
                tn = util.Namespace(
                    aa=self.g.aa[key] / self.energy_sum(key, "_aa"),
                    bb=self.g.bb[key] / self.energy_sum(key, "_aa"),
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
        self, amplitudes: Optional[Namespace[SpinArrayType]] = None
    ) -> Namespace[SpinArrayType]:
        """Initialise the cluster lambda amplitudes.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Initial cluster lambda amplitudes.
        """
        if not amplitudes:
            amplitudes = self.amplitudes
        lambdas: Namespace[SpinArrayType] = util.Namespace()

        # Build L amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            perm = list(range(n, 2 * n)) + list(range(n))
            lambdas[lname] = util.Namespace()
            for key in dict(amplitudes[name]).keys():
                ln = np.transpose(amplitudes[name][key], perm)
                lambdas[lname][key] = ln

        # Build LS amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            lambdas["l" + name] = amplitudes[name]  # type: ignore

        # Build LU amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            perm = list(range(nb)) + [nb + 1, nb]
            lambdas["l" + name] = util.Namespace()
            for key in dict(amplitudes[name]).keys():
                ln = np.transpose(amplitudes[name][key], perm)
                lambdas["l" + name][key] = ln

        return lambdas

    def update_amps(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
    ) -> Namespace[SpinArrayType]:
        """Update the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Updated cluster amplitudes.
        """
        amplitudes = self._get_amps(amplitudes=amplitudes)
        func, kwargs = self._load_function(
            "update_amps",
            eris=eris,
            amplitudes=amplitudes,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript, _ = util.combine_subscripts(key, comb)
                tn = res[name][comb]
                tn /= self.energy_sum(key, comb)
                tn += amplitudes[name][comb]
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                res[name][comb] = tn

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            res[name] /= self.energy_sum(key, "_" * n)  # type: ignore
            res[name] += amplitudes[name]  # type: ignore

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
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        lambdas_pert: Optional[Namespace[SpinArrayType]] = None,
        perturbative: bool = False,
    ) -> Namespace[SpinArrayType]:
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
        amplitudes = self._get_amps(amplitudes=amplitudes)
        lambdas = self._get_lams(lambdas=lambdas, amplitudes=amplitudes)
        func, kwargs = self._load_function(
            "update_lams",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
            lambdas_pert=lambdas_pert,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            for comb in util.generate_spin_combinations(n, unique=True):
                subscript, _ = util.combine_subscripts(key, comb)
                tn = res[name][comb]
                tn /= self.energy_sum(key, comb)
                tn += lambdas[name][comb]
                tn = util.symmetrise(subscript, tn, symmetry="-" * (2 * n))
                res[name][comb] = tn

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            res[name] /= self.energy_sum(key, "_" * n)  # type: ignore
            res[name] += lambdas[name]  # type: ignore

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            if nf != 1:
                raise util.ModelNotImplemented
            tn = res[name].aa
            tn /= self.energy_sum(key, "_" * nb + "aa")
            tn += lambdas[name].aa
            res[name].aa = tn
            tn = res[name].bb
            tn /= self.energy_sum(key, "_" * nb + "bb")
            tn += lambdas[name].bb
            res[name].bb = tn

        return res

    def make_rdm1_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        hermitise: bool = True,
    ) -> SpinArrayType:
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
        dm: SpinArrayType = func(**kwargs)

        if hermitise:
            dm.aa = (dm.aa + np.transpose(dm.aa)) * 0.5
            dm.bb = (dm.bb + np.transpose(dm.bb)) * 0.5

        return dm

    def make_rdm2_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        hermitise: bool = True,
    ) -> SpinArrayType:
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
        dm: SpinArrayType = func(**kwargs)

        if hermitise:

            def transpose1(dm: NDArray[T]) -> NDArray[T]:
                dm = (np.transpose(dm, (0, 1, 2, 3)) + np.transpose(dm, (2, 3, 0, 1))) * 0.5
                return dm

            def transpose2(dm: NDArray[T]) -> NDArray[T]:
                dm = (np.transpose(dm, (0, 1, 2, 3)) + np.transpose(dm, (1, 0, 3, 2))) * 0.5
                return dm

            dm.aaaa = transpose2(transpose1(dm.aaaa))
            dm.aabb = transpose2(dm.aabb)
            dm.bbbb = transpose2(transpose1(dm.bbbb))

        return dm

    def make_eb_coup_rdm(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        unshifted: bool = True,
        hermitise: bool = True,
    ) -> SpinArrayType:
        r"""Make the electron-boson coupling reduced density matrix.

        .. math::
            \langle b^+ i^+ j \rangle

        and

        .. math::
            \langle b i^+ j \rangle

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
        dm_eb: SpinArrayType = func(**kwargs)

        if hermitise:
            dm_eb.aa = np.array(
                [
                    (dm_eb.aa[0] + np.transpose(dm_eb.aa[1], (0, 2, 1))) * 0.5,
                    (dm_eb.aa[1] + np.transpose(dm_eb.aa[0], (0, 2, 1))) * 0.5,
                ]
            )
            dm_eb.bb = np.array(
                [
                    (dm_eb.bb[0] + np.transpose(dm_eb.bb[1], (0, 2, 1))) * 0.5,
                    (dm_eb.bb[1] + np.transpose(dm_eb.bb[0], (0, 2, 1))) * 0.5,
                ]
            )

        if unshifted and self.options.shift:
            rdm1_f = self.make_rdm1_f(hermitise=hermitise)
            shift = util.einsum("x,ij->xij", self.xi, rdm1_f.aa)
            dm_eb.aa -= shift[None]
            shift = util.einsum("x,ij->xij", self.xi, rdm1_f.bb)
            dm_eb.bb -= shift[None]

        return dm_eb

    def energy_sum(self, *args: str, signs_dict: Optional[dict[str, str]] = None) -> NDArray[T]:
        """Get a direct sum of energies.

        Args:
            *args: Energies to sum. Should specify a subscript and spins.
            signs_dict: Signs of the energies in the sum. Default sets `("o", "O", "i")` to be
                positive, and `("v", "V", "a", "b")` to be negative.

        Returns:
            Sum of energies.
        """
        subscript, spins = args
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
            factor = 1 if signs_dict[key] == "+" else -1
            if key == "b":
                assert self.omega is not None
                energies.append(self.omega * types[float](factor))
            else:
                energies.append(np.diag(self.fock[spin + spin][key + key]) * types[float](factor))

        subscript = ",".join([next_char() for k in subscript])
        energy_sum = util.dirsum(subscript, *energies)

        return energy_sum

    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[T]:
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
                subscript, _ = util.combine_subscripts(key, spin)
                vectors.append(np.ravel(util.compress_axes(subscript, tn)))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(np.ravel(amplitudes[name]))  # type: ignore

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            vectors.append(np.ravel(amplitudes[name].aa))
            vectors.append(np.ravel(amplitudes[name].bb))

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct a namespace of amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        amplitudes: Namespace[SpinArrayType] = util.Namespace()
        i0 = 0
        sizes: dict[tuple[str, ...], int] = {
            (o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")
        }

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
            size = self.nbos**n
            amplitudes[name] = np.reshape(vector[i0 : i0 + size], shape)  # type: ignore
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            amplitudes[name] = util.Namespace()
            shape = (self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:])
            size = util.prod(shape)
            amplitudes[name].aa = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size
            shape = (self.nbos,) * nb + tuple(self.space[1].size(k) for k in key[nb:])
            size = util.prod(shape)
            amplitudes[name].bb = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        assert i0 == len(vector)

        return amplitudes

    def lambdas_to_vector(self, lambdas: Namespace[SpinArrayType]) -> NDArray[T]:
        """Construct a vector containing all of the lambda amplitudes used in the given ansatz.

        Args:
            lambdas: Cluster lambda amplitudes.

        Returns:
            Cluster lambda amplitudes as a vector.
        """
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            for spin in util.generate_spin_combinations(n, unique=True):
                tn = lambdas[name][spin]
                subscript, _ = util.combine_subscripts(key, spin)
                vectors.append(np.ravel(util.compress_axes(subscript, tn)))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            vectors.append(np.ravel(lambdas[name]))  # type: ignore

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            if nf != 1:
                raise util.ModelNotImplemented
            vectors.append(np.ravel(lambdas[name].aa))
            vectors.append(np.ravel(lambdas[name].bb))

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct a namespace of lambda amplitudes from a vector.

        Args:
            vector: Cluster lambda amplitudes as a vector.

        Returns:
            Cluster lambda amplitudes.
        """
        lambdas: Namespace[SpinArrayType] = util.Namespace()
        i0 = 0
        sizes: dict[tuple[str, ...], int] = {
            (o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")
        }

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            lambdas[name] = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                tn_tril = vector[i0 : i0 + size]
                tn = util.decompress_axes(subscript, tn_tril, shape=shape)
                lambdas[name][spin] = tn
                i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            shape = (self.nbos,) * n
            size = self.nbos**n
            lambdas[name] = np.reshape(vector[i0 : i0 + size], shape)  # type: ignore
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            if nf != 1:
                raise util.ModelNotImplemented
            lambdas[name] = util.Namespace()
            shape = (self.nbos,) * nb + tuple(self.space[0].size(k) for k in key[nb:])
            size = util.prod(shape)
            lambdas[name].aa = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size
            shape = (self.nbos,) * nb + tuple(self.space[1].size(k) for k in key[nb:])
            size = util.prod(shape)
            lambdas[name].bb = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        assert i0 == len(vector)

        return lambdas

    def get_mean_field_G(self) -> NDArray[T]:
        """Get the mean-field boson non-conserving term.

        Returns:
            Mean-field boson non-conserving term.
        """
        # FIXME should this also sum in frozen orbitals?
        assert self.omega is not None
        assert self.g is not None
        boo: tuple[NDArray[T], NDArray[T]] = (self.g.aa.boo, self.g.bb.boo)
        val = util.einsum("Ipp->I", boo[0])
        val += util.einsum("Ipp->I", boo[1])
        val -= self.xi * self.omega
        if self.bare_G is not None:
            # Require bare_G to have a spin index for now:
            assert self.bare_G.shape == val.shape
            val += self.bare_G
        return val

    @property
    def xi(self) -> NDArray[T]:
        """Get the shift in the bosonic operators to diagonalise the photon Hamiltonian.

        Returns:
            Shift in the bosonic operators.
        """
        assert self.omega is not None
        if self.options.shift:
            assert self.g is not None
            boo: tuple[NDArray[T], NDArray[T]] = (self.g.aa.boo, self.g.bb.boo)
            xi = util.einsum("Iii->I", boo[0])
            xi += util.einsum("Iii->I", boo[1])
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros(self.omega.shape)
        return xi

    @property
    def nmo(self) -> int:
        """Get the number of molecular orbitals.

        Returns:
            Number of molecular orbitals.
        """
        assert self.mo_occ[0].size == self.mo_occ[1].size
        return cast(int, self.mo_occ[0].size)

    @property
    def nocc(self) -> tuple[int, int]:
        """Get the number of occupied molecular orbitals.

        Returns:
            Number of occupied molecular orbitals for each spin.
        """
        return cast(
            tuple[int, int],
            tuple(np.sum(np.asarray(mo_occ > 0, dtype=int)) for mo_occ in self.mo_occ),
        )

    @property
    def nvir(self) -> tuple[int, int]:
        """Get the number of virtual molecular orbitals.

        Returns:
            Number of virtual molecular orbitals for each spin.
        """
        return cast(tuple[int, int], tuple(self.nmo - nocc for nocc in self.nocc))

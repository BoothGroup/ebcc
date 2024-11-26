"""Restricted electron-boson coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.cc.base import BaseEBCC
from ebcc.core.precision import types
from ebcc.eom import EA_REOM, EE_REOM, IP_REOM
from ebcc.ham.cderis import RCDERIs
from ebcc.ham.elbos import RElectronBoson
from ebcc.ham.eris import RERIs
from ebcc.ham.fock import RFock
from ebcc.ham.space import Space
from ebcc.opt.rbrueckner import BruecknerREBCC

if TYPE_CHECKING:
    from typing import Any, Optional, TypeAlias, Union

    from numpy import float64
    from numpy.typing import NDArray
    from pyscf.scf.hf import RHF, SCF

    from ebcc.util import Namespace

    T = float64

    ERIsInputType: TypeAlias = Union[RERIs, RCDERIs, NDArray[T]]
    SpinArrayType: TypeAlias = NDArray[T]
    SpaceType: TypeAlias = Space


class REBCC(BaseEBCC):
    """Restricted electron-boson coupled cluster."""

    # Types
    ERIs = RERIs
    Fock = RFock
    CDERIs = RCDERIs
    ElectronBoson = RElectronBoson
    Brueckner = BruecknerREBCC

    # Attributes
    space: SpaceType
    amplitudes: Namespace[SpinArrayType]
    lambdas: Namespace[SpinArrayType]
    fock: RFock

    @property
    def spin_type(self) -> str:
        """Get a string representation of the spin type."""
        return "R"

    def ip_eom(self, **kwargs: Any) -> IP_REOM:
        """Get the IP-EOM object.

        Args:
            options: Options for the IP-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            IP-EOM object.
        """
        return IP_REOM(self, **kwargs)

    def ea_eom(self, **kwargs: Any) -> EA_REOM:
        """Get the EA-EOM object.

        Args:
            options: Options for the EA-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EA-EOM object.
        """
        return EA_REOM(self, **kwargs)

    def ee_eom(self, **kwargs: Any) -> EE_REOM:
        """Get the EE-EOM object.

        Args:
            options: Options for the EE-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EE-EOM object.
        """
        return EE_REOM(self, **kwargs)

    @staticmethod
    def _convert_mf(mf: SCF) -> RHF:
        """Convert the mean-field object to the appropriate type."""
        hf = mf.to_rhf()
        if hasattr(mf, "xc"):
            hf.e_tot = hf.energy_tot()
        return hf

    def init_space(self) -> SpaceType:
        """Initialise the fermionic space.

        Returns:
            Fermionic space. All fermionic degrees of freedom are assumed to be correlated.
        """
        space = Space(
            self.mo_occ > 0,
            np.zeros(self.mo_occ.shape, dtype=np.bool_),
            np.zeros(self.mo_occ.shape, dtype=np.bool_),
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
            nocc=self.space.ncocc,
            nvir=self.space.ncvir,
            nbos=self.nbos,
        )

        if isinstance(kwargs["v"], self.CDERIs):
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

        # Build T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                amplitudes[name] = self.fock[key] / self.energy_sum(key)
            elif n == 2:
                key_t = key[0] + key[2] + key[1] + key[3]
                amplitudes[name] = np.transpose(eris[key_t], (0, 2, 1, 3)) / self.energy_sum(key)
            else:
                shape = tuple(self.space.size(k) for k in key)
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        # Build S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            if self.omega is None or self.G is None:
                raise ValueError("Bosonic parameters not set.")
            if n == 1:
                amplitudes[name] = -self.G / self.omega
            else:
                shape = (self.nbos,) * n
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        # Build U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if self.omega is None or self.g is None:
                raise ValueError("Bosonic parameters not set.")
            if nf != 1:
                raise util.ModelNotImplemented
            if nb == 1:
                amplitudes[name] = self.g[key] / self.energy_sum(key)
            else:
                shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape, dtype=types[float])

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
            lambdas[lname] = np.transpose(amplitudes[name], perm)

        # Build LS amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            lname = "l" + name
            lambdas[lname] = amplitudes[name]

        # Build LU amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            lname = "l" + name
            perm = list(range(nb)) + [nb + 1, nb]
            lambdas[lname] = np.transpose(amplitudes[name], perm)

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
            res[name] /= self.energy_sum(key)
            res[name] += amplitudes[name]

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            res[name] /= self.energy_sum(key)
            res[name] += amplitudes[name]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            res[name] /= self.energy_sum(key)
            res[name] += amplitudes[name]

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
        if lambdas_pert is not None:
            lambdas.update(lambdas_pert)

        func, kwargs = self._load_function(
            "update_lams%s" % ("_perturbative" if perturbative else ""),
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            res[name] /= self.energy_sum(key)
            if not perturbative:
                res[name] += lambdas[name]

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            res[name] /= self.energy_sum(key)
            if not perturbative:
                res[name] += lambdas[name]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            if nf != 1:
                raise util.ModelNotImplemented
            res[name] /= self.energy_sum(key)
            if not perturbative:
                res[name] += lambdas[name]

        if perturbative:
            res = util.Namespace(**{key + "pert": val for key, val in res.items()})

        return res

    def make_rdm1_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        hermitise: bool = True,
    ) -> NDArray[T]:
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
        dm: NDArray[T] = func(**kwargs)

        if hermitise:
            dm = (dm + np.transpose(dm)) * 0.5

        return dm

    def make_rdm2_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        hermitise: bool = True,
    ) -> NDArray[T]:
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
        dm: NDArray[T] = func(**kwargs)

        if hermitise:
            dm = (np.transpose(dm, (0, 1, 2, 3)) + np.transpose(dm, (2, 3, 0, 1))) * 0.5
            dm = (np.transpose(dm, (0, 1, 2, 3)) + np.transpose(dm, (1, 0, 3, 2))) * 0.5

        return dm

    def make_eb_coup_rdm(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        unshifted: bool = True,
        hermitise: bool = True,
    ) -> NDArray[T]:
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
        dm_eb: NDArray[T] = func(**kwargs)

        if hermitise:
            dm_eb = np.array(
                [
                    (dm_eb[0] + np.transpose(dm_eb[1], (0, 2, 1))) * 0.5,
                    (dm_eb[1] + np.transpose(dm_eb[0], (0, 2, 1))) * 0.5,
                ]
            )

        if unshifted and self.options.shift:
            rdm1_f = self.make_rdm1_f(hermitise=hermitise)
            shift = util.einsum("x,ij->xij", self.xi, rdm1_f)
            dm_eb -= shift[None]

        return dm_eb

    def energy_sum(self, *args: str, signs_dict: Optional[dict[str, str]] = None) -> NDArray[T]:
        """Get a direct sum of energies.

        Args:
            *args: Energies to sum. Should specify a subscript only.
            signs_dict: Signs of the energies in the sum. Default sets `("o", "O", "i")` to be
                positive, and `("v", "V", "a", "b")` to be negative.

        Returns:
            Sum of energies.
        """
        (subscript,) = args
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
        for key in subscript:
            factor = 1 if signs_dict[key] == "+" else -1
            if key == "b":
                assert self.omega is not None
                energies.append(self.omega * types[float](factor))
            else:
                energies.append(np.diag(self.fock[key + key]) * types[float](factor))

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
            vectors.append(np.ravel(amplitudes[name]))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(np.ravel(amplitudes[name]))

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            vectors.append(np.ravel(amplitudes[name]))

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            shape = tuple(self.space.size(k) for k in key)
            size = util.prod(shape)
            amplitudes[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * n
            size = util.prod(shape)
            amplitudes[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
            size = util.prod(shape)
            amplitudes[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

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
            vectors.append(np.ravel(lambdas[name]))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            vectors.append(np.ravel(lambdas[name]))

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            vectors.append(np.ravel(lambdas[name]))

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            shape = tuple(self.space.size(k) for k in key)
            size = util.prod(shape)
            lambdas[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            shape = (self.nbos,) * n
            size = util.prod(shape)
            lambdas[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
            size = util.prod(shape)
            lambdas[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        return lambdas

    def get_mean_field_G(self) -> NDArray[T]:
        """Get the mean-field boson non-conserving term.

        Returns:
            Mean-field boson non-conserving term.
        """
        # FIXME should this also sum in frozen orbitals?
        assert self.omega is not None
        assert self.g is not None
        boo: NDArray[T] = self.g.boo
        val = util.einsum("Ipp->I", boo) * 2.0
        val -= self.xi * self.omega
        if self.bare_G is not None:
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
            boo: NDArray[T] = self.g.boo
            xi = util.einsum("Iii->I", boo) * 2.0
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros(self.omega.shape, dtype=types[float])
        return xi

    @property
    def nmo(self) -> int:
        """Get the number of molecular orbitals.

        Returns:
            Number of molecular orbitals.
        """
        return self.space.nmo

    @property
    def nocc(self) -> int:
        """Get the number of occupied molecular orbitals.

        Returns:
            Number of occupied molecular orbitals.
        """
        return self.space.nocc

    @property
    def nvir(self) -> int:
        """Get the number of virtual molecular orbitals.

        Returns:
            Number of virtual molecular orbitals.
        """
        return self.space.nvir

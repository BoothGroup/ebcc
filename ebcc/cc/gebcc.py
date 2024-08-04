"""Generalised electron-boson coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import lib, scf

from ebcc import numpy as np
from ebcc import util
from ebcc.cc.base import BaseEBCC
from ebcc.core.precision import types
from ebcc.eom import EA_GEOM, EE_GEOM, IP_GEOM
from ebcc.ham.eris import GERIs
from ebcc.ham.fock import GFock
from ebcc.ham.space import Space
from ebcc.opt.gbrueckner import BruecknerGEBCC

if TYPE_CHECKING:
    from typing import Any, Optional, cast

    from pyscf.scf.ghf import GHF
    from pyscf.scf.hf import SCF

    from ebcc.cc.base import BaseOptions, ERIsInputType
    from ebcc.cc.rebcc import REBCC
    from ebcc.cc.uebcc import UEBCC
    from ebcc.numpy.typing import NDArray
    from ebcc.util import Namespace

    AmplitudeType = NDArray[float]


class GEBCC(BaseEBCC):
    """Restricted electron-boson coupled cluster.

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
    ERIs = GERIs
    Fock = GFock
    CDERIs = None
    Brueckner = BruecknerGEBCC

    @property
    def spin_type(self):
        """Get a string representation of the spin type."""
        return "G"

    def ip_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> IP_GEOM:
        """Get the IP-EOM object.

        Args:
            options: Options for the IP-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            IP-EOM object.
        """
        return IP_GEOM(self, options=options, **kwargs)

    def ea_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> EA_GEOM:
        """Get the EA-EOM object.

        Args:
            options: Options for the EA-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EA-EOM object.
        """
        return EA_GEOM(self, options=options, **kwargs)

    def ee_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> EE_GEOM:
        """Get the EE-EOM object.

        Args:
            options: Options for the EE-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EE-EOM object.
        """
        return EE_GEOM(self, options=options, **kwargs)

    @staticmethod
    def _convert_mf(mf: SCF) -> GHF:
        """Convert a mean-field object to a GHF object.

        Note:
            Converts to UHF first to ensure consistent ordering.
        """
        if isinstance(mf, scf.ghf.GHF):
            return mf
        return mf.to_uhf().to_ghf()

    @classmethod
    def from_uebcc(cls, ucc: UEBCC) -> GEBCC:
        """Initialise a `GEBCC` object from an `UEBCC` object.

        Args:
            ucc: Unrestricted electron-boson coupled cluster object.

        Returns:
            GEBCC object.
        """
        orbspin = scf.addons.get_ghf_orbspin(ucc.mf.mo_energy, ucc.mf.mo_occ, False)
        nocc = ucc.space[0].nocc + ucc.space[1].nocc
        nvir = ucc.space[0].nvir + ucc.space[1].nvir
        nbos = ucc.nbos
        sa = np.where(orbspin == 0)[0]
        sb = np.where(orbspin == 1)[0]

        occupied = np.zeros((nocc + nvir,), dtype=bool)
        occupied[sa] = ucc.space[0]._occupied.copy()
        occupied[sb] = ucc.space[1]._occupied.copy()
        frozen = np.zeros((nocc + nvir,), dtype=bool)
        frozen[sa] = ucc.space[0]._frozen.copy()
        frozen[sb] = ucc.space[1]._frozen.copy()
        active = np.zeros((nocc + nvir,), dtype=bool)
        active[sa] = ucc.space[0]._active.copy()
        active[sb] = ucc.space[1]._active.copy()
        space = Space(occupied, frozen, active)

        slices = util.Namespace(
            a=util.Namespace(**{k: np.where(orbspin[space.mask(k)] == 0)[0] for k in "oOivVa"}),
            b=util.Namespace(**{k: np.where(orbspin[space.mask(k)] == 1)[0] for k in "oOivVa"}),
        )

        if ucc.bare_g is not None:
            if np.asarray(ucc.bare_g).ndim == 3:
                bare_g_a = bare_g_b = ucc.bare_g
            else:
                bare_g_a, bare_g_b = ucc.bare_g
            g = np.zeros((ucc.nbos, ucc.nmo * 2, ucc.nmo * 2))
            g[np.ix_(range(ucc.nbos), sa, sa)] = bare_g_a.copy()
            g[np.ix_(range(ucc.nbos), sb, sb)] = bare_g_b.copy()
        else:
            g = None

        gcc = cls(
            ucc.mf,
            log=ucc.log,
            ansatz=ucc.ansatz,
            space=space,
            omega=ucc.omega,
            g=g,
            G=ucc.bare_G,
            options=ucc.options,
        )

        gcc.e_corr = ucc.e_corr
        gcc.converged = ucc.converged
        gcc.converged_lambda = ucc.converged_lambda

        has_amps = bool(ucc.amplitudes)
        has_lams = bool(ucc.lambdas)

        if has_amps:
            amplitudes = util.Namespace()

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                shape = tuple(space.size(k) for k in key)
                amplitudes[name] = np.zeros(shape, dtype=types[float])
                for comb in util.generate_spin_combinations(n, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(n))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(n))):
                            combn = util.permute_string(comb[:n], lperm)
                            combn += util.permute_string(comb[n:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(*[slices[s][k] for s, k in zip(combn, key)])
                            transpose = tuple(lperm) + tuple(p + n for p in uperm)
                            amp = (
                                getattr(ucc.amplitudes[name], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    amplitudes[name][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = ucc.amplitudes[name].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                shape = (nbos,) * nb + tuple(space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape, dtype=types[float])
                for comb in util.generate_spin_combinations(nf):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(nf))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(nf))):
                            combn = util.permute_string(comb[:nf], lperm)
                            combn += util.permute_string(comb[nf:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([range(nbos)] * nb),
                                *[slices[s][k] for s, k in zip(combn, key[nb:])],
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in lperm)
                                + tuple(p + nb + nf for p in uperm)
                            )
                            amp = (
                                getattr(ucc.amplitudes[name], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                                transpose = (
                                    tuple(range(nb))
                                    + tuple(p + nb for p in perm)
                                    + tuple(range(nb + nf, nb + 2 * nf))
                                )
                                if util.permute_string(comb[:nf], perm) == comb[:nf]:
                                    amplitudes[name][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            gcc.amplitudes = amplitudes

        if has_lams:
            lambdas = gcc.init_lams()  # Easier this way - but have to build ERIs...

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                lname = name.replace("t", "l")
                shape = tuple(space.size(k) for k in key[n:] + key[:n])
                lambdas[lname] = np.zeros(shape, dtype=types[float])
                for comb in util.generate_spin_combinations(n, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(n))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(n))):
                            combn = util.permute_string(comb[:n], lperm)
                            combn += util.permute_string(comb[n:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(*[slices[s][k] for s, k in zip(combn, key[n:] + key[:n])])
                            transpose = tuple(lperm) + tuple(p + n for p in uperm)
                            amp = (
                                getattr(ucc.lambdas[lname], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    lambdas[lname][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = ucc.lambdas[lname].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                shape = (nbos,) * nb + tuple(
                    space.size(k) for k in key[nb + nf :] + key[nb : nb + nf]
                )
                lambdas[lname] = np.zeros(shape, dtype=types[float])
                for comb in util.generate_spin_combinations(nf, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(nf))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(nf))):
                            combn = util.permute_string(comb[:nf], lperm)
                            combn += util.permute_string(comb[nf:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([range(nbos)] * nb),
                                *[
                                    slices[s][k]
                                    for s, k in zip(combn, key[nb + nf :] + key[nb : nb + nf])
                                ],
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in lperm)
                                + tuple(p + nb + nf for p in uperm)
                            )
                            amp = (
                                getattr(ucc.lambdas[lname], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                                transpose = (
                                    tuple(range(nb))
                                    + tuple(p + nb for p in perm)
                                    + tuple(range(nb + nf, nb + 2 * nf))
                                )
                                if util.permute_string(comb[:nf], perm) == comb[:nf]:
                                    lambdas[lname][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            gcc.lambdas = lambdas

        return gcc

    @classmethod
    def from_rebcc(cls, rcc: REBCC) -> GEBCC:
        """Initialise a `GEBCC` object from an `REBCC` object.

        Args:
            rcc: Restricted electron-boson coupled cluster object.

        Returns:
            GEBCC object.
        """
        from ebcc.cc.uebcc import UEBCC

        ucc = UEBCC.from_rebcc(rcc)
        gcc = cls.from_uebcc(ucc)
        return gcc

    def init_space(self) -> Space:
        """Initialise the fermionic space.

        Returns:
            Fermionic space. All fermionic degrees of freedom are assumed to be correlated.
        """
        space = Space(
            self.mo_occ > 0,
            np.zeros_like(self.mo_occ, dtype=bool),
            np.zeros_like(self.mo_occ, dtype=bool),
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

        # Build T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                amplitudes[name] = getattr(self.fock, key) / self.energy_sum(key)
            elif n == 2:
                amplitudes[name] = getattr(eris, key) / self.energy_sum(key)
            else:
                shape = tuple(self.space.size(k) for k in key)
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        if self.boson_ansatz:
            # Only true for real-valued couplings:
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
            if n == 1:
                amplitudes[name] = h[key] / self.energy_sum(key)
            else:
                shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape, dtype=types[float])

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
            lambdas[lname] = amplitudes[name].transpose(perm)

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
            lambdas[lname] = amplitudes[name].transpose(perm)

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
        res = cast(Namespace[AmplitudeType], func(**kwargs))
        res = {key.rstrip("new"): val for key, val in res.items()}

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
        if lambdas_pert is not None:
            lambdas.update(lambdas_pert)

        func, kwargs = self._load_function(
            "update_lams%s" % ("_perturbative" if perturbative else ""),
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        res = cast(Namespace[AmplitudeType], func(**kwargs))
        res = {key.rstrip("new"): val for key, val in res.items()}

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            res[lname] /= self.energy_sum(key[n:] + key[:n])
            if not perturbative:
                res[lname] += lambdas[lname]

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            lname = "l" + name
            res[lname] /= self.energy_sum(key[n:] + key[:n])
            if not perturbative:
                res[lname] += lambdas[lname]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            lname = "l" + name
            res[lname] /= self.energy_sum(key[:nb] + key[nb + nf :] + key[nb : nb + nf])
            if not perturbative:
                res[lname] += lambdas[lname]

        if perturbative:
            res = {key + "pert": val for key, val in res.items()}

        return res

    def make_rdm1_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> NDArray[float]:
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
        dm = cast(NDArray[float], func(**kwargs))

        if hermitise:
            dm = 0.5 * (dm + dm.T)

        return dm

    def make_rdm2_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> NDArray[float]:
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
        dm = cast(NDArray[float], func(**kwargs))

        if hermitise:
            dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2, 3, 0, 1))
            dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(1, 0, 3, 2))

        return dm

    def make_eb_coup_rdm(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        unshifted: bool = True,
        hermitise: bool = True,
    ) -> NDArray[float]:
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
        dm_eb = cast(NDArray[float], func(**kwargs))

        if hermitise:
            dm_eb[0] = 0.5 * (dm_eb[0] + dm_eb[1].transpose(0, 2, 1))
            dm_eb[1] = dm_eb[0].transpose(0, 2, 1).copy()

        if unshifted and self.options.shift:
            rdm1_f = self.make_rdm1_f(hermitise=hermitise)
            shift = util.einsum("x,ij->xij", self.xi, rdm1_f)
            dm_eb -= shift[None]

        return dm_eb

    def energy_sum(self, subscript: str, signs_dict: dict[str, int] = None) -> NDArray[float]:
        """Get a direct sum of energies.

        Args:
            subscript: Subscript for the direct sum.
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
        for key in subscript:
            if key == "b":
                energies.append(self.omega)
            else:
                energies.append(np.diag(self.fock[key + key]))

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
            vectors.append(amplitudes[name].ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(amplitudes[name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            vectors.append(amplitudes[name].ravel())

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            shape = tuple(self.space.size(k) for k in key)
            size = np.prod(shape)
            amplitudes[name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes[name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
            size = np.prod(shape)
            amplitudes[name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

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
            vectors.append(lambdas[name.replace("t", "l")].ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(lambdas["l" + name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            vectors.append(lambdas["l" + name].ravel())

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            key = key[n:] + key[:n]
            shape = tuple(self.space.size(k) for k in key)
            size = np.prod(shape)
            lambdas[lname] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["l" + name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            key = key[:nb] + key[nb + nf :] + key[nb : nb + nf]
            shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
            size = np.prod(shape)
            lambdas["l" + name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

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
            key = key[:-1]
            vectors.append(util.compress_axes(key, excitations[m]).ravel())
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
        return self.excitations_to_vector_ip(*excitations)

    def excitations_to_vector_ee(self, *excitations):
        """Construct a vector containing all of the EE-EOM excitations.

        Args:
            excitations: EE-EOM excitations.

        Returns:
            EE-EOM excitations as a vector.
        """
        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(util.compress_axes(key, excitations[m]).ravel())
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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            excitations.append(vn)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            excitations.append(vn)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            excitations.append(vn)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return tuple(excitations)

    def get_mean_field_G(self) -> NDArray[float]:
        """Get the mean-field boson non-conserving term.

        Returns:
            Mean-field boson non-conserving term.
        """
        # FIXME should this also sum in frozen orbitals?
        val = lib.einsum("Ipp->I", self.g.boo)
        val -= self.xi * self.omega
        if self.bare_G is not None:
            val += self.bare_G
        return val

    def get_g(self, g: NDArray[float]) -> Namespace[NDArray[float]]:
        """Get the blocks of the electron-boson coupling matrix.

        This matrix corresponds to the bosonic annihilation operator.

        Args:
            g: Electron-boson coupling matrix.

        Returns:
            Blocks of the electron-boson coupling matrix.
        """
        # TODO make a proper class for this
        slices = {
            "x": self.space.correlated,
            "o": self.space.correlated_occupied,
            "v": self.space.correlated_virtual,
            "O": self.space.active_occupied,
            "V": self.space.active_virtual,
            "i": self.space.inactive_occupied,
            "a": self.space.inactive_virtual,
        }

        class Blocks(util.Namespace):
            def __getitem__(selffer, key):
                assert key[0] == "b"
                i = slices[key[1]]
                j = slices[key[2]]
                return g[:, i][:, :, j].copy()

            __getattr__ = __getitem__

        return Blocks()

    @property
    def bare_fock(self) -> NDArray[float]:
        """Get the mean-field Fock matrix in the MO basis, including frozen parts.

        Returns an array and not a `BaseFock` object.

        Returns:
            Mean-field Fock matrix.
        """
        fock_ao = self.mf.get_fock().astype(types[float])
        fock = util.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff, self.mo_coeff)
        return fock

    @property
    def xi(self) -> NDArray[float]:
        """Get the shift in the bosonic operators to diagonalise the photon Hamiltonian.

        Returns:
            Shift in the bosonic operators.
        """
        if self.options.shift:
            xi = lib.einsum("Iii->I", self.g.boo)
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros_like(self.omega)
        return xi

    def get_fock(self) -> GFock:
        """Get the Fock matrix.

        Returns:
            Fock matrix.
        """
        return self.Fock(self, array=self.bare_fock)

    def get_eris(self, eris: Optional[ERIsInputType] = None) -> GERIs:
        """Get the electron repulsion integrals.

        Args:
            eris: Input electron repulsion integrals.

        Returns:
            Electron repulsion integrals.
        """
        if (eris is None) or isinstance(eris, np.ndarray):
            return self.ERIs(self, array=eris)
        else:
            return eris

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

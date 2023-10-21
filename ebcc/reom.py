"""Restricted equation-of-motion solver."""

import dataclasses
import warnings

from pyscf import lib

from ebcc import numpy as np
from ebcc import util
from ebcc.precision import types


class EOM:
    """Base class for equation-of-motion methods."""

    pass


@dataclasses.dataclass
class Options:
    """
    Options for EOM calculations.

    Attributes
    ----------
    nroots : int, optional
        Number of roots to solve for. Default value is `5`.
    e_tol : float, optional
        Threshold for convergence in the correlation energy. Default value
        is inherited from `self.ebcc.options`.
    max_iter : int, optional
        Maximum number of iterations. Default value is inherited from
        `self.ebcc.options`.
    max_space : int, optional
        Maximum size of Lanczos vector space. Default value is `12`.
    """

    nroots: int = 5
    koopmans: bool = False
    e_tol: float = util.Inherited
    max_iter: int = util.Inherited
    max_space: int = 12


class REOM(EOM):
    """Restricted equation-of-motion base class."""

    Options = Options

    def __init__(self, ebcc, options=None, **kwargs):
        self.ebcc = ebcc
        self.space = ebcc.space
        self.ansatz = ebcc.ansatz
        self.log = ebcc.log

        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)
        for key, val in self.options.__dict__.items():
            if val is util.Inherited:
                setattr(self.options, key, getattr(self.ebcc.options, key))

        self.log.info("%s", self.name)
        self.log.info("%s", "*" * len(self.name))
        self.log.debug("")
        self.log.debug("Options:")
        self.log.info(" > nroots:     %s", self.options.nroots)
        self.log.info(" > e_tol:      %s", self.options.e_tol)
        self.log.info(" > max_iter:   %s", self.options.max_iter)
        self.log.info(" > max_space:  %s", self.options.max_space)
        self.log.debug("")

        self.converged = False
        self.e = None
        self.v = None

    def amplitudes_to_vector(self, *amplitudes):
        """Convert the amplitudes to a vector."""
        raise NotImplementedError

    def vector_to_amplitudes(self, vector):
        """Convert the vector to amplitudes."""
        raise NotImplementedError

    def matvec(self, vector, eris=None):
        """Apply the EOM Hamiltonian to a vector."""
        raise NotImplementedError

    def diag(self, eris=None):
        """Find the diagonal of the EOM Hamiltonian."""
        raise NotImplementedError

    def bras(self, eris=None):
        """Construct the bra vectors."""
        raise NotImplementedError

    def kets(self, eris=None):
        """Construct the ket vectors."""
        raise NotImplementedError

    def dot_braket(self, bra, ket):
        """Find the dot-product between the bra and the ket."""
        return np.dot(bra, ket)

    def get_pick(self, guesses=None, real_system=True):
        """Pick eigenvalues which match the criterion."""

        if self.options.koopmans:
            assert guesses is not None
            g = np.asarray(guesses)

            def pick(w, v, nroots, envs):
                x0 = lib.linalg_helper._gen_x0(envs["v"], envs["xs"])
                x0 = np.asarray(x0)
                s = np.dot(g.conj(), x0.T)
                s = np.einsum("pi,qi->i", s.conj(), s)
                idx = np.argsort(-s)[:nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)

        else:

            def pick(w, v, nroots, envs):
                real_idx = np.where(abs(w.imag) < 1e-3)[0]
                w, v, idx = lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
                mask = np.argsort(np.abs(w))
                w, v = w[mask], v[:, mask]
                return w, v, 0

        return pick

    def _argsort_guess(self, diag):
        if self.options.koopmans:
            r_mf = self.vector_to_amplitudes(diag)[0]
            size = r_mf.size
            arg = np.argsort(np.diag(diag[:size]))
        else:
            arg = np.argsort(diag)

        return arg

    def get_guesses(self, diag=None):
        """Generate guess vectors."""

        if diag is None:
            diag = self.diag()

        arg = self._argsort_guess(diag)

        nroots = min(self.options.nroots, diag.size)
        guesses = np.zeros((nroots, diag.size), dtype=diag.dtype)
        for root, guess in enumerate(arg[:nroots]):
            guesses[root, guess] = 1.0

        return list(guesses)

    def callback(self, envs):
        """Callback function for the Davidson solver."""  # noqa: D401

        pass

    def _quasiparticle_weight(self, r1):
        return np.linalg.norm(r1) ** 2

    def davidson(self, guesses=None):
        """Solve the EOM Hamiltonian using the Davidson solver."""

        self.log.output(
            "Solving for %s excitations using the Davidson solver.", self.excitation_type.upper()
        )

        eris = self.ebcc.get_eris()
        matvecs = lambda vs: [self.matvec(v, eris=eris) for v in vs]
        diag = self.diag(eris=eris)

        if guesses is None:
            guesses = self.get_guesses(diag=diag)

        nroots = min(len(guesses), self.options.nroots)
        pick = self.get_pick(guesses=guesses)

        converged, e, v = lib.davidson_nosym1(
            matvecs,
            guesses,
            diag,
            tol=self.options.e_tol,
            nroots=nroots,
            pick=pick,
            max_cycle=self.options.max_iter,
            max_space=self.options.max_space,
            callback=self.callback,
            verbose=0,
        )

        if all(converged):
            self.log.output("Converged.")
        else:
            self.log.warning("Failed to converge %d roots." % sum(not c for c in converged))

        self.log.debug("")

        self.log.output("%4s %16s %16s", "Root", "Energy", "QP Weight")
        for n, (en, vn) in enumerate(zip(e, v)):
            r1n = self.vector_to_amplitudes(vn)[0]
            qpwt = self._quasiparticle_weight(r1n)
            self.log.output("%4d %16.10f %16.5g" % (n, en, qpwt))

        self.log.debug("")

        self.converged = converged
        self.e = e
        self.v = v

        return e

    kernel = davidson

    def moments(self, nmom, eris=None, amplitudes=None, hermitise=True):
        """Construct the moments of the EOM Hamiltonian."""

        if eris is None:
            eris = self.ebcc.get_eris()
        if amplitudes is None:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments = np.zeros((nmom, self.nmo, self.nmo), dtype=types[float])

        for j in range(self.nmo):
            ket = kets[j]
            for n in range(nmom):
                for i in range(self.nmo):
                    bra = bras[i]
                    moments[n, i, j] = self.dot_braket(bra, ket)
                if n != (nmom - 1):
                    ket = self.matvec(ket, eris=eris)

        if hermitise:
            moments = 0.5 * (moments + moments.swapaxes(1, 2))

        return moments

    @property
    def name(self):
        """Get a string representation of the method name."""
        return f"{self.excitation_type.upper()}-{self.spin_type}EOM-{self.ebcc.name}"

    @property
    def spin_type(self):
        """Get a string representation of the spin channel."""
        return self.ebcc.spin_type

    @property
    def excitation_type(self):
        """Get a string representation of the excitation type."""
        raise NotImplementedError

    @property
    def nmo(self):
        """Get the number of MOs."""
        return self.ebcc.nmo

    @property
    def nocc(self):
        """Get the number of occupied MOs."""
        return self.ebcc.nocc

    @property
    def nvir(self):
        """Get the number of virtual MOs."""
        return self.ebcc.nvir


class IP_REOM(REOM, metaclass=util.InheritDocstrings):
    """Restricted equation-of-motion class for ionisation potentials."""

    @util.has_docstring
    def amplitudes_to_vector(self, *amplitudes):
        return self.ebcc.excitations_to_vector_ip(*amplitudes)

    @util.has_docstring
    def vector_to_amplitudes(self, vector):
        return self.ebcc.vector_to_excitations_ip(vector)

    @util.has_docstring
    def matvec(self, vector, eris=None):
        amplitudes = self.vector_to_amplitudes(vector)
        amplitudes = self.ebcc.hbar_matvec_ip(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*amplitudes)

    @util.has_docstring
    def diag(self, eris=None):
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            parts.append(self.ebcc.energy_sum(key))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def _bras(self, eris=None):
        return self.ebcc.make_ip_mom_bras(eris=eris)

    def _kets(self, eris=None):
        return self.ebcc.make_ip_mom_kets(eris=eris)

    @util.has_docstring
    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    @util.has_docstring
    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[b[..., i] for b in kets_raw]) for i in range(self.nmo)]
        )
        return kets

    @property
    @util.has_docstring
    def excitation_type(self):
        return "ip"


class EA_REOM(REOM, metaclass=util.InheritDocstrings):
    """Equation-of-motion class for electron affinities."""

    @util.has_docstring
    def amplitudes_to_vector(self, *amplitudes):
        return self.ebcc.excitations_to_vector_ea(*amplitudes)

    @util.has_docstring
    def vector_to_amplitudes(self, vector):
        return self.ebcc.vector_to_excitations_ea(vector)

    @util.has_docstring
    def matvec(self, vector, eris=None):
        amplitudes = self.vector_to_amplitudes(vector)
        amplitudes = self.ebcc.hbar_matvec_ea(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*amplitudes)

    @util.has_docstring
    def diag(self, eris=None):
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
            parts.append(-self.ebcc.energy_sum(key))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def _bras(self, eris=None):
        return self.ebcc.make_ea_mom_bras(eris=eris)

    def _kets(self, eris=None):
        return self.ebcc.make_ea_mom_kets(eris=eris)

    @util.has_docstring
    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    @util.has_docstring
    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[b[..., i] for b in kets_raw]) for i in range(self.nmo)]
        )
        return kets

    @property
    @util.has_docstring
    def excitation_type(self):
        return "ea"


class EE_REOM(REOM, metaclass=util.InheritDocstrings):
    """Equation-of-motion class for neutral excitations."""

    @util.has_docstring
    def amplitudes_to_vector(self, *amplitudes):
        return self.ebcc.excitations_to_vector_ee(*amplitudes)

    @util.has_docstring
    def vector_to_amplitudes(self, vector):
        return self.ebcc.vector_to_excitations_ee(vector)

    @util.has_docstring
    def matvec(self, vector, eris=None):
        amplitudes = self.vector_to_amplitudes(vector)
        amplitudes = self.ebcc.hbar_matvec_ee(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*amplitudes)

    @util.has_docstring
    def diag(self, eris=None):
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            parts.append(-self.ebcc.energy_sum(key))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    @util.has_docstring
    def bras(self, eris=None):
        bras_raw = list(self.ebcc.make_ee_mom_bras(eris=eris))
        bras = np.array(
            [
                [self.amplitudes_to_vector(*[b[i, j] for b in bras_raw]) for j in range(self.nmo)]
                for i in range(self.nmo)
            ]
        )
        return bras

    @util.has_docstring
    def kets(self, eris=None):
        kets_raw = list(self.ebcc.make_ee_mom_kets(eris=eris))
        kets = np.array(
            [
                [
                    self.amplitudes_to_vector(*[k[..., i, j] for k in kets_raw])
                    for j in range(self.nmo)
                ]
                for i in range(self.nmo)
            ]
        )
        return kets

    @util.has_docstring
    def moments(self, nmom, eris=None, amplitudes=None, hermitise=True, diagonal_only=True):
        """Construct the moments of the EOM Hamiltonian."""

        if not diagonal_only:
            warnings.warn(
                "Constructing EE moments with `diagonal_only=False` will be very slow.",
                stacklevel=2,
            )

        if eris is None:
            eris = self.ebcc.get_eris()
        if amplitudes is None:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments = np.zeros((nmom, self.nmo, self.nmo, self.nmo, self.nmo), dtype=types[float])

        for k in range(self.nmo):
            for l in [k] if diagonal_only else range(self.nmo):
                ket = kets[k, l]
                for n in range(nmom):
                    for i in range(self.nmo):
                        for j in [i] if diagonal_only else range(self.nmo):
                            bra = bras[i, j]
                            moments[n, i, j, k, l] = self.dot_braket(bra, ket)
                    if n != (nmom - 1):
                        ket = self.matvec(ket, eris=eris)

        if hermitise:
            moments = 0.5 * (moments + moments.transpose(0, 3, 4, 1, 2))

        return moments

    @property
    @util.has_docstring
    def excitation_type(self):
        return "ee"

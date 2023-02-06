"""Restricted equation-of-motion solver.
"""

import dataclasses
import functools
import warnings

import numpy as np
from pyscf import lib

from ebcc import util

# TODO split into reom, ueom, geom
# TODO docstrings


@dataclasses.dataclass
class Options:
    """Options for EOM calculations.

    Attributes
    ----------
    nroots : int, optional
        Number of roots to solve for. Default value is 5.
    e_tol : float, optional
        Threshold for convergence in the correlation energy. Default
        value is inherited from `self.ebcc.options`.
    max_iter : int, optional
        Maximum number of iterations. Default value is inherited from
        `self.ebcc.options`.
    max_space : int, optional
        Maximum size of Lanczos vector space. Default value is 12.
    """

    nroots: int = 5
    koopmans: bool = False
    e_tol: float = util.Inherited
    max_iter: int = util.Inherited
    max_space: int = 12


class REOM:
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
        raise NotImplementedError

    def vector_to_amplitudes(self, vector):
        raise NotImplementedError

    def matvec(self, vector, eris=None):
        raise NotImplementedError

    def diag(self, eris=None):
        raise NotImplementedError

    def bras(self, eris=None):
        raise NotImplementedError

    def kets(self, eris=None):
        raise NotImplementedError

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
        guesses = np.zeros((nroots, diag.size))
        for root, guess in enumerate(arg[:nroots]):
            guesses[root, guess] = 1.0

        return list(guesses)

    def callback(self, envs):
        """Callback function between iterations."""

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

        moments = np.zeros((nmom, self.nmo, self.nmo))

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
        return self.excitation_type.upper() + "-REOM-" + self.ebcc.name

    @property
    def excitation_type(self):
        raise NotImplementedError

    @property
    def nmo(self):
        return self.ebcc.nmo

    @property
    def nocc(self):
        return self.ebcc.nocc

    @property
    def nvir(self):
        return self.ebcc.nvir


@util.inherit_docstrings
class IP_REOM(REOM):
    """Restricted equation-of-motion class for ionisation potentials."""

    def amplitudes_to_vector(self, *amplitudes):
        return self.ebcc.excitations_to_vector_ip(*amplitudes)

    def vector_to_amplitudes(self, vector):
        return self.ebcc.vector_to_excitations_ip(vector)

    def matvec(self, vector, eris=None):
        amplitudes = self.vector_to_amplitudes(vector)
        amplitudes = self.ebcc.hbar_matvec_ip(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*amplitudes)

    def diag(self, eris=None):
        parts = []
        e_ia = lib.direct_sum("i-a->ia", self.ebcc.eo, self.ebcc.ev)
        e_i = self.ebcc.eo

        for n in self.ansatz.correlated_cluster_ranks[0]:
            perm = list(range(0, n * 2, 2)) + list(range(1, (n - 1) * 2, 2))
            d = functools.reduce(np.add.outer, [e_ia] * (n - 1) + [e_i])
            d = d.transpose(perm)
            parts.append(d)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def _bras(self, eris=None):
        return self.ebcc.make_ip_mom_bras(eris=eris)

    def _kets(self, eris=None):
        return self.ebcc.make_ip_mom_kets(eris=eris)

    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[b[..., i] for b in kets_raw]) for i in range(self.nmo)]
        )
        return kets

    def dot_braket(self, bra, ket):
        return np.dot(bra, ket)

    @property
    def excitation_type(self):
        return "ip"


@util.inherit_docstrings
class EA_REOM(REOM):
    """Equation-of-motion class for electron affinities."""

    def amplitudes_to_vector(self, *amplitudes):
        return self.ebcc.excitations_to_vector_ea(*amplitudes)

    def vector_to_amplitudes(self, vector):
        return self.ebcc.vector_to_excitations_ea(vector)

    def matvec(self, vector, eris=None):
        amplitudes = self.vector_to_amplitudes(vector)
        amplitudes = self.ebcc.hbar_matvec_ea(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*amplitudes)

    def diag(self, eris=None):
        parts = []
        e_ai = lib.direct_sum("a-i->ai", self.ebcc.ev, self.ebcc.eo)
        e_a = self.ebcc.ev

        for n in self.ansatz.correlated_cluster_ranks[0]:
            perm = list(range(0, n * 2, 2)) + list(range(1, (n - 1) * 2, 2))
            d = functools.reduce(np.add.outer, [e_ai] * (n - 1) + [e_a])
            d = d.transpose(perm)
            parts.append(d)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def _bras(self, eris=None):
        return self.ebcc.make_ea_mom_bras(eris=eris)

    def _kets(self, eris=None):
        return self.ebcc.make_ea_mom_kets(eris=eris)

    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[b[..., i] for b in kets_raw]) for i in range(self.nmo)]
        )
        return kets

    def dot_braket(self, bra, ket):
        return np.dot(bra, ket)

    @property
    def excitation_type(self):
        return "ea"


@util.inherit_docstrings
class EE_REOM(REOM):
    """Equation-of-motion class for neutral excitations."""

    def amplitudes_to_vector(self, *amplitudes):
        return self.ebcc.excitations_to_vector_ee(*amplitudes)

    def vector_to_amplitudes(self, vector):
        return self.ebcc.vector_to_excitations_ee(vector)

    def matvec(self, vector, eris=None):
        amplitudes = self.vector_to_amplitudes(vector)
        amplitudes = self.ebcc.hbar_matvec_ee(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*amplitudes)

    def diag(self, eris=None):
        parts = []
        e_ia = lib.direct_sum("a-i->ia", self.ebcc.ev, self.ebcc.eo)

        for n in self.ansatz.correlated_cluster_ranks[0]:
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
            d = functools.reduce(np.add.outer, [e_ia] * n)
            d = d.transpose(perm)
            parts.append(d)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def bras(self, eris=None):
        bras_raw = list(self.ebcc.make_ee_mom_bras(eris=eris))
        bras = np.array(
            [
                [self.amplitudes_to_vector(*[b[i, j] for b in bras_raw]) for j in range(self.nmo)]
                for i in range(self.nmo)
            ]
        )
        return bras

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

    def dot_braket(self, bra, ket):
        return np.dot(bra, ket)

    def moments(self, nmom, eris=None, amplitudes=None, hermitise=True, diagonal_only=True):
        """Construct the moments of the EOM Hamiltonian."""

        if not diagonal_only:
            warnings.warn("Constructing EE moments with `diagonal_only=False` will be very slow.")

        if eris is None:
            eris = self.ebcc.get_eris()
        if amplitudes is None:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments = np.zeros((nmom, self.nmo, self.nmo, self.nmo, self.nmo))

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
    def excitation_type(self):
        return "ee"

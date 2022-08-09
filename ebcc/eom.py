"""Equation-of-motion solver.
"""

import dataclasses

import numpy as np
from pyscf import lib

from ebcc import util

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
    e_tol: float = util.Inherited
    max_iter: int = util.Inherited
    max_space: int = 12


class EOM:
    """Equation-of-motion base class."""

    def __init__(self, ebcc, options=None, **kwargs):
        self.ebcc = ebcc
        self.log = ebcc.log

        if options is None:
            options = Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)
        for key, val in self.options.__dict__.items():
            if val is util.Inherited:
                setattr(self.options, key, getattr(self.ebcc.options, key))

        self.log.info("%s", self.name)
        self.log.info("%s", "*" * len(self.name))
        self.log.info(" > nroots:     %s", self.options.nroots)
        self.log.info(" > e_tol:      %s", self.options.e_tol)
        self.log.info(" > max_iter:   %s", self.options.max_iter)
        self.log.info(" > max_space:  %s", self.options.max_space)

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

    def get_pick(self, guesses=None, koopmans=False, real_system=True):
        """Pick eigenvalues which match the criterion."""

        if koopmans:
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
                return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)

        return pick

    def get_guesses(self, diag=None, use_mean_field=False):
        """Generate guess vectors."""

        if diag is None:
            diag = self.diag()

        if use_mean_field:
            r_mf = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(np.diag(diag[: r_mf.size]))
        else:
            arg = np.argsort(np.abs(diag))

        nroots = min(self.options.nroots, diag.size)
        guesses = np.zeros((nroots, diag.size))
        for root, guess in enumerate(arg[:nroots]):
            guesses[root, guess] = 1.0

        return list(guesses)

    def callback(self, envs):
        """Callback function between iterations."""

        pass

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
            nroots=self.options.nroots,
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

        self.log.output("%4s %16s %16s", "Root", "Energy", "QP Weight")
        for n, (en, vn) in enumerate(zip(e, v)):
            r1n = self.vector_to_amplitudes(vn)[0]
            qpwt = np.linalg.norm(r1n) ** 2
            self.log.output("%4d %16.10f %16.5g" % (n, en, qpwt))

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

        bras = list(self.bras(eris=eris))
        kets = list(self.kets(eris=eris))

        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras]) for i in range(self.ebcc.nmo)]
        )
        kets = np.array(
            [self.amplitudes_to_vector(*[k[..., i] for k in kets]) for i in range(self.ebcc.nmo)]
        )

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
        return self.excitation_type.upper() + "-EOM-" + self.ebcc.name

    @property
    def excitation_type(self):
        raise NotImplementedError

    @property
    def nmo(self):
        return self.ebcc.nmo


class IP_EOM(EOM):
    """Equation-of-motion class for ionisation potentials."""

    def amplitudes_to_vector(self, *amplitudes):
        return self.ebcc.excitations_to_vector_ip(*amplitudes)

    def vector_to_amplitudes(self, vector):
        return self.ebcc.vector_to_excitations_ip(vector)

    def matvec(self, vector, eris=None):
        amplitudes = self.vector_to_amplitudes(vector)
        amplitudes = self.ebcc.hbar_matvec_ip(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*amplitudes)

    def diag(self, eris=None):
        return self.amplitudes_to_vector(*self.ebcc.hbar_diag_ip(eris=eris))

    def hbar(self, eris=None):
        return self.ebcc.hbar_ip(eris=eris)

    def bras(self, eris=None):
        return self.ebcc.make_ip_mom_bras(eris=eris)

    def kets(self, eris=None):
        return self.ebcc.make_ip_mom_kets(eris=eris)

    def dot_braket(self, bra, ket):
        # TODO generalise
        b1, b2 = self.vector_to_amplitudes(bra)
        k1, k2 = self.vector_to_amplitudes(ket)
        # TODO move factor to bra
        fac = 0.5 if self.ebcc.name.startswith("G") else 1.0
        out = 1.0 * np.dot(b1, k1) + fac * np.einsum("ija,ija->", b2, k2)
        return out

    @property
    def excitation_type(self):
        return "ip"


class EA_EOM(EOM):
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
        return self.amplitudes_to_vector(*self.ebcc.hbar_diag_ea(eris=eris))

    def hbar(self, eris=None):
        return self.ebcc.hbar_ea(eris=eris)

    def bras(self, eris=None):
        return self.ebcc.make_ea_mom_bras(eris=eris)

    def kets(self, eris=None):
        return self.ebcc.make_ea_mom_kets(eris=eris)

    def dot_braket(self, bra, ket):
        # TODO generalise
        b1, b2 = self.vector_to_amplitudes(bra)
        k1, k2 = self.vector_to_amplitudes(ket)
        # TODO move factor to bra
        fac = 0.5 if self.ebcc.name.startswith("G") else 1.0
        out = +1.0 * np.dot(b1, k1) + fac * np.einsum("abi,abi->", b2, k2)
        return out

    @property
    def excitation_type(self):
        return "ea"


class EE_EOM(EOM):
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
        return self.amplitudes_to_vector(*self.ebcc.hbar_diag_ee(eris=eris))

    def hbar(self, eris=None):
        return self.ebcc.hbar_ee(eris=eris)

    def bras(self, eris=None):
        return self.ebcc.make_ee_mom_bras(eris=eris)

    def kets(self, eris=None):
        return self.ebcc.make_ee_mom_kets(eris=eris)

    @property
    def excitation_type(self):
        return "ee"

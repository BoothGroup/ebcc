"""Equation-of-motion solver.
"""

import dataclasses
import warnings
from types import SimpleNamespace

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
            size = (r_mf.a.size + r_mf.b.size) if self.ebcc.name.startswith("U") else r_mf.size
            arg = np.argsort(np.diag(diag[: size]))
        else:
            arg = np.argsort(diag)

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

        self.log.output("%4s %16s %16s", "Root", "Energy", "QP Weight")
        for n, (en, vn) in enumerate(zip(e, v)):
            r1n = self.vector_to_amplitudes(vn)[0]
            if isinstance(r1n, SimpleNamespace):
                r1n = 0.5 * (r1n.a + r1n.b)
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

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        if not self.ebcc.name.startswith("U"):
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

        else:
            moments = SimpleNamespace(
                    aa=np.zeros((nmom, self.nmo, self.nmo)),
                    bb=np.zeros((nmom, self.nmo, self.nmo)),
            )

            for spin in util.generate_spin_combinations(1):
                for j in range(self.nmo):
                    ket = getattr(kets, spin[1])[j]
                    for n in range(nmom):
                        for i in range(self.nmo):
                            bra = getattr(bras, spin[0])[i]
                            getattr(moments, spin)[n, i, j] = self.dot_braket(bra, ket)
                        if n != (nmom - 1):
                            ket = self.matvec(ket, eris=eris)

                if hermitise:
                    t = getattr(moments, spin)
                    setattr(moments, spin, 0.5 * (t + t.swapaxes(1, 2)))

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

    @property
    def nocc(self):
        return self.ebcc.nocc

    @property
    def nvir(self):
        return self.ebcc.nvir


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

    def _bras(self, eris=None):
        return self.ebcc.make_ip_mom_bras(eris=eris)

    def _kets(self, eris=None):
        return self.ebcc.make_ip_mom_kets(eris=eris)

    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        if not self.ebcc.name.startswith("U"):
            bras = np.array(
                    [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
            )
        else:
            bras = SimpleNamespace(a=[], b=[])

            for i in range(self.nmo):
                amps_a = []
                amps_b = []

                m = 0
                for n in self.ebcc.rank_numeric[0]:
                    amp_a = SimpleNamespace()
                    amp_b = SimpleNamespace()
                    for spin in util.generate_spin_combinations(n, excited=True):
                        shape = tuple(self.nocc["ab".index(s)] for s in spin[:n]) + tuple(self.nvir["ab".index(s)] for s in spin[n:])
                        setattr(amp_a, spin, getattr(bras_raw[m], "a"+spin, {i: np.zeros(shape)})[i])
                        setattr(amp_b, spin, getattr(bras_raw[m], "b"+spin, {i: np.zeros(shape)})[i])
                    amps_a.append(amp_a)
                    amps_b.append(amp_b)
                    m += 1

                for n in self.ebcc.rank_numeric[1]:
                    raise NotImplementedError

                for nf in self.ebcc.rank_numeric[2]:
                    for nb in self.ebcc.rank_numeric[3]:
                        raise NotImplementedError

                bras.a.append(self.amplitudes_to_vector(*amps_a))
                bras.b.append(self.amplitudes_to_vector(*amps_b))

            bras.a = np.array(bras.a)
            bras.b = np.array(bras.b)

        return bras

    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        if not self.ebcc.name.startswith("U"):
            kets = np.array(
                    [self.amplitudes_to_vector(*[b[..., i] for b in kets_raw]) for i in range(self.nmo)]
            )
        else:
            kets = SimpleNamespace(a=[], b=[])

            for i in range(self.nmo):
                j = (Ellipsis, i)
                amps_a = []
                amps_b = []

                m = 0
                for n in self.ebcc.rank_numeric[0]:
                    amp_a = SimpleNamespace()
                    amp_b = SimpleNamespace()
                    for spin in util.generate_spin_combinations(n, excited=True):
                        shape = tuple(self.nocc["ab".index(s)] for s in spin[:n]) + tuple(self.nvir["ab".index(s)] for s in spin[n:])
                        setattr(amp_a, spin, getattr(kets_raw[m], spin+"a", {j: np.zeros(shape)})[j])
                        setattr(amp_b, spin, getattr(kets_raw[m], spin+"b", {j: np.zeros(shape)})[j])
                    amps_a.append(amp_a)
                    amps_b.append(amp_b)
                    m += 1

                for n in self.ebcc.rank_numeric[1]:
                    raise NotImplementedError

                for nf in self.ebcc.rank_numeric[2]:
                    for nb in self.ebcc.rank_numeric[3]:
                        raise NotImplementedError

                kets.a.append(self.amplitudes_to_vector(*amps_a))
                kets.b.append(self.amplitudes_to_vector(*amps_b))

            kets.a = np.array(kets.a)
            kets.b = np.array(kets.b)

        return kets

    def dot_braket(self, bra, ket):
        return np.dot(bra, ket)

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

    def _bras(self, eris=None):
        return self.ebcc.make_ea_mom_bras(eris=eris)

    def _kets(self, eris=None):
        return self.ebcc.make_ea_mom_kets(eris=eris)

    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        if not self.ebcc.name.startswith("U"):
            bras = np.array(
                    [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
            )
        else:
            bras = SimpleNamespace(a=[], b=[])

            for i in range(self.nmo):
                amps_a = []
                amps_b = []

                m = 0
                for n in self.ebcc.rank_numeric[0]:
                    amp_a = SimpleNamespace()
                    amp_b = SimpleNamespace()
                    for spin in util.generate_spin_combinations(n, excited=True):
                        shape = tuple(self.nvir["ab".index(s)] for s in spin[:n]) + tuple(self.nocc["ab".index(s)] for s in spin[n:])
                        setattr(amp_a, spin, getattr(bras_raw[m], "a"+spin, {i: np.zeros(shape)})[i])
                        setattr(amp_b, spin, getattr(bras_raw[m], "b"+spin, {i: np.zeros(shape)})[i])
                    amps_a.append(amp_a)
                    amps_b.append(amp_b)
                    m += 1

                for n in self.ebcc.rank_numeric[1]:
                    raise NotImplementedError

                for nf in self.ebcc.rank_numeric[2]:
                    for nb in self.ebcc.rank_numeric[3]:
                        raise NotImplementedError

                bras.a.append(self.amplitudes_to_vector(*amps_a))
                bras.b.append(self.amplitudes_to_vector(*amps_b))

            bras.a = np.array(bras.a)
            bras.b = np.array(bras.b)

        return bras

    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        if not self.ebcc.name.startswith("U"):
            kets = np.array(
                    [self.amplitudes_to_vector(*[b[..., i] for b in kets_raw]) for i in range(self.nmo)]
            )
        else:
            kets = SimpleNamespace(a=[], b=[])

            for i in range(self.nmo):
                j = (Ellipsis, i)
                amps_a = []
                amps_b = []

                m = 0
                for n in self.ebcc.rank_numeric[0]:
                    amp_a = SimpleNamespace()
                    amp_b = SimpleNamespace()
                    for spin in util.generate_spin_combinations(n, excited=True):
                        shape = tuple(self.nvir["ab".index(s)] for s in spin[:n]) + tuple(self.nocc["ab".index(s)] for s in spin[n:])
                        setattr(amp_a, spin, getattr(kets_raw[m], spin+"a", {j: np.zeros(shape)})[j])
                        setattr(amp_b, spin, getattr(kets_raw[m], spin+"b", {j: np.zeros(shape)})[j])
                    amps_a.append(amp_a)
                    amps_b.append(amp_b)
                    m += 1

                for n in self.ebcc.rank_numeric[1]:
                    raise NotImplementedError

                for nf in self.ebcc.rank_numeric[2]:
                    for nb in self.ebcc.rank_numeric[3]:
                        raise NotImplementedError

                kets.a.append(self.amplitudes_to_vector(*amps_a))
                kets.b.append(self.amplitudes_to_vector(*amps_b))

            kets.a = np.array(kets.a)
            kets.b = np.array(kets.b)

        return kets

    def dot_braket(self, bra, ket):
        return np.dot(bra, ket)

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
        bras_raw = list(self.ebcc.make_ee_mom_bras(eris=eris))
        if not self.ebcc.name.startswith("U"):
            bras = np.array(
                [
                    [self.amplitudes_to_vector(*[b[i, j] for b in bras_raw])
                    for j in range(self.nmo)]
                    for i in range(self.nmo)
                ]
            )
        else:
            bras = SimpleNamespace(aa=[], bb=[])

            for i in range(self.nmo):
                for j in range(self.nmo):
                    amps_aa = []
                    amps_bb = []

                    m = 0
                    for n in self.ebcc.rank_numeric[0]:
                        amp_aa = SimpleNamespace()
                        amp_bb = SimpleNamespace()
                        for spin in util.generate_spin_combinations(n):
                            shape = tuple([
                                *[self.nocc["ab".index(s)] for s in spin[:n]],
                                *[self.nvir["ab".index(s)] for s in spin[n:]],
                            ])
                            setattr(amp_aa, spin, getattr(bras_raw[m], "aa"+spin, {(i, j): np.zeros(shape)})[i, j])
                            setattr(amp_bb, spin, getattr(bras_raw[m], "bb"+spin, {(i, j): np.zeros(shape)})[i, j])
                        amps_aa.append(amp_aa)
                        amps_bb.append(amp_bb)
                        m += 1

                    for n in self.ebcc.rank_numeric[1]:
                        raise NotImplementedError

                    for nf in self.ebcc.rank_numeric[2]:
                        for nb in self.ebcc.rank_numeric[3]:
                            raise NotImplementedError

                    bras.aa.append(self.amplitudes_to_vector(*amps_aa))
                    bras.bb.append(self.amplitudes_to_vector(*amps_bb))

            bras.aa = np.array(bras.aa)
            bras.bb = np.array(bras.bb)

            bras.aa = bras.aa.reshape(self.nmo, self.nmo, *bras.aa.shape[1:])
            bras.bb = bras.bb.reshape(self.nmo, self.nmo, *bras.bb.shape[1:])

        return bras

    def kets(self, eris=None):
        kets_raw = list(self.ebcc.make_ee_mom_kets(eris=eris))
        if not self.ebcc.name.startswith("U"):
            kets = np.array(
                [
                    [self.amplitudes_to_vector(*[k[..., i, j] for k in kets_raw])
                    for j in range(self.nmo)]
                    for i in range(self.nmo)
                ]
            )
        else:
            kets = SimpleNamespace(aa=[], bb=[])

            for i in range(self.nmo):
                for j in range(self.nmo):
                    k = (Ellipsis, i, j)
                    amps_aa = []
                    amps_bb = []

                    m = 0
                    for n in self.ebcc.rank_numeric[0]:
                        amp_aa = SimpleNamespace()
                        amp_bb = SimpleNamespace()
                        for spin in util.generate_spin_combinations(n):
                            shape = tuple([
                                *[self.nocc["ab".index(s)] for s in spin[:n]],
                                *[self.nvir["ab".index(s)] for s in spin[n:]],
                            ])
                            setattr(amp_aa, spin, getattr(kets_raw[m], spin+"aa", {k: np.zeros(shape)})[k])
                            setattr(amp_bb, spin, getattr(kets_raw[m], spin+"bb", {k: np.zeros(shape)})[k])
                        amps_aa.append(amp_aa)
                        amps_bb.append(amp_bb)
                        m += 1

                    for n in self.ebcc.rank_numeric[1]:
                        raise NotImplementedError

                    for nf in self.ebcc.rank_numeric[2]:
                        for nb in self.ebcc.rank_numeric[3]:
                            raise NotImplementedError

                    kets.aa.append(self.amplitudes_to_vector(*amps_aa))
                    kets.bb.append(self.amplitudes_to_vector(*amps_bb))

            kets.aa = np.array(kets.aa)
            kets.bb = np.array(kets.bb)

            kets.aa = kets.aa.reshape(self.nmo, self.nmo, *kets.aa.shape[1:])
            kets.bb = kets.bb.reshape(self.nmo, self.nmo, *kets.bb.shape[1:])

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

        if not self.ebcc.name.startswith("U"):
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

        else:
            moments = SimpleNamespace(
                    aaaa=np.zeros((nmom, self.nmo, self.nmo, self.nmo, self.nmo)),
                    aabb=np.zeros((nmom, self.nmo, self.nmo, self.nmo, self.nmo)),
                    bbaa=np.zeros((nmom, self.nmo, self.nmo, self.nmo, self.nmo)),
                    bbbb=np.zeros((nmom, self.nmo, self.nmo, self.nmo, self.nmo)),
            )

            for spin in util.generate_spin_combinations(2):
                spin = util.permute_string(spin, (0, 2, 1, 3))
                for k in range(self.nmo):
                    for l in [k] if diagonal_only else range(self.nmo):
                        ket = getattr(kets, spin[2:])[k, l]
                        for n in range(nmom):
                            for i in range(self.nmo):
                                for j in [i] if diagonal_only else range(self.nmo):
                                    bra = getattr(bras, spin[:2])[i, j]
                                    getattr(moments, spin)[n, i, j, k, l] = self.dot_braket(bra, ket)
                            if n != (nmom - 1):
                                ket = self.matvec(ket, eris=eris)

        return moments

    @property
    def excitation_type(self):
        return "ee"

"""Restricted electron-boson coupled cluster.
"""

import sys
import logging
import functools
import importlib
import dataclasses
from typing import Tuple
from types import SimpleNamespace
import numpy as np
from pyscf import lib, ao2mo
from ebcc import util

# TODO generalise dispatch of functions from self._eqns
# TODO rename ranks from i.e. 0, 1, 2 -> "", "S", "SD"
# TODO interface to GF solver
# TODO dd moments
# NOTE CCSD-21 lambdas don't seem to work


@dataclasses.dataclass
class Options:
    """Options for EBCC calculations.
    """

    rank: Tuple[int] = ("SD", "", "")
    omega: np.ndarray = None
    g: np.ndarray = None
    G: np.ndarray = None
    shift: bool = True
    e_tol: float = 1e-8
    t_tol: float = 1e-8
    max_iter: int = 200
    diis_space: int = 20


class REBCC:
    Options = Options

    def __init__(
            self,
            mf,
            log: logging.Logger = None,
            rank: Tuple[int] = ("SD", "", ""),
            omega: np.ndarray = None,
            g: np.ndarray = None,
            G: np.ndarray = None,
            options: Options = None,
            **kwargs,
    ):
        self.log = self._get_log(log)
        self.mf = self._convert_mf(mf)
        self.rank = rank
        self._eqns = self._get_eqns()

        self.log.info("%s", self.name)
        self.log.info("%s", "*" * len(self.name))
        self.log.info(" > Fermion rank:   %s", self.rank[0])
        self.log.info(" > Boson rank:     %s", self.rank[1] if len(self.rank[1]) else None)
        self.log.info(" > Coupling rank:  %s", self.rank[2] if len(self.rank[2]) else None)

        self.omega = omega
        self.bare_G = G

        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)

        self.e_corr = None
        self.amplitudes = None
        self.converged = False
        self.lambdas = None
        self.converged_lambda = False

        if not (self.rank[1] == self.rank[2] == ""):
            assert self.omega is not None
            assert g.shape == (self.nbos, self.nmo, self.nmo)

            self.g = self.get_g(g)
            self.G = self.get_mean_field_G()

            if self.options.shift:
                self.log.info(" > Energy shift due to polaritonic basis:  %.10f", self.const)
        else:
            assert self.nbos == 0
            self.options.shift = False
            self.g = None
            self.G = None

        self.fock = self.get_fock()

        self.log.info(" > nmo:   %d", self.nmo)
        self.log.info(" > nocc:  %d", self.nocc)
        self.log.info(" > nvir:  %d", self.nvir)
        self.log.info(" > nbos:  %d", self.nbos)
        self.log.info(" > e_tol:  %s", self.options.e_tol)
        self.log.info(" > t_tol:  %s", self.options.t_tol)
        self.log.info(" > max_iter:  %s", self.options.max_iter)
        self.log.info(" > diis_space:  %s", self.options.diis_space)

    @staticmethod
    def _get_log(log):
        if log is None:
            log = logging.getLogger(__name__)
            log.setLevel(logging.INFO)
            errh = logging.StreamHandler(sys.stderr)
            log.addHandler(errh)

        logging.addLevelName(25, "OUTPUT")

        def output(self, msg, *args, **kwargs):
            if self.isEnabledFor(25):
                self._log(25, msg, args, **kwargs)

        logging.Logger.output = output

        return log

    @staticmethod
    def _convert_mf(mf):
        return mf

    def _get_eqns(self):
        name = self.name.replace("-", "_")
        eqns = importlib.import_module("ebcc.codegen.%s" % name)
        return eqns

    def init_amps(self, eris=None):
        """Initialise amplitudes.
        """

        if eris is None:
            eris = self.get_eris()

        amplitudes = dict()
        e_ia = lib.direct_sum("i-a->ia", self.eo, self.ev)

        # Build T amplitudes:
        for n in self.rank_numeric[0]:
            if n == 1:
                amplitudes["t%d" % n] = self.fock.vo.T / e_ia
            elif n == 2:
                e_ijab = lib.direct_sum("ia,jb->ijab", e_ia, e_ia)
                amplitudes["t%d" % n] = eris.ovov.swapaxes(1, 2) / e_ijab
            else:
                amplitudes["t%d" % n] = np.zeros((self.nocc,) * n + (self.nvir,) * n)

        if not (self.rank[1] == self.rank[2] == ""):
            # Only true for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for n in self.rank_numeric[1]:
            if n == 1:
                amplitudes["s%d" % n] = -H / self.omega
            else:
                amplitudes["s%d" % n] = np.zeros((self.nbos,) * n)

        # Build U amplitudes:
        for n in self.rank_numeric[2]:
            if n == 1:
                e_xia = lib.direct_sum("ia-x->xia", e_ia, self.omega)
                amplitudes["u1%d" % n] = h.bov / e_xia
            else:
                amplitudes["u1%d" % n] = np.zeros((self.nbos,) * n + (self.nocc, self.nvir))

        return amplitudes

    def init_lams(self, amplitudes=None):
        """Initialise lambda amplitudes.
        """

        if amplitudes is None:
            amplitudes = self.amplitudes

        lambdas = dict()

        # Build L amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(n, 2*n)) + list(range(n))
            lambdas["l%d" % n] = amplitudes["t%d" % n].transpose(perm)

        # Build LS amplitudes:
        for n in self.rank_numeric[1]:
            # FIXME should these be transposed?
            lambdas["ls%d" % n] = amplitudes["s%d" % n]

        # Build LU amplitudes:
        for n in self.rank_numeric[2]:
            perm = list(range(n)) + [n+1, n]
            lambdas["lu1%d" % n] = amplitudes["u1%d" % n].transpose(perm)

        return lambdas

    def kernel(self, eris=None):
        """Run calculation.
        """

        if eris is None:
            eris = self.get_eris()

        amplitudes = self.init_amps(eris=eris)
        e_cc = e_init = self.energy(amplitudes=amplitudes, eris=eris)
        converged = False

        diis = lib.diis.DIIS()
        diis.space = self.options.diis_space

        self.log.output("Solving for excitation amplitudes.")
        self.log.info("%4s %16s %16s %16s", "Iter", "Energy (corr.)", "Δ(Energy)", "Δ(Amplitudes)")
        self.log.info("%4d %16.10f", 0, e_init)

        for niter in range(1, self.options.max_iter+1):
            amplitudes_prev = amplitudes
            amplitudes = self.update_amps(amplitudes=amplitudes, eris=eris)
            vector = self.amplitudes_to_vector(amplitudes)
            vector = diis.update(vector)
            amplitudes = self.vector_to_amplitudes(vector)
            dt = np.linalg.norm(vector - self.amplitudes_to_vector(amplitudes_prev))**2

            e_prev = e_cc
            e_cc = self.energy(amplitudes=amplitudes, eris=eris)
            de = abs(e_prev - e_cc)

            self.log.info("%4d %16.10f %16.5g %16.5g", niter, e_cc, de, dt)

            converged = de < self.options.e_tol and dt < self.options.t_tol
            if converged:
                self.log.output("Converged.")
                break
        else:
            self.log.warning("Failed to converge.")

        self.e_corr = e_cc
        self.amplitudes = amplitudes
        self.converged = converged

        self.log.output("E(corr) = %.10f", self.e_corr)
        self.log.output("E(tot)  = %.10f", self.e_tot)

        return e_cc

    def solve_lambda(self, amplitudes=None, eris=None):
        """Solve lammbda equations.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)  # TODO warn?

        lambdas = self.init_lams(amplitudes=amplitudes)

        diis = lib.diis.DIIS()
        diis.space = self.options.diis_space

        self.log.output("Solving for de-excitation (lambda) amplitudes.")
        self.log.info("%4s %16s", "Iter", "Δ(Amplitudes)")

        for niter in range(1, self.options.max_iter+1):
            lambdas_prev = lambdas
            lambdas = self.update_lams(amplitudes=amplitudes, lambdas=lambdas, eris=eris)
            vector = self.lambdas_to_vector(lambdas)
            vector = diis.update(vector)
            lambdas = self.vector_to_lambdas(vector)
            dl = np.linalg.norm(vector - self.lambdas_to_vector(lambdas_prev))**2

            self.log.info("%4d %16.5g", niter, dl)

            converged = dl < self.options.t_tol
            if converged:
                self.log.output("Converged.")
                break
        else:
            self.log.warning("Failed to converge.")

        self.lambdas = lambdas
        self.converged_lambda = converged

        return None

    def _pack_codegen_kwargs(self, *extra_kwargs, eris=None):
        """Pack all the possible keyword arguments for generated code
        into a dictionary.
        """

        if eris is None:
            eris = self.get_eris()

        omega = np.diag(self.omega) if self.omega is not None else None

        kwargs = dict(
                f=self.fock,
                v=eris,
                g=self.g,
                G=self.G,
                w=omega,
                nocc=self.nocc,
                nvir=self.nvir,
                nbos=self.nbos,
        )
        for kw in extra_kwargs:
            if kw is not None:
                kwargs.update(kw)

        return kwargs

    def _load_function(self, name, eris=False, amplitudes=False, lambdas=False, **kwargs):
        """Load a function from the generated code, and return a dict
        of arguments.
        """

        if not (eris is False):
            if eris is None:
                eris = self.get_eris()
        else:
            eris = None

        dicts = []

        if not (amplitudes is False):
            if amplitudes is None:
                amplitudes = self.amplitudes
            if amplitudes is None:
                amplitudes = self.init_amps(eris=eris)
            dicts.append(amplitudes)

        if not (lambdas is False):
            if lambdas is None:
                lambdas = self.lambdas
            if lambdas is None:
                lambdas = self.init_lams(amplitudes=amplitudes)
            dicts.append(lambdas)

        if kwargs:
            dicts.append(kwargs)

        func = getattr(self._eqns, name, None)

        if func is None:
            raise NotImplementedError("%s for rank = %s" % (name, self.rank))

        kwargs = self._pack_codegen_kwargs(*dicts, eris=eris)

        return func, kwargs

    def energy(self, eris=None, amplitudes=None):
        """Compute the energy.
        """

        func, kwargs = self._load_function(
                "energy",
                eris=eris,
                amplitudes=amplitudes,
        )

        return func(**kwargs)

    def update_amps(self, eris=None, amplitudes=None):
        """Update the amplitudes.
        """

        func, kwargs = self._load_function(
                "update_amps",
                eris=eris,
                amplitudes=amplitudes,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        e_ia = lib.direct_sum("i-a->ia", self.eo, self.ev)

        # Divide T amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(0, n*2, 2)) + list(range(1, n*2, 2))
            d = functools.reduce(np.add.outer, [e_ia] * n)
            d = d.transpose(perm)
            res["t%d" % n] /= d
            res["t%d" % n] += amplitudes["t%d" % n]

        # Divide S amplitudes:
        for n in self.rank_numeric[1]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n))
            res["s%d" % n] /= d
            res["s%d" % n] += amplitudes["s%d" % n]

        # Divide U amplitudes:
        for n in self.rank_numeric[2]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ia])
            res["u1%d" % n] /= d
            res["u1%d" % n] += amplitudes["u1%d" % n]

        return res

    def update_lams(self, eris=None, amplitudes=None, lambdas=None):
        """Update the lambda amplitudes.
        """

        func, kwargs = self._load_function(
                "update_lams",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        e_ai = lib.direct_sum("i-a->ai", self.eo, self.ev)

        # Divide T amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(0, n*2, 2)) + list(range(1, n*2, 2))
            d = functools.reduce(np.add.outer, [e_ai] * n)
            d = d.transpose(perm)
            res["l%d" % n] /= d
            res["l%d" % n] += lambdas["l%d" % n]

        # Divide S amplitudes:
        for n in self.rank_numeric[1]:
            d = functools.reduce(np.add.outer, [-self.omega] * n)
            res["ls%d" % n] /= d
            res["ls%d" % n] += lambdas["ls%d" % n]

        # Divide U amplitudes:
        for n in self.rank_numeric[2]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ai])
            res["lu1%d" % n] /= d
            res["lu1%d" % n] += lambdas["lu1%d" % n]

        return res

    def make_sing_b_dm(self, eris=None, amplitudes=None, lambdas=None):
        """Build the single boson DM <b†> and <b>.
        """

        func, kwargs = self._load_function(
                "make_sing_b_dm",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_rdm1_b(self, eris=None, amplitudes=None, lambdas=None):
        """Build the bosonic 1RDM <b† b>.
        """

        func, kwargs = self._load_function(
                "make_rdm1_b",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_rdm1_f(self, eris=None, amplitudes=None, lambdas=None):
        """Build the fermionic 1RDM.
        """

        func, kwargs = self._load_function(
                "make_rdm1_f",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_rdm2_f(self, eris=None, amplitudes=None, lambdas=None):
        """Build the fermionic 2RDM.
        """

        func, kwargs = self._load_function(
                "make_rdm2_f",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_eb_coup_rdm(self, eris=None, amplitudes=None, lambdas=None):
        """Build the electron-boson coupling RDMs <b† i† j> and <b i† j>.
        """

        func, kwargs = self._load_function(
                "make_eb_coup_rdm",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def hbar_matvec_ip(self, r1, r2, eris=None, amplitudes=None):
        """Compute the product between a state vector and the EOM
        Hamiltonian for the IP.
        """

        func, kwargs = self._load_function(
                "hbar_matvec_ip",
                eris=eris,
                amplitudes=amplitudes,
                r1=r1,
                r2=r2,
        )

        return func(**kwargs)

    def hbar_matvec_ea(self, r1, r2, eris=None, amplitudes=None):
        """Compute the product between a state vector and the EOM
        Hamiltonian for the EA.
        """

        func, kwargs = self._load_function(
                "hbar_matvec_ea",
                eris=eris,
                amplitudes=amplitudes,
                r1=r1,
                r2=r2,
        )

        return func(**kwargs)

    def hbar_matvec_dd(self, r1, r2, eris=None, amplitudes=None):
        """Compute the product between a state vector and the EOM
        Hamiltonian for the DD.
        """

        raise NotImplementedError  # TODO

    def hbar_diag_ip(self, eris=None, amplitudes=None):
        """Compute the diagonal of the EOM Hamiltonian for the IP.
        """

        func, kwargs = self._load_function(
                "hbar_diag_ip",
                eris=eris,
                amplitudes=amplitudes,
        )

        return func(**kwargs)

    def hbar_diag_ea(self, eris=None, amplitudes=None):
        """Compute the diagonal of the EOM Hamiltonian for the EA.
        """

        func, kwargs = self._load_function(
                "hbar_diag_ea",
                eris=eris,
                amplitudes=amplitudes,
        )

        return func(**kwargs)

    def hbar_diag_dd(self, eris=None, amplitudes=None):
        """Compute the of the EOM Hamiltonian for the DD.
        """

        raise NotImplementedError  # TODO

    def make_ip_mom_bras(self, eris=None, amplitudes=None, lambdas=None):
        """Get the bra IP vectors to construct EOM moments.
        """

        func, kwargs = self._load_function(
                "make_ip_mom_bras",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ea_mom_bras(self, eris=None, amplitudes=None, lambdas=None):
        """Get the bra EA vectors to construct EOM moments.
        """

        func, kwargs = self._load_function(
                "make_ea_mom_bras",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_dd_mom_bras(self, eris=None, amplitudes=None, lambdas=None):
        """Get the bra DD vectors to construct EOM moments.
        """

        raise NotImplementedError  # TODO

    def make_ip_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """Get the ket IP vectors to construct EOM moments.
        """

        func, kwargs = self._load_function(
                "make_ip_mom_kets",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ea_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """Get the ket IP vectors to construct EOM moments.
        """

        func, kwargs = self._load_function(
                "make_ea_mom_kets",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        return func(**kwargs)

    def make_dd_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """Get the ket DD vectors to construct EOM moments.
        """

        raise NotImplementedError  # TODO

    def make_ip_1mom(self, eris=None, amplitudes=None, lambdas=None):
        """Build the first fermionic hole single-particle moment.

            T_{p, q} = <c†_p (H - E) c_q>
        """

        raise NotImplementedError  # TODO

    def make_ea_1mom(self, eris=None, amplitudes=None, lambdas=None):
        """Build the first fermionic particle single-particle moment.

            T_{p, q} = <c_p (H - E) c†_q>
        """

        raise NotImplementedError  # TODO

    def eom_ip(self, nroots=5, eris=None, amplitudes=None):
        """Solve the similarity-transformed hamiltonian for the IP
        with the equation-of-motion approach.
        """

        amplitudes_to_vector = self.excitations_to_vector_ip
        vector_to_amplitudes = self.vector_to_excitations_ip

        diag_parts = self.hbar_diag_ip(eris=eris, amplitudes=amplitudes)
        diag = amplitudes_to_vector(*diag_parts)

        def matvec(v):
            r = self.hbar_matvec_ip(*vector_to_amplitudes(v), eris=eris, amplitudes=amplitudes)
            return amplitudes_to_vector(*r)
        matvecs = lambda vs: [matvec(v) for v in vs]

        def pick(w, v, nroots, envs):
            w, v, idx = lib.linalg_helper.pick_real_eigs(w, v, nroots, envs)
            mask = w > 0  # FIXME
            w, v = w[mask], v[:, mask]
            mask = np.argsort(w)
            w, v = w[mask], v[:, mask]
            return w, v, 0

        guesses = np.zeros((nroots, diag.size))
        arg = np.argsort(np.absolute(diag))
        for root, guess in enumerate(arg[:nroots]):
            guesses[root, guess] = 1.0
        guesses = list(guesses)

        conv, e, v = lib.davidson_nosym1(
                matvecs,
                guesses,
                diag,
                tol=self.options.e_tol,
                nroots=nroots,
                pick=pick,
                max_cycle=self.options.max_iter,
                max_space=12,
                verbose=0,
        )

        return e, v

    def eom_ea(self, nroots=5, eris=None, amplitudes=None):
        """Solve the similarity-transformed hamiltonian for the EA
        with the equation-of-motion approach.
        """
        # TODO move to kernel function and combine with above

        amplitudes_to_vector = self.excitations_to_vector_ea
        vector_to_amplitudes = self.vector_to_excitations_ea

        diag_parts = self.hbar_diag_ea(eris=eris, amplitudes=amplitudes)
        diag = amplitudes_to_vector(*diag_parts)

        def matvec(v):
            r = self.hbar_matvec_ea(*vector_to_amplitudes(v), eris=eris, amplitudes=amplitudes)
            return amplitudes_to_vector(*r)
        matvecs = lambda vs: [matvec(v) for v in vs]

        def pick(w, v, nroots, envs):
            w, v, idx = lib.linalg_helper.pick_real_eigs(w, v, nroots, envs)
            mask = w > 0  # FIXME
            if np.any(mask):
                w, v = w[mask], v[:, mask]
            mask = np.argsort(w)
            w, v = w[mask], v[:, mask]
            return w, v, 0

        guesses = np.zeros((nroots, diag.size))
        arg = np.argsort(np.absolute(diag))
        for root, guess in enumerate(arg[:nroots]):
            guesses[root, guess] = 1.0
        guesses = list(guesses)

        conv, e, v = lib.davidson_nosym1(
                matvecs,
                guesses,
                diag,
                tol=self.options.e_tol,
                nroots=nroots,
                pick=pick,
                max_cycle=self.options.max_iter,
                max_space=12,
                verbose=0,
        )

        return e, v

    def eom_ee(self, nroots=5, eris=None, amplitudes=None):
        """Solve the similarity-transformed hamiltonian for the EA
        with the equation-of-motion approach.
        """

        raise NotImplementedError  # TODO

    def get_mean_field_G(self):
        val = lib.einsum("Ipp->I", self.g.boo) * 2.0
        val -= self.xi * self.omega

        if self.bare_G is not None:
            val += self.bare_G

        return val

    def get_g(self, g):
        boo = g[:, :self.nocc, :self.nocc]
        bov = g[:, :self.nocc, self.nocc:]
        bvo = g[:, self.nocc:, :self.nocc]
        bvv = g[:, self.nocc:, self.nocc:]

        g = SimpleNamespace(boo=boo, bov=bov, bvo=bvo, bvv=bvv)

        return g

    @property
    def bare_fock(self):
        fock = lib.einsum("pq,pi,qj->ij", self.mf.get_fock(), self.mf.mo_coeff, self.mf.mo_coeff)
        return fock

    def get_fock(self):
        """Get blocks of the Fock matrix, shifted due to bosons where
        the ansatz requires. The diagonal of the bare Fock matrix is
        subtracted.
        """

        fock = self.bare_fock

        oo = fock[:self.nocc, :self.nocc]
        ov = fock[:self.nocc, self.nocc:]
        vo = fock[self.nocc:, :self.nocc]
        vv = fock[self.nocc:, self.nocc:]

        if self.options.shift:
            xi = self.xi
            oo -= lib.einsum("I,Iij->ij", xi, self.g.boo + self.g.boo.transpose(0, 2, 1))
            ov -= lib.einsum("I,Iia->ia", xi, self.g.bov + self.g.bvo.transpose(0, 2, 1))
            vo -= lib.einsum("I,Iai->ai", xi, self.g.bvo + self.g.bov.transpose(0, 2, 1))
            vv -= lib.einsum("I,Iab->ab", xi, self.g.bvv + self.g.bvv.transpose(0, 2, 1))

        assert np.allclose(oo, oo.T)
        assert np.allclose(vv, vv.T)
        assert np.allclose(ov, vo.T)

        f = SimpleNamespace(oo=oo, ov=ov, vo=vo, vv=vv)

        return f

    def get_eris(self):
        """Get blocks of the ERIs.
        """

        o = slice(None, self.nocc)
        v = slice(self.nocc, None)
        slices = {"o": o, "v": v}

        # JIT namespace
        class two_e_blocks:
            def __getattr__(blocks, key):
                if key not in blocks.__dict__:
                    coeffs = [self.mf.mo_coeff[:, slices[k]] for k in key]
                    block = ao2mo.incore.general(self.mf._eri, coeffs, compact=False)
                    block = block.reshape([c.shape[-1] for c in coeffs])
                    blocks.__dict__[key] = block
                return blocks.__dict__[key]

        return two_e_blocks()

    def amplitudes_to_vector(self, amplitudes):
        """Construct a vector containing all of the amplitudes used in
        the given ansatz.
        """

        vectors = []

        for n in self.rank_numeric[0]:
            vectors.append(amplitudes["t%d" % n].ravel())

        for n in self.rank_numeric[1]:
            vectors.append(amplitudes["s%d" % n].ravel())

        for n in self.rank_numeric[2]:
            vectors.append(amplitudes["u1%d" % n].ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        """Construct all of the amplitudes used in the given ansatz
        from a vector.
        """

        amplitudes = {}
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nocc,) * n + (self.nvir,) * n
            size = np.prod(shape)
            amplitudes["t%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes["s%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[2]:
            shape = (self.nbos,) * n + (self.nocc, self.nvir)
            size = np.prod(shape)
            amplitudes["u1%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        return amplitudes

    def lambdas_to_vector(self, lambdas):
        """Construct a vector containing all of the lambda amplitudes
        used in the given ansatz.
        """

        vectors = []

        for n in self.rank_numeric[0]:
            vectors.append(lambdas["l%d" % n].ravel())

        for n in self.rank_numeric[1]:
            vectors.append(lambdas["ls%d" % n].ravel())

        for n in self.rank_numeric[2]:
            vectors.append(lambdas["lu1%d" % n].ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        """Construct all of the lambdas used in the given ansatz
        from a vector.
        """

        lambdas = {}
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nvir,) * n + (self.nocc,) * n
            size = np.prod(shape)
            lambdas["l%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["ls%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[2]:
            shape = (self.nbos,) * n + (self.nvir, self.nocc)
            size = np.prod(shape)
            lambdas["lu1%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        return lambdas

    def excitations_to_vector_ip(self, *excitations):
        """Construct a vector containing all of the excitation
        amplitudes used in the given ansatz.
        """

        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            vectors.append(excitations[m].ravel())
            m += 1

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for n in self.rank_numeric[2]:
            raise NotImplementedError

        return np.concatenate(vectors)

    excitations_to_vector_ea = excitations_to_vector_ip

    def vector_to_excitations_ip(self, vector):
        """Construct all of the excitation amplitudes used in the
        given ansatz from a vector.
        """

        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nocc,) * n + (self.nvir,) * (n-1)
            size = np.prod(shape)
            excitations.append(vector[i0:i0+size].reshape(shape))
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for n in self.rank_numeric[2]:
            raise NotImplementedError

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        """Construct all of the excitation amplitudes used in the
        given ansatz from a vector.
        """

        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nvir,) * n + (self.nocc,) * (n-1)
            size = np.prod(shape)
            excitations.append(vector[i0:i0+size].reshape(shape))
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for n in self.rank_numeric[2]:
            raise NotImplementedError

        return tuple(excitations)

    @property
    def xi(self):
        if self.options.shift:
            xi = lib.einsum("Iii->I", self.g.boo) * 2.0
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros_like(self.omega)
        return xi

    @property
    def const(self):
        if self.options.shift:
            return lib.einsum("I,I->", self.omega, self.xi**2)
        else:
            return 0.0

    @property
    def name(self):
        return "RCC" + "-".join(self.rank).rstrip("-")

    @property
    def rank_numeric(self):
        values = {"S": 1, "D": 2, "T": 3, "Q": 4}
        return tuple(tuple(values[i] for i in j) for j in self.rank)

    @property
    def nmo(self):
        return self.mf.mo_occ.size

    @property
    def nocc(self):
        return np.sum(self.mf.mo_occ > 0)

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def nbos(self):
        if self.omega is None:
            return 0
        return self.omega.shape[0]

    @property
    def eo(self):
        # NOTE NOT this:
        #return self.mf.mo_energy[:self.nocc]
        return np.diag(self.fock.oo)

    @property
    def ev(self):
        # NOTE NOT this:
        #return self.mf.mo_energy[self.nocc:]
        return np.diag(self.fock.vv)

    @property
    def e_tot(self):
        return self.mf.e_tot + self.e_corr

    @property
    def t1(self):
        return self.amplitudes["t1"]

    @property
    def t2(self):
        return self.amplitudes["t2"]

    @property
    def l1(self):
        return self.lambdas["l1"]

    @property
    def l2(self):
        return self.lambdas["l2"]



if __name__ == "__main__":
    from pyscf import gto, scf, cc
    import numpy as np

    mol = gto.Mole()
    #mol.atom = "He 0 0 0"
    #mol.basis = "cc-pvdz"
    mol.atom = "H 0 0 0; F 0 0 1.1"
    mol.basis = "6-31g"
    mol.verbose = 5
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    ccsd_ref = cc.CCSD(mf)
    ccsd_ref.kernel()
    ccsd_ref.solve_lambda()

    #ccsd = REBCC(mf, rank=(2, 0, 0))
    #ccsd.kernel()
    #ccsd.solve_lambda()

    #print("e ", np.abs(ccsd.e_corr - ccsd_ref.e_corr))
    #print("t1", np.max(np.abs(ccsd.t1 - ccsd_ref.t1)))
    #print("t2", np.max(np.abs(ccsd.t2 - ccsd_ref.t2)))
    #print("l1", np.max(np.abs(ccsd.l1 - ccsd_ref.l1.T)))
    #print("l2", np.max(np.abs(ccsd.l2 - ccsd_ref.l2.transpose(2, 3, 0, 1))))
    #print("rdm1", np.max(np.abs(ccsd.make_rdm1_f() - ccsd_ref.make_rdm1())))
    #print("rdm2", np.max(np.abs(ccsd.make_rdm2_f() - ccsd_ref.make_rdm2())))

    ## Transpose issue I think:
    ##print(ccsd.make_rdm2_f())
    ##print(ccsd_ref.make_rdm2())

    #v1 = np.ones((ccsd.nocc,))
    #v2 = np.zeros((ccsd.nocc, ccsd.nocc, ccsd.nvir))

    #ra = ccsd_ref.eomip_method().gen_matvec()[0]([np.concatenate([v1.ravel(), v2.ravel()])])
    #rb1, rb2 = ccsd.hbar_matvec_ip(r1=v1, r2=v2)
    #rb = np.concatenate([rb1.ravel(), rb2.ravel()])
    #print("r", np.allclose(ra, rb/2))

    nbos = 5
    np.random.seed(1)
    g = np.random.random((nbos, mol.nao, mol.nao)) * 0.03
    g = g + g.transpose(0, 2, 1)
    omega = np.random.random((nbos)) * 0.5

    np.set_printoptions(edgeitems=1000, linewidth=1000, precision=8)
    ccsd = REBCC(mf, rank=(2, 1, 1), omega=omega, g=g)
    ccsd.kernel()
    ccsd.solve_lambda()

    #amps = ccsd.init_amps()
    #amps = ccsd.update_amps(amplitudes=amps)
    #print(amps["t1"])
    #print(amps["t2"])
    #print(amps["s1"])
    #print(amps["u11"])


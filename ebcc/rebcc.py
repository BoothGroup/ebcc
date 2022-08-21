"""Restricted electron-boson coupled cluster.
"""

import dataclasses
import functools
import importlib
import logging
from types import SimpleNamespace
from typing import Tuple

import numpy as np
from pyscf import ao2mo, lib

from ebcc import default_log, eom, util

# TODO math in docstrings
# TODO resolve G vs bare_G confusion


class Amplitudes(dict):
    """Amplitude container class. Consists of a dictionary with keys
    that are strings of the name of each amplitude, and values are
    arrays whose dimension depends on the particular amplitude.
    """

    pass


class ERIs(SimpleNamespace):
    """Electronic repulsion integral container class. Consists of a
    just-in-time namespace containing blocks of the integrals, with
    keys that are length-4 strings of `"o"` or `"v"` signifying
    whether the corresponding dimension is occupied or virtual.
    """

    def __init__(self, ebcc, slices=None, mo_coeff=None):
        self.mf = ebcc.mf
        self.slices = slices
        self.mo_coeff = mo_coeff

        if self.mo_coeff is None:
            self.mo_coeff = self.mf.mo_coeff
        if not (isinstance(self.mo_coeff, (tuple, list)) or self.mo_coeff.ndim == 3):
            self.mo_coeff = [self.mo_coeff] * 4

        if self.slices is None:
            o = slice(None, ebcc.nocc)
            v = slice(ebcc.nocc, None)
            self.slices = {"o": o, "v": v}
        if not isinstance(self.slices, (tuple, list)):
            self.slices = [self.slices] * 4

    def __getattr__(self, key):
        """Just-in-time attribute getter."""

        if key not in self.__dict__:
            coeffs = []
            for i, k in enumerate(key):
                coeffs.append(self.mo_coeff[i][:, self.slices[i][k]])
            block = ao2mo.incore.general(self.mf._eri, coeffs, compact=False)
            block = block.reshape([c.shape[-1] for c in coeffs])
            self.__dict__[key] = block

        return self.__dict__[key]


@dataclasses.dataclass
class Options:
    """Options for EBCC calculations.

    Attributes
    ----------
    shift : bool, optional
        If `True`, shift the boson operators such that the Hamiltonian
        is normal-ordered with respect to a coherent state. This
        removes the bosonic coupling to the static mean-field density,
        introducing a constant energy shift. Default value is `True`.
    e_tol : float, optional
        Threshold for convergence in the correlation energy. Default
        value is 1e-8.
    t_tol : float, optional
        Threshold for convergence in the amplitude norm. Default value
        is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. Default value is 200.
    diis_space : int, optional
        Number of amplitudes to use in DIIS extrapolation. Default
        value is 12.
    """

    shift: bool = True
    e_tol: float = 1e-8
    t_tol: float = 1e-8
    max_iter: int = 200
    diis_space: int = 12


class REBCC:
    """Restricted electron-boson coupled cluster class.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF mean-field object.
    log : logging.Logger, optional
        Log to print output to. Default value is the global logger
        which outputs to `sys.stderr`.
    rank : tuple(str), optional
        Rank of (fermionic, bosonic, fermion-boson coupling) cluster
        operators. Default value is `("SD", "", "")`.
    omega : numpy.ndarray (nbos,), optional
        Bosonic frequencies. Default value is None.
    g : numpy.ndarray (nbos, nmo, nmo), optional
        Electron-boson coupling matrix corresponding to the bosonic
        annihilation operator i.e.

        .. math:: g_{xpq} p^\\dagger q b

        The creation part is assume to be the fermionic transpose of
        this tensor to retain hermiticity in the overall Hamiltonian.
        Default value is None.
    G : numpy.ndarray (nbos,), optional
        Boson non-conserving term of the Hamiltonian i.e.

        .. math:: G_x (b^\\dagger + b)

        Default value is None.
    options : dataclasses.dataclass
        Object containing the options. Default value is `Options`.
    **kwargs : dict
        Additional keyword arguments used to update `options`.

    Attributes
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF mean-field object.
    log : logging.Logger
        Log to print output to.
    options : dataclasses.dataclass
        Object containing the options.
    rank : tuple(str)
        Rank of (fermionic, bosonic, fermion-boson coupling) cluster
        operators.
    fermion_excitations : str, optional
        Rank of fermionic excitations. Default value is "SD".
    boson_excitations : str, optional
        Rank of bosonic excitations. Default is "".
    fermion_coupling_rank : int, optional
        Rank of fermionic term in coupling. Default is 0.
    boson_coupling_rank : int, optional
        Rank of bosonic term in coupling. Default is 0.
    e_corr : float
        Correlation energy.
    amplitudes : Amplitudes
        Cluster amplitudes.
    lambdas : Amplitudes
        Cluster lambda amplitudes.
    converged : bool
        Whether the coupled cluster equations converged.
    converged_lambda : bool
        Whether the lambda coupled cluster equations converged.
    omega : numpy.ndarray (nbos,)
        Bosonic frequencies.
    g : SimpleNamespace
        Namespace containing blocks of the electron-boson coupling
        matrix. Each attribute should be a length-3 string of `b`,
        `o` or `v` signifying whether the corresponding axis is
        bosonic, occupied or virtual.
    G : numpy.ndarray (nbos,)
        Mean-field boson non-conserving term of the Hamiltonian.
    bare_G : numpy.ndarray (nbos,)
        Boson non-conserving term of the Hamiltonian.
    fock : SimpleNamespace
        Namespace containing blocks of the Fock matrix. Each
        attribute should be a length-2 string of `o` or `v`
        signifying whether the corresponding axis is occupied or
        virtual.
    bare_fock : numpy.ndarray (nmo, nmo)
        The mean-field Fock matrix in the MO basis.
    xi : numpy.ndarray (nbos,)
        Shift in bosonic operators to diagonalise the phononic
        Hamiltonian.
    const : float
        Shift in the energy from moving to polaritonic basis.
    name : str
        Name of the method.
    rank_numeric : tuple of int
        Numeric form of rank tuple.
    nmo : int
        Number of molecular orbitals.
    nocc : int
        Number of occupied molecular orbitals.
    nvir : int
        Number of virtual molecular orbitals.
    nbos : int
        Number of bosonic degrees of freedom.

    Methods
    -------
    init_amps(eris=None)
        Initialise the amplitudes.
    init_lams(amplitudes=None)
        Initialise the lambda amplitudes.
    kernel(eris=None)
        Run the coupled cluster calculation.
    solve_lambda(amplitudes=None, eris=None)
        Solve the lambda coupled cluster equations.
    energy(eris=None, amplitudes=None)
        Compute the correlation energy.
    update_amps(eris=None, amplitudes=None)
        Update the amplitudes.
    update_lams(eris=None, amplitudes=None, lambdas=None)
        Update the lambda amplitudes.
    make_sing_b_dm(eris=None, amplitudes=None, lambdas=None)
        Build the single boson density matrix.
    make_rdm1_b(eris=None, amplitudes=None, lambdas=None,
                unshifted=None, hermitise=None)
        Build the bosonic one-particle reduced density matrix.
    make_rdm1_f(eris=None, amplitudes=None, lambdas=None,
                hermitise=None)
        Build the fermionic one-particle reduced density matrix.
    make_rdm2_f(eris=None, amplitudes=None, lambdas=None,
                hermitise=None)
        Build the fermionic two-particle reduced density matrix.
    make_eb_coup_rdm(eris=None, amplitudes=None, lambdas=None,
                     unshifted=True, hermitise=True)
        Build the electron-boson coupling reduced density matrices.
    hbar_matvec_ip(r1, r2, eris=None, amplitudes=None)
        Compute the product between a state vector and the EOM
        Hamiltonian for the IP.
    hbar_matvec_ea(r1, r2, eris=None, amplitudes=None)
        Compute the product between a state vector and the EOM
        Hamiltonian for the EA.
    hbar_matvec_ee(r1, r2, eris=None, amplitudes=None)
        Compute the product between a state vector and the EOM
        Hamiltonian for the EE.
    hbar_diag_ip(eris=None, amplitudes=None)
        Compute the diagonal of the EOM Hamiltonian for the IP.
    hbar_diag_ea(eris=None, amplitudes=None)
        Compute the diagonal of the EOM Hamiltonian for the EA.
    hbar_diag_ee(eris=None, amplitudes=None)
        Compute the diagonal of the EOM Hamiltonian for the EE.
    make_ip_mom_bras(eris=None, amplitudes=None, lambdas=None)
        Get the bra IP vectors to construct EOM moments.
    make_ea_mom_bras(eris=None, amplitudes=None, lambdas=None)
        Get the bra EA vectors to construct EOM moments.
    make_ee_mom_bras(eris=None, amplitudes=None, lambdas=None)
        Get the bra EE vectors to construct EOM moments.
    make_ip_mom_kets(eris=None, amplitudes=None, lambdas=None)
        Get the ket IP vectors to construct EOM moments.
    make_ea_mom_kets(eris=None, amplitudes=None, lambdas=None)
        Get the ket EA vectors to construct EOM moments.
    make_ee_mom_kets(eris=None, amplitudes=None, lambdas=None)
        Get the ket EE vectors to construct EOM moments.
    make_ip_1mom(eris=None, amplitudes=None, lambdas=None)
        Build the first fermionic hole single-particle moment.
    make_ea_1mom(eris=None, amplitudes=None, lambdas=None)
        Build the first fermionic particle single-particle moment.
    amplitudes_to_vector(amplitudes)
        Construct a vector containing all of the amplitudes used in
        the given ansatz.
    vector_to_amplitudes(vector)
        Construct all of the amplitudes used in the given ansatz from
        a vector.
    lambdas_to_vector(lambdas)
        Construct a vector containing all of the lambda amplitudes
        used in the given ansatz.
    vector_to_lambdas(vector)
        Construct all of the lambdas used in the given ansatz from a
        vector.
    excitations_to_vector_ip(*excitations)
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the IP.
    excitations_to_vector_ea(*excitations)
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EA.
    excitations_to_vector_ee(*excitations)
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EE.
    vector_to_excitations_ip(vector)
        Construct all of the excitation amplitudes used in the given
        ansatz from a vector for the IP.
    vector_to_excitations_ea(vector)
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EA.
    vector_to_excitations_ee(vector)
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EE.
    get_mean_field_G()
        Get the mean-field boson non-conserving term of the
        Hamiltonian.
    get_g(g)
        Get the blocks of the electron-boson coupling matrix
        corresponding to the bosonic annihilation operator.
    get_fock()
        Get the blocks of the Fock matrix, shifted due to bosons
        where the ansatz requires.
    get_eris()
        Get blocks of the ERIs.
    """

    Options = Options
    Amplitudes = Amplitudes
    ERIs = ERIs

    def __init__(
        self,
        mf,
        log: logging.Logger = None,
        fermion_excitations: str = "SD",
        boson_excitations: str = "",
        fermion_coupling_rank: int = 0,
        boson_coupling_rank: int = 0,
        omega: np.ndarray = None,
        g: np.ndarray = None,
        G: np.ndarray = None,
        options: Options = None,
        **kwargs,
    ):
        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)

        self.log = default_log if log is None else log
        self.mf = self._convert_mf(mf)
        self.rank = (
            fermion_excitations,
            boson_excitations,
            fermion_coupling_rank,
            boson_coupling_rank,
        )
        self._eqns = self._get_eqns()

        if bool(self.rank[2]) != bool(self.rank[3]):
            raise ValueError(
                "Fermionic and bosonic coupling ranks must both be zero, " "or both non-zero."
            )

        self.log.info("%s", self.name)
        self.log.info("%s", "*" * len(self.name))
        self.log.info(" > Fermion excitations:    %s", self.rank[0])
        self.log.info(" > Boson excitations:      %s", self.rank[1] if len(self.rank[1]) else None)
        self.log.info(" > Fermion coupling rank:  %s", self.rank[2])
        self.log.info(" > Boson coupling rank:    %s", self.rank[3])

        self.omega = omega
        self.bare_g = g
        self.bare_G = G

        self.e_corr = None
        self.amplitudes = None
        self.converged = False
        self.lambdas = None
        self.converged_lambda = False

        if self.rank[1] != "":
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
        self.log.info(" > nocc:  %s", self.nocc)
        self.log.info(" > nvir:  %s", self.nvir)
        self.log.info(" > nbos:  %d", self.nbos)
        self.log.info(" > e_tol:  %s", self.options.e_tol)
        self.log.info(" > t_tol:  %s", self.options.t_tol)
        self.log.info(" > max_iter:  %s", self.options.max_iter)
        self.log.info(" > diis_space:  %s", self.options.diis_space)

    def kernel(self, eris=None):
        """Run the coupled cluster calculation.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.

        Returns
        -------
        e_cc : float
            Correlation energy.
        """

        if eris is None:
            eris = self.get_eris()

        if self.amplitudes is None:
            amplitudes = self.init_amps(eris=eris)
        else:
            amplitudes = self.amplitudes

        e_cc = e_init = self.energy(amplitudes=amplitudes, eris=eris)

        diis = lib.diis.DIIS()
        diis.space = self.options.diis_space

        self.log.output("Solving for excitation amplitudes.")
        self.log.info("%4s %16s %16s %16s", "Iter", "Energy (corr.)", "Δ(Energy)", "Δ(Amplitudes)")
        self.log.info("%4d %16.10f", 0, e_init)

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            amplitudes_prev = amplitudes
            amplitudes = self.update_amps(amplitudes=amplitudes, eris=eris)
            vector = self.amplitudes_to_vector(amplitudes)
            vector = diis.update(vector)
            amplitudes = self.vector_to_amplitudes(vector)
            dt = np.linalg.norm(vector - self.amplitudes_to_vector(amplitudes_prev)) ** 2

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
        """Solve the lambda coupled cluster equations.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)  # TODO warn?

        if self.lambdas is None:
            lambdas = self.init_lams(amplitudes=amplitudes)
        else:
            lambdas = self.lambdas

        diis = lib.diis.DIIS()
        diis.space = self.options.diis_space

        self.log.output("Solving for de-excitation (lambda) amplitudes.")
        self.log.info("%4s %16s", "Iter", "Δ(Amplitudes)")

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            lambdas_prev = lambdas
            lambdas = self.update_lams(amplitudes=amplitudes, lambdas=lambdas, eris=eris)
            vector = self.lambdas_to_vector(lambdas)
            vector = diis.update(vector)
            lambdas = self.vector_to_lambdas(vector)
            dl = np.linalg.norm(vector - self.lambdas_to_vector(lambdas_prev)) ** 2

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

    @staticmethod
    def _convert_mf(mf):
        """Convert the input PySCF mean-field object to the one
        required for the current class.
        """
        return mf.to_rhf()

    def _get_eqns(self):
        """Get the module which contains the generated equations for
        the current model.
        """
        name = self.name.replace("-", "_")
        eqns = importlib.import_module("ebcc.codegen.%s" % name)
        return eqns

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

    def init_amps(self, eris=None):
        """Initialise the amplitudes.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.

        Returns
        -------
        amplitudes : Amplitudes
            Cluster amplitudes.
        """

        if eris is None:
            eris = self.get_eris()

        amplitudes = self.Amplitudes()
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
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                if nb == 1:
                    e_xia = lib.direct_sum("ia-x->xia", e_ia, self.omega)
                    amplitudes["u%d%d" % (nf, nb)] = h.bov / e_xia
                else:
                    amplitudes["u%d%d" % (nf, nb)] = np.zeros(
                        (self.nbos,) * nb + (self.nocc, self.nvir)
                    )

        return amplitudes

    def init_lams(self, amplitudes=None):
        """Initialise the lambda amplitudes.

        Parameters
        ----------
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        lambdas : Amplitudes
            Updated cluster lambda amplitudes.
        """

        if amplitudes is None:
            amplitudes = self.amplitudes

        lambdas = self.Amplitudes()

        # Build L amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(n, 2 * n)) + list(range(n))
            lambdas["l%d" % n] = amplitudes["t%d" % n].transpose(perm)

        # Build LS amplitudes:
        for n in self.rank_numeric[1]:
            lambdas["ls%d" % n] = amplitudes["s%d" % n]

        # Build LU amplitudes:
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                perm = list(range(nb)) + [nb + 1, nb]
                lambdas["lu%d%d" % (nf, nb)] = amplitudes["u%d%d" % (nf, nb)].transpose(perm)

        return lambdas

    def energy(self, eris=None, amplitudes=None):
        """Compute the correlation energy.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        e_cc : float
            Correlation energy.
        """

        func, kwargs = self._load_function(
            "energy",
            eris=eris,
            amplitudes=amplitudes,
        )

        return func(**kwargs)

    def update_amps(self, eris=None, amplitudes=None):
        """Update the amplitudes.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        amplitudes : Amplitudes
            Updated cluster amplitudes.
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
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
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
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ia] * nf))
                res["u%d%d" % (nf, nb)] /= d
                res["u%d%d" % (nf, nb)] += amplitudes["u%d%d" % (nf, nb)]

        return res

    def update_lams(self, eris=None, amplitudes=None, lambdas=None):
        """Update the lambda amplitudes.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        lambdas : Amplitudes
            Updated cluster lambda amplitudes.
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
            perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
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
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                d = functools.reduce(np.add.outer, ([-self.omega] * nb) + ([e_ai] * nf))
                res["lu%d%d" % (nf, nb)] /= d
                res["lu%d%d" % (nf, nb)] += lambdas["lu%d%d" % (nf, nb)]

        return res

    def make_sing_b_dm(self, eris=None, amplitudes=None, lambdas=None):
        """Build the single boson density matrix:

        ..math :: \\langle b^+ \\rangle

        and

        ..math :: \\langle b \\rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        dm_b : numpy.ndarray (nbos,)
            Single boson density matrix.
        """

        func, kwargs = self._load_function(
            "make_sing_b_dm",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_rdm1_b(self, eris=None, amplitudes=None, lambdas=None, unshifted=True, hermitise=True):
        """Build the bosonic one-particle reduced density matrix:

        ..math :: \\langle b^+ b \\rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.
        unshifted : bool, optional
            If `self.shift` is `True`, then `unshifted=True` applies
            the reverse transformation such that the bosonic operators
            are defined with respect to the unshifted bosons. Default
            value is True. Has no effect if `self.shift` is `False`.
        hermitise : bool, optional
            Force Hermiticity in the output. Default value is `True`.

        Returns
        -------
        rdm1_b : numpy.ndarray (nbos, nbos)
            Bosonic one-particle reduced density matrix.
        """

        func, kwargs = self._load_function(
            "make_rdm1_b",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:
            dm = 0.5 * (dm + dm.T)

        if unshifted and self.options.shift:
            dm_cre, dm_ann = self.make_sing_b_dm()
            xi = self.xi
            dm[np.diag_indices_from(dm)] -= xi * (dm_cre + dm_ann) - xi ** 2

        return dm

    def make_rdm1_f(self, eris=None, amplitudes=None, lambdas=None, hermitise=True):
        """Build the fermionic one-particle reduced density matrix:

        ..math :: \\langle i^+ j \\rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.
        hermitise : bool, optional
            Force Hermiticity in the output. Default value is `True`.

        Returns
        -------
        rdm1_f : numpy.ndarray (nmo, nmo)
            Fermionic one-particle reduced density matrix.
        """

        func, kwargs = self._load_function(
            "make_rdm1_f",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:
            dm = 0.5 * (dm + dm.T)

        return dm

    def make_rdm2_f(self, eris=None, amplitudes=None, lambdas=None, hermitise=True):
        """Build the fermionic two-particle reduced density matrix:

        ..math :: \\Gamma_{ijkl} = \\langle i^+ j^+ l k \\rangle

        which is stored in Chemist's notation.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.
        hermitise : bool, optional
            Force Hermiticity in the output. Default value is `True`.

        Returns
        -------
        rdm2_f : numpy.ndarray (nmo, nmo, nmo, nmo)
            Fermionic two-particle reduced density matrix.
        """

        func, kwargs = self._load_function(
            "make_rdm2_f",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:
            dm = 0.5 * (+dm.transpose(0, 1, 2, 3) + dm.transpose(2, 3, 0, 1))
            dm = 0.5 * (+dm.transpose(0, 1, 2, 3) + dm.transpose(1, 0, 3, 2))

        return dm

    def make_eb_coup_rdm(
        self, eris=None, amplitudes=None, lambdas=None, unshifted=True, hermitise=True
    ):
        """Build the electron-boson coupling reduced density matrices:

        ..math :: \\langle b^+ i^+ j \\rangle

        and

        ..math :: \\langle b i^+ j \\rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.
        unshifted : bool, optional
            If `self.shift` is `True`, then `unshifted=True` applies
            the reverse transformation such that the bosonic operators
            are defined with respect to the unshifted bosons. Default
            value is True. Has no effect if `self.shift` is `False`.
        hermitise : bool, optional
            Force Hermiticity in the output. Default value is `True`.

        Returns
        -------
        dm_eb : numpy.ndarray (2, nbos, nmo, nmo)
            Electron-boson coupling reduce density matrices. First
            index corresponds to creation and second to annihilation
            of the bosonic index.
        """

        func, kwargs = self._load_function(
            "make_eb_coup_rdm",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm_eb = func(**kwargs)

        if hermitise:
            dm_eb[0] = 0.5 * (dm_eb[0] + dm_eb[1].transpose(0, 2, 1))
            dm_eb[1] = dm_eb[0].transpose(0, 2, 1).copy()

        if unshifted and self.options.shift:
            rdm1_f = self.make_rdm1_f(hermitise=hermitise)
            shift = lib.einsum("x,ij->xij", self.xi, rdm1_f)
            dm_eb -= shift[None]

        return dm_eb

    def hbar_ip(self, eris=None, amplitudes=None):
        """Compute the EOM Hamiltonian for the IP.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        h : numpy.ndarray
            EOM Hamiltonian for the IP.
        """
        # TODO generalise

        func, kwargs = self._load_function(
            "hbar_ip",
            eris=eris,
            amplitudes=amplitudes,
        )

        blocks = func(**kwargs)
        nocc = self.nocc
        nvir = self.nvir

        h11 = blocks["h11"].reshape(nocc, nocc)
        h12 = blocks["h12"].reshape(nocc, nocc * nocc * nvir)
        h21 = blocks["h21"].reshape(nocc * nocc * nvir, nocc)
        h22 = blocks["h22"].reshape(nocc * nocc * nvir, nocc * nocc * nvir)

        h = np.block([[h11, h12], [h21, h22]])

        return h

    def hbar_ea(self, eris=None, amplitudes=None):
        """Compute the EOM Hamiltonian for the EA.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        h : numpy.ndarray
            EOM Hamiltonian for the EA.
        """
        # TODO generalise

        func, kwargs = self._load_function(
            "hbar_ea",
            eris=eris,
            amplitudes=amplitudes,
        )

        blocks = func(**kwargs)
        nocc = self.nocc
        nvir = self.nvir

        h11 = blocks["h11"].reshape(nvir, nvir)
        h12 = blocks["h12"].reshape(nvir, nvir * nvir * nocc)
        h21 = blocks["h21"].reshape(nvir * nvir * nocc, nvir)
        h22 = blocks["h22"].reshape(nvir * nvir * nocc, nvir * nvir * nocc)

        h = np.block([[h11, h12], [h21, h22]])

        return h

    def hbar_matvec_ip(self, r1, r2, eris=None, amplitudes=None):
        """Compute the product between a state vector and the EOM
        Hamiltonian for the IP.

        Parameters
        ----------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector. Keys are
            strings of the name of each vector, and values are arrays
            whose dimension depends on the particular sector.
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector resulting
            from the matrix-vector product with the input vectors.
            Keys are strings of the name of each vector, and values
            are arrays whose dimension depends on the particular
            sector.
        """
        # TODO generalise vectors input

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

        Parameters
        ----------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector. Keys are
            strings of the name of each vector, and values are arrays
            whose dimension depends on the particular sector.
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector resulting
            from the matrix-vector product with the input vectors.
            Keys are strings of the name of each vector, and values
            are arrays whose dimension depends on the particular
            sector.
        """

        func, kwargs = self._load_function(
            "hbar_matvec_ea",
            eris=eris,
            amplitudes=amplitudes,
            r1=r1,
            r2=r2,
        )

        return func(**kwargs)

    def hbar_matvec_ee(self, r1, r2, eris=None, amplitudes=None):
        """Compute the product between a state vector and the EOM
        Hamiltonian for the EE.

        Parameters
        ----------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector. Keys are
            strings of the name of each vector, and values are arrays
            whose dimension depends on the particular sector.
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector resulting
            from the matrix-vector product with the input vectors.
            Keys are strings of the name of each vector, and values
            are arrays whose dimension depends on the particular
            sector.
        """

        func, kwargs = self._load_function(
            "hbar_matvec_ee",
            eris=eris,
            amplitudes=amplitudes,
            r1=r1,
            r2=r2,
        )

        return func(**kwargs)

    def hbar_diag_ip(self, eris=None, amplitudes=None):
        """Compute the diagonal of the EOM Hamiltonian for the IP.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        diag : dict of (str, numpy.ndarray)
            Dictionary containing the diagonal in each sector of the
            matrix. Keys are strings of the name of each vector, and
            values are arrays whose dimension depends on the
            particular sector.
        """

        func, kwargs = self._load_function(
            "hbar_diag_ip",
            eris=eris,
            amplitudes=amplitudes,
        )

        return func(**kwargs)

    def hbar_diag_ea(self, eris=None, amplitudes=None):
        """Compute the diagonal of the EOM Hamiltonian for the EA.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        diag : dict of (str, numpy.ndarray)
            Dictionary containing the diagonal in each sector of the
            matrix. Keys are strings of the name of each vector, and
            values are arrays whose dimension depends on the
            particular sector.
        """

        func, kwargs = self._load_function(
            "hbar_diag_ea",
            eris=eris,
            amplitudes=amplitudes,
        )

        return func(**kwargs)

    def hbar_diag_ee(self, eris=None, amplitudes=None):
        """Compute the of the EOM Hamiltonian for the EE.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        diag : dict of (str, numpy.ndarray)
            Dictionary containing the diagonal in each sector of the
            matrix. Keys are strings of the name of each vector, and
            values are arrays whose dimension depends on the
            particular sector.
        """

        func, kwargs = self._load_function(
            "hbar_diag_ee",
            eris=eris,
            amplitudes=amplitudes,
        )

        return func(**kwargs)

    def make_ip_mom_bras(self, eris=None, amplitudes=None, lambdas=None):
        """Get the bra IP vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        bras : dict of (str, numpy.ndarray)
            Dictionary containing the bra vectors in each sector. Keys
            are strings of the name of each sector, and values are
            arrays whose dimension depends on the particular sector.
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

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        bras : dict of (str, numpy.ndarray)
            Dictionary containing the bra vectors in each sector. Keys
            are strings of the name of each sector, and values are
            arrays whose dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ea_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ee_mom_bras(self, eris=None, amplitudes=None, lambdas=None):
        """Get the bra EE vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        bras : dict of (str, numpy.ndarray)
            Dictionary containing the bra vectors in each sector. Keys
            are strings of the name of each sector, and values are
            arrays whose dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ee_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ip_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """Get the ket IP vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        kets : dict of (str, numpy.ndarray)
            Dictionary containing the ket vectors in each sector. Keys
            are strings of the name of each sector, and values are
            arrays whose dimension depends on the particular sector.
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

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        kets : dict of (str, numpy.ndarray)
            Dictionary containing the ket vectors in each sector. Keys
            are strings of the name of each sector, and values are
            arrays whose dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ea_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ee_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """Get the ket EE vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        kets : dict of (str, numpy.ndarray)
            Dictionary containing the ket vectors in each sector. Keys
            are strings of the name of each sector, and values are
            arrays whose dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ee_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ip_1mom(self, eris=None, amplitudes=None, lambdas=None):
        """Build the first fermionic hole single-particle moment.

        .. math:: T_{pq} = \\langle c_p^+ (H - E_0) c_q \\rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        mom : numpy.ndarray (nmo, nmo)
            Array of the first moment.
        """

        raise NotImplementedError  # TODO

    def make_ea_1mom(self, eris=None, amplitudes=None, lambdas=None):
        """Build the first fermionic particle single-particle moment.

        .. math:: T_{pq} = \\langle c_p (H - E_0) c_q^+ \\rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        mom : numpy.ndarray (nmo, nmo)
            Array of the first moment.
        """

        raise NotImplementedError  # TODO

    def ip_eom(self, options=None, **kwargs):
        return eom.IP_EOM(self, options=options, **kwargs)

    def ea_eom(self, options=None, **kwargs):
        return eom.EA_EOM(self, options=options, **kwargs)

    def ee_eom(self, options=None, **kwargs):
        return eom.EE_EOM(self, options=options, **kwargs)

    def amplitudes_to_vector(self, amplitudes):
        """Construct a vector containing all of the amplitudes used in
        the given ansatz.

        Parameters
        ----------
        amplitudes : Amplitudes, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the amplitudes flattened
            and concatenated. Size depends on the ansatz.
        """

        vectors = []

        for n in self.rank_numeric[0]:
            vectors.append(amplitudes["t%d" % n].ravel())

        for n in self.rank_numeric[1]:
            vectors.append(amplitudes["s%d" % n].ravel())

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                vectors.append(amplitudes["u%d%d" % (nf, nb)].ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        """Construct all of the amplitudes used in the given ansatz
        from a vector.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the amplitudes flattened
            and concatenated. Size depends on the ansatz.

        Returns
        -------
        amplitudes : Amplitudes
            Cluster amplitudes.
        """

        amplitudes = self.Amplitudes()
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nocc,) * n + (self.nvir,) * n
            size = np.prod(shape)
            amplitudes["t%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes["s%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                shape = (self.nbos,) * nb + (self.nocc, self.nvir) * nf
                size = np.prod(shape)
                amplitudes["u%d%d" % (nf, nb)] = vector[i0 : i0 + size].reshape(shape)
                i0 += size

        return amplitudes

    def lambdas_to_vector(self, lambdas):
        """Construct a vector containing all of the lambda amplitudes
        used in the given ansatz.

        Parameters
        ----------
        lambdas : Amplitudes, optional
            Cluster lambda amplitudes. Default value is generated
            using `self.init_lams()`.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the lambdas flattened
            and concatenated. Size depends on the ansatz.
        """

        vectors = []

        for n in self.rank_numeric[0]:
            vectors.append(lambdas["l%d" % n].ravel())

        for n in self.rank_numeric[1]:
            vectors.append(lambdas["ls%d" % n].ravel())

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                vectors.append(lambdas["lu%d%d" % (nf, nb)].ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        """Construct all of the lambdas used in the given ansatz
        from a vector.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the lambdas flattened
            and concatenated. Size depends on the ansatz.

        Returns
        -------
        lambdas : Amplitudes
            Cluster lambda amplitudes.
        """

        lambdas = self.Amplitudes()
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nvir,) * n + (self.nocc,) * n
            size = np.prod(shape)
            lambdas["l%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["ls%d" % n] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                shape = (self.nbos,) * nb + (self.nvir, self.nocc) * nf
                size = np.prod(shape)
                lambdas["lu%d%d" % (nf, nb)] = vector[i0 : i0 + size].reshape(shape)
                i0 += size

        return lambdas

    def excitations_to_vector_ip(self, *excitations):
        """Construct a vector containing all of the excitation
        amplitudes used in the given ansatz for the IP.

        Parameters
        ----------
        *excitations : iterable of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of
            the name of each excitations, and values are arrays whose
            dimension depends on the particular excitation amplitude.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened
            and concatenated. Size depends on the ansatz.
        """

        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            vectors.append(excitations[m].ravel())
            m += 1

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return np.concatenate(vectors)

    def excitations_to_vector_ea(self, *excitations):
        """Construct a vector containing all of the excitation
        amplitudes used in the given ansatz for the EA.

        Parameters
        ----------
        *excitations : iterable of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of
            the name of each excitations, and values are arrays whose
            dimension depends on the particular excitation amplitude.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened
            and concatenated. Size depends on the ansatz.
        """
        return self.excitations_to_vector_ip(*excitations)

    def excitations_to_vector_ee(self, *excitations):
        """Construct a vector containing all of the excitation
        amplitudes used in the given ansatz for the EE.

        Parameters
        ----------
        *excitations : iterable of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of
            the name of each excitations, and values are arrays whose
            dimension depends on the particular excitation amplitude.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened
            and concatenated. Size depends on the ansatz.
        """
        return self.excitations_to_vector_ip(*excitations)

    def vector_to_excitations_ip(self, vector):
        """Construct all of the excitation amplitudes used in the
        given ansatz from a vector for the IP.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened
            and concatenated. Size depends on the ansatz.

        Returns
        -------
        excitations : tuple of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of
            the name of each excitations, and values are arrays whose
            dimension depends on the particular excitation amplitude.
        """

        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nocc,) * n + (self.nvir,) * (n - 1)
            size = np.prod(shape)
            excitations.append(vector[i0 : i0 + size].reshape(shape))
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        """Construct a vector containing all of the excitation
        amplitudes used in the given ansatz for the EA.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened
            and concatenated. Size depends on the ansatz.

        Returns
        -------
        excitations : tuple of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of
            the name of each excitations, and values are arrays whose
            dimension depends on the particular excitation amplitude.
        """

        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nvir,) * n + (self.nocc,) * (n - 1)
            size = np.prod(shape)
            excitations.append(vector[i0 : i0 + size].reshape(shape))
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return tuple(excitations)

    def vector_to_excitations_ee(self, vector):
        """Construct a vector containing all of the excitation
        amplitudes used in the given ansatz for the EE.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened
            and concatenated. Size depends on the ansatz.

        Returns
        -------
        excitations : tuple of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of
            the name of each excitations, and values are arrays whose
            dimension depends on the particular excitation amplitude.
        """

        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            shape = (self.nocc,) * n + (self.nvir,) * n
            size = np.prod(shape)
            excitations.append(vector[i0 : i0 + size].reshape(shape))
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return tuple(excitations)

    def get_mean_field_G(self):
        """Get the mean-field boson non-conserving term of the
        Hamiltonian.

        Returns
        -------
        G_mf : numpy.ndarray (nbos,)
            Mean-field boson non-conserving term of the Hamiltonian.
        """

        val = lib.einsum("Ipp->I", self.g.boo) * 2.0
        val -= self.xi * self.omega

        if self.bare_G is not None:
            val += self.bare_G

        return val

    def get_g(self, g):
        """Get blocks of the electron-boson coupling matrix
        corresponding to the bosonic annihilation operator.

        Parameters
        ----------
        g : numpy.ndarray (nbos, nmo, nmo)
            Array of the electron-boson coupling matrix.

        Returns
        -------
        g : SimpleNamespace
            Namespace containing blocks of the electron-boson coupling
            matrix. Each attribute should be a length-3 string of
            `b`, `o` or `v` signifying whether the corresponding axis
            is bosonic, occupied or virtual.
        """

        boo = g[:, : self.nocc, : self.nocc]
        bov = g[:, : self.nocc, self.nocc :]
        bvo = g[:, self.nocc :, : self.nocc]
        bvv = g[:, self.nocc :, self.nocc :]

        g = SimpleNamespace(boo=boo, bov=bov, bvo=bvo, bvv=bvv)

        return g

    def get_fock(self):
        """Get blocks of the Fock matrix, shifted due to bosons where
        the ansatz requires.

        Returns
        -------
        fock : SimpleNamespace
            Namespace containing blocks of the Fock matrix. Each
            attribute should be a length-2 string of `o` or `v`
            signifying whether the corresponding axis is occupied or
            virtual.
        """

        fock = self.bare_fock

        oo = fock[: self.nocc, : self.nocc]
        ov = fock[: self.nocc, self.nocc :]
        vo = fock[self.nocc :, : self.nocc]
        vv = fock[self.nocc :, self.nocc :]

        if self.options.shift:
            xi = self.xi
            oo -= lib.einsum("I,Iij->ij", xi, self.g.boo + self.g.boo.transpose(0, 2, 1))
            ov -= lib.einsum("I,Iia->ia", xi, self.g.bov + self.g.bvo.transpose(0, 2, 1))
            vo -= lib.einsum("I,Iai->ai", xi, self.g.bvo + self.g.bov.transpose(0, 2, 1))
            vv -= lib.einsum("I,Iab->ab", xi, self.g.bvv + self.g.bvv.transpose(0, 2, 1))

        f = SimpleNamespace(oo=oo, ov=ov, vo=vo, vv=vv)

        return f

    def get_eris(self):
        """Get blocks of the ERIs.

        Returns
        -------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        """

        return self.ERIs(self)

    @property
    def bare_fock(self):
        """Get the mean-field Fock matrix in the MO basis.

        Returns
        -------
        bare_fock : numpy.ndarray (nmo, nmo)
            The mean-field Fock matrix in the MO basis.
        """

        fock_ao = self.mf.get_fock()
        mo_coeff = self.mf.mo_coeff

        fock = lib.einsum("pq,pi,qj->ij", fock_ao, mo_coeff, mo_coeff)

        return fock

    @property
    def xi(self):
        """Get the shift in bosonic operators to diagonalise the
        phononic Hamiltonian.

        Returns
        -------
        xi : numpy.ndarray (nbos,)
            Shift in bosonic operators to diagonalise the phononic
            Hamiltonian.
        """

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
        """Get the shift in the energy from moving to polaritonic
        basis.

        Returns
        -------
        const : float
            Shift in the energy from moving to polaritonic basis.
        """
        if self.options.shift:
            return lib.einsum("I,I->", self.omega, self.xi ** 2)
        else:
            return 0.0

    @property
    def fermion_excitations(self):
        """Return a string representation of the fermionic excitations
        in the ansatz.

        Returns
        -------
        fermion_excitations : str
            String representation of the fermion excitations.
        """
        return self.rank[0]

    @property
    def boson_excitations(self):
        """Return a string representation of the bosonic excitations
        in the ansatz.

        Returns
        -------
        boson_excitations : str
            String representation of the boson excitations.
        """
        return self.rank[1]

    @property
    def fermion_coupling_rank(self):
        """Return the fermionic rank of the fermion-boson coupling
        in the ansatz.

        Returns
        -------
        fermion_coupling_rank : int
            Integer representation of the fermion coupling rank.
        """
        return self.rank[2]

    @property
    def boson_coupling_rank(self):
        """Return the bosonic rank of the fermion-boson coupling
        in the ansatz.

        Returns
        -------
        boson_coupling_rank : int
            Integer representation of the boson coupling rank.
        """
        return self.rank[3]

    @property
    def name(self):
        """Get a string with the name of the method.

        Returns
        -------
        name : str
            Name of the method.
        """
        name = "RCC%s" % self.fermion_excitations
        if self.boson_excitations:
            name += "-%s" % self.boson_excitations
        if self.fermion_coupling_rank or self.boson_coupling_rank:
            name += "-%d" % self.fermion_coupling_rank
            name += "-%d" % self.boson_coupling_rank
        return name

    @property
    def rank_numeric(self):
        """Get a list of cluster operator rank numbers for each of
        the fermionic, bosonic and coupling ansatz.

        Returns
        -------
        rank_numeric : tuple of tuple of int
            Numeric form of rank tuple.
        """
        # TODO this won't support i.e. SDt

        rank = []

        standard_notation = {"S": 1, "D": 2, "T": 3, "Q": 4}
        partial_notation = {"2": [1, 2], "3": [1, 2, 3], "4": [1, 2, 3, 4]}

        for op in self.rank[:2]:
            rank_entry = []
            for char in op:
                if char in standard_notation:
                    rank_entry.append(standard_notation[char])
                elif char in partial_notation:
                    rank_entry += partial_notation[char]
            rank.append(tuple(rank_entry))

        for op in self.rank[2:]:
            rank.append(tuple(range(1, op + 1)))

        return tuple(rank)

    @property
    def nmo(self):
        """Get the number of molecular orbitals.

        Returns
        -------
        nmo : int
            Number of molecular orbitals.
        """
        return self.mf.mo_occ.size

    @property
    def nocc(self):
        """Get the number of occupied molecular orbitals.

        Returns
        -------
        nocc : int
            Number of occupied molecular orbitals.
        """
        return np.sum(self.mf.mo_occ > 0)

    @property
    def nvir(self):
        """Get the number of virtual molecular orbitals.

        Returns
        -------
        nvir : int
            Number of virtual molecular orbitals.
        """
        return self.nmo - self.nocc

    @property
    def nbos(self):
        """Get the number of bosonic degrees of freedom.

        Returns
        -------
        nbos : int
            Number of bosonic degrees of freedom.
        """
        if self.omega is None:
            return 0
        return self.omega.shape[0]

    @property
    def eo(self):
        """Get the diagonal of the occupied Fock matrix in MO basis,
        shifted due to bosons where the ansatz requires.

        Returns
        -------
        eo : numpy.ndarray (nocc,)
            Diagonal of the occupied Fock matrix in MO basis.
        """
        return np.diag(self.fock.oo)

    @property
    def ev(self):
        """Get the diagonal of the virtual Fock matrix in MO basis,
        shifted due to bosons where the ansatz requires.

        Returns
        -------
        ev : numpy.ndarray (nvir,)
            Diagonal of the virtual Fock matrix in MO basis.
        """
        return np.diag(self.fock.vv)

    @property
    def e_tot(self):
        """Return the total energy (mean-field plus correlation).

        Returns
        -------
        e_tot : float
            Total energy.
        """
        return self.mf.e_tot + self.e_corr

    @property
    def t1(self):
        """Return the T1 amplitude.

        Returns
        -------
        t1 : numpy.ndarray (nocc, nvir)
            T1 amplitude.
        """
        return self.amplitudes["t1"]

    @property
    def t2(self):
        """Return the T2 amplitude.

        Returns
        -------
        t2 : numpy.ndarray (nocc, nocc, nvir, nvir)
            T2 amplitude.
        """
        return self.amplitudes["t2"]

    @property
    def l1(self):
        """Return the L1 amplitude.

        Returns
        -------
        l1 : numpy.ndarray (nvir, nocc)
            L1 amplitude.
        """
        return self.lambdas["l1"]

    @property
    def l2(self):
        """Return the L1 amplitude.

        Returns
        -------
        l1 : numpy.ndarray (nvir, nocc)
            L1 amplitude.
        """
        return self.lambdas["l2"]


if __name__ == "__main__":
    import numpy as np
    from pyscf import cc, gto, scf

    mol = gto.Mole()
    # mol.atom = "He 0 0 0"
    # mol.basis = "cc-pvdz"
    mol.atom = "H 0 0 0; F 0 0 1.1"
    mol.basis = "6-31g"
    mol.verbose = 5
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    ccsd_ref = cc.CCSD(mf)
    ccsd_ref.kernel()
    ccsd_ref.solve_lambda()

    # ccsd = REBCC(mf, rank=("SD", "", ""))
    # ccsd.kernel()
    # ccsd.solve_lambda()

    # print("e ", np.abs(ccsd.e_corr - ccsd_ref.e_corr))
    # print("t1", np.max(np.abs(ccsd.t1 - ccsd_ref.t1)))
    # print("t2", np.max(np.abs(ccsd.t2 - ccsd_ref.t2)))
    # print("l1", np.max(np.abs(ccsd.l1 - ccsd_ref.l1.T)))
    # print("l2", np.max(np.abs(ccsd.l2 - ccsd_ref.l2.transpose(2, 3, 0, 1))))
    # print("rdm1", np.max(np.abs(ccsd.make_rdm1_f() - ccsd_ref.make_rdm1())))
    # print("rdm2", np.max(np.abs(ccsd.make_rdm2_f() - ccsd_ref.make_rdm2())))

    ## Transpose issue I think:
    ##print(ccsd.make_rdm2_f())
    ##print(ccsd_ref.make_rdm2())

    # v1 = np.ones((ccsd.nocc,))
    # v2 = np.zeros((ccsd.nocc, ccsd.nocc, ccsd.nvir))

    # ra = ccsd_ref.eomip_method().gen_matvec()[0]([np.concatenate([v1.ravel(), v2.ravel()])])
    # rb1, rb2 = ccsd.hbar_matvec_ip(r1=v1, r2=v2)
    # rb = np.concatenate([rb1.ravel(), rb2.ravel()])
    # print("r", np.allclose(ra, rb/2))

    nbos = 5
    np.random.seed(1)
    g = np.random.random((nbos, mol.nao, mol.nao)) * 0.03
    g = g + g.transpose(0, 2, 1)
    omega = np.random.random((nbos)) * 0.5

    np.set_printoptions(edgeitems=1000, linewidth=1000, precision=8)
    ccsd = REBCC(mf, rank=("SD", "S", "S"), omega=omega, g=g, shift=False)
    ccsd.kernel()
    ccsd.solve_lambda()
    np.savetxt("tmp2.dat", ccsd.make_rdm1_b())

    # amps = ccsd.init_amps()
    # amps = ccsd.update_amps(amplitudes=amps)
    # print(amps["t1"])
    # print(amps["t2"])
    # print(amps["s1"])
    # print(amps["u11"])

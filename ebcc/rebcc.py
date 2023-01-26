"""Restricted electron-boson coupled cluster.
"""

import dataclasses
import functools
import importlib
import logging
import types
from typing import Any, Sequence, Union

import numpy as np
from pyscf import ao2mo, lib, scf

from ebcc import METHOD_TYPES, default_log, init_logging, reom, util

# TODO test bosonic RDMs and lambdas - only regression atm
# TODO math in docstrings
# TODO resolve G vs bare_G confusion
# TODO warnings if amplitudes/lambdas are not converged when calling i.e. DM funcs
# TODO update docstrings
# TODO orbspin with cluster spaces?
# TODO fix TensorSymm and benchmark third order again


class AbstractEBCC:
    """Abstract base class for EBCC objects."""

    pass


class Amplitudes(dict):
    """Amplitude container class. Consists of a dictionary with keys
    that are strings of the name of each amplitude, and values are
    arrays whose dimension depends on the particular amplitude.
    """

    pass


class ERIs(types.SimpleNamespace):
    """Electronic repulsion integral container class. Consists of a
    just-in-time namespace containing blocks of the integrals, with
    keys that are length-4 strings of `"o"` or `"v"` signifying
    whether the corresponding dimension is occupied or virtual.
    """

    def __init__(
        self,
        ebcc: AbstractEBCC,
        array: np.ndarray = None,
        slices: Sequence[slice] = None,
        mo_coeff: np.ndarray = None,
    ):
        self.mf = ebcc.mf
        self.slices = slices
        self.mo_coeff = mo_coeff
        self.array = array

        if self.mo_coeff is None:
            self.mo_coeff = ebcc.mo_coeff
        if not (isinstance(self.mo_coeff, (tuple, list)) or self.mo_coeff.ndim == 3):
            self.mo_coeff = [self.mo_coeff] * 4

        if self.slices is None:
            o = slice(None, ebcc.nocc)
            v = slice(ebcc.nocc, None)
            self.slices = {"o": o, "v": v}
        if not isinstance(self.slices, (tuple, list)):
            self.slices = [self.slices] * 4

    def __getattr__(self, key: str) -> np.ndarray:
        """Just-in-time attribute getter."""

        if self.array is None:
            if key not in self.__dict__.keys():
                coeffs = []
                for i, k in enumerate(key):
                    coeffs.append(self.mo_coeff[i][:, self.slices[i][k]])
                block = ao2mo.incore.general(self.mf._eri, coeffs, compact=False)
                block = block.reshape([c.shape[-1] for c in coeffs])
                self.__dict__[key] = block
            return self.__dict__[key]
        else:
            slices = []
            for i, k in enumerate(key):
                slices.append(self.slices[i][k])
            block = self.array[tuple(slices)]
            return block


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


class REBCC(AbstractEBCC):
    """Restricted electron-boson coupled cluster class.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF mean-field object.
    log : logging.Logger, optional
        Log to print output to. Default value is the global logger
        which outputs to `sys.stderr`.
    ansatz : str, optional
        Fermionic ansatz. Default value is "CCSD".
    boson_excitations : str, optional
        Rank of bosonic excitations. Default is "".
    fermion_coupling_rank : int, optional
        Rank of fermionic term in coupling. Default is 0.
    boson_coupling_rank : int, optional
        Rank of bosonic term in coupling. Default is 0.
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
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. Default value is `mf.mo_coeff`.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupancies. Default value is `mf.mo_occ`.
    fock : util.Namespace, optional
        Fock input. Default value is calculated using `get_fock()`.
    options : dataclasses.dataclass, optional
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
    g : util.Namespace
        Namespace containing blocks of the electron-boson coupling
        matrix. Each attribute should be a length-3 string of `b`,
        `o` or `v` signifying whether the corresponding axis is
        bosonic, occupied or virtual.
    G : numpy.ndarray (nbos,)
        Mean-field boson non-conserving term of the Hamiltonian.
    bare_G : numpy.ndarray (nbos,)
        Boson non-conserving term of the Hamiltonian.
    fock : util.Namespace
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
        mf: scf.hf.SCF,
        log: logging.Logger = None,
        ansatz: str = "CCSD",
        boson_excitations: str = "",
        fermion_coupling_rank: int = 0,
        boson_coupling_rank: int = 0,
        omega: np.ndarray = None,
        g: np.ndarray = None,
        G: np.ndarray = None,
        mo_coeff: np.ndarray = None,
        mo_occ: np.ndarray = None,
        fock: util.Namespace = None,
        options: Options = None,
        **kwargs,
    ):
        # Options:
        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)

        # Parameters:
        self.log = default_log if log is None else log
        self.mf = self._convert_mf(mf)
        self._mo_coeff = mo_coeff
        self._mo_occ = mo_occ
        self.rank = (
            ansatz,
            boson_excitations,
            fermion_coupling_rank,
            boson_coupling_rank,
        )
        self._eqns = self._get_eqns()

        # Boson parameters:
        if bool(self.rank[2]) != bool(self.rank[3]):
            raise ValueError(
                "Fermionic and bosonic coupling ranks must both be zero, or both non-zero."
            )
        self.omega = omega
        self.bare_g = g
        self.bare_G = G
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

        # Fock matrix:
        if fock is None:
            self.fock = self.get_fock()
        else:
            self.fock = fock

        # Attributes:
        self.e_corr = None
        self.amplitudes = None
        self.converged = False
        self.lambdas = None
        self.converged_lambda = False

        # Logging:
        init_logging(self.log)
        self.log.info("%s", self.name)
        self.log.info("%s", "*" * len(self.name))
        self.log.debug("")
        self.log.debug("Options:")
        self.log.info(" > Fermion ansatz:         %s", self.rank[0])
        self.log.info(" > Boson excitations:      %s", self.rank[1] if len(self.rank[1]) else None)
        self.log.info(" > Fermion coupling rank:  %s", self.rank[2])
        self.log.info(" > Boson coupling rank:    %s", self.rank[3])
        self.log.info(" > nmo:   %d", self.nmo)
        self.log.info(" > nocc:  %s", self.nocc)
        self.log.info(" > nvir:  %s", self.nvir)
        self.log.info(" > nbos:  %d", self.nbos)
        self.log.info(" > e_tol:  %s", self.options.e_tol)
        self.log.info(" > t_tol:  %s", self.options.t_tol)
        self.log.info(" > max_iter:  %s", self.options.max_iter)
        self.log.info(" > diis_space:  %s", self.options.diis_space)
        self.log.debug("")

    def kernel(self, eris: Union[ERIs, np.ndarray] = None) -> float:
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

        # Get the ERIs:
        eris = self.get_eris(eris)

        # Get the amplitude guesses:
        if self.amplitudes is None:
            amplitudes = self.init_amps(eris=eris)
        else:
            amplitudes = self.amplitudes

        # Get the initial energy:
        e_cc = e_init = self.energy(amplitudes=amplitudes, eris=eris)

        # Set up DIIS:
        diis = lib.diis.DIIS()
        diis.space = self.options.diis_space

        self.log.output("Solving for excitation amplitudes.")
        self.log.info("%4s %16s %16s %16s", "Iter", "Energy (corr.)", "Δ(Energy)", "Δ(Amplitudes)")
        self.log.info("%4d %16.10f", 0, e_init)

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            # Update the amplitudes, extrapolate with DIIS and calculate change:
            amplitudes_prev = amplitudes
            amplitudes = self.update_amps(amplitudes=amplitudes, eris=eris)
            vector = self.amplitudes_to_vector(amplitudes)
            vector = diis.update(vector)
            amplitudes = self.vector_to_amplitudes(vector)
            dt = np.linalg.norm(vector - self.amplitudes_to_vector(amplitudes_prev), ord=np.inf)

            # Update the energy and calculate change:
            e_prev = e_cc
            e_cc = self.energy(amplitudes=amplitudes, eris=eris)
            de = abs(e_prev - e_cc)

            self.log.info("%4d %16.10f %16.5g %16.5g", niter, e_cc, de, dt)

            # Check for convergence:
            converged = de < self.options.e_tol and dt < self.options.t_tol
            if converged:
                self.log.output("Converged.")
                break
        else:
            self.log.warning("Failed to converge.")

        # Include perturbative correction if required:
        if self.ansatz_is_perturbative:
            self.log.info("Computing perturbative energy correction.")
            e_pert = self.energy_perturbative(amplitudes=amplitudes, eris=eris)
            e_cc += e_pert
            self.log.info("E(pert) = %.10f", e_pert)

        # Update attributes:
        self.e_corr = e_cc
        self.amplitudes = amplitudes
        self.converged = converged

        self.log.debug("")
        self.log.output("E(corr) = %.10f", self.e_corr)
        self.log.output("E(tot)  = %.10f", self.e_tot)
        self.log.debug("")

        return e_cc

    def solve_lambda(self, amplitudes: Amplitudes = None, eris: Union[ERIs, np.ndarray] = None):
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

        # Get the ERIs:
        eris = self.get_eris(eris)

        # Get the amplitudes:
        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        # Get the lambda amplitude guesses:
        if self.lambdas is None:
            lambdas = self.init_lams(amplitudes=amplitudes)
        else:
            lambdas = self.lambdas

        # Set up DIIS:
        diis = lib.diis.DIIS()
        diis.space = self.options.diis_space

        self.log.output("Solving for de-excitation (lambda) amplitudes.")
        self.log.info("%4s %16s", "Iter", "Δ(Amplitudes)")

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            # Update the lambda amplitudes, extrapolate with DIIS and calculate change:
            lambdas_prev = lambdas
            lambdas = self.update_lams(amplitudes=amplitudes, lambdas=lambdas, eris=eris)
            vector = self.lambdas_to_vector(lambdas)
            vector = diis.update(vector)
            lambdas = self.vector_to_lambdas(vector)
            dl = np.linalg.norm(vector - self.lambdas_to_vector(lambdas_prev), ord=np.inf)

            self.log.info("%4d %16.5g", niter, dl)

            # Check for convergence:
            converged = dl < self.options.t_tol
            if converged:
                self.log.output("Converged.")
                break
        else:
            self.log.warning("Failed to converge.")

        self.log.debug("")

        # Update attributes:
        self.lambdas = lambdas
        self.converged_lambda = converged

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
        name = name.replace("(", "_").replace(")", "")
        eqns = importlib.import_module("ebcc.codegen.%s" % name)
        return eqns

    def _pack_codegen_kwargs(self, *extra_kwargs, eris=None):
        """Pack all the possible keyword arguments for generated code
        into a dictionary.
        """

        eris = self.get_eris(eris)

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
            eris = self.get_eris(eris)
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
                self.log.warning("No lambda amplitudes found, guesses will be used.")
                lambdas = self.init_lams(amplitudes=amplitudes)
            dicts.append(lambdas)

        if kwargs:
            dicts.append(kwargs)

        func = getattr(self._eqns, name, None)

        if func is None:
            raise util.ModelNotImplemented("%s for rank = %s" % (name, self.rank))

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

        eris = self.get_eris(eris)

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
                raise util.ModelNotImplemented
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
                raise util.ModelNotImplemented
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

    def energy_perturbative(self, eris=None, amplitudes=None):
        """Compute the perturbative part to the correlation energy.

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
        e_pert : float
            Perturbative correction to the correlation energy.
        """

        func, kwargs = self._load_function(
            "energy_perturbative",
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
                raise util.ModelNotImplemented
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
                raise util.ModelNotImplemented
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
            dm[np.diag_indices_from(dm)] -= xi * (dm_cre + dm_ann) - xi**2

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

        raise util.ModelNotImplemented  # TODO

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

        raise util.ModelNotImplemented  # TODO

    def ip_eom(self, options=None, **kwargs):
        return reom.IP_REOM(self, options=options, **kwargs)

    def ea_eom(self, options=None, **kwargs):
        return reom.EA_REOM(self, options=options, **kwargs)

    def ee_eom(self, options=None, **kwargs):
        return reom.EE_REOM(self, options=options, **kwargs)

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
            raise util.ModelNotImplemented

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise util.ModelNotImplemented

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
            raise util.ModelNotImplemented

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise util.ModelNotImplemented

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
            raise util.ModelNotImplemented

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise util.ModelNotImplemented

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
            raise util.ModelNotImplemented

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise util.ModelNotImplemented

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
        g : util.Namespace
            Namespace containing blocks of the electron-boson coupling
            matrix. Each attribute should be a length-3 string of
            `b`, `o` or `v` signifying whether the corresponding axis
            is bosonic, occupied or virtual.
        """

        boo = g[:, : self.nocc, : self.nocc]
        bov = g[:, : self.nocc, self.nocc :]
        bvo = g[:, self.nocc :, : self.nocc]
        bvv = g[:, self.nocc :, self.nocc :]

        g = util.Namespace(boo=boo, bov=bov, bvo=bvo, bvv=bvv)

        return g

    def get_fock(self):
        """Get blocks of the Fock matrix, shifted due to bosons where
        the ansatz requires.

        Returns
        -------
        fock : util.Namespace
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

        f = util.Namespace(oo=oo, ov=ov, vo=vo, vv=vv)

        return f

    def get_eris(self, eris=None):
        """Get blocks of the ERIs.

        Parameters
        ----------
        eris : np.ndarray or ERIs, optional.
            Electronic repulsion integrals, either in the form of a
            dense array or an ERIs object. Default value is `None`.

        Returns
        -------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.ERIs()`.
        """
        if (eris is None) or isinstance(eris, np.ndarray):
            return self.ERIs(self, array=eris)
        else:
            return eris

    @property
    def bare_fock(self):
        """Get the mean-field Fock matrix in the MO basis.

        Returns
        -------
        bare_fock : numpy.ndarray (nmo, nmo)
            The mean-field Fock matrix in the MO basis.
        """

        fock_ao = self.mf.get_fock()
        mo_coeff = self.mo_coeff

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
            return lib.einsum("I,I->", self.omega, self.xi**2)
        else:
            return 0.0

    @property
    def ansatz(self):
        """Return a string representation of the fermionic ansatz.

        Returns
        -------
        ansatz : str
            String representation of the fermion ansatz.
        """
        return self.rank[0]

    @property
    def ansatz_is_perturbative(self):
        """Return a boolean indicating whether the ansatz includes a
        perturbative correction e.g. CCSD(T).

        Returns
        -------
        perturbative : bool
            Boolean indicating if the ansatz is perturbatively
            corrected.
        """
        return "(" in self.ansatz and ")" in self.ansatz

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
        name = "R" + self.ansatz
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

        for i, op in enumerate(self.rank[:2]):
            # Remove any perturbative corrections
            while "(" in op:
                start = op.index("(")
                end = op.index(")")
                op = op[:start]
                if (end + 1) < len(op):
                    op += op[end+1:]

            if i == 0:
                # Check in order of longest -> shortest string in case
                # one method name starts with a substring equal to the
                # name of another method:
                for method_type in sorted(METHOD_TYPES, key=len)[::-1]:
                    if op.startswith(method_type):
                        op = op.replace(method_type, "")
                        break
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
    def mo_coeff(self):
        """Get the molecular orbital coefficients.

        Returns
        -------
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients.
        """
        if self._mo_coeff is None:
            return self.mf.mo_coeff
        return self._mo_coeff

    @property
    def mo_occ(self):
        """Get the molecular orbital occupancies.

        Returns
        -------
        mo_occ : numpy.ndarray
            Molecular orbital occupancies.
        """
        if self._mo_occ is None:
            return self.mf.mo_occ
        return self._mo_occ

    @property
    def nmo(self):
        """Get the number of molecular orbitals.

        Returns
        -------
        nmo : int
            Number of molecular orbitals.
        """
        return self.mo_occ.size

    @property
    def nocc(self):
        """Get the number of occupied molecular orbitals.

        Returns
        -------
        nocc : int
            Number of occupied molecular orbitals.
        """
        return np.sum(self.mo_occ > 0)

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

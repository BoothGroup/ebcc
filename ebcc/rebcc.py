"""Restricted electron-boson coupled cluster."""

import dataclasses

from pyscf import lib

from ebcc import default_log, init_logging
from ebcc import numpy as np
from ebcc import reom, util
from ebcc.ansatz import Ansatz
from ebcc.brueckner import BruecknerREBCC
from ebcc.cderis import RCDERIs
from ebcc.dump import Dump
from ebcc.eris import RERIs
from ebcc.fock import RFock
from ebcc.precision import types
from ebcc.space import Space


class EBCC:
    """Base class for EBCC."""

    pass


@dataclasses.dataclass
class Options:
    """
    Options for EBCC calculations.

    Attributes
    ----------
    shift : bool, optional
        If `True`, shift the boson operators such that the Hamiltonian is
        normal-ordered with respect to a coherent state. This removes the
        bosonic coupling to the static mean-field density, introducing a
        constant energy shift. Default value is `True`.
    e_tol : float, optional
        Threshold for convergence in the correlation energy. Default value
        is `1e-8`.
    t_tol : float, optional
        Threshold for convergence in the amplitude norm. Default value is
        `1e-8`.
    max_iter : int, optional
        Maximum number of iterations. Default value is `200`.
    diis_space : int, optional
        Number of amplitudes to use in DIIS extrapolation. Default value is
        `12`.
    """

    shift: bool = True
    e_tol: float = 1e-8
    t_tol: float = 1e-8
    max_iter: int = 200
    diis_space: int = 12


class REBCC(EBCC):
    r"""
    Restricted electron-boson coupled cluster class.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF mean-field object.
    log : logging.Logger, optional
        Log to print output to. Default value is the global logger which
        outputs to `sys.stderr`.
    ansatz : str or Ansatz, optional
        Overall ansatz, as a string representation or an `Ansatz` object.
        If `None`, construct from the individual ansatz parameters,
        otherwise override them. Default value is `None`.
    space : Space, optional
        Space object defining the size of frozen, correlated and active
        fermionic spaces. If `None`, all fermionic degrees of freedom are
        assumed correlated. Default value is `None`.
    omega : numpy.ndarray (nbos,), optional
        Bosonic frequencies. Default value is `None`.
    g : numpy.ndarray (nbos, nmo, nmo), optional
        Electron-boson coupling matrix corresponding to the bosonic
        annihilation operator i.e.

        .. math:: g_{xpq} p^\dagger q b

        The creation part is assume to be the fermionic transpose of this
        tensor to retain hermiticity in the overall Hamiltonian. Default
        value is `None`.
    G : numpy.ndarray (nbos,), optional
        Boson non-conserving term of the Hamiltonian i.e.

        .. math:: G_x (b^\dagger + b)

        Default value is `None`.
    mo_coeff : numpy.ndarray, optional
        Molecular orbital coefficients. Default value is `mf.mo_coeff`.
    mo_occ : numpy.ndarray, optional
        Molecular orbital occupancies. Default value is `mf.mo_occ`.
    fock : util.Namespace, optional
        Fock input. Default value is calculated using `get_fock()`.
    options : dataclasses.dataclass, optional
        Object containing the options. Default value is `Options()`.
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
    amplitudes : Namespace
        Cluster amplitudes.
    lambdas : Namespace
        Cluster lambda amplitudes.
    converged : bool
        Whether the coupled cluster equations converged.
    converged_lambda : bool
        Whether the lambda coupled cluster equations converged.
    omega : numpy.ndarray (nbos,)
        Bosonic frequencies.
    g : util.Namespace
        Namespace containing blocks of the electron-boson coupling matrix.
        Each attribute should be a length-3 string of `b`, `o` or `v`
        signifying whether the corresponding axis is bosonic, occupied, or
        virtual.
    G : numpy.ndarray (nbos,)
        Mean-field boson non-conserving term of the Hamiltonian.
    bare_G : numpy.ndarray (nbos,)
        Boson non-conserving term of the Hamiltonian.
    fock : util.Namespace
        Namespace containing blocks of the Fock matrix. Each attribute
        should be a length-2 string of `o` or `v` signifying whether the
        corresponding axis is occupied or virtual.
    bare_fock : numpy.ndarray (nmo, nmo)
        The mean-field Fock matrix in the MO basis.
    xi : numpy.ndarray (nbos,)
        Shift in bosonic operators to diagonalise the phononic Hamiltonian.
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
    make_rdm1_b(eris=None, amplitudes=None, lambdas=None, unshifted=None,
                hermitise=None)
        Build the bosonic one-particle reduced density matrix.
    make_rdm1_f(eris=None, amplitudes=None, lambdas=None, hermitise=None)
        Build the fermionic one-particle reduced density matrix.
    make_rdm2_f(eris=None, amplitudes=None, lambdas=None, hermitise=None)
        Build the fermionic two-particle reduced density matrix.
    make_eb_coup_rdm(eris=None, amplitudes=None, lambdas=None,
                     unshifted=True, hermitise=True)
        Build the electron-boson coupling reduced density matrices.
    hbar_matvec_ip(r1, r2, eris=None, amplitudes=None)
        Compute the product between a state vector and the EOM Hamiltonian
        for the IP.
    hbar_matvec_ea(r1, r2, eris=None, amplitudes=None)
        Compute the product between a state vector and the EOM Hamiltonian
        for the EA.
    hbar_matvec_ee(r1, r2, eris=None, amplitudes=None)
        Compute the product between a state vector and the EOM Hamiltonian
        for the EE.
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
        Construct a vector containing all of the amplitudes used in the
        given ansatz.
    vector_to_amplitudes(vector)
        Construct all of the amplitudes used in the given ansatz from a
        vector.
    lambdas_to_vector(lambdas)
        Construct a vector containing all of the lambda amplitudes used in
        the given ansatz.
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
        Get the mean-field boson non-conserving term of the Hamiltonian.
    get_g(g)
        Get the blocks of the electron-boson coupling matrix corresponding
        to the bosonic annihilation operator.
    get_fock()
        Get the blocks of the Fock matrix, shifted due to bosons where the
        ansatz requires.
    get_eris()
        Get blocks of the ERIs.
    """

    Options = Options
    ERIs = RERIs
    Fock = RFock
    CDERIs = RCDERIs
    Brueckner = BruecknerREBCC

    def __init__(
        self,
        mf,
        log=None,
        ansatz="CCSD",
        space=None,
        omega=None,
        g=None,
        G=None,
        mo_coeff=None,
        mo_occ=None,
        fock=None,
        options=None,
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
        self._mo_coeff = np.asarray(mo_coeff).astype(types[float]) if mo_coeff is not None else None
        self._mo_occ = np.asarray(mo_occ).astype(types[float]) if mo_occ is not None else None

        # Ansatz:
        if isinstance(ansatz, Ansatz):
            self.ansatz = ansatz
        else:
            self.ansatz = Ansatz.from_string(
                ansatz, density_fitting=getattr(self.mf, "with_df", None) is not None
            )
        self._eqns = self.ansatz._get_eqns(self.spin_type)

        # Space:
        if space is not None:
            self.space = space
        else:
            self.space = self.init_space()

        # Boson parameters:
        if bool(self.fermion_coupling_rank) != bool(self.boson_coupling_rank):
            raise ValueError(
                "Fermionic and bosonic coupling ranks must both be zero, or both non-zero."
            )
        self.omega = omega.astype(types[float]) if omega is not None else None
        self.bare_g = g.astype(types[float]) if g is not None else None
        self.bare_G = G.astype(types[float]) if G is not None else None
        if self.boson_ansatz != "":
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
        self.log.info("Options:")
        self.log.info(" > e_tol:  %s", self.options.e_tol)
        self.log.info(" > t_tol:  %s", self.options.t_tol)
        self.log.info(" > max_iter:  %s", self.options.max_iter)
        self.log.info(" > diis_space:  %s", self.options.diis_space)
        self.log.debug("")
        self.log.info("Ansatz: %s", self.ansatz)
        self.log.debug("")
        self.log.info("Space: %s", self.space)
        self.log.debug("")

    def kernel(self, eris=None):
        """
        Run the coupled cluster calculation.

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

        # Start a timer:
        timer = util.Timer()

        # Get the ERIs:
        eris = self.get_eris(eris)

        # Get the amplitude guesses:
        if self.amplitudes is None:
            amplitudes = self.init_amps(eris=eris)
        else:
            amplitudes = self.amplitudes

        # Get the initial energy:
        e_cc = e_init = self.energy(amplitudes=amplitudes, eris=eris)

        self.log.output("Solving for excitation amplitudes.")
        self.log.info("%4s %16s %16s %16s", "Iter", "Energy (corr.)", "Δ(Energy)", "Δ(Amplitudes)")
        self.log.info("%4d %16.10f", 0, e_init)

        if not self.ansatz.is_one_shot:
            # Set up DIIS:
            diis = lib.diis.DIIS()
            diis.space = self.options.diis_space

            converged = False
            for niter in range(1, self.options.max_iter + 1):
                # Update the amplitudes, extrapolate with DIIS and
                # calculate change:
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
            if self.ansatz.has_perturbative_correction:
                self.log.info("Computing perturbative energy correction.")
                e_pert = self.energy_perturbative(amplitudes=amplitudes, eris=eris)
                e_cc += e_pert
                self.log.info("E(pert) = %.10f", e_pert)

        else:
            converged = True

        # Update attributes:
        self.e_corr = e_cc
        self.amplitudes = amplitudes
        self.converged = converged

        self.log.debug("")
        self.log.output("E(corr) = %.10f", self.e_corr)
        self.log.output("E(tot)  = %.10f", self.e_tot)
        self.log.debug("")
        self.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.log.debug("")
        self.log.debug("")

        return e_cc

    def solve_lambda(self, amplitudes=None, eris=None):
        """
        Solve the lambda coupled cluster equations.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        """

        # Start a timer:
        timer = util.Timer()

        # Get the ERIs:
        eris = self.get_eris(eris)

        # Get the amplitudes:
        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        # If needed, precompute the perturbative part of the lambda
        # amplitudes:
        if self.ansatz.has_perturbative_correction:
            lambdas_pert = self.update_lams(eris=eris, amplitudes=amplitudes, perturbative=True)
        else:
            lambdas_pert = None

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
            # Update the lambda amplitudes, extrapolate with DIIS and
            # calculate change:
            lambdas_prev = lambdas
            lambdas = self.update_lams(
                amplitudes=amplitudes,
                lambdas=lambdas,
                lambdas_pert=lambdas_pert,
                eris=eris,
            )
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
        self.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.log.debug("")
        self.log.debug("")

        # Update attributes:
        self.lambdas = lambdas
        self.converged_lambda = converged

    def brueckner(self, *args, **kwargs):
        """
        Run a Brueckner orbital coupled cluster calculation.

        Returns
        -------
        e_cc : float
            Correlation energy.
        """

        bcc = self.Brueckner(self, *args, **kwargs)

        return bcc.kernel()

    def write(self, file):
        """
        Write the EBCC data to a file.

        Parameters
        ----------
        file : str
            Path of file to write to.
        """

        writer = Dump(file)
        writer.write(self)

    @classmethod
    def read(cls, file, log=None):
        """
        Read the data from a file.

        Parameters
        ----------
        file : str
            Path of file to read from.
        log : Logger, optional
            Logger to assign to the EBCC object.

        Returns
        -------
        ebcc : EBCC
            The EBCC object loaded from the file.
        """

        reader = Dump(file)
        cc = reader.read(cls=cls, log=log)

        return cc

    @staticmethod
    def _convert_mf(mf):
        """
        Convert the input PySCF mean-field object to the one required for
        the current class.
        """
        return mf.to_rhf()

    def _load_function(self, name, eris=False, amplitudes=False, lambdas=False, **kwargs):
        """
        Load a function from the generated code, and return a dict of
        arguments.
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
                self.log.warning("Using Λ = T* for %s", name)
                lambdas = self.init_lams(amplitudes=amplitudes)
            dicts.append(lambdas)

        if kwargs:
            dicts.append(kwargs)

        func = getattr(self._eqns, name, None)

        if func is None:
            raise util.ModelNotImplemented("%s for rank = %s" % (name, self.name))

        kwargs = self._pack_codegen_kwargs(*dicts, eris=eris)

        return func, kwargs

    def _pack_codegen_kwargs(self, *extra_kwargs, eris=None):
        """
        Pack all the possible keyword arguments for generated code
        into a dictionary.
        """
        # TODO change all APIs to take the space object instead of
        # nocc, nvir, nbos, etc.

        eris = self.get_eris(eris)

        omega = np.diag(self.omega) if self.omega is not None else None

        kwargs = dict(
            f=self.fock,
            v=eris,
            g=self.g,
            G=self.G,
            w=omega,
            space=self.space,
            nocc=self.space.ncocc,  # FIXME rename?
            nvir=self.space.ncvir,  # FIXME rename?
            nbos=self.nbos,
        )
        if isinstance(eris, self.CDERIs):
            kwargs["naux"] = self.mf.with_df.get_naoaux()
        for kw in extra_kwargs:
            if kw is not None:
                kwargs.update(kw)

        return kwargs

    def init_space(self):
        """
        Initialise the default `Space` object.

        Returns
        -------
        space : Space
            Space object in which all fermionic degrees of freedom are
            considered inactive.
        """

        space = Space(
            self.mo_occ > 0,
            np.zeros_like(self.mo_occ, dtype=bool),
            np.zeros_like(self.mo_occ, dtype=bool),
        )

        return space

    def init_amps(self, eris=None):
        """
        Initialise the amplitudes.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.

        Returns
        -------
        amplitudes : Namespace
            Cluster amplitudes.
        """

        eris = self.get_eris(eris)
        amplitudes = util.Namespace()

        # Build T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                amplitudes[name] = self.fock[key] / self.energy_sum(key)
            elif n == 2:
                key_t = key[0] + key[2] + key[1] + key[3]
                amplitudes[name] = eris[key_t].swapaxes(1, 2) / self.energy_sum(key)
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
            if nb == 1:
                amplitudes[name] = h[key] / self.energy_sum(key)
            else:
                shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        return amplitudes

    def init_lams(self, amplitudes=None):
        """
        Initialise the lambda amplitudes.

        Parameters
        ----------
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        lambdas : Namespace
            Cluster lambda amplitudes.
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

    def energy(self, eris=None, amplitudes=None):
        """
        Compute the correlation energy.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
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

    def energy_perturbative(self, eris=None, amplitudes=None, lambdas=None):
        """
        Compute the perturbative part to the correlation energy.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
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
            lambdas=lambdas,
        )

        return func(**kwargs)

    def update_amps(self, eris=None, amplitudes=None):
        """
        Update the amplitudes.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        amplitudes : Namespace
            Updated cluster amplitudes.
        """

        func, kwargs = self._load_function(
            "update_amps",
            eris=eris,
            amplitudes=amplitudes,
        )
        res = func(**kwargs)
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
        eris=None,
        amplitudes=None,
        lambdas=None,
        lambdas_pert=None,
        perturbative=False,
    ):
        """
        Update the lambda amplitudes.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.
        perturbative : bool, optional
            Whether to compute the perturbative part of the lambda
            amplitudes. Default value is `False`.

        Returns
        -------
        lambdas : Namespace
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
        res = func(**kwargs)
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

    def make_sing_b_dm(self, eris=None, amplitudes=None, lambdas=None):
        r"""
        Build the single boson density matrix:

        ..math :: \langle b^+ \rangle

        and

        ..math :: \langle b \rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

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
        r"""
        Build the bosonic one-particle reduced density matrix:

        ..math :: \langle b^+ b \rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.
        unshifted : bool, optional
            If `self.shift` is `True`, then `unshifted=True` applies the
            reverse transformation such that the bosonic operators are
            defined with respect to the unshifted bosons. Default value is
            `True`. Has no effect if `self.shift` is `False`.
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
        r"""
        Build the fermionic one-particle reduced density matrix:

        ..math :: \langle i^+ j \rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.
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
        r"""
        Build the fermionic two-particle reduced density matrix:

        ..math :: \Gamma_{ijkl} = \langle i^+ j^+ l k \rangle

        which is stored in Chemist's notation.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.
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
            dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2, 3, 0, 1))
            dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(1, 0, 3, 2))

        return dm

    def make_eb_coup_rdm(
        self,
        eris=None,
        amplitudes=None,
        lambdas=None,
        unshifted=True,
        hermitise=True,
    ):
        r"""
        Build the electron-boson coupling reduced density matrices:

        ..math :: \langle b^+ i^+ j \rangle

        and

        ..math :: \langle b i^+ j \rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.
        unshifted : bool, optional
            If `self.shift` is `True`, then `unshifted=True` applies the
            reverse transformation such that the bosonic operators are
            defined with respect to the unshifted bosons. Default value is
            `True`. Has no effect if `self.shift` is `False`.
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
            shift = util.einsum("x,ij->xij", self.xi, rdm1_f)
            dm_eb -= shift[None]

        return dm_eb

    def hbar_matvec_ip(self, r1, r2, eris=None, amplitudes=None):
        """
        Compute the product between a state vector and the EOM Hamiltonian
        for the IP.

        Parameters
        ----------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector. Keys are
            strings of the name of each vector, and values are arrays
            whose dimension depends on the particular sector.
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector resulting from
            the matrix-vector product with the input vectors. Keys are
            strings of the name of each vector, and values are arrays whose
            dimension depends on the particular sector.
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
        """
        Compute the product between a state vector and the EOM Hamiltonian
        for the EA.

        Parameters
        ----------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector. Keys are
            strings of the name of each vector, and values are arrays
            whose dimension depends on the particular sector.
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector resulting from
            the matrix-vector product with the input vectors. Keys are
            strings of the name of each vector, and values are arrays whose
            dimension depends on the particular sector.
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
        """
        Compute the product between a state vector and the EOM Hamiltonian
        for the EE.

        Parameters
        ----------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector. Keys are
            strings of the name of each vector, and values are arrays
            whose dimension depends on the particular sector.
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vectors : dict of (str, numpy.ndarray)
            Dictionary containing the vectors in each sector resulting from
            the matrix-vector product with the input vectors. Keys are
            strings of the name of each vector, and values are arrays whose
            dimension depends on the particular sector.
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
        """
        Get the bra IP vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        bras : dict of (str, numpy.ndarray)
            Dictionary containing the bra vectors in each sector. Keys are
            strings of the name of each sector, and values are arrays whose
            dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ip_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ea_mom_bras(self, eris=None, amplitudes=None, lambdas=None):
        """
        Get the bra EA vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        bras : dict of (str, numpy.ndarray)
            Dictionary containing the bra vectors in each sector. Keys are
            strings of the name of each sector, and values are arrays whose
            dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ea_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ee_mom_bras(self, eris=None, amplitudes=None, lambdas=None):
        """
        Get the bra EE vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        bras : dict of (str, numpy.ndarray)
            Dictionary containing the bra vectors in each sector. Keys are
            strings of the name of each sector, and values are arrays whose
            dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ee_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ip_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """
        Get the ket IP vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        kets : dict of (str, numpy.ndarray)
            Dictionary containing the ket vectors in each sector. Keys are
            strings of the name of each sector, and values are arrays whose
            dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ip_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ea_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """
        Get the ket IP vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        kets : dict of (str, numpy.ndarray)
            Dictionary containing the ket vectors in each sector. Keys are
            strings of the name of each sector, and values are arrays whose
            dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ea_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ee_mom_kets(self, eris=None, amplitudes=None, lambdas=None):
        """
        Get the ket EE vectors to construct EOM moments.

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        kets : dict of (str, numpy.ndarray)
            Dictionary containing the ket vectors in each sector. Keys are
            strings of the name of each sector, and values are arrays whose
            dimension depends on the particular sector.
        """

        func, kwargs = self._load_function(
            "make_ee_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        return func(**kwargs)

    def make_ip_1mom(self, eris=None, amplitudes=None, lambdas=None):
        r"""
        Build the first fermionic hole single-particle moment.

        .. math:: T_{pq} = \langle c_p^+ (H - E_0) c_q \rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        mom : numpy.ndarray (nmo, nmo)
            Array of the first moment.
        """

        raise util.ModelNotImplemented  # TODO

    def make_ea_1mom(self, eris=None, amplitudes=None, lambdas=None):
        r"""
        Build the first fermionic particle single-particle moment.

        .. math:: T_{pq} = \langle c_p (H - E_0) c_q^+ \rangle

        Parameters
        ----------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.get_eris()`.
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        mom : numpy.ndarray (nmo, nmo)
            Array of the first moment.
        """

        raise util.ModelNotImplemented  # TODO

    def ip_eom(self, options=None, **kwargs):
        """Get the IP EOM object."""
        return reom.IP_REOM(self, options=options, **kwargs)

    def ea_eom(self, options=None, **kwargs):
        """Get the EA EOM object."""
        return reom.EA_REOM(self, options=options, **kwargs)

    def ee_eom(self, options=None, **kwargs):
        """Get the EE EOM object."""
        return reom.EE_REOM(self, options=options, **kwargs)

    def amplitudes_to_vector(self, amplitudes):
        """
        Construct a vector containing all of the amplitudes used in the
        given ansatz.

        Parameters
        ----------
        amplitudes : Namespace, optional
            Cluster amplitudes. Default value is generated using
            `self.init_amps()`.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the amplitudes flattened and
            concatenated. Size depends on the ansatz.
        """

        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(amplitudes[name].ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(amplitudes[name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            vectors.append(amplitudes[name].ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        """
        Construct all of the amplitudes used in the given ansatz from a
        vector.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the amplitudes flattened
            and concatenated. Size depends on the ansatz.

        Returns
        -------
        amplitudes : Namespace
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

    def lambdas_to_vector(self, lambdas):
        """
        Construct a vector containing all of the lambda amplitudes used in
        the given ansatz.

        Parameters
        ----------
        lambdas : Namespace, optional
            Cluster lambda amplitudes. Default value is generated using
            `self.init_lams()`.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the lambdas flattened and
            concatenated. Size depends on the ansatz.
        """

        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(lambdas[name.replace("t", "l")].ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(lambdas["l" + name].ravel())

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            vectors.append(lambdas["l" + name].ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        """
        Construct all of the lambdas used in the given ansatz from a
        vector.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the lambdas flattened and
            concatenated. Size depends on the ansatz.

        Returns
        -------
        lambdas : Namespace
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

    def excitations_to_vector_ip(self, *excitations):
        """
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the IP.

        Parameters
        ----------
        *excitations : iterable of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of the
            name of each excitations, and values are arrays whose dimension
            depends on the particular excitation amplitude.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened and
            concatenated. Size depends on the ansatz.
        """

        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(excitations[m].ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def excitations_to_vector_ea(self, *excitations):
        """
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EA.

        Parameters
        ----------
        *excitations : iterable of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of the
            name of each excitations, and values are arrays whose dimension
            depends on the particular excitation amplitude.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened and
            concatenated. Size depends on the ansatz.
        """
        return self.excitations_to_vector_ip(*excitations)

    def excitations_to_vector_ee(self, *excitations):
        """
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EE.

        Parameters
        ----------
        *excitations : iterable of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of the
            name of each excitations, and values are arrays whose dimension
            depends on the particular excitation amplitude.

        Returns
        -------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened and
            concatenated. Size depends on the ansatz.
        """
        return self.excitations_to_vector_ip(*excitations)

    def vector_to_excitations_ip(self, vector):
        """
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the IP.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened and
            concatenated. Size depends on the ansatz.

        Returns
        -------
        excitations : tuple of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of the
            name of each excitations, and values are arrays whose dimension
            depends on the particular excitation amplitude.
        """

        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            shape = tuple(self.space.size(k) for k in key)
            size = np.prod(shape)
            excitations.append(vector[i0 : i0 + size].reshape(shape))
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        """
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EA.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened and
            concatenated. Size depends on the ansatz.

        Returns
        -------
        excitations : tuple of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of the
            name of each excitations, and values are arrays whose dimension
            depends on the particular excitation amplitude.
        """

        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
            shape = tuple(self.space.size(k) for k in key)
            size = np.prod(shape)
            excitations.append(vector[i0 : i0 + size].reshape(shape))
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return tuple(excitations)

    def vector_to_excitations_ee(self, vector):
        """
        Construct a vector containing all of the excitation amplitudes
        used in the given ansatz for the EE.

        Parameters
        ----------
        vector : numpy.ndarray
            Single vector consisting of all the excitations flattened and
            concatenated. Size depends on the ansatz.

        Returns
        -------
        excitations : tuple of numpy.ndarray
            Dictionary containing the excitations. Keys are strings of the
            name of each excitations, and values are arrays whose dimension
            depends on the particular excitation amplitude.
        """

        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            shape = tuple(self.space.size(k) for k in key)
            size = np.prod(shape)
            excitations.append(vector[i0 : i0 + size].reshape(shape))
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return tuple(excitations)

    def get_mean_field_G(self):
        """
        Get the mean-field boson non-conserving term of the Hamiltonian.

        Returns
        -------
        G_mf : numpy.ndarray (nbos,)
            Mean-field boson non-conserving term of the Hamiltonian.
        """

        # FIXME should this also sum in frozen orbitals?
        val = lib.einsum("Ipp->I", self.g.boo) * 2.0
        val -= self.xi * self.omega

        if self.bare_G is not None:
            val += self.bare_G

        return val

    def get_g(self, g):
        """
        Get blocks of the electron-boson coupling matrix corresponding to
        the bosonic annihilation operator.

        Parameters
        ----------
        g : numpy.ndarray (nbos, nmo, nmo)
            Array of the electron-boson coupling matrix.

        Returns
        -------
        g : Namespace
            Namespace containing blocks of the electron-boson coupling
            matrix. Each attribute should be a length-3 string of
            `b`, `o` or `v` signifying whether the corresponding axis
            is bosonic, occupied or virtual.
        """

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

    def get_fock(self):
        """
        Get blocks of the Fock matrix, shifted due to bosons where the
        ansatz requires.

        Returns
        -------
        fock : Namespace
            Namespace containing blocks of the Fock matrix. Each attribute
            should be a length-2 string of `o` or `v` signifying whether
            the corresponding axis is occupied or virtual.
        """
        return self.Fock(self, array=self.bare_fock)

    def get_eris(self, eris=None):
        """Get blocks of the ERIs.

        Parameters
        ----------
        eris : np.ndarray or ERIs, optional.
            Electronic repulsion integrals, either in the form of a dense
            array or an `ERIs` object. Default value is `None`.

        Returns
        -------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.ERIs()`.
        """
        if (eris is None) or isinstance(eris, np.ndarray):
            if (isinstance(eris, np.ndarray) and eris.ndim == 3) or getattr(
                self.mf, "with_df", None
            ):
                return self.CDERIs(self, array=eris)
            else:
                return self.ERIs(self, array=eris)
        else:
            return eris

    @property
    def bare_fock(self):
        """
        Get the mean-field Fock matrix in the MO basis, including frozen
        parts.

        Returns
        -------
        bare_fock : numpy.ndarray (nmo, nmo)
            The mean-field Fock matrix in the MO basis.
        """

        fock_ao = self.mf.get_fock().astype(types[float])
        mo_coeff = self.mo_coeff

        fock = util.einsum("pq,pi,qj->ij", fock_ao, mo_coeff, mo_coeff)

        return fock

    @property
    def xi(self):
        """
        Get the shift in bosonic operators to diagonalise the photonic
        Hamiltonian.

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
            xi = np.zeros_like(self.omega, dtype=types[float])

        return xi

    @property
    def const(self):
        """
        Get the shift in the energy from moving to polaritonic basis.

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
    def fermion_ansatz(self):
        """Get a string representation of the fermion ansatz."""
        return self.ansatz.fermion_ansatz

    @property
    def boson_ansatz(self):
        """Get a string representation of the boson ansatz."""
        return self.ansatz.boson_ansatz

    @property
    def fermion_coupling_rank(self):
        """Get an integer representation of the fermion coupling rank."""
        return self.ansatz.fermion_coupling_rank

    @property
    def boson_coupling_rank(self):
        """Get an integer representation of the boson coupling rank."""
        return self.ansatz.boson_coupling_rank

    @property
    def spin_type(self):
        """Get a string represent of the spin channel."""
        return "R"

    @property
    def name(self):
        """Get a string representation of the method name."""
        return self.spin_type + self.ansatz.name

    @property
    def mo_coeff(self):
        """
        Get the molecular orbital coefficients.

        Returns
        -------
        mo_coeff : numpy.ndarray
            Molecular orbital coefficients.
        """
        if self._mo_coeff is None:
            return np.asarray(self.mf.mo_coeff).astype(types[float])
        return self._mo_coeff

    @property
    def mo_occ(self):
        """
        Get the molecular orbital occupancies.

        Returns
        -------
        mo_occ : numpy.ndarray
            Molecular orbital occupancies.
        """
        if self._mo_occ is None:
            return np.asarray(self.mf.mo_occ).astype(types[float])
        return self._mo_occ

    @property
    def nmo(self):
        """Get the number of MOs."""
        return self.space.nmo

    @property
    def nocc(self):
        """Get the number of occupied MOs."""
        return self.space.nocc

    @property
    def nvir(self):
        """Get the number of virtual MOs."""
        return self.space.nvir

    @property
    def nbos(self):
        """
        Get the number of bosonic degrees of freedom.

        Returns
        -------
        nbos : int
            Number of bosonic degrees of freedom.
        """
        if self.omega is None:
            return 0
        return self.omega.shape[0]

    def energy_sum(self, subscript, signs_dict=None):
        """
        Get a direct sum of energies.

        Parameters
        ----------
        subscript : str
            The direct sum subscript, where each character indicates the
            sector for each energy. For the default slice characters, see
            `Space`. Occupied degrees of freedom are assumed to be
            positive, virtual and bosonic negative (the signs can be
            changed via the `signs_dict` keyword argument).
        signs_dict : dict, optional
            Dictionary defining custom signs for each sector. If `None`,
            initialised such that `["o", "O", "i"]` are positive, and
            `["v", "V", "a", "b"]` negative. Default value is `None`.

        Returns
        -------
        energy_sum : numpy.ndarray
            Array of energy sums.
        """

        n = 0

        def next_char():
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

    @property
    def e_tot(self):
        """
        Return the total energy (mean-field plus correlation).

        Returns
        -------
        e_tot : float
            Total energy.
        """
        return types[float](self.mf.e_tot) + self.e_corr

    @property
    def t1(self):
        """Get the T1 amplitudes."""
        return self.amplitudes["t1"]

    @property
    def t2(self):
        """Get the T2 amplitudes."""
        return self.amplitudes["t2"]

    @property
    def t3(self):
        """Get the T3 amplitudes."""
        return self.amplitudes["t3"]

    @property
    def l1(self):
        """Get the L1 amplitudes."""
        return self.lambdas["l1"]

    @property
    def l2(self):
        """Get the L2 amplitudes."""
        return self.lambdas["l2"]

    @property
    def l3(self):
        """Get the L3 amplitudes."""
        return self.lambdas["l3"]

import functools
import numpy as np
from typing import Tuple
from types import SimpleNamespace
from pyscf import lib, ao2mo


_supported_methods = {
        (2, 0, 0): "CCSD",
        (2, 1, 1): "CCSD-11",
        (2, 2, 1): "CCSD-21",
        (2, 2, 2): "CCSD-22",
}


class EBCC:
    def __init__(
            self,
            mf,
            rank: Tuple[int] = (2, 0, 0),
            omega: np.ndarray = None,
            g: np.ndarray = None,
            G: np.ndarray = None,
            shift: bool = True,
            opt_code: bool = False,
            e_tol: float = 1e-8,
            t_tol: float = 1e-8,
            max_iter: int = 500,
            diis_space: int = 12,
    ):
        self.mf = mf
        if rank not in _supported_methods:
            raise NotImplementedError
        self.rank = rank

        print("%s" % _supported_methods[self.rank])
        print(" > Rank of fermionic cluster excitations: %d" % self.rank[0])
        print(" > Rank of bosonic cluster excitations:   %d" % self.rank[1])
        print(" > Rank of bosonic-1e cluster coupling:   %d" % self.rank[2])

        # Cleanup - do we need all these states?
        self.omega = omega
        self.bare_G = G
        self.shift = shift

        self.e_tol = e_tol
        self.t_tol = t_tol
        self.max_iter = max_iter
        self.diis_space = diis_space

        self.amplitudes = None
        self.e_corr = None
        self.converged = False

        self.lambdas = None
        self.converged_lambda = False

        if not (self.rank[1] == self.rank[2] == 0):
            assert self.omega is not None
            assert g.shape == (self.nbos, self.nmo, self.nmo)

            self.g = self.get_g(g)
            self.G = self.get_mean_field_G()

            if self.shift:
                print(" > Energy shift due to polaritonic basis: %.10f" % self.const)
        else:
            assert self.nbos == 0
            self.shift = False
            self.g = None
            self.G = None

        print(" > Number of bosonic modes: %d" % self.nbos)
        print(" > e_tol: %s" % self.e_tol)
        print(" > t_tol: %s" % self.t_tol)
        print(" > max_iter: %s" % self.max_iter)
        print(" > diis_space: %s" % self.diis_space)

        self.fock = self.get_fock()

        # TODO generalise:
        if opt_code:
            raise NotImplementedError
        if self.rank == (2, 0, 0):
            import ebcc.codegen.ccsd as _eqns
        elif self.rank == (2, 1, 1):
            import ebcc.codegen.ccsd_1_1 as _eqns
        elif self.rank == (2, 2, 1):
            import ebcc.codegen.ccsd_2_1 as _eqns
        else:
            raise NotImplementedError
        self._eqns = _eqns

    def init_amps(self, eris=None):
        """Initialise amplitudes.
        """

        if eris is None:
            eris = self.get_eris()

        amplitudes = dict()
        e_ia = lib.direct_sum("i-a->ia", self.eo, self.ev)

        # Build T amplitudes:
        for n in range(1, self.rank[0]+1):
            if n == 1:
                amplitudes["t%d" % n] = self.fock.vo.T / e_ia
            elif n == 2:
                e_ijab = lib.direct_sum("ia,jb->ijab", e_ia, e_ia)
                amplitudes["t%d" % n] = eris.ovov.swapaxes(1, 2) / e_ijab
            else:
                amplitudes["t%d" % n] = np.zeros((self.nocc,) * n + (self.nvir,) * n)

        if not (self.rank[1] == self.rank[2] == 0):
            # Only true for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for n in range(1, self.rank[1]+1):
            if n == 1:
                amplitudes["s%d" % n] = -H / self.omega
            else:
                amplitudes["s%d" % n] = np.zeros((self.nbos,) * n)

        # Build U amplitudes:
        for n in range(1, self.rank[2]+1):
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
        for n in range(1, self.rank[0]+1):
            perm = list(range(n, 2*n)) + list(range(n))
            lambdas["l%d" % n] = amplitudes["t%d" % n].transpose(perm)

        # Build LS amplitudes:
        for n in range(1, self.rank[1]+1):
            # FIXME should these be transposed?
            lambdas["ls%d" % n] = amplitudes["s%d" % n]

        # Build LU amplitudes:
        for n in range(1, self.rank[2]+1):
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
        diis.space = self.diis_space

        print("Solving for excitation amplitudes.")
        print("%4s %16s %16s %16s" % ("Iter", "Energy (corr.)", "Δ(Energy)", "Δ(Amplitudes)"))
        print("%4d %16.10f" % (0, e_init))

        for niter in range(1, self.max_iter+1):
            amplitudes_prev = amplitudes
            amplitudes = self.update_amps(amplitudes=amplitudes, eris=eris)
            vector = self.amplitudes_to_vector(amplitudes)
            vector = diis.update(vector)
            amplitudes = self.vector_to_amplitudes(vector)
            dt = np.linalg.norm(vector - self.amplitudes_to_vector(amplitudes_prev))**2

            e_prev = e_cc
            e_cc = self.energy(amplitudes=amplitudes, eris=eris)
            de = abs(e_prev - e_cc)

            print("%4d %16.10f %16.5g %16.5g" % (niter, e_cc, de, dt))

            converged = de < self.e_tol and dt < self.t_tol
            if converged:
                print("Converged.")
                break
        else:
            print("Failed to converge.")

        self.e_corr = e_cc
        self.amplitudes = amplitudes
        self.converged = converged

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
        diis.space = self.diis_space

        print("Solving for de-excitation (lambda) amplitudes.")
        print("%4s %16s" % ("Iter", "Δ(Amplitudes)"))

        for niter in range(1, self.max_iter+1):
            lambdas_prev = lambdas
            lambdas = self.update_lams(amplitudes=amplitudes, lambdas=lambdas, eris=eris)
            vector = self.lambdas_to_vector(lambdas)
            vector = diis.update(vector)
            lambdas = self.vector_to_lambdas(vector)
            dl = np.linalg.norm(vector - self.lambdas_to_vector(lambdas_prev))**2

            print("%4d %16.5g" % (niter, dl))

            converged = dl < self.t_tol
            if converged:
                print("Converged.")
                break
        else:
            print("Failed to converge.")

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

    def energy(self, eris=None, amplitudes=None):
        """Compute the energy.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        try:
            func = self._eqns.energy
        except AttributeError:
            raise NotImplementedError("energy for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, eris=eris)

        res = func(**kwargs)

        return res["e_cc"]

    def update_amps(self, eris=None, amplitudes=None):
        """Update the amplitudes.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        try:
            func = self._eqns.update_amps
        except AttributeError:
            raise NotImplementedError("update_amps for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, eris=eris)

        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}
        e_ia = lib.direct_sum("i-a->ia", self.eo, self.ev)

        # Divide T amplitudes:
        for n in range(1, self.rank[0]+1):
            perm = list(range(0, n*2, 2)) + list(range(1, n*2, 2))
            d = functools.reduce(np.add.outer, [e_ia] * n)
            d = d.transpose(perm)
            res["t%d" % n] /= d
            res["t%d" % n] += amplitudes["t%d" % n]

        # Divide S amplitudes:
        for n in range(1, self.rank[1]+1):
            d = functools.reduce(np.add.outer, ([-self.omega] * n))
            res["s%d" % n] /= d
            res["s%d" % n] += amplitudes["s%d" % n]

        # Divide U amplitudes:
        for n in range(1, self.rank[2]+1):
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ia])
            res["u1%d" % n] /= d
            res["u1%d" % n] += amplitudes["u1%d" % n]

        return res

    def update_lams(self, eris=None, amplitudes=None, lambdas=None):
        """Update the lambda amplitudes.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        if lambdas is None:
            lambdas = self.lambdas
        if lambdas is None:
            lambdas = self.init_lams(amplitudes=amplitudes)

        try:
            func = self._eqns.update_lams
        except AttributeError:
            raise NotImplementedError("update_lams for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, lambdas, eris=eris)

        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}
        e_ai = lib.direct_sum("i-a->ai", self.eo, self.ev)

        # Divide T amplitudes:
        for n in range(1, self.rank[0]+1):
            perm = list(range(0, n*2, 2)) + list(range(1, n*2, 2))
            d = functools.reduce(np.add.outer, [e_ai] * n)
            d = d.transpose(perm)
            res["l%d" % n] /= d
            res["l%d" % n] += lambdas["l%d" % n]

        # Divide S amplitudes:
        for n in range(1, self.rank[1]+1):
            d = functools.reduce(np.add.outer, [-self.omega] * n)
            res["ls%d" % n] /= d
            res["ls%d" % n] += lambdas["ls%d" % n]

        # Divide U amplitudes:
        for n in range(1, self.rank[2]+1):
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ai])
            res["lu1%d" % n] /= d
            res["lu1%d" % n] += lambdas["lu1%d" % n]

        return res

    def make_sing_b_dm(self, eris=None, amplitudes=None, lambdas=None):
        """Build the single boson DM <b†> and <b>.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        if lambdas is None:
            lambdas = self.lambdas
        if lambdas is None:
            lambdas = self.init_lams(eris=eris)

        try:
            func = self._eqns.make_sing_b_dm
        except AttributeError:
            raise NotImplementedError("make_sing_b_dm for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, lambdas, eris=eris)

        res = func(**kwargs)

        return res["dm_b"]

    def make_rdm1_b(self, eris=None, amplitudes=None, lambdas=None):
        """Build the bosonic 1RDM <b† b>.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        if lambdas is None:
            lambdas = self.lambdas
        if lambdas is None:
            lambdas = self.init_lams(eris=eris)

        try:
            func = self._eqns.make_rdm1_b
        except AttributeError:
            raise NotImplementedError("make_rdm1_b for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, lambdas, eris=eris)

        res = func(**kwargs)

        return res["rdm1_b"]

    def make_rdm1_f(self, eris=None, amplitudes=None, lambdas=None):
        """Build the fermionic 1RDM.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        if lambdas is None:
            lambdas = self.lambdas
        if lambdas is None:
            lambdas = self.init_lams(amplitudes=amplitudes)

        try:
            func = self._eqns.make_rdm1_f
        except AttributeError:
            raise NotImplementedError("make_rdm1_f for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, lambdas, eris=eris)

        res = func(**kwargs)

        return res["rdm1_f"]

    def make_rdm2_f(self, eris=None, amplitudes=None, lambdas=None):
        """Build the fermionic 2RDM.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        if lambdas is None:
            lambdas = self.lambdas
        if lambdas is None:
            lambdas = self.init_lams(eris=eris)

        try:
            func = self._eqns.make_rdm2_f
        except AttributeError:
            raise NotImplementedError("make_rdm2_f for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, lambdas, eris=eris)

        res = func(**kwargs)

        return res["rdm2_f"]

    def make_eb_coup_rdm(self, eris=None, amplitudes=None, lambdas=None):
        """Build the electron-boson coupling RDMs <b† i† j> and <b i† j>.
        """

        if eris is None:
            eris = self.get_eris()

        if amplitudes is None:
            amplitudes = self.amplitudes
        if amplitudes is None:
            amplitudes = self.init_amps(eris=eris)

        if lambdas is None:
            lambdas = self.lambdas
        if lambdas is None:
            lambdas = self.init_lams(eris=eris)

        try:
            func = self._eqns.make_eb_coup_rdm
        except AttributeError:
            raise NotImplementedError("make_eb_coup_rdm for rank = %s" % self.rank)

        kwargs = self._pack_codegen_kwargs(amplitudes, lambdas, eris=eris)

        res = func(**kwargs)

        return res["rdm_eb"]

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

    def make_ip_eom_moms(self, order, eris=None, amplitudes=None, lambdas=None):
        """Build the fermionic hole single-particle EOM moments.

            T_{n, p, q} = <c†_p (H - E)^n c_q>
        """

        raise NotImplementedError  # TODO

    def make_ea_eom_moms(self, order, eris=None, amplitudes=None, lambdas=None):
        """Build the fermionic particle single-particle EOM moments.

            T_{n, p, q} = <c_p (H - E)^n c†_q>
        """

        raise NotImplementedError  # TODO

    def make_dd_eom_moms(self, order, eris=None, amplitudes=None, lambdas=None):
        """Build the fermionic density-density moments.
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

        if self.shift:
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

        for n in range(1, self.rank[0]+1):
            vectors.append(amplitudes["t%d" % n].ravel())

        for n in range(1, self.rank[1]+1):
            vectors.append(amplitudes["s%d" % n].ravel())

        for n in range(1, self.rank[2]+1):
            vectors.append(amplitudes["u1%d" % n].ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        """Construct all of the amplitudes used in the given ansatz
        from a vector.
        """

        amplitudes = {}
        i0 = 0

        for n in range(1, self.rank[0]+1):
            shape = (self.nocc,) * n + (self.nvir,) * n
            size = np.prod(shape)
            amplitudes["t%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in range(1, self.rank[1]+1):
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes["s%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in range(1, self.rank[2]+1):
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

        for n in range(1, self.rank[0]+1):
            vectors.append(lambdas["l%d" % n].ravel())

        for n in range(1, self.rank[1]+1):
            vectors.append(lambdas["ls%d" % n].ravel())

        for n in range(1, self.rank[2]+1):
            vectors.append(lambdas["lu1%d" % n].ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        """Construct all of the lambdas used in the given ansatz
        from a vector.
        """

        lambdas = {}
        i0 = 0

        for n in range(1, self.rank[0]+1):
            shape = (self.nvir,) * n + (self.nocc,) * n
            size = np.prod(shape)
            lambdas["l%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in range(1, self.rank[1]+1):
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["ls%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in range(1, self.rank[2]+1):
            shape = (self.nbos,) * n + (self.nvir, self.nocc)
            size = np.prod(shape)
            lambdas["lu1%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        return lambdas

    @property
    def xi(self):
        if self.shift:
            xi = lib.einsum("Iii->I", self.g.boo) * 2.0
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros_like(self.omega)
        return xi

    @property
    def const(self):
        if self.shift:
            return lib.einsum("I,I->", self.omega, self.xi**2)
        else:
            return 0.0

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
        return self.amplitudes["l1"]

    @property
    def l2(self):
        return self.amplitudes["l2"]



if __name__ == "__main__":
    from pyscf import gto, scf, cc
    import numpy as np

    mol = gto.Mole()
    #mol.atom = "He 0 0 0"
    #mol.basis = "cc-pvdz"
    mol.atom = "H 0 0 0; F 0 0 1.1"
    mol.basis = "6-31g"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    ccsd_ref = cc.CCSD(mf)
    ccsd_ref.kernel()

    ccsd = EBCC(mf, rank=(2, 0, 0))
    ccsd.kernel()
    ccsd.solve_lambda()

    print(np.abs(ccsd.e_corr - ccsd_ref.e_corr))
    print(np.max(np.abs(ccsd.t1 - ccsd_ref.t1)))
    print(np.max(np.abs(ccsd.t2 - ccsd_ref.t2)))
    print(np.max(np.abs(ccsd.make_rdm1_f() - ccsd_ref.make_rdm1())))
    print(np.max(np.abs(ccsd.make_rdm2_f() - ccsd_ref.make_rdm2())))

    # Transpose issue I think:
    #print(ccsd.make_rdm2_f())
    #print(ccsd_ref.make_rdm2())

    nbos = 5
    np.random.seed(1)
    g = np.random.random((nbos, mol.nao, mol.nao)) * 0.03
    g = g + g.transpose(0, 2, 1)
    omega = np.random.random((nbos)) * 0.5

    np.set_printoptions(edgeitems=1000, linewidth=1000, precision=8)
    ccsd = EBCC(mf, rank=(2, 2, 1), omega=omega, g=g)
    ccsd.kernel()
    ccsd.solve_lambda()

    #amps = ccsd.init_amps()
    #amps = ccsd.update_amps(amplitudes=amps)
    #print(amps["t1"])
    #print(amps["t2"])
    #print(amps["s1"])
    #print(amps["s2"])
    #print(amps["u11"])


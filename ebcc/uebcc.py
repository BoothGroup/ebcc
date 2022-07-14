"""Unrestricted electron-boson coupled cluster.
"""

import itertools
import functools
from types import SimpleNamespace
import numpy as np
from pyscf import ao2mo, lib
from ebcc import util, rebcc


class Amplitudes(rebcc.Amplitudes):
    """Amplitude container class. Consists of a dictionary with keys
    that are strings of the name of each amplitude. Values are
    namespaces with keys indicating whether each fermionic dimension
    is alpha (`"a"`) or beta (`"b"`) spin, and values are arrays whose
    dimension depends on the particular amplitude. For purely bosonic
    amplitudes the values of `Amplitudes` are simply arrays, with no
    fermionic spins to index.
    """

    pass


class ERIs(SimpleNamespace):
    """Electronic repulsion integral container class. Consists of a
    namespace with keys that are length-4 string of `"a"` or `"b"`
    signifying whether the corresponding dimension is alpha or beta
    spin, and values are of type `rebcc.ERIs`.
    """

    def __init__(self, ebcc):
        self.mf = ebcc.mf
        o = [slice(None, n) for n in ebcc.nocc]
        v = [slice(n, None) for n in ebcc.nocc]
        slices = [{"o": o1, "v": v1} for o1, v1 in zip(o, v)]
        mo_coeff = self.mf.mo_coeff

        self.aaaa = rebcc.ERIs(
                ebcc,
                slices=[slices[i] for i in (0, 0, 0, 0)],
                mo_coeff=[mo_coeff[i] for i in (0, 0, 0, 0)],
        )
        self.aabb = rebcc.ERIs(
                ebcc,
                slices=[slices[i] for i in (0, 0, 1, 1)],
                mo_coeff=[mo_coeff[i] for i in (0, 0, 1, 1)],
        )
        self.bbaa = rebcc.ERIs(
                ebcc,
                slices=[slices[i] for i in (1, 1, 0, 0)],
                mo_coeff=[mo_coeff[i] for i in (1, 1, 0, 0)],
        )
        self.bbbb = rebcc.ERIs(
                ebcc,
                slices=[slices[i] for i in (1, 1, 1, 1)],
                mo_coeff=[mo_coeff[i] for i in (1, 1, 1, 1)],
        )


def generate_spin_combinations(n):
    """Generate combinations of spin components for a given number
    of occupied and virtual axes.

    Parameters
    ----------
    n : int
        Order of cluster amplitude.

    Yields
    ------
    spin : str
        String of spin combination.

    Examples
    --------
    >>> generate_spin_combinations(1)
    ['aa', 'bb']
    >>> generate_spin_combinations(2)
    ['aaaa', 'abab', 'baba', 'bbbb']
    """

    for tup in itertools.product(("a", "b"), repeat=n):
        yield "".join(list(tup) * 2)


@util.inherit_docstrings
class UEBCC(rebcc.REBCC):
    Amplitudes = Amplitudes
    ERIs = ERIs

    @staticmethod
    def _convert_mf(mf):
        return mf.to_uhf()

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.get_eris()

        amplitudes = self.Amplitudes()
        e_ia = SimpleNamespace(
                aa=lib.direct_sum("i-a->ia", self.eo.a, self.ev.a),
                bb=lib.direct_sum("i-a->ia", self.eo.b, self.ev.b),
        )

        # Build T amplitudes
        for n in self.rank_numeric[0]:
            if n == 1:
                tn = SimpleNamespace(
                        aa=self.fock.aa.vo.T / e_ia.aa,
                        bb=self.fock.bb.vo.T / e_ia.bb,
                )
                amplitudes["t%d" % n] = tn
            elif n == 2:
                e_ijab = SimpleNamespace(
                        aaaa=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.aa),
                        abab=lib.direct_sum("ia,jb->ijab", e_ia.aa, e_ia.bb),
                        baba=lib.direct_sum("ia,jb->ijab", e_ia.bb, e_ia.aa),
                        bbbb=lib.direct_sum("ia,jb->ijab", e_ia.bb, e_ia.bb),
                )
                tn = SimpleNamespace(
                        aaaa=eris.aaaa.ovov.swapaxes(1, 2) / e_ijab.aaaa,
                        abab=eris.aabb.ovov.swapaxes(1, 2) / e_ijab.abab,
                        baba=eris.bbaa.ovov.swapaxes(1, 2) / e_ijab.baba,
                        bbbb=eris.bbbb.ovov.swapaxes(1, 2) / e_ijab.bbbb,
                )
                amplitudes["t%d" % n] = tn
            else:
                raise NotImplementedError  # TODO

        if not (self.rank[1] == self.rank[2] == ""):
            # Only tue for real-valued couplings:
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
                e_xia = SimpleNamespace(
                        aa=lib.direct_sum("ia-x->xia", e_ia.aa, self.omega),
                        bb=lib.direct_sum("ia-x->xia", e_ia.bb, self.omega),
                )
                u1n = SimpleNamespace(
                        aa=h.aa.bov / e_xia.aa,
                        bb=h.bb.bov / e_xia.bb,
                )
                amplitudes["u1%d" % n] = u1n
            else:
                u1n = SimpleNamespace(
                        aa=np.zeros((self.nbos,) * n + (self.nocc[0], self.nvir[0])),
                        bb=np.zeros((self.nbos,) * n + (self.nocc[1], self.nvir[1])),
                )
                amplitudes["u1%d" % n] = u1n

        return amplitudes

    def init_lams(self, amplitudes=None):
        if amplitudes is None:
            amplitudes = self.amplitudes

        lambdas = self.Amplitudes()

        # Build L amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(n, 2*n)) + list(range(n))
            lambdas["l%d" % n] = SimpleNamespace()
            for key in amplitudes["t%d" % n].__dict__.keys():
                ln = getattr(amplitudes["t%d" % n], key).transpose(perm)
                setattr(lambdas["l%d" % n], key, ln)

        # Build LS amplitudes:
        for n in self.rank_numeric[1]:
            lambdas["ls%d" % n] = amplitudes["s%d" % n]

        # Build LU amplitudes:
        for n in self.rank_numeric[2]:
            perm = list(range(n)) + [n+1, n]
            lambdas["lu1%d" % n] = SimpleNamespace()
            for key in amplitudes["u1%d" % n].__dict__.keys():
                lu1n = getattr(amplitudes["u1%d" % n], key).transpose(perm)
                setattr(lambdas["lu1%d" % n], key, lu1n)

        return lambdas

    def update_amps(self, eris=None, amplitudes=None):
        func, kwargs = self._load_function(
                "update_amps",
                eris=eris,
                amplitudes=amplitudes,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        e_ia = SimpleNamespace(
                aa=lib.direct_sum("i-a->ia", self.eo.a, self.ev.a),
                bb=lib.direct_sum("i-a->ia", self.eo.b, self.ev.b),
        )

        # Divide T amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(0, n*2, 2)) + list(range(1, n*2, 2))
            for key in generate_spin_combinations(n):
                es = [getattr(e_ia, key[i]+key[i+n]) for i in range(n)]
                d = functools.reduce(np.add.outer, es)
                d = d.transpose(perm)
                tn = getattr(res["t%d" % n], key)
                tn /= d
                tn += getattr(amplitudes["t%d" % n], key)
                setattr(res["t%d" % n], key, tn)

        # Divide S amplitudes:
        for n in self.rank_numeric[1]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n))
            res["s%d" % n] /= d
            res["s%d" % n] += amplitudes["s%d" % n]

        # Divide U amplitudes:
        for n in self.rank_numeric[2]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ia.aa])
            tn = res["u1%d" % n].aa
            tn /= d
            tn += amplitudes["u1%d" % n].aa
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ia.bb])
            res["u1%d" % n].aa = tn
            tn = res["u1%d" % n].bb
            tn /= d
            tn += amplitudes["u1%d" % n].bb
            res["u1%d" % n].bb = tn

        return res

    def update_lams(self, eris=None, amplitudes=None, lambdas=None):
        func, kwargs = self._load_function(
                "update_lams",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )
        res = func(**kwargs)
        res = {key.rstrip("new"): val for key, val in res.items()}

        e_ai = SimpleNamespace(
                aa=lib.direct_sum("i-a->ai", self.eo.a, self.ev.a),
                bb=lib.direct_sum("i-a->ai", self.eo.b, self.ev.b),
        )

        # Divide T amplitudes:
        for n in self.rank_numeric[0]:
            perm = list(range(0, n*2, 2)) + list(range(1, n*2, 2))
            for key in generate_spin_combinations(n):
                es = [getattr(e_ai, key[i]+key[i+n]) for i in range(n)]
                d = functools.reduce(np.add.outer, es)
                d = d.transpose(perm)
                tn = getattr(res["l%d" % n], key)
                tn /= d
                tn += getattr(lambdas["l%d" % n], key)
                setattr(res["l%d" % n], key, tn)

        # Divide S amplitudes:
        for n in self.rank_numeric[1]:
            d = functools.reduce(np.add.outer, [-self.omega] * n)
            res["ls%d" % n] /= d
            res["ls%d" % n] += lambdas["ls%d" % n]

        # Divide U amplitudes:
        for n in self.rank_numeric[2]:
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ai.aa])
            tn = res["lu1%d" % n].aa
            tn /= d
            tn += lambdas["lu1%d" % n].aa
            d = functools.reduce(np.add.outer, ([-self.omega] * n) + [e_ai.bb])
            res["lu1%d" % n].aa = tn
            tn = res["lu1%d" % n].bb
            tn /= d
            tn += lambdas["lu1%d" % n].bb
            res["lu1%d" % n].bb = tn

        return res

    def make_rdm1_f(self, eris=None, amplitudes=None, lambdas=None, hermitise=True):
        func, kwargs = self._load_function(
                "make_rdm1_f",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:
            dm.aa = 0.5 * (dm.aa + dm.aa.T)
            dm.bb = 0.5 * (dm.bb + dm.bb.T)

        return dm

    def make_rdm2_f(self, eris=None, amplitudes=None, lambdas=None, hermitise=True):
        func, kwargs = self._load_function(
                "make_rdm2_f",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:
            transpose = lambda x: 0.125 * (
                    + x.transpose(0, 1, 2, 3)
                    + x.transpose(1, 0, 2, 3)
                    + x.transpose(0, 1, 3, 2)
                    + x.transpose(1, 0, 3, 2)
                    + x.transpose(2, 3, 0, 1)
                    + x.transpose(2, 3, 1, 0)
                    + x.transpose(3, 2, 0, 1)
                    + x.transpose(3, 2, 1, 0)
            )
            dm.aaaa = transpose(dm.aaaa)
            dm.aabb = transpose(dm.aabb)
            dm.bbaa = transpose(dm.bbaa)
            dm.bbbb = transpose(dm.bbbb)

        return dm

    def make_eb_coup_rdm(self, eris=None, amplitudes=None, lambdas=None, unshifted=True, hermitise=True):
        func, kwargs = self._load_function(
                "make_eb_coup_rdm",
                eris=eris,
                amplitudes=amplitudes,
                lambdas=lambdas,
        )

        dm_eb = func(**kwargs)

        if hermitise:
            dm_eb[0].aa = 0.5 * (dm_eb[0].aa + dm_eb[1].aa.transpose(0, 2, 1))
            dm_eb[0].bb = 0.5 * (dm_eb[0].bb + dm_eb[1].bb.transpose(0, 2, 1))
            dm_eb[1].aa = dm_eb[0].aa.transpose(0, 2, 1).copy()
            dm_eb[1].bb = dm_eb[0].bb.transpose(0, 2, 1).copy()

        if unshifted and self.options.shift:
            rdm1_f = self.make_rdm1_f(hermitise=hermitise)
            shift = lib.einsum("x,ij->xij", self.xi, rdm1_f.aa)
            dm_eb.aa -= shift[None]
            shift = lib.einsum("x,ij->xij", self.xi, rdm1_f.bb)
            dm_eb.bb -= shift[None]

        return dm_eb

    def get_mean_field_G(self):
        val  = lib.einsum("Ipp->I", self.g.aa.boo)
        val += lib.einsum("Ipp->I", self.g.bb.boo)
        val -= self.xi * self.omega

        if self.bare_G is not None:
            # Require bare_G to have a spin index for now:
            assert np.shape(self.bare_G) == val.shape
            val += self.bare_G

        return val

    def get_g(self, g):
        if np.array(g).ndim != 4:
            g = np.array([g, g])

        gs = SimpleNamespace()

        boo = g[0][:, :self.nocc[0], :self.nocc[0]]
        bov = g[0][:, :self.nocc[0], self.nocc[0]:]
        bvo = g[0][:, self.nocc[0]:, :self.nocc[0]]
        bvv = g[0][:, self.nocc[0]:, self.nocc[0]:]
        gs.aa = SimpleNamespace(boo=boo, bov=bov, bvo=bvo, bvv=bvv)

        boo = g[1][:, :self.nocc[1], :self.nocc[1]]
        bov = g[1][:, :self.nocc[1], self.nocc[1]:]
        bvo = g[1][:, self.nocc[1]:, :self.nocc[1]]
        bvv = g[1][:, self.nocc[1]:, self.nocc[1]:]
        gs.bb = SimpleNamespace(boo=boo, bov=bov, bvo=bvo, bvv=bvv)

        return gs

    @property
    def bare_fock(self):
        fock = lib.einsum("npq,npi,nqj->nij", self.mf.get_fock(), self.mf.mo_coeff, self.mf.mo_coeff)
        fock = SimpleNamespace(aa=fock[0], bb=fock[1])
        return fock

    def get_fock(self):
        fock = self.bare_fock
        if self.options.shift:
            xi = self.xi

        f = SimpleNamespace()

        oo = fock.aa[:self.nocc[0], :self.nocc[0]]
        ov = fock.aa[:self.nocc[0], self.nocc[0]:]
        vo = fock.aa[self.nocc[0]:, :self.nocc[0]]
        vv = fock.aa[self.nocc[0]:, self.nocc[0]:]

        if self.options.shift:
            oo -= lib.einsum("I,Iij->ij", xi, g.aa.boo + g.aa.boo.transpose(0, 2, 1))
            ov -= lib.einsum("I,Iia->ia", xi, g.aa.bov + g.aa.bvo.transpose(0, 2, 1))
            vo -= lib.einsum("I,Iai->ai", xi, g.aa.bvo + g.aa.bov.transpose(0, 2, 1))
            vv -= lib.einsum("I,Iab->ab", xi, g.aa.bvv + g.aa.bvv.transpose(0, 2, 1))

        f.aa = SimpleNamespace(oo=oo, ov=ov, vo=vo, vv=vv)

        oo = fock.bb[:self.nocc[1], :self.nocc[1]]
        ov = fock.bb[:self.nocc[1], self.nocc[1]:]
        vo = fock.bb[self.nocc[1]:, :self.nocc[1]]
        vv = fock.bb[self.nocc[1]:, self.nocc[1]:]

        if self.options.shift:
            oo -= lib.einsum("I,Iij->ij", xi, g.bb.boo + g.bb.boo.transpose(0, 2, 1))
            ov -= lib.einsum("I,Iia->ia", xi, g.bb.bov + g.bb.bvo.transpose(0, 2, 1))
            vo -= lib.einsum("I,Iai->ai", xi, g.bb.bvo + g.bb.bov.transpose(0, 2, 1))
            vv -= lib.einsum("I,Iab->ab", xi, g.bb.bvv + g.bb.bvv.transpose(0, 2, 1))

        f.bb = SimpleNamespace(oo=oo, ov=ov, vo=vo, vv=vv)

        return f

    def get_eris(self):
        return self.ERIs(self)

    def amplitudes_to_vector(self, amplitudes):
        vectors = []

        for n in self.rank_numeric[0]:
            for key in generate_spin_combinations(n):
                tn = getattr(amplitudes["t%d" % n], key)
                vectors.append(tn.ravel())

        for n in self.rank_numeric[1]:
            vectors.append(amplitudes["s%d" % n].ravel())

        for n in self.rank_numeric[2]:
            vectors.append(amplitudes["u1%d" % n].aa.ravel())
            vectors.append(amplitudes["u1%d" % n].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector):
        amplitudes = self.Amplitudes()
        i0 = 0
        spin_indices = {"a": 0, "b": 1}

        for n in self.rank_numeric[0]:
            amplitudes["t%d" % n] = SimpleNamespace()
            for key in generate_spin_combinations(n):
                shape = tuple(self.nocc[spin_indices[s]] for s in key[:n]) \
                        + tuple(self.nvir[spin_indices[s]] for s in key[n:])
                size = np.prod(shape)
                tn = vector[i0:i0+size].reshape(shape)
                setattr(amplitudes["t%d" % n], key, tn)
                i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            amplitudes["s%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[2]:
            amplitudes["u1%d" % n] = SimpleNamespace()
            shape = (self.nbos,) * n + (self.nocc[0], self.nvir[0])
            size = np.prod(shape)
            amplitudes["u1%d" % n].aa = vector[i0:i0+size].reshape(shape)
            i0 += size
            shape = (self.nbos,) * n + (self.nocc[1], self.nvir[1])
            size = np.prod(shape)
            amplitudes["u1%d" % n].bb = vector[i0:i0+size].reshape(shape)
            i0 += size

        return amplitudes

    def lambdas_to_vector(self, lambdas):
        vectors = []

        for n in self.rank_numeric[0]:
            for key in generate_spin_combinations(n):
                tn = getattr(lambdas["l%d" % n], key)
                vectors.append(tn.ravel())

        for n in self.rank_numeric[1]:
            vectors.append(lambdas["ls%d" % n].ravel())

        for n in self.rank_numeric[2]:
            vectors.append(lambdas["lu1%d" % n].aa.ravel())
            vectors.append(lambdas["lu1%d" % n].bb.ravel())

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector):
        lambdas = self.Amplitudes()
        i0 = 0
        spin_indices = {"a": 0, "b": 1}

        for n in self.rank_numeric[0]:
            lambdas["l%d" % n] = SimpleNamespace()
            for key in generate_spin_combinations(n):
                shape = tuple(self.nvir[spin_indices[s]] for s in key[:n]) \
                        + tuple(self.nocc[spin_indices[s]] for s in key[n:])
                size = np.prod(shape)
                tn = vector[i0:i0+size].reshape(shape)
                setattr(lambdas["l%d" % n], key, tn)
                i0 += size

        for n in self.rank_numeric[1]:
            shape = (self.nbos,) * n
            size = np.prod(shape)
            lambdas["ls%d" % n] = vector[i0:i0+size].reshape(shape)
            i0 += size

        for n in self.rank_numeric[2]:
            lambdas["lu1%d" % n] = SimpleNamespace()
            shape = (self.nbos,) * n + (self.nvir[0], self.nocc[0])
            size = np.prod(shape)
            lambdas["lu1%d" % n].aa = vector[i0:i0+size].reshape(shape)
            i0 += size
            shape = (self.nbos,) * n + (self.nvir[1], self.nocc[1])
            size = np.prod(shape)
            lambdas["lu1%d" % n].bb = vector[i0:i0+size].reshape(shape)
            i0 += size

        return lambdas

    @property
    def name(self):
        return "UCC" + "-".join(self.rank).rstrip("-")

    @property
    def nmo(self):
        assert self.mf.mo_occ[0].size == self.mf.mo_occ[1].size
        return self.mf.mo_occ[0].size

    @property
    def nocc(self):
        return tuple(np.sum(mo_occ > 0) for mo_occ in self.mf.mo_occ)

    @property
    def nvir(self):
        return tuple(self.nmo - nocc for nocc in self.nocc)

    @property
    def eo(self):
        eo = SimpleNamespace(
                a=np.diag(self.fock.aa.oo),
                b=np.diag(self.fock.bb.oo),
        )
        return eo

    @property
    def ev(self):
        ev = SimpleNamespace(
                a=np.diag(self.fock.aa.vv),
                b=np.diag(self.fock.bb.vv),
        )
        return ev



if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = "H 0 0 0; F 0 0 1.1"
    mol.basis = "6-31g"
    mol.verbose = 5
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    nbos = 5
    np.random.seed(1)
    g = np.random.random((nbos, mol.nao, mol.nao)) * 0.03
    g = g + g.transpose(0, 2, 1)
    omega = np.random.random((nbos)) * 0.5

    cc = UEBCC(mf, rank=("SD", "SD", "S"), omega=omega, g=g, shift=False)
    cc.kernel()
    cc.solve_lambda()

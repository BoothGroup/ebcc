"""Unrestricted equation-of-motion solver.
"""

import dataclasses
import functools
import warnings

import numpy as np
from pyscf import lib

from ebcc import reom, util


@util.inherit_docstrings
class UEOM(reom.REOM):
    """Unrestricted equation-of-motion base class."""

    def _argsort_guess(self, diag):
        if self.options.koopmans:
            r_mf = self.vector_to_amplitudes(diag)[0]
            size = r_mf.a.size + r_mf.b.size
            arg = np.argsort(np.diag(diag[:size]))
        else:
            arg = np.argsort(diag)

        return arg

    def _quasiparticle_weight(self, r1):
        return np.linalg.norm(r1.a) ** 2 + np.linalg.norm(r1.b) ** 2

    def moments(self, nmom, eris=None, amplitudes=None, hermitise=True):
        if eris is None:
            eris = self.ebcc.get_eris()
        if amplitudes is None:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments = util.Namespace(
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
        return self.excitation_type.upper() + "-UEOM-" + self.ebcc.name


@util.inherit_docstrings
class IP_UEOM(UEOM, reom.IP_REOM):
    """Unrestricted equation-of-motion class for ionisation potentials."""

    def diag(self, eris=None):
        parts = []
        e_ia = util.Namespace(
            aa=lib.direct_sum("i-a->ia", self.ebcc.eo.a, self.ebcc.ev.a),
            bb=lib.direct_sum("i-a->ia", self.ebcc.eo.b, self.ebcc.ev.b),
        )
        e_i = self.ebcc.eo

        for n in self.ansatz.correlated_cluster_ranks[0]:
            spin_part = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                tensors = []
                for i, s in enumerate(comb[:n]):
                    if i == (n - 1):
                        tensors.append(getattr(e_i, s))
                    else:
                        tensors.append(getattr(e_ia, s + s))
                perm = list(range(0, n * 2, 2)) + list(range(1, (n - 1) * 2, 2))
                d = functools.reduce(np.add.outer, tensors)
                d = d.transpose(perm)
                setattr(spin_part, comb, d)
            parts.append(spin_part)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        bras = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            amps_a = []
            amps_b = []

            m = 0
            for n in self.ansatz.correlated_cluster_ranks[0]:
                amp_a = util.Namespace()
                amp_b = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncocc for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncvir for s in spin[n:]
                    )
                    setattr(amp_a, spin, getattr(bras_raw[m], "a" + spin, {i: np.zeros(shape)})[i])
                    setattr(amp_b, spin, getattr(bras_raw[m], "b" + spin, {i: np.zeros(shape)})[i])
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for n in self.ansatz.correlated_cluster_ranks[1]:
                raise util.ModelNotImplemented

            for nf in self.ansatz.correlated_cluster_ranks[2]:
                for nb in self.ansatz.correlated_cluster_ranks[3]:
                    raise util.ModelNotImplemented

            bras.a.append(self.amplitudes_to_vector(*amps_a))
            bras.b.append(self.amplitudes_to_vector(*amps_b))

        bras.a = np.array(bras.a)
        bras.b = np.array(bras.b)

        return bras

    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        kets = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            j = (Ellipsis, i)
            amps_a = []
            amps_b = []

            m = 0
            for n in self.ansatz.correlated_cluster_ranks[0]:
                amp_a = util.Namespace()
                amp_b = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncocc for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncvir for s in spin[n:]
                    )
                    setattr(amp_a, spin, getattr(kets_raw[m], spin + "a", {j: np.zeros(shape)})[j])
                    setattr(amp_b, spin, getattr(kets_raw[m], spin + "b", {j: np.zeros(shape)})[j])
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for n in self.ansatz.correlated_cluster_ranks[1]:
                raise util.ModelNotImplemented

            for nf in self.ansatz.correlated_cluster_ranks[2]:
                for nb in self.ansatz.correlated_cluster_ranks[3]:
                    raise util.ModelNotImplemented

            kets.a.append(self.amplitudes_to_vector(*amps_a))
            kets.b.append(self.amplitudes_to_vector(*amps_b))

        kets.a = np.array(kets.a)
        kets.b = np.array(kets.b)

        return kets


@util.inherit_docstrings
class EA_UEOM(UEOM, reom.EA_REOM):
    """Unrestricted equation-of-motion class for electron affinities."""

    def diag(self, eris=None):
        parts = []
        e_ai = util.Namespace(
            aa=lib.direct_sum("a-i->ai", self.ebcc.ev.a, self.ebcc.eo.a),
            bb=lib.direct_sum("a-i->ai", self.ebcc.ev.b, self.ebcc.eo.b),
        )
        e_a = self.ebcc.ev

        for n in self.ansatz.correlated_cluster_ranks[0]:
            spin_part = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                tensors = []
                for i, s in enumerate(comb[:n]):
                    if i == (n - 1):
                        tensors.append(getattr(e_a, s))
                    else:
                        tensors.append(getattr(e_ai, s + s))
                perm = list(range(0, n * 2, 2)) + list(range(1, (n - 1) * 2, 2))
                d = functools.reduce(np.add.outer, tensors)
                d = d.transpose(perm)
                setattr(spin_part, comb, d)
            parts.append(spin_part)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def bras(self, eris=None):
        bras_raw = list(self._bras(eris=eris))
        bras = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            amps_a = []
            amps_b = []

            m = 0
            for n in self.ansatz.correlated_cluster_ranks[0]:
                amp_a = util.Namespace()
                amp_b = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncvir for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncocc for s in spin[n:]
                    )
                    setattr(amp_a, spin, getattr(bras_raw[m], "a" + spin, {i: np.zeros(shape)})[i])
                    setattr(amp_b, spin, getattr(bras_raw[m], "b" + spin, {i: np.zeros(shape)})[i])
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for n in self.ansatz.correlated_cluster_ranks[1]:
                raise util.ModelNotImplemented

            for nf in self.ansatz.correlated_cluster_ranks[2]:
                for nb in self.ansatz.correlated_cluster_ranks[3]:
                    raise util.ModelNotImplemented

            bras.a.append(self.amplitudes_to_vector(*amps_a))
            bras.b.append(self.amplitudes_to_vector(*amps_b))

        bras.a = np.array(bras.a)
        bras.b = np.array(bras.b)

        return bras

    def kets(self, eris=None):
        kets_raw = list(self._kets(eris=eris))
        kets = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            j = (Ellipsis, i)
            amps_a = []
            amps_b = []

            m = 0
            for n in self.ansatz.correlated_cluster_ranks[0]:
                amp_a = util.Namespace()
                amp_b = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncvir for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncocc for s in spin[n:]
                    )
                    setattr(amp_a, spin, getattr(kets_raw[m], spin + "a", {j: np.zeros(shape)})[j])
                    setattr(amp_b, spin, getattr(kets_raw[m], spin + "b", {j: np.zeros(shape)})[j])
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for n in self.ansatz.correlated_cluster_ranks[1]:
                raise util.ModelNotImplemented

            for nf in self.ansatz.correlated_cluster_ranks[2]:
                for nb in self.ansatz.correlated_cluster_ranks[3]:
                    raise util.ModelNotImplemented

            kets.a.append(self.amplitudes_to_vector(*amps_a))
            kets.b.append(self.amplitudes_to_vector(*amps_b))

        kets.a = np.array(kets.a)
        kets.b = np.array(kets.b)

        return kets


@util.inherit_docstrings
class EE_UEOM(UEOM, reom.EE_REOM):
    """Unrestricted equation-of-motion class for neutral excitations."""

    def _quasiparticle_weight(self, r1):
        return np.linalg.norm(r1.aa) ** 2 + np.linalg.norm(r1.bb) ** 2

    def diag(self, eris=None):
        parts = []
        e_ia = util.Namespace(
            aa=lib.direct_sum("i-a->ia", self.ebcc.eo.a, self.ebcc.ev.a),
            bb=lib.direct_sum("i-a->ia", self.ebcc.eo.b, self.ebcc.ev.b),
        )

        for n in self.ansatz.correlated_cluster_ranks[0]:
            spin_part = util.Namespace()
            for comb in util.generate_spin_combinations(n):
                tensors = []
                for i, s in enumerate(comb[:n]):
                    tensors.append(getattr(e_ia, s + s))
                perm = list(range(0, n * 2, 2)) + list(range(1, n * 2, 2))
                d = functools.reduce(np.add.outer, tensors)
                d = d.transpose(perm)
                setattr(spin_part, comb, d)
            parts.append(spin_part)

        for n in self.ansatz.correlated_cluster_ranks[1]:
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def bras(self, eris=None):  # pragma: no cover
        raise util.ModelNotImplemented("EE moments for UEBCC not working.")

        bras_raw = list(self.ebcc.make_ee_mom_bras(eris=eris))
        bras = util.Namespace(aa=[], bb=[])

        for i in range(self.nmo):
            for j in range(self.nmo):
                amps_aa = []
                amps_bb = []

                m = 0
                for n in self.ansatz.correlated_cluster_ranks[0]:
                    amp_aa = util.Namespace()
                    amp_bb = util.Namespace()
                    for spin in util.generate_spin_combinations(n):
                        shape = tuple(
                            [
                                *[self.space["ab".index(s)].ncocc for s in spin[:n]],
                                *[self.space["ab".index(s)].ncvir for s in spin[n:]],
                            ]
                        )
                        setattr(
                            amp_aa,
                            spin,
                            getattr(bras_raw[m], "aa" + spin, {(i, j): np.zeros(shape)})[i, j],
                        )
                        setattr(
                            amp_bb,
                            spin,
                            getattr(bras_raw[m], "bb" + spin, {(i, j): np.zeros(shape)})[i, j],
                        )
                    amps_aa.append(amp_aa)
                    amps_bb.append(amp_bb)
                    m += 1

                for n in self.ansatz.correlated_cluster_ranks[1]:
                    raise util.ModelNotImplemented

                for nf in self.ansatz.correlated_cluster_ranks[2]:
                    for nb in self.ansatz.correlated_cluster_ranks[3]:
                        raise util.ModelNotImplemented

                bras.aa.append(self.amplitudes_to_vector(*amps_aa))
                bras.bb.append(self.amplitudes_to_vector(*amps_bb))

        bras.aa = np.array(bras.aa)
        bras.bb = np.array(bras.bb)

        bras.aa = bras.aa.reshape(self.nmo, self.nmo, *bras.aa.shape[1:])
        bras.bb = bras.bb.reshape(self.nmo, self.nmo, *bras.bb.shape[1:])

        return bras

    def kets(self, eris=None):  # pragma: no cover
        raise util.ModelNotImplemented("EE moments for UEBCC not working.")

        kets_raw = list(self.ebcc.make_ee_mom_kets(eris=eris))
        kets = util.Namespace(aa=[], bb=[])

        for i in range(self.nmo):
            for j in range(self.nmo):
                k = (Ellipsis, i, j)
                amps_aa = []
                amps_bb = []

                m = 0
                for n in self.ansatz.correlated_cluster_ranks[0]:
                    amp_aa = util.Namespace()
                    amp_bb = util.Namespace()
                    for spin in util.generate_spin_combinations(n):
                        shape = tuple(
                            [
                                *[self.space["ab".index(s)].ncocc for s in spin[:n]],
                                *[self.space["ab".index(s)].ncvir for s in spin[n:]],
                            ]
                        )
                        setattr(
                            amp_aa,
                            spin,
                            getattr(kets_raw[m], spin + "aa", {k: np.zeros(shape)})[k],
                        )
                        setattr(
                            amp_bb,
                            spin,
                            getattr(kets_raw[m], spin + "bb", {k: np.zeros(shape)})[k],
                        )
                    amps_aa.append(amp_aa)
                    amps_bb.append(amp_bb)
                    m += 1

                for n in self.ansatz.correlated_cluster_ranks[1]:
                    raise util.ModelNotImplemented

                for nf in self.ansatz.correlated_cluster_ranks[2]:
                    for nb in self.ansatz.correlated_cluster_ranks[3]:
                        raise util.ModelNotImplemented

                kets.aa.append(self.amplitudes_to_vector(*amps_aa))
                kets.bb.append(self.amplitudes_to_vector(*amps_bb))

        kets.aa = np.array(kets.aa)
        kets.bb = np.array(kets.bb)

        kets.aa = kets.aa.reshape(self.nmo, self.nmo, *kets.aa.shape[1:])
        kets.bb = kets.bb.reshape(self.nmo, self.nmo, *kets.bb.shape[1:])

        return kets

    def moments(
        self, nmom, eris=None, amplitudes=None, hermitise=True, diagonal_only=True
    ):  # pragma: no cover
        raise util.ModelNotImplemented("EE moments for UEBCC not working.")

        if not diagonal_only:
            warnings.warn("Constructing EE moments with `diagonal_only=False` will be very slow.")

        if eris is None:
            eris = self.ebcc.get_eris()
        if amplitudes is None:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments = util.Namespace(
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

            if hermitise:
                m = getattr(moments, spin)
                setattr(moments, spin, 0.5 * (m + m.transpose(0, 3, 4, 1, 2)))

        return moments

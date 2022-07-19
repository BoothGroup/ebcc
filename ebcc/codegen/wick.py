"""Generate sympy expressions in spin-orbital notation using qwick.
"""

import os
import sympy
from fractions import Fraction
from qwick.wick import apply_wick as _apply_wick
from qwick.index import Idx
from qwick.operator import FOperator, BOperator
from qwick.expression import *
from qwick.convenience import *
from qwick.codegen import ALPHA, BETA, FERMION, SCALAR_BOSON
from ebcc.codegen.convenience_extra import *


def get_factor(*indices):
    """Return the factor for a given operator, where a factor
    1/2 is raised to the power of the number of identical
    indices.
    """

    counts = {"occ": 0, "vir": 0, "nm": 0}

    for index in indices:
        counts[index.space] += 1

    n = sum(max(0, count-1) for count in counts.values())

    return Fraction(1, 2**n)


def get_rank(rank=("SD", "", "")):
    """Return ranks for each string.
    """

    values = {
        "S": 1,
        "D": 2,
        "T": 3,
        "Q": 4,
    }

    return tuple(tuple(values[i] for i in j) for j in rank)


def get_hamiltonian(rank=("SD", "", ""), compress=False):
    """Define the core Hamiltonian.
    """

    # fermions
    h1e = one_e("f", ["occ", "vir"], norder=True)
    h2e = two_e("v", ["occ", "vir"], norder=True, compress=compress)
    h = h1e + h2e

    particles = {
            "f": ((FERMION, 0), (FERMION, 0)),
            "v": ((FERMION, 0), (FERMION, 1), (FERMION, 0), (FERMION, 1)),
    }

    # bosons
    if rank[1] or rank[2]:
        hp = two_p("w") + one_p("G")
        hep = ep11("g", ["occ", "vir"], ["nm"], norder=True, name2="gc")
        h += hp + hep

        particles["G"] = ((SCALAR_BOSON, 0),)
        particles["w"] = ((SCALAR_BOSON, 0), (SCALAR_BOSON, 0))
        particles["g"] = ((SCALAR_BOSON, 0), (FERMION, 1), (FERMION, 1)),
        particles["gc"] = ((SCALAR_BOSON, 0), (FERMION, 1), (FERMION, 1)),

    return h, particles


def get_bra_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left projection spaces.
    """

    rank = get_rank(rank)
    bras = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(i, True) for i in occ] + [FOperator(a, False) for a in vir[::-1]]
        tensors = [Tensor(occ + vir, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, False) for x in nm]
        tensors = [Tensor(nm, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # fermion-boson coupling
    for n in rank[2]:
        i = Idx(0, "occ") if occs is None else occs[0]
        a = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, False) for x in nm] + \
                [FOperator(i, True), FOperator(a, False)]
        tensors = [Tensor(nm + [i, a], "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    return tuple(bras)


def get_bra_ip_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces.
    """

    rank = get_rank(rank)
    bras = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n-1)] if virs is None else virs[:n-1]
        operators = [FOperator(i, True) for i in occ] + [FOperator(a, False) for a in vir[::-1]]
        tensors = [Tensor(occ + vir, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(bras)


def get_bra_ea_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces.
    """

    rank = get_rank(rank)
    bras = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n-1)] if occs is None else occs[:n-1]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(i, True) for i in occ] + [FOperator(a, False) for a in vir[::-1]]
        tensors = [Tensor(vir + occ, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(bras)


def get_ket_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define right projection spaces.
    """

    rank = get_rank(rank)
    kets = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        tensors = [Tensor(occ + vir, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, True) for x in nm]
        tensors = [Tensor(nm, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # fermion-boson coupling
    for n in rank[2]:
        i = Idx(0, "occ") if occs is None else occs[0]
        a = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        operators = [BOperator(x, True) for x in nm] + \
                [FOperator(a, True), FOperator(i, False)]
        tensors = [Tensor(nm + [i, a], "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    return tuple(kets)


def get_ket_ip_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces.
    """

    rank = get_rank(rank)
    kets = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n-1)] if virs is None else virs[:n-1]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        tensors = [Tensor(occ + vir, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(kets)


def get_ket_ea_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define left IP projection spaces.
    """

    rank = get_rank(rank)
    kets = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n-1)] if occs is None else occs[:n-1]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        operators = [FOperator(a, True) for a in vir[::-1]] + [FOperator(i, False) for i in occ]
        tensors = [Tensor(vir + occ, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(kets)


def get_r_ip_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define space of trial vector to apply an IP hamiltonian to.
    """

    rank = get_rank(rank)
    rs = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n-1)] if virs is None else virs[:n-1]
        name = "r{n}".format(n=n)
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        tensors = [Tensor(occ + vir, name, sym=TensorSym([[0, 1, 2], [1, 0, 2]], [1, -1]) if n == 2 else TensorSym([], []))]  # FIXME symm
        rs.append(Expression([Term(scalar, sums, tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(rs)


def get_r_ea_spaces(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define space of trial vector to apply an EA hamiltonian to.
    """

    rank = get_rank(rank)
    rs = []

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n-1)] if occs is None else occs[:n-1]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        name = "r{n}".format(n=n)
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(a) for a in vir] + [Sigma(i) for i in occ]
        operators = [FOperator(a, True) for a in vir[::-1]] + [FOperator(i, False) for i in occ]
        tensors = [Tensor(vir + occ, name, sym=TensorSym([[0, 1, 2], [1, 0, 2]], [1, -1]) if n == 2 else TensorSym([], []))]  # FIXME symm
        rs.append(Expression([Term(scalar, sums, tensors, operators, [])]))

    # boson
    for n in rank[1]:
        raise NotImplementedError  # TODO

    # fermion-boson coupling
    for n in rank[2]:
        raise NotImplementedError  # TODO

    return tuple(rs)


def get_symm(particles):
    raise NotImplementedError


def get_excitation_ansatz(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define excitation amplitudes for the given ansatz.
    """

    rank = get_rank(rank)
    t = []
    particles = {}

    # fermion
    for n in rank[0]:
        occ = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        vir = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        name = "t{n}".format(n=n)
        tensors = [Tensor(occ + vir, name, sym=get_sym(True) if n == 2 else TensorSym([], []))]  # FIXME get symmetry for all n
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        t.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((FERMION, x) for x in range(n)) * 2

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(*nm)
        sums = [Sigma(x) for x in nm]
        name = "s{n}".format(n=n)
        tensors = [Tensor(nm, name, sym=TensorSym([(0, 1), (1, 0)], [1, 1]) if n == 2 else TensorSym([], []))]  # FIXME symmetry
        operators = [BOperator(x, True) for x in nm]
        t.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n))

    # fermion-boson coupling
    for n in rank[2]:
        i = Idx(0, "occ") if occs is None else occs[0]
        a = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(i, a, *nm)
        sums = [Sigma(x) for x in nm] + [Sigma(i), Sigma(a)]
        name = "u1{n}".format(n=n)
        tensors = [Tensor(nm + [i, a], name, sym=TensorSym([(0, 1, 2, 3), (1, 0, 2, 3)], [1, 1]) if n == 2 else TensorSym([], []))]
        operators = [BOperator(x, True) for x in nm] + [FOperator(a, True), FOperator(i, False)]
        t.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n)) + ((FERMION, n), (FERMION, n))

    return Expression(t), particles


def get_deexcitation_ansatz(rank=("SD", "", ""), occs=None, virs=None, nms=None):
    """Define de-excitation amplitudes for the given ansatz.
    """

    rank = get_rank(rank)
    l = []
    particles = {}

    # fermion
    for n in rank[0]:
        # Swapped variables names so I can copy the code:
        vir = [Idx(i, "occ") for i in range(n)] if occs is None else occs[:n]
        occ = [Idx(a, "vir") for a in range(n)] if virs is None else virs[:n]
        scalar = get_factor(*occ, *vir)
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        name = "l{n}".format(n=n)
        tensors = [Tensor(occ + vir, name, sym=get_sym(True) if n == 2 else TensorSym([], []))]  # FIXME get symmetry for all n
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        l.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((FERMION, x) for x in range(n)) * 2

    # boson
    for n in rank[1]:
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(*nm)
        sums = [Sigma(x) for x in nm]
        name = "ls{n}".format(n=n)
        tensors = [Tensor(nm, name, sym=TensorSym([(0, 1), (1, 0)], [1, 1]) if n == 2 else TensorSym([], []))]  # FIXME symmetry
        operators = [BOperator(x, False) for x in nm]
        l.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n))

    # fermion-boson coupling
    for n in rank[2]:
        # Swapped variables names so I can copy the code:
        a = Idx(0, "occ") if occs is None else occs[0]
        i = Idx(0, "vir") if virs is None else virs[0]
        nm = [Idx(x, "nm", fermion=False) for x in range(n)] if nms is None else nms[:n]
        scalar = get_factor(i, a, *nm)
        sums = [Sigma(x) for x in nm] + [Sigma(i), Sigma(a)]
        name = "lu1{n}".format(n=n)
        tensors = [Tensor(nm + [i, a], name, sym=TensorSym([(0, 1, 2, 3), (1, 0, 2, 3)], [1, 1]) if n == 2 else TensorSym([], []))]
        operators = [BOperator(x, False) for x in nm] + [FOperator(a, True), FOperator(i, False)]
        l.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n)) + ((FERMION, n), (FERMION, n))

    return Expression(l), particles


def bch(h, t, max_commutator=4):
    """Construct successive orders of \\bar{H} and return a list.
    """

    def factorial(n):
        if n in (0, 1):
            return 1
        elif n > 1:
            return n * factorial(n-1)
        else:
            raise ValueError("{n}!".format(n=n))

    comms = [h]
    for i in range(max_commutator):
        comms.append(commute(comms[-1], t))

    hbars = [h]
    for i in range(1, len(comms)):
        scalar = Fraction(1, factorial(i))
        hbars.append(hbars[-1] + comms[i] * scalar)

    return hbars

construct_hbar = bch


def apply_wick(expr):
    """Apply wick.
    """

    return _apply_wick(expr)

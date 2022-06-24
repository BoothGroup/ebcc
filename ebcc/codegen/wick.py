"""Generate sympy expressions in spin-orbital notation using qwick.
"""

import sympy
from fractions import Fraction
from qwick.wick import apply_wick
from qwick.index import Idx
from qwick.operator import FOperator, BOperator
from qwick.expression import *
from qwick.convenience import *
from qwick.codegen import ALPHA, BETA, FERMION, SCALAR_BOSON
from ebcc.codegen.convenience_extra import *


def get_hamiltonian(rank=(2, 0, 0), compress=False):
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
    if rank[0] > 0 or rank[1] > 0:
        hp = two_p("w") + one_p("G")
        hep = ep11("g", ["occ", "vir"], ["nm"], norder=True, name2="gc")
        h += hp + hep

        particles["G"] = ((SCALAR_BOSON, 0),)
        particles["w"] = ((SCALAR_BOSON, 0), (SCALAR_BOSON, 0))
        particles["g"] = ((SCALAR_BOSON, 0), (FERMION, 1), (FERMION, 1)),
        particles["gc"] = ((SCALAR_BOSON, 0), (FERMION, 1), (FERMION, 1)),

    return h, particles


def get_bra_spaces(rank=(2, 0, 0)):
    """Define left projection spaces.
    """

    bras = []

    # fermion
    for n in range(1, rank[0]+1):
        occ = [Idx(i, "occ") for i in range(n)]
        vir = [Idx(a, "vir") for a in range(n)]
        operators = [FOperator(i, True) for i in occ] + [FOperator(a, False) for a in vir[::-1]]
        tensors = [Tensor(vir + occ, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in range(1, rank[1]+1):
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        operators = [BOperator(x, False) for x in nm]
        tensors = [Tensor(nm, "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    # fermion-boson coupling
    for n in range(1, rank[2]+1):
        i = Idx(0, "occ")
        a = Idx(0, "vir")
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        operators = [BOperator(x, False) for x in nm] + \
                [FOperator(i, True), FOperator(a, False)]
        tensors = [Tensor(nm + [a, i], "")]
        bras.append(Expression([Term(1, [], tensors, operators, [])]))

    return tuple(bras)


def get_ket_spaces(rank=(2, 0, 0)):
    """Define right projection spaces.
    """

    kets = []

    # fermion
    for n in range(1, rank[0]+1):
        occ = [Idx(i, "occ") for i in range(n)]
        vir = [Idx(a, "vir") for a in range(n)]
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        tensors = [Tensor(occ + vir, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # boson
    for n in range(1, rank[1]+1):
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        operators = [BOperator(x, True) for x in nm]
        tensors = [Tensor(nm, "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    # fermion-boson coupling
    for n in range(1, rank[2]+1):
        i = Idx(0, "occ")
        a = Idx(0, "vir")
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        operators = [BOperator(x, True) for x in nm] + \
                [FOperator(a, True), FOperator(i, False)]
        tensors = [Tensor(nm + [i, a], "")]
        kets.append(Expression([Term(1, [], tensors, operators, [])]))

    return tuple(kets)


def get_symm(particles):
    raise NotImplementedError


def get_excitation_ansatz(rank=(2, 0, 0)):
    """Define excitation amplitudes for the given ansatz.
    """

    t = []
    particles = {}

    # fermion
    for n in range(1, rank[0]+1):
        occ = [Idx(i, "occ") for i in range(n)]
        vir = [Idx(a, "vir") for a in range(n)]
        scalar = Fraction(1, 4**(n-1))
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        name = "t{n}".format(n=n)
        tensors = [Tensor(vir + occ, name, sym=get_sym(True) if n == 2 else TensorSym([], []))]  # FIXME get symmetry for all n
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        t.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((FERMION, x) for x in range(n)) * 2

    # boson
    for n in range(1, rank[1]+1):
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        scalar = Fraction(1, 2**(n-1))
        sums = [Sigma(x) for x in nm]
        name = "s{n}".format(n=n)
        tensors = [Tensor(nm, name, sym=TensorSym([(0, 1), (1, 0)], [1, 1]) if n == 2 else TensorSym([], []))]  # FIXME symmetry
        operators = [BOperator(x, True) for x in nm]
        t.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n))

    # fermion-boson coupling
    for n in range(1, rank[2]+1):
        i = Idx(0, "occ")
        a = Idx(0, "vir")
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        scalar = Fraction(1, 2**(n-1))
        sums = [Sigma(x) for x in nm] + [Sigma(i), Sigma(a)]
        name = "u1{n}".format(n=n)
        tensors = [Tensor(nm + [a, i], name, sym=TensorSym([(0, 1, 2, 3), (1, 0, 2, 3)], [1, 1]) if n == 2 else TensorSym([], []))]
        operators = [BOperator(x, True) for x in nm] + [FOperator(a, True), FOperator(i, False)]
        t.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n)) + ((FERMION, n), (FERMION, n))

    return Expression(t), particles


def get_deexcitation_ansatz(rank=(2, 0, 0)):
    """Define de-excitation amplitudes for the given ansatz.
    """

    l = []
    particles = {}

    # fermion
    for n in range(1, rank[0]+1):
        # Swapped variables names so I can copy the code:
        vir = [Idx(i, "occ") for i in range(n)]
        occ = [Idx(a, "vir") for a in range(n)]
        scalar = Fraction(1, 4**(n-1))
        sums = [Sigma(i) for i in occ] + [Sigma(a) for a in vir]
        name = "l{n}".format(n=n)
        tensors = [Tensor(vir + occ, name, sym=get_sym(True) if n == 2 else TensorSym([], []))]  # FIXME get symmetry for all n
        operators = [FOperator(a, True) for a in vir] + [FOperator(i, False) for i in occ[::-1]]
        l.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((FERMION, x) for x in range(n)) * 2

    # boson
    for n in range(1, rank[1]+1):
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        scalar = Fraction(1, 2**(n-1))
        sums = [Sigma(x) for x in nm]
        name = "ls{n}".format(n=n)
        tensors = [Tensor(nm, name, sym=TensorSym([(0, 1), (1, 0)], [1, 1]) if n == 2 else TensorSym([], []))]  # FIXME symmetry
        operators = [BOperator(x, False) for x in nm]
        l.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n))

    # fermion-boson coupling
    for n in range(1, rank[2]+1):
        # Swapped variables names so I can copy the code:
        a = Idx(0, "occ")
        i = Idx(0, "vir")
        nm = [Idx(x, "nm", fermion=False) for x in range(n)]
        scalar = Fraction(1, 2**(n-1))
        sums = [Sigma(x) for x in nm] + [Sigma(i), Sigma(a)]
        name = "lu1{n}".format(n=n)
        tensors = [Tensor(nm + [a, i], name, sym=TensorSym([(0, 1, 2, 3), (1, 0, 2, 3)], [1, 1]) if n == 2 else TensorSym([], []))]
        operators = [BOperator(x, False) for x in nm] + [FOperator(a, True), FOperator(i, False)]
        l.append(Term(scalar, sums, tensors, operators, []))
        particles[name] = tuple((SCALAR_BOSON, x) for x in range(n)) + ((FERMION, n), (FERMION, n))

    return Expression(l), particles


def construct_hbar(h, t, max_commutator=4):
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

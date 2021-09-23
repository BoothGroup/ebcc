''' Extra functionality not available in WICK '''
from fractions import Fraction
from wick.index import Idx
from wick.operator import Projector, BOperator, FOperator
from wick.operator import TensorSym, Tensor, Sigma, normal_ordered
from wick.expression import Term, Expression

def P_dexit1(name, spaces, index_key=None):
    """
    Return the tensor representation of a Boson de-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    for s in spaces:
        x = Idx(0, s, fermion=False)
        sums = [Sigma(x)]
        tensors = [Tensor([x], name)]
        operators = [BOperator(x, False)]
        e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
        terms.append(e1)
    return Expression(terms)

def EPS_dexit1(name, bspaces, ospaces, vspaces, index_key=None):
    """
    Return the tensor representation of a coupled
    Fermion excitation operator and Boson deexcitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    for bs in bspaces:
        for os in ospaces:
            for vs in vspaces:
                x = Idx(0, bs, fermion=False)
                i = Idx(0, os)
                a = Idx(0, vs)
                sums = [Sigma(x), Sigma(i), Sigma(a)]
                tensors = [Tensor([x, a, i], name)]
                operators = [BOperator(x, False), FOperator(a, True), 
                             FOperator(i, False)]
                e1 = Term(1, sums, tensors, operators, [], index_key=index_key)
                terms.append(e1)
    return Expression(terms)

def PB1(bspace, index_key=None):
    """
    Return the projector onto a space of single boson occupations 

    bspace (str): boson space
    """
    x = Idx(0, bspace, fermion=False)
    P = Projector()
    operators = [BOperator(x, True), P, BOperator(x, False)]
    exp = Expression([
        Term(1, [Sigma(x)], [], operators, [], index_key=index_key)])
    return exp

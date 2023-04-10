'''
Generate Hugenholtz diagrams
Scales roughly 
'''
#TODO test energies

import numpy as np
import itertools
from collections import defaultdict
from pyscf.agf2 import mpi_helper

OCC_INDS = 'ijklmnopIJKLMNOP'
VIR_INDS = 'abcdefghABCDEFGH'


def index_to_ov(idx):
    ''' Convert i.e. j->o, b->v etc.
    '''

    if idx in OCC_INDS:
        return 'o'
    elif idx in VIR_INDS:
        return 'v'
    else:
        raise ValueError


def iter_edges(graph):
    ''' Iterate over the edges (out, in)
    '''

    for i in range(len(graph)):
        yield graph[i-1], graph[i]


def direction(start, end):
    ''' 
    Return the direction of an edge:

         1 : up (occupied)
        -1 : down (virtual)
         0 : level (bubble)
    '''

    return int(bool(end > start) - bool(start > end))


def classify_graph(graph, ignore_bubble=False):
    ''' 
    Return a unique classifier for a graph allowing equality - only
    unique among graphs with the same size.
    '''

    edges = defaultdict(int)
    step = len(graph) // 2

    for start, end in iter_edges(graph):
        if ignore_bubble:
            if direction(start, end) == 0:
                return False

        edge = start * step + end
        edges[edge] += 1

    return frozenset(edges.items())


def equal_graphs(a, b):
    ''' Return True if two graphs are equal
    '''

    return classify_graph(a) == classify_graph(b)


def get_connectivity(graph):
    ''' 
    Return ∑(number of edges going in) - ∑(number of edges going out)
    for each node in a graph.
    '''

    count = [0,] * (len(graph) // 2)
    
    for start, end in iter_edges(graph):
        count[start] += 1
        count[end] -= 1

    return count


def permutations(n=2, ignore_bubble=False):
    '''
    Return permutations of [0,1,...,n,0,1,...n]
    From more_itertools
    '''

    perm = sorted(range(n)) * 2
    size = n * 2

    if n < 2:
        yield perm

    while True:
        # Find largest i such that i < i+1
        for i in range(size-2, -1, -1):
            if perm[i] < perm[i+1]:
                break
        else:
            return

        # Find largest j such that i < j
        for j in range(size-1, i, -1):
            if perm[i] < perm[j]:
                break

        # Swap i, j and reverse from i+1
        perm[i], perm[j] = perm[j], perm[i]
        perm[i+1:] = perm[:i-size:-1]

        # To skip bubbles at this point:
        #
        #                0  1       i-1    i  i+1       j-1    j  j+1       2n-2 2n-1 
        #               -------------------------------------------------------------
        # Start with:  [ 0, 1, ..., a-1,   a, a+1, ..., b-1,   b, b+1, ..., c-2, c-1 ]
        # Swap i,j:    [ 0, 1, ..., a-1,   b, a+1, ..., b-1,   a, b+1, ..., c-2, c-1 ]
        # Reverse i+1: [ 0, 1, ..., a-1,   b, c-1, ..., b+1,   a, b-1, ..., a+2, a+1 ]
        #
        # Therefore new neighbours exist at:
        #
        #   (i-1, i), (i, i+1), (j-1, j), (j, j+1), (2n-1, 0)
        #
        # and the first permutation will always have no bubbles. It's probably
        # possible to "jump ahead" in the algorithm to the next non-bubble but
        # that would take some thinking.
        if ignore_bubble:
            new_neighbours = set([
                    (i-1, i%size), (i, (i+1)%size),
                    (j-1, j), (j, (j+1)%size), (size-1, 0),
            ])
            if any(perm[a] == perm[b] for a, b in new_neighbours):
                continue

        yield perm


def get_hugenholtz_diagrams(n=2, ignore_bubble=True):
    ''' 
    Get a list of directed graphs representing Hugenholtz diagrams
    for a given order of perturbation theory.
    '''

    seen = set()

    for graph in permutations(n):
        classification = classify_graph(graph, ignore_bubble=ignore_bubble)
        
        # Check if diagram is a duplicate
        if classification is False or classification in seen:
            continue
        seen.add(classification)

        # Check the connectivity is balanced
        if any([x != 0 for x in get_connectivity(graph)]):
            continue

        # If we get here, it's a valid diagram
        yield graph


def sort_energy(term):
    ''' Sort the energy repr into a canonical order
    '''

    occ_part = [char for char in term if char in OCC_INDS]
    vir_part = [char for char in term if char in VIR_INDS]

    term_out = ''.join(occ_part) + ''.join(vir_part)

    return term_out


def sort_integral(term):
    ''' Sort the integral repr into a canonical order
    '''

    term_out = term
    score_min = np.inf
    sign = 1
    allowed = [
        ((0,1,2,3),  1), ((1,0,3,2),  1), ((2,3,0,1),  1), ((3,2,1,0),  1),
        ((0,1,3,2), -1), ((1,0,2,3), -1), ((3,2,0,1), -1), ((2,3,1,0), -1),
    ]

    for allow, sign_flip in allowed:
        perm = [term[x] for x in allow]
        string = ''.join(['0' if s in OCC_INDS else '1' for s in perm])
        score = int(string, 2)
        if score < score_min:
            term_out = ''.join(perm)
            score_min = score
            sign *= sign_flip

    return term_out, sign


def label_edges(graph):
    ''' Return a list of index labels for each edge
    '''

    labels = []
    o = list(OCC_INDS)
    v = list(VIR_INDS)

    for start, end in iter_edges(graph):
        d = direction(start, end)

        if d == 1:
            labels.append(o.pop(0))
        elif d == -1:
            labels.append(v.pop(0))
        else:
            labels.append(None)

    return labels


def direction_dicts(graph, labels):
    ''' 
    Return dictionaries with keys corresponding to each vertex and
    values containing a list of labels corresponding to edges going
    in and out of the vertex.
    '''

    outs = defaultdict(list)
    ins = defaultdict(list)

    last = graph[-1]
    for n in range(len(graph)):
        outs[last].append(labels[n])
        last = graph[n]
        ins[last].append(labels[n])

    ins = { k:v for k,v in ins.items() }
    outs = { k:v for k,v in outs.items() }

    return ins, outs


def get_hugenholtz_energies(graph, labels):
    ''' Get the energy contribution from a given Hugenholtz diagram
    '''

    order = len(graph) // 2
    terms = []

    for n in range(order-1):
        x = 0.5 + n
        term = ''

        for label, (start, end) in zip(labels, iter_edges(graph)):
            if (start < x and end > x) or (start > x and end < x):
                term = term + label

        term = sort_energy(term)
        terms.append(term)

    return ','.join(terms)


def get_hugenholtz_integrals(graph, labels):
    ''' Get the integral contribution from a given Hugentholtz diagram
    '''

    ins, outs = direction_dicts(graph, labels)
    terms = []
    sign = 1

    for point in set(graph):
        term = ''.join([*ins[point], *outs[point]])
        term, sf = sort_integral(term)
        terms.append(term)
        sign *= sf

    return ','.join(terms), sign


def get_hugenholtz_sign(graph, labels, integrals):
    ''' Get the sign contribution from a given Hugenholtz diagram
    '''

    integrals = integrals.split(',')
    rule = [x[0]+x[2] for x in integrals] + [x[1]+x[3] for x in integrals]
    paths = [rule.pop(0)]

    while len(rule):
        r = rule.pop(0)
        for i in range(len(paths)):
            if r[-1] == paths[i][0]:
                paths[i] = r + paths[i]
                break
            elif r[0] == paths[i][-1]:
                paths[i] = paths[i] + r
                break
        else:
            paths.append(r)

    l = len(paths)
    h = len([x for x in labels if x in OCC_INDS])
    sign = pow(-1, h+l)

    return sign


def get_hugenholtz_factor(graph):
    ''' Get the factor contribution from a given Hugenholtz diagram
    '''

    count = defaultdict(int)

    for start, end in iter_edges(graph):
        count[(start, end)] += 1

    n = sum([x // 2 for x in count.values()])

    return pow(0.5, n)


def get_pdaggerq(
        graph,
        use_t2=True,
):
    labels = label_edges(graph)
    integrals, sign = get_hugenholtz_integrals(graph, labels)
    energies = get_hugenholtz_energies(graph, labels)
    sign *= get_hugenholtz_sign(graph, labels, integrals)
    factor = get_hugenholtz_factor(graph)

    if use_t2:
        amplitudes = []
        energies_to_remove = []
        integrals_to_remove = []

        for i, integral in enumerate(integrals.split(',')):
            for j, energy in enumerate(energies.split(',')):
                if integral == energy:
                    inds = tuple(index_to_ov(x) for x in integral)
                    if inds == ('o', 'o', 'v', 'v'):
                        if i in integrals_to_remove or j in energies_to_remove:
                            continue
                        amplitudes.append(integral)
                        integrals_to_remove.append(i)
                        energies_to_remove.append(j)

        energies = [x for i,x in enumerate(energies.split(','))
                    if i not in energies_to_remove]
        integrals = [x for i,x in enumerate(integrals.split(','))
                     if i not in integrals_to_remove]

        energies = ','.join(energies)
        integrals = ','.join(integrals)
        amplitudes = ','.join(amplitudes)

    term = [("+" if sign > 0 else "-") + str(factor)]
    if integrals.strip():
        for integral in integrals.split(","):
            term += ["<%s,%s||%s,%s>" % tuple(integral)]
    if energies.strip():
        for energy in energies.split(","):
            term += ["denom%d(%s)" % (len(energy)//2, ",".join(energy))]
    if amplitudes.strip():
        for amplitude in amplitudes.split(","):
            n = len(amplitude) // 2
            term += ["t2(%s,%s)" % (",".join(amplitude[n:]), ",".join(amplitude[:n]))]

    return term


if __name__ == '__main__':
    import time
    from ojb.plotting import get_scaling

    graphs = get_hugenholtz_diagrams(2)
    for graph in graphs:
        term = get_pdaggerq(graph)
        print(term)

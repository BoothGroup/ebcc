"""Common functionality for the bootstrap scripts for `ebcc`.
"""

from albert.qc._pdaggerq import import_from_pdaggerq
from albert.qc.spin import ghf_to_rhf, ghf_to_uhf, get_amplitude_spins, get_density_spins
from albert.qc.decomp import density_fit
from albert.opt import optimise
from albert.opt.tools import optimise_eom
from albert.tensor import Tensor


def spin_integrate(exprs, spin):
    """Perform the spin integration on the given expressions."""
    if spin == "ghf":
        return exprs
    elif spin == "uhf":
        return ghf_to_uhf(exprs)
    else:
        return (ghf_to_rhf(exprs),)


def get_energy(terms, spin, strategy="exhaust", density_fit=False):
    """Get the energy expressions from `pdaggerq` terms."""
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    if density_fit:
        expr = tuple(density_fit(e) for e in expr)
    output = tuple(Tensor(name="e_cc") for _ in range(len(expr)))
    output_expr = optimise(output, expr, strategy=strategy)
    returns = (Tensor(name="e_cc"),)
    return output_expr, returns


def get_amplitudes(terms_grouped, spin, strategy="exhaust", which="t", orders=None, density_fit=False):
    """Get the amplitude expressions from `pdaggerq` terms."""
    expr = []
    output = []
    returns = []
    if orders is None:
        orders = range(1, len(terms_grouped) + 1)
    for order, terms in zip(orders, terms_grouped):
        n = order - 1
        if which == "t":
            indices = "ijkl"[: n + 1] + "abcd"[: n + 1]
        else:
            indices = "abcd"[: n + 1] + "ijkl"[: n + 1]
        for index_spins in get_amplitude_spins(indices[: n + 1], indices[n + 1 :], spin):
            expr_n = import_from_pdaggerq(terms, index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            output_n = [Tensor(*tuple(sorted(e.external_indices, key=lambda i: indices.index(i.name))), name=f"{which}{order}new") for e in expr_n]
            returns_n = (output_n[0],)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)
    if strategy is not None:
        output_expr = optimise(output, expr, strategy="exhaust")
    else:
        output_expr = list(zip(output, expr))
    return output_expr, returns


def get_rdm1(terms_sectors, spin, strategy="exhaust"):
    """Get the one-body reduced density matrix expressions from `pdaggerq` terms."""
    if spin == "ghf":
        from albert.qc.ghf import Delta, T1
    elif spin == "uhf":
        from albert.qc.uhf import Delta, T1
    else:
        from albert.qc.rhf import Delta, T1
    expr = []
    output = []
    returns = []
    deltas = []
    deltas_sources = []
    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
        for index_spins in get_density_spins(indices, spin):
            expr_n = import_from_pdaggerq(terms_sectors[sectors, indices], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            if spin == "rhf":
                expr_n = tuple(e * 2 for e in expr_n)
            if density_fit:
                expr_n = tuple(density_fit(e) for e in expr_n)
            output_n = [Tensor(*tuple(sorted(e.external_indices, key=lambda i: indices.index(i.name))), name=f"rdm1") for e in expr_n]
            returns_n = (output_n[0],)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)
            if len(set(sectors)) == 1:
                delta = Delta(*tuple(sorted(expr_n[0].external_indices, key=lambda i: indices.index(i.name))))
                deltas.append(delta)
                deltas_sources.append(next(expr_n[0].search_leaves(T1)))
    output_expr = optimise(output, expr, strategy=strategy)
    return output_expr, returns, deltas, deltas_sources


def get_rdm2(terms_sectors, spin, strategy="exhaust", density_fit=False):
    """Get the two-body reduced density matrix expressions from `pdaggerq` terms."""
    if spin == "ghf":
        from albert.qc.ghf import Delta, T1
    elif spin == "uhf":
        from albert.qc.uhf import Delta, T1
    else:
        from albert.qc.rhf import Delta, T1
    expr = []
    output = []
    returns = []
    deltas = []
    deltas_sources = []
    for sectors, indices in [
        ("oooo", "ijkl"),
        ("ooov", "ijka"),
        ("oovo", "ijak"),
        ("ovoo", "iajk"),
        ("vooo", "aijk"),
        ("oovv", "ijab"),
        ("ovov", "iajb"),
        ("ovvo", "iabj"),
        ("voov", "aijb"),
        ("vovo", "aibj"),
        ("vvoo", "abij"),
        ("ovvv", "iabc"),
        ("vovv", "aibc"),
        ("vvov", "abic"),
        ("vvvo", "abci"),
        ("vvvv", "abcd"),
    ]:
        for index_spins in get_density_spins(indices, spin):
            expr_n = import_from_pdaggerq(terms_sectors[sectors, indices], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            if density_fit:
                expr_n = tuple(density_fit(e) for e in expr_n)
            output_n = [Tensor(*tuple(sorted(e.external_indices, key=lambda i: indices.index(i.name))), name=f"rdm2") for e in expr_n]
            returns_n = (output_n[0],)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)
            if len(set(sectors)) == 1:
                delta = Delta(*tuple(sorted(expr_n[0].external_indices, key=lambda i: indices.index(i.name))[:2]))
                deltas.append(delta)
                deltas_sources.append(next(expr_n[0].search_leaves(T1)))
    output_expr = optimise(output, expr, strategy=strategy)
    return output_expr, returns, deltas, deltas_sources


def get_eom(terms_grouped, spin, strategy="exhaust", which="ip", orders=None, density_fit=False):
    """Get the EOM expressions from `pdaggerq` terms."""
    expr = []
    output = []
    returns = []
    if orders is None:
        orders = range(1, len(terms_grouped) + 1)
    for order, terms in zip(orders, terms_grouped):
        n = order - 1
        if which == "ip":
            indices = "ijkl"[: n + 1] + "abcd"[: n]
        elif which == "ea":
            indices = "abcd"[: n + 1] + "ijkl"[: n]
        else:
            indices = "ijkl"[: n + 1] + "abcd"[: n + 1]
        for index_spins in get_amplitude_spins(indices[: n + 1], indices[n + 1 :], spin):
            expr_n = import_from_pdaggerq(terms, index_spins=index_spins, l_is_lambda=False)
            expr_n = spin_integrate(expr_n, spin)
            if density_fit:
                expr_n = tuple(density_fit(e) for e in expr_n)
            output_n = [Tensor(*sorted(e.external_indices, key=lambda i: indices.index(i.name)), name=f"r{order}new") for e in expr_n]
            returns_n = (output_n[0],)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)
    (returns_nr, output_expr_nr), (returns_r, output_expr_r) = optimise_eom(returns, output, expr, strategy=strategy)
    if spin == "uhf":
        # R amplitudes may get wrong spins
        output_expr_nr = [(output, expr.canonicalise()) for output, expr in output_expr_nr]
        output_expr_r = [(output, expr.canonicalise()) for output, expr in output_expr_r]
    return output_expr_nr, returns_nr, output_expr_r, returns_r

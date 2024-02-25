"""
Generate the CC2 code.
"""

import sys
import pdaggerq
from ebcc.codegen.bootstrap_common import *

from albert.qc._pdaggerq import import_from_pdaggerq
from albert.canon import canonicalise_indices
from albert.tensor import Tensor

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}CC2.py", "w"),
        einsum_func="einsum",
        einsum_kwargs={},
        name_generator=name_generators[spin],
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

# Set up pdaggerq
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# Get the energy contractions in pdaggerq format
pq.clear()
pq.set_left_operators([["1"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()
terms = pq.fully_contracted_strings()
terms = remove_hf_energy(terms)

# Get the energy in albert format
expr = import_from_pdaggerq(terms)
expr = canonicalise_indices(expr, *default_indices.values())
expr = spin_integrate(expr, spin)
output = tuple(Tensor(name="e_cc") for _ in range(len(expr)))
output, expr = optimise(output, expr, spin, strategy="exhaust")
returns = (Tensor(name="e_cc"),)

# Generate the energy code
for codegen in code_generators.values():
    codegen(
        "energy",
        returns,
        output,
        expr,
    )

# Get the T1 contractions in pdaggerq format
pq.clear()
pq.set_left_operators([["e1(i,a)"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
pq.simplify()
terms_t1 = pq.fully_contracted_strings()

# Get the T2 contractions in pdaggerq format
pq.clear()
pq.set_left_operators([["e2(i,j,b,a)"]])
pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v"], ["t1"])
pq.simplify()
terms_t2 = pq.fully_contracted_strings()

# Get the T amplitudes in albert format
terms = [terms_t1, terms_t2]
expr = []
output = []
returns = []
for n in range(2):
    for index_spins in get_amplitude_spins(n + 1, spin):
        expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
        expr_n = canonicalise_indices(expr_n, *default_indices.values())
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_t_amplitude_outputs(expr_n, f"t{n+1}new")
        returns_n = (Tensor(*default_indices["o"][:n+1], *default_indices["v"][:n+1], name=f"t{n+1}new"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
output, expr = optimise(output, expr, spin, strategy="exhaust")

# Generate the T amplitude code
for name, codegen in code_generators.items():
    if name == "einsum":
        kwargs = {
            "preamble": "t1new = Namespace()\nt2new = Namespace()" if spin == "uhf" else None,
            "as_dict": True,
        }
    else:
        kwargs = {}
    codegen(
        "update_amps",
        returns,
        output,
        expr,
        **kwargs,
    )

# Get the L1 contractions in pdaggerq format
pq.clear()
pq.set_left_operators([["1"]])
pq.set_right_operators([["1"]])
pq.add_st_operator(1.0, ["f", "e1(a,i)"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v", "e1(a,i)"], ["t1", "t2"])
pq.set_left_operators([["l1"], ["l2"]])
pq.add_st_operator( 1.0, ["f", "e1(a,i)"], ["t1", "t2"])
pq.add_st_operator( 1.0, ["v", "e1(a,i)"], ["t1", "t2"])
pq.add_st_operator(-1.0, ["e1(a,i)", "f"], ["t1", "t2"])
pq.add_st_operator(-1.0, ["e1(a,i)", "v"], ["t1", "t2"])
pq.simplify()
terms_l1 = pq.fully_contracted_strings()

# Get the L2 contractions in pdaggerq format
pq.clear()
pq.set_left_operators([["1"]])
pq.set_right_operators([["1"]])
pq.add_st_operator(1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2"])
pq.add_st_operator(1.0, ["v", "e2(a,b,j,i)"], ["t1"])
pq.set_left_operators([["l1"], ["l2"]])
pq.add_st_operator( 1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2"])
pq.add_st_operator( 1.0, ["v", "e2(a,b,j,i)"], ["t1"])
pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "f"], ["t1", "t2"])
pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "v"], ["t1"])
pq.simplify()
terms_l2 = pq.fully_contracted_strings()

# Get the L amplitudes in albert format
terms = [terms_l1, terms_l2]
expr = []
output = []
returns = []
for n in range(2):
    for index_spins in get_amplitude_spins(n + 1, spin):
        expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
        expr_n = canonicalise_indices(expr_n, *default_indices.values())
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_l_amplitude_outputs(expr_n, f"l{n+1}new")
        returns_n = (Tensor(*default_indices["v"][:n+1], *default_indices["o"][:n+1], name=f"l{n+1}new"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
output, expr = optimise(output, expr, spin, strategy="opt")

# Generate the L amplitude code
for name, codegen in code_generators.items():
    if name == "einsum":
        kwargs = {
            "preamble": "l1new = Namespace()\nl2new = Namespace()" if spin == "uhf" else None,
            "as_dict": True,
        }
    else:
        kwargs = {}
    codegen(
        "update_lams",
        returns,
        output,
        expr,
        **kwargs,
    )

# Get the 1RDM contractions in pdaggerq format
terms = {}
for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
    pq.clear()
    pq.set_left_operators([["1"], ["l1"], ["l2"]])
    pq.add_st_operator(1.0, [f"e1({','.join(indices)})"], ["t1", "t2"])
    pq.simplify()
    terms[sectors, indices] = pq.fully_contracted_strings()

# Get the 1RDM in albert format
expr = []
output = []
returns = []
for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
    for index_spins in get_density_spins(1, spin):
        expr_n = import_from_pdaggerq(terms[sectors, indices], index_spins=index_spins)
        expr_n = canonicalise_indices(expr_n, *default_indices.values())
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_density_outputs(expr_n, f"d", indices)
        returns_n = (Tensor(*indices, name=f"d"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
output, expr = optimise(output, expr, spin, strategy="exhaust")

# Generate the 1RDM code
for name, codegen in code_generators.items():
    if name == "einsum":
        preamble = "rdm1 = Namespace()"
        if spin == "uhf":
            preamble += "\nrdm1.aa = Namespace()"
            preamble += "\nrdm1.bb = Namespace()"
            preamble += "\ndelta = Namespace("
            preamble += "\n    aa=Namespace(oo=np.eye(t1.aa.shape[0]), vv=np.eye(t1.aa.shape[1])),"
            preamble += "\n    bb=Namespace(oo=np.eye(t1.bb.shape[0]), vv=np.eye(t1.bb.shape[1])),"
            preamble += "\n)"
            postamble = "rdm1.aa = np.block([[rdm1.aa.oo, rdm1.aa.ov], [rdm1.aa.vo, rdm1.aa.vv]])"
            postamble += "\nrdm1.bb = np.block([[rdm1.bb.oo, rdm1.bb.ov], [rdm1.bb.vo, rdm1.bb.vv]])"
        else:
            preamble += "\ndelta = Namespace("
            preamble += "\n    oo=np.eye(t1.shape[0]),"
            preamble += "\n    vv=np.eye(t1.shape[1]),"
            preamble += "\n)"
            postamble = "rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])"
        kwargs = {
            "preamble": preamble,
            "postamble": postamble,
        }
    else:
        kwargs = {}
    codegen(
        "make_rdm1_f",
        returns,
        output,
        expr,
        **kwargs,
    )

# Get the 2RDM contractions in pdaggerq format
terms = {}
for sectors, indices in [
    ("oooo", "ijkl"), ("ooov", "ijka"), ("oovo", "ijak"), ("ovoo", "iajk"),
    ("vooo", "aijk"), ("oovv", "ijab"), ("ovov", "iajb"), ("ovvo", "iabj"),
    ("voov", "aijb"), ("vovo", "aibj"), ("vvoo", "abij"), ("ovvv", "iabc"),
    ("vovv", "aibc"), ("vvov", "abic"), ("vvvo", "abci"), ("vvvv", "abcd"),
]:
    pq.clear()
    pq.set_left_operators([["1"], ["l1"], ["l2"]])
    pq.add_st_operator(1.0, [f"e2({indices[0]},{indices[1]},{indices[3]},{indices[2]})"], ["t1", "t2"])
    pq.simplify()
    terms[sectors, indices] = pq.fully_contracted_strings()

# Get the 2RDM in albert format
expr = []
output = []
returns = []
for sectors, indices in [
    ("oooo", "ijkl"), ("ooov", "ijka"), ("oovo", "ijak"), ("ovoo", "iajk"),
    ("vooo", "aijk"), ("oovv", "ijab"), ("ovov", "iajb"), ("ovvo", "iabj"),
    ("voov", "aijb"), ("vovo", "aibj"), ("vvoo", "abij"), ("ovvv", "iabc"),
    ("vovv", "aibc"), ("vvov", "abic"), ("vvvo", "abci"), ("vvvv", "abcd"),
]:
    for index_spins in get_density_spins(2, spin):
        expr_n = import_from_pdaggerq(terms[sectors, indices], index_spins=index_spins)
        expr_n = canonicalise_indices(expr_n, *default_indices.values())
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_density_outputs(expr_n, f"Γ", indices)
        returns_n = (Tensor(*indices, name=f"Γ"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
output, expr = optimise(output, expr, spin, strategy="exhaust")

# Generate the 2RDM code
for name, codegen in code_generators.items():
    if name == "einsum":
        preamble = "rdm2 = Namespace()"
        if spin == "uhf":
            preamable += "\nrdm2.aaaa = Namespace()"
            preamable += "\nrdm2.abab = Namespace()"
            preamable += "\nrdm2.bbbb = Namespace()"
            preamble += "\ndelta = Namespace("
            preamble += "\n    aa=Namespace(oo=np.eye(t1.aa.shape[0]), vv=np.eye(t1.aa.shape[1])),"
            preamble += "\n    bb=Namespace(oo=np.eye(t1.bb.shape[0]), vv=np.eye(t1.bb.shape[1])),"
            preamble += "\n)"
            postamble = "rdm2.aaaa = pack_2e(%s)" % ", ".join(f"rdm2.aaaa.{perm}" for perm in ov_2e)
            postamble += "\nrdm2.abab = pack_2e(%s)" % ", ".join(f"rdm2.abab.{perm}" for perm in ov_2e)
            postamble += "\nrdm2.bbbb = pack_2e(%s)" % ", ".join(f"rdm2.bbbb.{perm}" for perm in ov_2e)
            postamble += "\nrdm2 = Namespace("
            postamble += "\n    aaaa=rdm2.aaaa.swapaxes(1, 2),"
            postamble += "\n    aabb=rdm2.abab.swapaxes(1, 2),"
            postamble += "\n    bbbb=rdm2.bbbb.swapaxes(1, 2),"
            postamble += "\n)"
        else:
            preamble += "\ndelta = Namespace("
            preamble += "\n    oo=np.eye(t1.shape[0]),"
            preamble += "\n    vv=np.eye(t1.shape[1]),"
            preamble += "\n)"
            postamble = "rdm2 = pack_2e(%s)" % ", ".join(f"rdm2.{perm}" for perm in ov_2e)
            postamble += "\nrdm2 = rdm2.swapaxes(1, 2)"
        kwargs = {
            "preamble": preamble,
            "postamble": postamble,
        }
    codegen(
        "make_rdm2_f",
        returns,
        output,
        expr,
        **kwargs,
    )

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()

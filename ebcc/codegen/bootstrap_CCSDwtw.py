"""
Generate the CCSDt code.
"""

import sys
sys.setrecursionlimit(100000)

import pdaggerq
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.base import Base
from albert.tensor import Tensor
from albert.algebra import Mul, Add

from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
def name_generator(tensor, add_spaces=True):
    name = name_generators[spin](tensor, add_spaces=add_spaces)
    if isinstance(tensor, Tensor) and tensor.name.startswith("t") and not tensor.name.startswith("tmp"):
        spaces = tuple(i.space for i in tensor.indices)
        if not any(s is None for s in spaces):  # Otherwise, it's a "return" tensor
            if spaces not in (("o", "v"), ("o", "o", "v", "v"), ("o", "o", "O", "v", "v", "V")):
                name = name.replace(".", "_")
                name = f"{name}_{''.join(spaces)}"
    return name

class _EinsumCodeGen(EinsumCodeGen):
    def ignore_argument(self, arg):
        if "_" in name_generator(arg):
            return True
        return super().ignore_argument(arg)

    def tensor_cleanup(self, *args):
        args = tuple(self.get_name(arg) if isinstance(arg, Tensor) else arg for arg in args)
        args = tuple(arg for arg in args if "_" not in arg)
        return super().tensor_cleanup(*args)

    def preamble(self, imports=None):
        if imports is None:
            imports = "from ebcc import numpy as np\n"
            imports += "from ebcc.util import pack_2e, einsum, dirsum, Namespace\n"
            imports += "from ebcc.backend import _inflate"
        super().preamble(imports=imports)

code_generators = {
    "einsum": _EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}CCSDwtw.py", "w"),
        name_generator=name_generator,
        spin=spin,
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

# Set up pdaggerq
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

with Stopwatch("Energy"):
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

with Stopwatch("T amplitudes"):
    # Get the T1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_t1 = pq.fully_contracted_strings()

    # Get the T2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_t2 = pq.fully_contracted_strings()

    # Get the T3 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e3(i,j,k,c,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_t3 = pq.fully_contracted_strings()

    # Get the T amplitudes in albert format
    terms = [terms_t1, terms_t2, terms_t3]
    expr = []
    output = []
    returns = []
    for n in range(3):
        for index_spins in get_amplitude_spins(n + 1, spin):
            indices = default_indices["o"][: n + 1] + default_indices["v"][: n + 1]
            expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            output_n = get_t_amplitude_outputs(expr_n, f"t{n+1}new", indices=indices)
            returns_n = (Tensor(*tuple(Index(i) for i in indices), name=f"t{n+1}new"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

    # Make the T3 indices active
    new_expr = []
    new_output = []
    for i, (o, e) in enumerate(zip(output, expr)):
        # Get the indices
        indices = set()
        if o.name.startswith("t3"):
            indices = indices.union({o.indices[2], o.indices[5]})
        for a in e.nested_view():
            indices_a = indices.copy()
            for t in a:
                if isinstance(t, Tensor) and t.name.startswith("t3"):
                    indices_a = indices_a.union({t.indices[2], t.indices[5]})

            # Substitute the indices
            subs = {i: Index(i.name.upper(), spin=i.spin, space=i.space.upper()) for i in indices_a}
            new_args = []
            for t in a:
                if isinstance(t, Tensor):
                    new_args.append(t.map_indices(subs))
                else:
                    new_args.append(t)
            new_output.append(o.map_indices(subs))
            new_expr.append(Mul(*new_args))
    output = new_output
    expr = new_expr

    ## Spin integrate
    #new_output = []
    #new_expr = []
    #for o, e in zip(output, expr):
    #    new_expr_spin = spin_integrate(e, spin)
    #    for ee in new_expr_spin:
    #        if isinstance(ee, Base) and len(ee.without_coefficient().args):
    #            new_expr.append(ee)
    #            new_output.append(get_t_amplitude_outputs((ee,), o.name, indices=[i.name for i in sorted(ee.external_indices)])[0])
    #expr = new_expr
    #output = new_output

    # Optimise
    #output, expr = optimise(output, expr, spin, strategy="exhaust" if spin != "uhf" else "greedy")

    # Generate the T amplitude code
    for name, codegen in code_generators.items():
        ignore_arguments = []
        if name == "einsum":
            preamble = "space = kwargs[\"space\"]"
            preamble += "\nt1new = Namespace()"
            preamble += "\nt2new = Namespace()"
            preamble += "\nt3new = Namespace()"
            if spin == "uhf":
                preamble += "\nsoa = np.ones((t1.aa.shape[0],), dtype=bool)"
                preamble += "\nsva = np.ones((t1.aa.shape[1],), dtype=bool)"
                preamble += "\nsob = np.ones((t1.bb.shape[0],), dtype=bool)"
                preamble += "\nsvb = np.ones((t1.bb.shape[1],), dtype=bool)"
                preamble += "\nsOa = space[0].active[space[0].correlated][space[0].occupied[space[0].correlated]]"
                preamble += "\nsVa = space[0].active[space[0].correlated][space[0].virtual[space[0].correlated]]"
                preamble += "\nsOb = space[1].active[space[1].correlated][space[1].occupied[space[1].correlated]]"
                preamble += "\nsVb = space[1].active[space[1].correlated][space[1].virtual[space[1].correlated]]"
                preamble += "\n"
                postamble = ""
                inp_keys = {"aa": set(), "bb": set(), "aaaa": set(), "abab": set(), "bbbb": set()}
                out_keys = {"aa": set(), "bb": set(), "aaaa": set(), "abab": set(), "bbbb": set()}
                for o, e in zip(output, expr):
                    if o.name.startswith("t") and not (o.name.startswith("tmp") or o.name.startswith("t3")):
                        spaces = "".join(tuple(i.space for i in o.indices))
                        spins = "".join(tuple({"α": "a", "β": "b"}[i.spin] for i in o.indices))
                        if set(spaces) != {"o", "v"}:
                            inp_keys[spins].add(spaces)
                    for a in e.nested_view():
                        for t in a:
                            if isinstance(t, Tensor) and t.name.startswith("t") and not (t.name.startswith("tmp") or t.name.startswith("t3")):
                                spaces = "".join(tuple(i.space for i in t.indices))
                                spins = "".join(tuple({"α": "a", "β": "b"}[i.spin] for i in t.indices))
                                inp_keys[spins].add(spaces)
                for spins, inp_keys in inp_keys.items():
                    for key in inp_keys:
                        slices = ", ".join(f"s{c}{s}" for c, s in zip(key, spins))
                        preamble += f"\nt{len(key)//2}_{spins}_{key} = np.copy(t{len(key)//2}.{spins}[np.ix_({slices})])"
                        ignore_arguments.append(f"t{len(key)//2}_{spins}_{key}")
                for spins, out_keys in out_keys.items():
                    for key in out_keys:
                        slices = ", ".join(f"s{c}{s}" for c, s in zip(key, spins))
                        postamble += f"\nt{len(key)//2}new.{spins} += _inflate(t{len(key)//2}new.{spins}.shape, np.ix_({slices}), t{len(key)//2}new_{spins}_{key})"
            else:
                preamble += "\nso = np.ones((t1.shape[0],), dtype=bool)"
                preamble += "\nsv = np.ones((t1.shape[1],), dtype=bool)"
                preamble += "\nsi = np.ones((t3.shape[2],), dtype=bool)"
                preamble += "\nsa = np.ones((t3.shape[5],), dtype=bool)"
                preamble += "\nsO = space.active[space.correlated][space.occupied[space.correlated]]"
                preamble += "\nsV = space.active[space.correlated][space.virtual[space.correlated]]"
                postamble = ""
                inp_keys = set()
                out_keys = set()
                for o, e in zip(output, expr):
                    if o.name.startswith("t") and not o.name.startswith("tmp"):
                        spaces = "".join(tuple(i.space for i in o.indices))
                        if set(spaces) != {"o", "v"} and spaces != "ooOvvV":
                            out_keys.add(spaces)
                    for a in e.nested_view():
                        for t in a:
                            if isinstance(t, Tensor) and t.name.startswith("t") and not t.name.startswith("tmp"):
                                spaces = "".join(tuple(i.space for i in t.indices))
                                if set(spaces) != {"o", "v"} and spaces != "ooOvvV":
                                    inp_keys.add(spaces)
                for key in sorted(inp_keys, key=lambda x: (len(x), x)):
                    if len(key) == 6:
                        slices = ", ".join(f"s{c}" for c in key[:2]) + ", si, " + ", ".join(f"s{c}" for c in key[3:5]) + ", sa"
                    else:
                        slices = ", ".join(f"s{c}" for c in key)
                    preamble += f"\nt{len(key)//2}_{key} = np.copy(t{len(key)//2}[np.ix_({slices})])"
                    ignore_arguments.append(f"t{len(key)//2}_{key}")
                for key in sorted(out_keys, key=lambda x: (len(x), x)):
                    if len(key) == 6:
                        slices = ", ".join(f"s{c}" for c in key[:2]) + ", si, " + ", ".join(f"s{c}" for c in key[3:5]) + ", sa"
                    else:
                        slices = ", ".join(f"s{c}" for c in key)
                    postamble += f"\nt{len(key)//2}new += _inflate(t{len(key)//2}new.shape, np.ix_({slices}), t{len(key)//2}new_{key})"
                preamble += "\n"
            kwargs = {
                "preamble": preamble,
                "postamble": postamble,
                "as_dict": True,
                "ignore_arguments": ignore_arguments,
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

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()

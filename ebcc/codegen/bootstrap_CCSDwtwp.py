"""
Generate the CCSDt' code.
"""

import sys

import pdaggerq
from albert.qc._pdaggerq import remove_reference_energy, remove_reference_energy_eom
from albert.qc.spin import ghf_to_uhf, ghf_to_rhf
from albert.qc import ghf, uhf, rhf
from albert.tensor import Tensor
from albert.index import Index
from albert.code._ebcc import EBCCCodeGenerator
from albert.misc import Stopwatch
from albert.opt.tools import _tensor_info
from albert.opt import optimise
from albert.algebra import Mul
from albert.misc import ExclusionSet

from ebcc.codegen.bootstrap_common import get_energy, get_amplitudes, get_rdm1, get_rdm2, get_eom

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
class CodeGenerator(EBCCCodeGenerator):
    _add_spaces = ExclusionSet()

    def get_name(self, tensor, add_spins=None, add_spaces=None, spin_delimiter=".", space_delimiter="."):
        if isinstance(tensor, Tensor) and tensor.name.startswith("t") and not tensor.name.startswith("tmp"):
            spaces = tuple(i.space for i in tensor.indices)
            if spaces not in (("o", "v"), ("o", "o", "v", "v"), ("O", "O", "O", "V", "V", "V")):
                space_delimiter = "_"
            else:
                add_spaces = False
        return super().get_name(tensor, add_spins=add_spins, add_spaces=add_spaces, spin_delimiter=spin_delimiter, space_delimiter=space_delimiter)

    def module_imports(self):
        super().module_imports()
        self.write("from ebcc.backend import _inflate")

    def ignore_argument(self, tensor):
        if "_" in self.get_name(tensor):
            return True
        return super().ignore_argument(tensor)

    def tensor_cleanup(self, *args):
        names = tuple(self.get_name(arg) for arg in args)
        args = tuple(arg for arg, name in zip(args, names) if "_" not in name)
        super().tensor_cleanup(*args)

code_generators = {
    "einsum": CodeGenerator(
        stdout=open(f"{spin[0].upper()}CCSDwtwp.py", "w"),
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
    terms = remove_reference_energy(terms)

    # Get the energy in albert format
    output_expr, returns = get_energy(terms, spin)

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output_expr,
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
    output_expr, returns = get_amplitudes([terms_t1, terms_t2, terms_t3], spin, strategy=None)

    # Make the T3 indices active
    new_expr = []
    new_output = []
    for i, (o, e) in enumerate(output_expr):
        # Get the indices
        indices = set()
        if o.name.startswith("t3"):
            indices = indices.union(set(o.external_indices))

        for add in e.expand()._children:
            indices_a = indices.copy()
            def _get_indices(tensor):
                global indices_a
                if tensor.name.startswith("t3"):
                    indices_a = indices_a.union(set(tensor.external_indices))
                return tensor

            add = add.apply(_get_indices, Tensor)

            # Substitute the indices
            subs = {i: Index(i.name.upper(), spin=i.spin, space=i.space.upper()) for i in indices_a}
            def _substitute(tensor):
                return tensor.map_indices(subs)

            new_output.append(o.map_indices(subs))
            new_expr.append(add.apply(_substitute, Tensor))

    # Optimise
    output_expr = optimise(new_output, new_expr, strategy="exhaust" if spin != "uhf" else "greedy")

    # Generate the T amplitude code
    for name, codegen in code_generators.items():
        if name != "einsum":
            raise NotImplementedError  # FIXME remove packing
        ignore_arguments = []
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
            for o, e in output_expr:
                if o.name.startswith("t") and not (o.name.startswith("tmp") or o.name.startswith("t3")):
                    spaces = "".join(tuple(i.space for i in o.indices))
                    spins = "".join(tuple({"α": "a", "β": "b"}[i.spin] for i in o.indices))
                    if set(spaces) != {"o", "v"}:
                        inp_keys[spins].add(spaces)
                for t in e.search_leaves(Tensor):
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
            preamble += "\nsO = space.active[space.correlated][space.occupied[space.correlated]]"
            preamble += "\nsV = space.active[space.correlated][space.virtual[space.correlated]]"
            postamble = ""
            inp_keys = set()
            out_keys = set()
            for o, e in output_expr:
                if o.name.startswith("t") and not (o.name.startswith("tmp") or o.name.startswith("t3")):
                    spaces = "".join(tuple(i.space for i in o.indices))
                    if set(spaces) != {"o", "v"}:
                        out_keys.add(spaces)
                for t in e.search_leaves(Tensor):
                    if isinstance(t, Tensor) and t.name.startswith("t") and not (t.name.startswith("tmp") or t.name.startswith("t3")):
                        spaces = "".join(tuple(i.space for i in t.indices))
                        if set(spaces) != {"o", "v"}:
                            inp_keys.add(spaces)
            for key in sorted(inp_keys):
                slices = ", ".join(f"s{c}" for c in key)
                preamble += f"\nt{len(key)//2}_{key} = np.copy(t{len(key)//2}[np.ix_({slices})])"
                ignore_arguments.append(f"t{len(key)//2}_{key}")
            for key in sorted(out_keys):
                slices = ", ".join(f"s{c}" for c in key)
                postamble += f"\nt{len(key)//2}new += _inflate(t{len(key)//2}new.shape, np.ix_({slices}), t{len(key)//2}new_{key})"
            preamble += "\n"
        codegen(
            "update_amps",
            returns,
            output_expr,
            preamble = lambda: codegen.write(preamble),
            postamble = lambda: codegen.write(postamble),
            as_dict=True,
        )

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()

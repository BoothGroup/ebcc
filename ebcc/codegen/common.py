import numpy as np
import os
import sys
from types import MethodType
from timeit import default_timer as timer
from qwick import codegen

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

PYTHON_HEADER = """# Code generated by qwick.

import numpy as np
from pyscf import lib
from types import SimpleNamespace
from ebcc.codegen import common

"""

PYTHON_FOOTER = ""

LATEX_HEADER = r"""\documentclass{article}

\begin{document}

\title{Equations generated by {\tt qwick}.}
\maketitle

"""

LATEX_FOOTER = r"""

\end{document}"""


ov_2e = ["oooo", "ooov", "oovo", "ovoo", "vooo", "oovv", "ovov", "ovvo", "voov", "vovo", "vvoo", "ovvv", "vovv", "vvov", "vvvo", "vvvv"]
ov_1e = ["oo", "ov", "vo", "vv"]


CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.pkl")

def cache(func):
    """Cache output of function in a pickle file.
    """

    def wrapper(*args, **kwargs):
        key = hash((func, *args, *tuple(kwargs.items())))

        if not os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({}, f)

        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)

        if key not in data:
            res = func(*args, **kwargs)
            data[key] = res
            with open(CACHE_FILE, "wb") as f:
                pickle.write(data, f)

        return data[key]

    return wrapper


# Default particle types:
particles = {
        # Fermionic hamiltonian elements:
        "f": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "v": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        # Bosonic hamiltonian elements:
        "G": ((codegen.SCALAR_BOSON, 0),),
        "w": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 1)),
        # Fermion-boson coupling:
        "g": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "gc": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        # Amplitudes:
        "t1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "s1": ((codegen.SCALAR_BOSON, 0),),
        "s2": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0)),
        "u11": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "u12": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "r1": ((codegen.FERMION, 0),),
        "r2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0)),
        # Lambda amplitudes:
        "l1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "ls1": ((codegen.SCALAR_BOSON, 0),),
        "ls2": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0)),
        "lu11": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "lu12": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        # Updates:
        "t1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "s1new": ((codegen.SCALAR_BOSON, 0),),
        "s2new": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0)),
        "u11new": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "u12new": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "l1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "ls1new": ((codegen.SCALAR_BOSON, 0),),
        "ls2new": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0)),
        "lu11new": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "lu12new": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "r1new": ((codegen.FERMION, 0),),
        "r2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0)),
        # Delta function:
        "delta": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        # Density matrices:
        **{"rdm1_f_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 0)) for x in ov_1e},
        **{"rdm2_f_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)) for x in ov_2e},
        "rdm1_b": ((codegen.BOSON, 0), (codegen.BOSON, 0)),
        **{"dm_b%s" % x: ((codegen.BOSON, 0),) for x in ("", "_cre", "_des")},
        **{"rdm_eb_%s_%s" % (x, y): ((codegen.BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)) for y in ov_1e for x in ("cre", "des")},
        # Bras and kets:
        **{"bra1"+tag: ((codegen.FERMION, 0), (codegen.FERMION, 0)) for tag in ("", "_o", "_v")},
        **{"ket1"+tag: ((codegen.FERMION, 0), (codegen.FERMION, 0)) for tag in ("", "_o", "_v")},
        **{"bra2"+tag: ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)) for tag in ("", "_o", "_v")},
        **{"ket2"+tag: ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)) for tag in ("", "_o", "_v")},
        # Similarity transformed hamiltonian:
        "h11": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "h22": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 2), (codegen.FERMION, 3), (codegen.FERMION, 2)),  # FIXME?
}


# Default printer
def get_printer(spin):
    reorder_axes = {
            # TODO remove:
            "l1new": (1, 0),
            "l2new": (2, 3, 0, 1),
            "lu11new": (0, 2, 1),
            "lu12new": (0, 1, 3, 2),
    }

    if spin == "rhf" or spin == "uhf":
        reorder_axes["v"] = (0, 2, 1, 3)
        for x in ov_2e:
            reorder_axes["rdm2_f_%s" % x] = (0, 2, 1, 3)

    # This should be done earlier and the indices themselves manipulated:
    occ = "ijklmnop"
    vir = "abcdefgh"
    bos = "wxyzstuv"
    char_to_sector = {
            **{x: "o" for x in occ},
            **{x: "v" for x in vir},
            **{x: "b" for x in bos},
    }
    sector_to_char = {}
    for key, val in char_to_sector.items():
        if val not in sector_to_char:
            sector_to_char[val] = []
        sector_to_char[val].append(key)
    char_to_sector = {
            **char_to_sector,
            **{key.upper(): val for key, val in char_to_sector.items()},
    }
    char_to_sector = {
            **char_to_sector,
            **{key+"α": val for key, val in char_to_sector.items()},
            **{key+"β": val for key, val in char_to_sector.items()},
    }

    class EinsumPrinter(codegen.EinsumPrinter):
        def doprint(self, terms, *args, **kwargs):
            out = codegen.EinsumPrinter.doprint(self, terms, *args, **kwargs)
            lines = out.split("\n")
            for i, line in enumerate(lines):
                if "lib.einsum" in line:
                    subscript = line.split("\"")[1]
                    subscript_in = subscript.split("->")[0].split(",")
                    subscript_out = subscript.split("->")[1]

                    ins = []
                    for sin in subscript_in:
                        part = []
                        for j in range(len(sin)):
                            if sin[j] in ("α", "β"):
                                part[-1] += sin[j]
                            else:
                                part.append(sin[j])
                        ins.append(tuple(part))

                    out = []
                    for j in range(len(subscript_out)):
                        if subscript_out[j] in ("α", "β"):
                            out[-1] += subscript_out[j]
                        else:
                            out.append(subscript_out[j])
                    out = tuple(out)

                    sector_counts = {"o": 0, "v": 0, "b": 0}
                    index_map = {}
                    for sin in ins:
                        for s in sin:
                            if s not in index_map:
                                index_map[s] = (char_to_sector[s], sector_counts[char_to_sector[s]])
                                sector_counts[char_to_sector[s]] += 1
                    for s in out:
                        if s not in index_map:
                            index_map[s] = (char_to_sector[s], sector_counts[char_to_sector[s]])
                            sector_counts[char_to_sector[s]] += 1

                    new_ins = []
                    for sin in ins:
                        new_ins.append([])
                        for x in sin:
                            s, j = index_map[x]
                            new_ins[-1].append(sector_to_char[s][j])
                    new_out = []
                    for x in out:
                        s, j = index_map[x]
                        new_out.append(sector_to_char[s][j])

                    subscript_in = ",".join(["".join(x) for x in new_ins])
                    subscript_out = "".join(new_out)

                    new_subscript = subscript_in + "->" + subscript_out
                    line = line.replace(subscript, new_subscript)

                lines[i] = line

            return "\n".join(lines)

    printer = EinsumPrinter(
            occupancy_tags={
                "f": "{base}{spindelim}{spin}.{tags}",
                "v": "{base}{spindelim}{spin}.{tags}",
                "g": "{base}{spindelim}{spin}.{tags}",
                "gc": "{base}{spindelim}{spin}.{tags}",
                "delta": "delta_{tags}{spindelim}{spin}",
                **{"t%d" % n: "{base}{spindelim}{spin}" for n in range(1, 4)},
                **{"u1%d" % n: "{base}{spindelim}{spin}" for n in range(1, 4)},
                **{"l%d" % n: "{base}{spindelim}{spin}" for n in range(1, 4)},
                **{"lu1%d" % n: "{base}{spindelim}{spin}" for n in range(1, 4)},
            },
            reorder_axes=reorder_axes,
            remove_spacing=True,
            garbage_collection=True,
            base_indent=1,
            einsum="lib.einsum",
            zeros="np.zeros",
            dtype="np.float64",
    )

    return printer


# Prefix and spin transformation function
def get_transformation_function(spin):
    if spin == "rhf":
        prefix = "r"
        def transform_spin(terms, indices, **kwargs):
            project_onto = kwargs.pop("project_rhf", None)
            return codegen.ghf_to_rhf(terms, indices, **kwargs)
    elif spin == "uhf":
        prefix = "u"
        def transform_spin(terms, indices, **kwargs):
            kwargs.pop("project_rhf", None)
            return codegen.ghf_to_uhf(terms, indices, **kwargs)
    elif spin == "ghf":
        prefix = "g"
        def transform_spin(terms, indices, **kwargs):
            groups = [[terms[0]]]
            for term in terms[1:]:
                for i, group in enumerate(groups):
                    if group[0].lhs == term.lhs:
                        groups[i].append(term)
                        break
                else:
                    groups.append([term])

            return groups

    return transform_spin, prefix


class FilePrinter:
    def __init__(self, name):
        if rank != 0:
            return
        if len(sys.argv) == 1 or sys.argv[1] != "dummy":
            self.python_file = open("%s.py" % name, "w")
            self.latex_file = open("%s.tex" % name, "w")
        else:
            self.python_file = sys.stdout
            self.latex_file = open(os.devnull, "w")

    def __enter__(self):
        if rank != 0:
            return self
        # Initialise the Python file:
        self.python_file.write(PYTHON_HEADER)
        # Initialise the LaTeX file:
        self.latex_file.write(LATEX_HEADER)
        return self

    def __exit__(self, *args, **kwargs):
        if rank != 0:
            return
        # Exit the python file:
        self.python_file.write(PYTHON_FOOTER)
        self.python_file.close()
        # Exit the LaTeX file:
        self.latex_file.write(LATEX_FOOTER)
        self.latex_file.close()


class FunctionPrinter:
    def __init__(self, file_printer, name, args, res, return_dict=True, timer=None, spin_cases={}):
        self.file_printer = file_printer
        self.name = name
        self.args = args
        self.res = res
        self.return_dict = return_dict
        self.timer = timer
        self.spin_cases = spin_cases

    def write_python(self, string, comment=None):
        if rank != 0:
            return
        if comment:
            self.file_printer.python_file.write("    # %s\n" % comment)
        self.file_printer.python_file.write(string)
        self.file_printer.python_file.write("\n")
        self.file_printer.python_file.flush()

    def write_latex(self, string, comment=None):
        if rank != 0:
            return
        if comment:
            self.file_printer.latex_file.write(comment + ":\n\n")
        self.file_printer.latex_file.write("$$" + string + "$$")
        self.file_printer.latex_file.write("\n\n")
        self.file_printer.latex_file.flush()

    def __enter__(self):
        if rank != 0:
            return self
        # Initialise python function
        self.write_python("def %s(%s, **kwargs):" % (
            self.name, ", ".join(["%s=None" % arg for arg in self.args])
        ))
        # Unrestricted spin cases:
        break_line = False
        for res in self.res:
            if len(self.spin_cases.get(res, [])):
                self.write_python("    %s = SimpleNamespace()" % res)
                break_line = True
        if break_line:
            self.write_python("")
        return self

    def __exit__(self, *args, **kwargs):
        if rank != 0:
            return
        # Unrestricted spin cases:
        break_line = False
        for res in self.res:
            for case in self.spin_cases.get(res, []):
                break_line = True
                self.write_python("    %s.%s = %s_%s" % (res, case, res, case))
        if break_line:
            self.write_python("")
        # Return from python function
        if self.return_dict:
            res = "{" + ", ".join(["\"%s\": %s" % (v, v) for v in self.res]) + "}"
        else:
            res = ", ".join(self.res)
        self.write_python("    return %s\n" % res)
        if self.timer is not None:
            print("Time for %s: %.5f s" % (self.name, self.timer()))


# TODO this is a horrible horrible mess...
def get_function_printer(spin, has_bosons=False):
    if spin == "uhf":
        FunctionPrinter_ = FunctionPrinter
    else:
        class FunctionPrinter_(FunctionPrinter):
            def __init__(self, *args, **kwargs):
                kwargs["spin_cases"] = {}
                return FunctionPrinter.__init__(self, *args, **kwargs)

    if has_bosons:
        class FunctionPrinter__(FunctionPrinter_):
            def __init__(self, *args, init_gc=True, **kwargs):
                FunctionPrinter_.__init__(self, *args, **kwargs)
                self.init_gc = init_gc

            def __enter__(self):
                self = FunctionPrinter_.__enter__(self)
                if self.init_gc:
                    if spin == "uhf":
                        self.write_python(
                                "    # Get boson coupling creation array:\n"
                                "    gc = SimpleNamespace(\n"
                                "        aa = SimpleNamespace(\n"
                                "            boo=g.aa.boo.transpose(0, 2, 1),\n"
                                "            bov=g.aa.bvo.transpose(0, 2, 1),\n"
                                "            bvo=g.aa.bov.transpose(0, 2, 1),\n"
                                "            bvv=g.aa.bvv.transpose(0, 2, 1),\n"
                                "        ),\n"
                                "        bb = SimpleNamespace(\n"
                                "            boo=g.bb.boo.transpose(0, 2, 1),\n"
                                "            bov=g.bb.bvo.transpose(0, 2, 1),\n"
                                "            bvo=g.bb.bov.transpose(0, 2, 1),\n"
                                "            bvv=g.bb.bvv.transpose(0, 2, 1),\n"
                                "        ),\n"
                                "    )\n"
                        )
                    else:
                        self.write_python(
                                "    # Get boson coupling creation array:\n"
                                "    gc = SimpleNamespace(\n"
                                "        boo=g.boo.transpose(0, 2, 1),\n"
                                "        bov=g.bvo.transpose(0, 2, 1),\n"
                                "        bvo=g.bov.transpose(0, 2, 1),\n"
                                "        bvv=g.bvv.transpose(0, 2, 1),\n"
                                "    )\n"
                        )
                return self
    else:
        FunctionPrinter__ = FunctionPrinter_

    return FunctionPrinter__


def get_sizes(nocc, nvir, nbos, spin):
    if spin == "uhf":
        sizes = {"nocc[0]": nocc, "nocc[1]": nocc, "nvir[0]": nvir, "nvir[1]": nvir}
    else:
        sizes = {"nocc": nocc, "nvir": nvir}

    if nbos:
        sizes["nbos"] = nbos

    return sizes


class Stopwatch:
    def __init__(self):
        self._t0 = timer()
        self._t = timer()

    def split(self):
        t = timer() - self._t0
        return t

    def lap(self):
        t = timer() - self._t
        self._t = timer()
        return t

    def reset(self):
        self.__init__()
        return self

    __call__ = lap


def pack_2e(*args):
    # args should be in the order of ov_2e

    assert len(args) == len(ov_2e)

    nocc = args[0].shape[0]
    nvir = args[-1].shape[-1]
    occ = slice(None, nocc)
    vir = slice(nocc, None)
    out = np.zeros((nocc+nvir,) * 4)

    for key, arg in zip(ov_2e, args):
        slices = [occ if x == "o" else vir for x in key]
        out[tuple(slices)] = arg

    return out

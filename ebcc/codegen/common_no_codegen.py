
import numpy as np

ov_2e = ["oooo", "ooov", "oovo", "ovoo", "vooo", "oovv", "ovov", "ovvo", "voov", "vovo", "vvoo", "ovvv", "vovv", "vvov", "vvvo", "vvvv"]
ov_1e = ["oo", "ov", "vo", "vv"]

def pack_2e(*args):
    # args should be in the order of ov_2e

    assert len(args) == len(ov_2e)

    nocc = args[0].shape
    nvir = args[-1].shape
    occ = [slice(None, n) for n in nocc]
    vir = [slice(n, None) for n in nocc]
    out = np.zeros(tuple(no+nv for no, nv in zip(nocc, nvir)))

    for key, arg in zip(ov_2e, args):
        slices = [occ[i] if x == "o" else vir[i] for i, x in enumerate(key)]
        out[tuple(slices)] = arg

    return out

"""File dumping and reading functionality."""

from pyscf import scf
from pyscf.lib.chkfile import dump, dump_mol, load, load_mol

from ebcc import util
from ebcc.ansatz import Ansatz
from ebcc.space import Space


class Dump:
    """
    File handler for reading and writing EBCC calculations.

    Attributes
    ----------
    name : str
        The name of the file.
    """

    def __init__(self, name):
        self.name = name

    def write(self, ebcc):
        """
        Write the EBCC object to the file.

        Parameters
        ----------
        ebcc : EBCC
            The EBCC object to write.
        """

        # Write the options
        dic = {}
        for key, val in ebcc.options.__dict__.items():
            if val is not None:
                dic[key] = val
        dump(self.name, "options", dic)

        # Write the mean-field data
        dic = {
            "e_tot": ebcc.mf.e_tot,
            "mo_energy": ebcc.mf.mo_energy,
            "mo_coeff": ebcc.mf.mo_coeff,
            "mo_occ": ebcc.mf.mo_occ,
        }
        dump_mol(ebcc.mf.mol, self.name)
        dump(self.name, "mean-field", dic)

        # Write the MOs used
        dic = {
            "mo_coeff": ebcc.mo_coeff,
            "mo_occ": ebcc.mo_occ,
        }
        dump(self.name, "mo", dic)

        # Write the ansatz
        dic = {
            "fermion_ansatz": ebcc.ansatz.fermion_ansatz,
            "boson_ansatz": ebcc.ansatz.boson_ansatz,
            "fermion_coupling_rank": ebcc.ansatz.fermion_coupling_rank,
            "boson_coupling_rank": ebcc.ansatz.boson_coupling_rank,
        }
        if ebcc.ansatz.module_name is not None:
            dic["module_name"] = ebcc.ansatz.module_name
        dump(self.name, "ansatz", dic)

        # Write the space
        if ebcc.spin_type == "U":
            dic = {
                "occupied": (ebcc.space[0]._occupied, ebcc.space[1]._occupied),
                "frozen": (ebcc.space[0]._frozen, ebcc.space[1]._frozen),
                "active": (ebcc.space[0]._active, ebcc.space[1]._active),
            }
        else:
            dic = {
                "occupied": ebcc.space._occupied,
                "frozen": ebcc.space._frozen,
                "active": ebcc.space._active,
            }
        dump(self.name, "space", dic)

        # Write the bosonic parameters
        dic = {}
        if ebcc.omega is not None:
            dic["omega"] = ebcc.omega
        if ebcc.bare_g is not None:
            dic["bare_g"] = ebcc.bare_g
        if ebcc.bare_G is not None:
            dic["bare_G"] = ebcc.bare_G
        dump(self.name, "bosons", dic)

        # Write the Fock matrix
        # TODO write the Fock matrix class instead

        # Write miscellaneous data
        kwargs = {
            "spin_type": ebcc.spin_type,
        }
        if ebcc.e_corr is not None:
            kwargs["e_corr"] = ebcc.e_corr
        if ebcc.converged is not None:
            kwargs["converged"] = ebcc.converged
        if ebcc.converged_lambda is not None:
            kwargs["converged_lambda"] = ebcc.converged_lambda
        dump(self.name, "misc", kwargs)

        # Write the amplitudes
        if ebcc.spin_type == "U":
            if ebcc.amplitudes is not None:
                dump(
                    self.name,
                    "amplitudes",
                    {
                        key: ({**val} if isinstance(val, (util.Namespace, dict)) else val)
                        for key, val in ebcc.amplitudes.items()
                    },
                )
            if ebcc.lambdas is not None:
                dump(
                    self.name,
                    "lambdas",
                    {
                        key: ({**val} if isinstance(val, (util.Namespace, dict)) else val)
                        for key, val in ebcc.lambdas.items()
                    },
                )
        else:
            if ebcc.amplitudes is not None:
                dump(self.name, "amplitudes", {**ebcc.amplitudes})
            if ebcc.lambdas is not None:
                dump(self.name, "lambdas", {**ebcc.lambdas})

    def read(self, cls, log=None):
        """
        Load the file to an EBCC object.

        Parameters
        ----------
        cls : type
            EBCC class to load the file to.
        log : Logger, optional
            Logger to assign to the EBCC object.

        Returns
        -------
        ebcc : EBCC
            The EBCC object loaded from the file.
        """

        # Load the options
        dic = load(self.name, "options")
        options = cls.Options()
        for key, val in dic.items():
            setattr(options, key, val)

        # Load the miscellaneous data
        misc = load(self.name, "misc")
        spin_type = misc.pop("spin_type").decode("ascii")

        # Load the mean-field data
        mf_cls = {"G": scf.GHF, "U": scf.UHF, "R": scf.RHF}[spin_type]
        mol = load_mol(self.name)
        dic = load(self.name, "mean-field")
        mf = mf_cls(mol)
        mf.__dict__.update(dic)

        # Load the MOs used
        dic = load(self.name, "mo")
        mo_coeff = dic.get("mo_coeff", None)
        mo_occ = dic.get("mo_occ", None)

        # Load the ansatz
        dic = load(self.name, "ansatz")
        module_name = dic.get("module_name", None)
        if isinstance(module_name, str):
            module_name = module_name.encode("ascii")
        ansatz = Ansatz(
            dic.get("fermion_ansatz", b"CCSD").decode("ascii"),
            dic.get("boson_ansatz", b"").decode("ascii"),
            dic.get("fermion_coupling_rank", 0),
            dic.get("boson_coupling_rank", 0),
            module_name,
        )

        # Load the space
        dic = load(self.name, "space")
        if spin_type == "U":
            space = (
                Space(
                    dic.get("occupied", None)[0],
                    dic.get("frozen", None)[0],
                    dic.get("active", None)[0],
                ),
                Space(
                    dic.get("occupied", None)[1],
                    dic.get("frozen", None)[1],
                    dic.get("active", None)[1],
                ),
            )
        else:
            space = Space(
                dic.get("occupied", None),
                dic.get("frozen", None),
                dic.get("active", None),
            )

        # Load the bosonic parameters
        dic = load(self.name, "bosons")
        omega = dic.get("omega", None)
        bare_g = dic.get("bare_g", None)
        bare_G = dic.get("bare_G", None)

        # Load the Fock matrix
        # TODO load the Fock matrix class instead

        # Load the amplitudes
        amplitudes = load(self.name, "amplitudes")
        lambdas = load(self.name, "lambdas")
        if spin_type == "U":
            if amplitudes is not None:
                amplitudes = {
                    key: (util.Namespace(**val) if isinstance(val, dict) else val)
                    for key, val in amplitudes.items()
                }
                amplitudes = util.Namespace(**amplitudes)
            if lambdas is not None:
                lambdas = {
                    key: (util.Namespace(**val) if isinstance(val, dict) else val)
                    for key, val in lambdas.items()
                }
                lambdas = util.Namespace(**lambdas)
        else:
            if amplitudes is not None:
                amplitudes = util.Namespace(**amplitudes)
            if lambdas is not None:
                lambdas = util.Namespace(**lambdas)

        # Initialise the EBCC object
        cc = cls(
            mf,
            log=log,
            ansatz=ansatz,
            space=space,
            omega=omega,
            g=bare_g,
            G=bare_G,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            # fock=fock,
            options=options,
        )
        cc.__dict__.update(misc)
        cc.amplitudes = amplitudes
        cc.lambdas = lambdas

        return cc

import numpy as np
from pyscf import scf, ao2mo

def block_diag(A, B):
    """Return a block diagonal matrix
       A 0
       0 B
    """
    ma = A.shape[0]
    mb = B.shape[0]
    na = A.shape[1]
    nb = B.shape[1]

    z1 = np.zeros((ma, nb))
    z2 = np.zeros((mb, na))
    M1 = np.hstack((A, z1))
    M2 = np.hstack((z2, B))
    return np.vstack((M1, M2))

class one_e_blocks(object):
    def __init__(self, oo, ov, vo, vv):
        self.oo = oo
        self.ov = ov
        self.vo = vo
        self.vv = vv


class two_e_blocks(object):
    def __init__(self,
                 vvvv=None, vvvo=None, vovv=None, vvoo=None,
                 vovo=None, oovv=None, vooo=None, ooov=None, oooo=None):
        self.vvvv = vvvv
        self.vvvo = vvvo
        self.vovv = vovv
        self.vvoo = vvoo
        self.vovo = vovo
        self.oovv = oovv
        self.vooo = vooo
        self.ooov = ooov
        self.oooo = oooo

class FAKE_EPCC_MODEL:
    # Make a fake 'model' class for epcc, so that we can run their cc code

    def __init__(self, mol, mf, eri, omega=None, gmat=None, shift=True):

        if isinstance(mf, scf.rhf.RHF):
            self.mf = scf.addons.convert_to_uhf(mf)
        elif isinstance(mf, scf.uhf.UHF):
            self.mf = copy.copy(mf)
        else:
            raise NotImplementedError
        self.mol = mol

        # Number of spatial orbitals
        self.nao = len(self.mf.mo_occ[0])
        # Number of spinorbitals
        self.nso = 2*self.nao
        
        self.eri = eri
        assert(eri.size == self.nao**4)

        # Alpha and beta electron numbers
        self.na, self.nb = int(self.mf.mo_occ[0].sum()), int(self.mf.mo_occ[1].sum())
        # Number of alpha and beta virtual orbitals
        self.va, self.vb = self.nao - self.na, self.nao - self.nb
        # Number of occupied and virtual spinorbitals
        self.no = self.na + self.nb
        self.nv = self.nso - self.no
        # Alpha and beta density matrices
        self.pa = np.einsum('ai,bi->ab',self.mf.mo_coeff[0][:,:self.na], self.mf.mo_coeff[0][:,:self.na])
        self.pb = np.einsum('ai,bi->ab',self.mf.mo_coeff[1][:,:self.nb], self.mf.mo_coeff[1][:,:self.nb])
        self.ptot = block_diag(self.pa, self.pb)
        
        t = self.mf.get_hcore()
        # Create ghf like one-electron hamiltonian
        self.tmat = block_diag(t,t)

        # Construct bare fermionic Fock matrix in block diagonal spin-orbital form, in the AO basis
        fock_u = (t, t) + self.mf.get_veff(self.mol, dm=(self.pa, self.pb))
        self.bare_fock = block_diag(fock_u[0], fock_u[1])

        if omega is not None:
            self.gmatso = np.asarray([block_diag(gmat[i], gmat[i]) for i in range(len(gmat))])
            # Block diagonal alpha and beta occupied MOs
            Co = block_diag(self.mf.mo_coeff[0][:,:self.na], self.mf.mo_coeff[1][:,:self.nb])
            # Block diagonal alpha and beta virtual MOs
            Cv = block_diag(self.mf.mo_coeff[0][:,self.na:], self.mf.mo_coeff[1][:,self.nb:])

            # Transform each block in spinorbitals
            oo = np.einsum('Ipq,pi,qj->Iij',self.gmatso,Co,Co)
            ov = np.einsum('Ipq,pi,qa->Iia',self.gmatso,Co,Cv)
            vo = np.einsum('Ipq,pa,qi->Iai',self.gmatso,Cv,Co)
            vv = np.einsum('Ipq,pa,qb->Iab',self.gmatso,Cv,Cv)
            self.g = one_e_blocks(oo,ov,vo,vv)

            self.w = omega

            self.shift = shift
            if self.shift:
                self.xi = np.einsum('Iab,ab->I', self.gmatso, self.ptot) / self.w
                self.const = -np.einsum('I,I->',self.w, self.xi**2)
                self.G = np.zeros_like(self.w)
            else:
                self.G = np.einsum('Ipq,qp->I',self.gmatso,self.ptot)
                self.xi = np.zeros_like(self.w)
                self.const = 0.0
        else:
            self.shift = False
            self.const = 0.0

    def gint(self):
        return (self.g, self.g)

    def omega(self):
        return self.w

    def energies(self):
        f = self.g_fock()
        return f.oo.diagonal(), f.vv.diagonal()

    def mfG(self):
        return (self.G, self.G)

    def g_fock(self):

        # Block diagonal alpha and beta occupied MOs
        Co = block_diag(self.mf.mo_coeff[0][:,:self.na], self.mf.mo_coeff[1][:,:self.nb])
        # Block diagonal alpha and beta virtual MOs
        Cv = block_diag(self.mf.mo_coeff[0][:,self.na:], self.mf.mo_coeff[1][:,self.nb:])
        if self.shift:
            Foo = np.einsum('pi,pq,qj->ij',Co,self.bare_fock,Co) - 2*np.einsum('I,pi,Ipq,qj->ij',self.xi, Co, self.gmatso, Co)
            Fov = np.einsum('pi,pq,qa->ia',Co,self.bare_fock,Cv) - 2*np.einsum('I,pi,Ipq,qa->ia',self.xi, Co, self.gmatso, Cv)
            Fvo = np.einsum('pa,pq,qi->ai',Cv,self.bare_fock,Co) - 2*np.einsum('I,pa,Ipq,qi->ai',self.xi, Cv, self.gmatso, Co)
            Fvv = np.einsum('pa,pq,qb->ab',Cv,self.bare_fock,Cv) - 2*np.einsum('I,pa,Ipq,qb->ab',self.xi, Cv, self.gmatso, Cv)
        else:
            Foo = np.einsum('pi,pq,qj->ij',Co,self.bare_fock,Co)
            Fov = np.einsum('pi,pq,qa->ia',Co,self.bare_fock,Cv)
            Fvo = np.einsum('pa,pq,qi->ai',Cv,self.bare_fock,Co)
            Fvv = np.einsum('pa,pq,qb->ab',Cv,self.bare_fock,Cv)
        return one_e_blocks(Foo, Fov, Fvo, Fvv)

    def hf_energy(self):
        ehf = 0.5*(np.einsum('ij,ji->',self.ptot,self.bare_fock) + np.einsum('ij,ji->',self.ptot,self.tmat))
        ehf = ehf + self.mol.energy_nuc()
        if self.shift:
            ehf += self.const
        return ehf

    def g_aint(self):
        
        # Get full array of alpha then beta orbitals
        C = np.hstack((self.mf.mo_coeff[0], self.mf.mo_coeff[1]))
        
        eri1 = np.einsum('ai,abcd->ibcd',C,self.eri)
        eri2 = np.einsum('bj,ibcd->ijcd',C,eri1)
        eri3 = np.einsum('ck,ijcd->ijkd',C,eri2)
        # Integrals now transformed, and in MO spin-orbital basis
        eri_g = np.einsum('dl,ijkd->ijkl',C,eri3)

        eri_g[:self.nao, self.nao:] = 0.0
        eri_g[self.nao:, :self.nao] = 0.0
        eri_g[:,:,:self.nao, self.nao:] = 0.0
        eri_g[:,:,self.nao:, :self.nao] = 0.0

        ## Get chemical notation eris from pyscf in generalized basis (i.e. the full block of alpha, beta MOs in each dim)
        #eri = ao2mo.general(self.mol, [C,]*4, compact=False).reshape([self.nso,]*4)
        ## Zero spin forbidden sectors
        #eri[:self.nao, self.nao:] = eri[self.nao:, :self.nao] = eri[:,:,:self.nao, self.nao:] = eri[:,:,self.nao:,:self.nao] = 0.0

        #eri_g = np.zeros((self.nso, self.nso, self.nso, self.nso))
        #eri_g[:self.nao, :self.nao, :self.nao, :self.nao] = self.eri.reshape([self.nao,]*4)
        #eri_g[self.nao:, self.nao:, :self.nao, :self.nao] = self.eri.reshape([self.nao,]*4)
        #eri_g[:self.nao, :self.nao, self.nao:, self.nao:] = self.eri.reshape([self.nao,]*4)
        #eri_g[self.nao:, self.nao:, self.nao:, self.nao:] = self.eri.reshape([self.nao,]*4)

        # Convert to antisymmetrized physicist notation integrals
        Ua_mo = eri_g.transpose(0,2,1,3) - eri_g.transpose(0,2,3,1)

        # Reorder integrals, s.t. ordering is occupied,alpha, occupied,beta
        temp = [i for i in range(self.nso)]
        oidx = temp[:self.na] + temp[self.nao:self.nao + self.nb]
        vidx = temp[self.na:self.nao] + temp[self.nao + self.nb:]

        # Separate into GHF blocks
        vvvv = Ua_mo[np.ix_(vidx,vidx,vidx,vidx)]
        vvvo = Ua_mo[np.ix_(vidx,vidx,vidx,oidx)]
        vovv = Ua_mo[np.ix_(vidx,oidx,vidx,vidx)]
        vvoo = Ua_mo[np.ix_(vidx,vidx,oidx,oidx)]
        oovv = Ua_mo[np.ix_(oidx,oidx,vidx,vidx)]
        vovo = Ua_mo[np.ix_(vidx,oidx,vidx,oidx)]
        vooo = Ua_mo[np.ix_(vidx,oidx,oidx,oidx)]
        ooov = Ua_mo[np.ix_(oidx,oidx,oidx,vidx)]
        oooo = Ua_mo[np.ix_(oidx,oidx,oidx,oidx)]
        return two_e_blocks(vvvv=vvvv,vvvo=vvvo, vovv=vovv, vvoo=vvoo, oovv=oovv,vovo=vovo, vooo=vooo, ooov=ooov, oooo=oooo)

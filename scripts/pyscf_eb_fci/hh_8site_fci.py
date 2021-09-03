import numpy

def setup_hh(nsite, t, U, omega0, g, apbc):
    '''
    set up standard Hubbard-Holstein Hamiltonian
    '''
    idx = numpy.arange(nsite-1)
    tmat_hh = numpy.zeros((nsite,)*2)
    tmat_hh[idx+1,idx] = tmat_hh[idx,idx+1] = -t
    tmat_hh[0,nsite-1] = tmat_hh[nsite-1,0] = t if apbc else -t

    Umat_hh = numpy.zeros((nsite,)*4)
    for i in range(nsite):
        Umat_hh[i,i,i,i] = U

    Vmat_hh = numpy.zeros((nsite,)*3)
    for i in range(nsite):
        Vmat_hh[i,i,i] = g

    Omat_hh = numpy.zeros((nsite,)*2)
    for i in range(nsite):
        Omat_hh[i,i] = omega0

    return tmat_hh, Umat_hh, Vmat_hh, Omat_hh


def delta_zpm(g, nelec, omega0, nsite):
    '''
    get the constant shift in energy owing to the ZPM transformation
    '''
    return (g**2 * nelec**2) / (omega0*nsite)

def make_zpm_coeffs(tmat, nsite, nelec, g, omega0):
    '''
    effect the ZPM transformation on the one-electron integrals, and also return the "uncoupled" boson
    excitation/de-excitation coefficents, and the ZPM energy shift
    '''
    nav = nelec/float(nsite)
    tmat_trans = tmat.copy()
    tmat_trans -= numpy.eye(nsite)*(2.0*nav*g**2)/omega0
    return tmat_trans, numpy.ones(nsite)*nav*g, delta_zpm(g, nelec, omega0, nsite)


from eb_solve import kernel
nsite, nelec = 4, 4
t = 1.
U = 1.
omega0 = 1.5
g = numpy.sqrt(0.15)
nboson_max = 1
apbc = False

tmat, Umat, Vmat, Omat = setup_hh(nsite, t, U, omega0, g, apbc)

tmat_trans, Vmat_unc, delta_zpm = make_zpm_coeffs(tmat, nsite, nelec, g, omega0)
print('delta zpm: {}'.format(delta_zpm))

res0 = kernel(tmat, Umat, Vmat, Omat, nsite, nelec, nsite, nboson_max, adj_zero_pho = False)
e1 = res0[0]
print('without ZPM removal', e1)
res0 = kernel(tmat_trans, Umat, Vmat, Omat, nsite, nelec, nsite, nboson_max, adj_zero_pho = True)
e2 = res0[0]
print('with ZPM removal', e2+delta_zpm)

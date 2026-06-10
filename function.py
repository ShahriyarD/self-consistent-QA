
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def get_spin(state, position):
    '''Returns the spin value at the specified position.'''
    return 1 if (state >> position) & 1 == 0 else -1

def H_z(L=11, seed=93):
    np.random.seed(seed)
    mean, var = 0, 1/L
    std = np.sqrt(var)
    dim = 2**L
    H = sp.lil_matrix((dim, dim), dtype=np.float32)
    for state in range(dim):
        diag = 0
        for i in range(L):
            for j in range(i+1, L):
                diag += np.random.normal(mean, std) * get_spin(state, i) * get_spin(state, j)
        diff = bin(state).count('1') - (L - bin(state).count('1'))
        diag += np.random.normal(mean, std) * diff
        H[state, state] = diag
    return H.tocsc()

def H_x(L=11):
    dim = 2**L
    H = sp.lil_matrix((dim, dim), dtype=np.float32)
    for state in range(dim):
        for site in range(L):
            H[state, state ^ (1 << site)] = -1
    return H.tocsc()


def TVI(L=11):
    dim = 2**L
    H = sp.lil_matrix((dim, dim), dtype=np.float32)
    for state in range(dim):
        for i in range(L):
            for j in range(L):
                state_prime = state ^ (1 << i)
                state_dprime = state_prime ^ (1 << j)
                H[state, state_dprime] += 1/L
    return H.tocsc()

def qsim(L=11, T=1000, f=1000, seed=93, k=10000):
    
    Hx = H_x(L)
    mx_eigvals, mx_eigvecs = np.linalg.eigh(Hx.toarray())
    xx = TVI(L)
    dim = 2**L
    initial_vec = np.full(dim, -1/np.sqrt(dim), dtype=np.complex64)
    Hz = H_z(L, seed)
    ge, gs = spla.eigsh(Hz, k = 1, which='SA')
    
    dt = 0.001

    s = 0
    vec = initial_vec
    tvi = []
    
    for i in range(int(T/dt)):
        s = (i+1) * dt/T
        lmbd = s
        H = s*lmbd*Hz - s*(1-lmbd)*xx + (1-s)*Hx
        vec = spla.expm_multiply(-1j * H * dt, vec)
        vec = vec / np.linalg.norm(vec)    
        tvi.append(np.abs(np.vdot(gs, vec))**2)
    
    s = 0
    vec = initial_vec
    m_x =[-1]
    sc=[]

    for i in range(int(T/dt)):
        s = (i+1) * dt/T
        lmbd = s
        
        h = 2*s*(1-lmbd)*m_x[-1] + (1-s)

        H = s*lmbd*Hz + h*Hx
        vec = spla.expm_multiply(-1j * H * dt, vec)
        vec = vec / np.linalg.norm(vec)
        sc.append(np.abs(np.vdot(gs, vec))**2)
        
        if (i+1) % f == 0:
            print(s, sc[-1], m_x[-1])
            coeff = mx_eigvecs.conj().T @ vec
            prob     = np.abs(coeff)**2
            cum_prob = np.cumsum(prob)
            cum_prob[-1] = 1.0
            tickets = np.random.rand(k)
            idx     = np.searchsorted(cum_prob, tickets)
            m_x.append((mx_eigvals[idx] / L).mean())
            

    s = 0
    vec = initial_vec
    cqa = []
    
    for i in range(int(T/dt)):
        s = (i+1) * dt/T
        H = s*Hz + (1-s)*Hx
        vec = spla.expm_multiply(-1j * H * dt, vec)
        vec = vec/np.linalg.norm(vec)     
        cqa.append(np.abs(np.vdot(gs, vec))**2)
  
    return tvi, sc, cqa


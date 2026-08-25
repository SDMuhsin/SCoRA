"""VENDORED VERBATIM from LoCA (ICLR'25) -- github.com/TL-UESTC/LoCA,
peft/src/peft/tuners/loca/dct_utils.py.  Cloned to ./temp/LoCA.

NOT REIMPLEMENTED.  This file is byte-identical to the authors' original except
for this header; `src/verify_loca_adapter.py` gate G0 asserts that.  Keeping the
authors' own DCT/iDCT is what makes our LoCA arm THEIR method rather than our
reading of it (the [R.95] discipline: our FourierFT dW is bit-identical to
peft.tuners.fourierft, which is why every FourierFT number in this repo is
attributable to the official operator).
"""
import torch
import numpy as np

class DCTMatrixCache:
    def __init__(self):
        self.matrices = {}

    def get_matrix(self, N, device, dtype):
        key = (N, device, dtype)
        # print(key)
        if key not in self.matrices:
            # print(N, device, dtype)
            matrix = self._create_dct_matrix(N).to(device).to(dtype)
            self.matrices[key] = matrix.detach()
        return self.matrices[key]

    def _create_dct_matrix(self, N):
        i = torch.arange(N).unsqueeze(1)
        j = torch.arange(N).unsqueeze(0)
        
        matrix = torch.cos((2 * j + 1) * i * torch.pi / (2 * N))
        scale = torch.sqrt(torch.tensor(2.0 / N))
        matrix *= scale
        matrix[0] /= torch.sqrt(torch.tensor(2.0))
        return matrix

class FASTDCTMatrixCache:
    def __init__(self):
        self.matrices_W_dct = {}
        self.matrices_W_idct = {}
        self.matrices_V_t = {}
        self.matrices_result_x = {}
    
    def get_matrix_W_idct(self, N, device, dtype):
        key = (N, device, dtype)
        if key not in self.matrices_W_idct:
            matrix_W = self._create_idct_W_matrix(N).to(device).to(dtype)
            self.matrices_W_idct[key] = matrix_W.detach()
        return self.matrices_W_idct[key]
    

    def get_matrix_W_dct(self, N, device, dtype):
        key = (N, device, dtype)
        if key not in self.matrices_W_dct:
            matrix_W = self._create_dct_W_matrix(N).to(device).to(dtype)
            self.matrices_W_dct[key] = matrix_W.detach()
        return self.matrices_W_dct[key]

    def _create_idct_W_matrix(self, N):
        k = torch.arange(N)[None, :] * torch.pi / (2 * N)
        W = torch.exp(1j * k) 
        return W

    def _create_dct_W_matrix(self, N):
        k = -torch.arange(N)[None, :] * torch.pi / (2 * N)
        W = torch.exp(1j * k) 
        return W

FAST_DCT_CACHE = FASTDCTMatrixCache()
DCT_CACHE = DCTMatrixCache()

def sparse_idct2d(coefs, L, M, N):
    C_M_T = DCT_CACHE.get_matrix(M, coefs.device, coefs.dtype).T
    C_N = DCT_CACHE.get_matrix(N, coefs.device, coefs.dtype)

    C_M_T_selected = C_M_T[:, L[0,:]]
    C_N_selected = C_N[L[1,:], :]
    Y = (C_M_T_selected * coefs.unsqueeze(0)) @ C_N_selected
    return Y

def ori_idct2d(coefs, L, M, N):
    C_M_T = DCT_CACHE.get_matrix(M, coefs.device, coefs.dtype).T
    C_N = DCT_CACHE.get_matrix(N, coefs.device, coefs.dtype)
    dense_matrix = torch.sparse_coo_tensor(L, coefs, torch.Size([M, N])).to_dense()
    # print(C_M_T.dtype, sparse_coefs.dtype)
    # Y = torch.sparse.mm(C_M_T, sparse_coefs)
    # Y = torch.mm(Y.to_dense(), C_N)
    return C_M_T @ dense_matrix @ C_N

def fast_idct(X, norm=None):
    x_shape = X.shape
    M, N = x_shape[0], x_shape[1]
    X_v = X.contiguous().view(-1, N) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    
    V_t = torch.empty_like(X_v, dtype=torch.complex64)
    V_t.real = X_v
    V_t.imag[:, 1:] = -X_v.flip([1])[:, :-1]
    
    V = V_t * FAST_DCT_CACHE.get_matrix_W_idct(N, X.device, X.dtype)
    v = torch.fft.irfft(V, n=N, dim=1)

    return torch.stack([v[:, :N - (N // 2)], v.flip([1])[:, :N // 2]], dim=2).flatten(1)

def fast_idct2d(coefs, L, M, N, norm='ortho'):
    '''
    The current fast idct is based on torch.fft.irfft
    '''    
    X = torch.sparse_coo_tensor(L, coefs, torch.Size([M, N])).to_dense()
    x1 = fast_idct(X, norm=norm)
    x2 = fast_idct(x1.T, norm=norm)
    return x2.T

def fast_dct(x, norm='ortho'):
    '''
    The current fast dct is based on torch.fft.fft
    '''    
    N = x.shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v, dim=1)

    V = (Vc * FAST_DCT_CACHE.get_matrix_W_dct(N, x.device, x.dtype)).real

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    # V = 2 * V.view(*x_shape)
    return V * 2

def fast_dct_2d(x, norm='ortho'):
    X1 = fast_dct(x, norm=norm)
    X2 = fast_dct(X1.T, norm=norm)
    return X2.T

def ori_dct2d(x):
    M, N = x.shape
    C_M = DCT_CACHE.get_matrix(M, x.device, x.dtype)
    C_N_T = DCT_CACHE.get_matrix(N, x.device, x.dtype).T
    return C_M @ x @ C_N_T


def idct_2d_impl(coefs, L, M, N, mode):
    if mode == 'default':
        # perform 2d dct in its original form
        return ori_idct2d(coefs, L, M, N)
    elif mode == 'sparse':
        return sparse_idct2d(coefs, L, M, N)
    elif mode == 'fast':
        return fast_idct2d(coefs, L, M, N)
    else:
        raise NotImplementedError("{} mode has not been implemented yet.".format(mode))    

def dct_2d_impl(matrix, mode):
    if mode == 'default' or mode == 'sparse':
        # perform 2d dct in its original form
        return ori_dct2d(matrix) 
    elif mode == 'fast':
        return fast_dct_2d(matrix)
    else:
        raise NotImplementedError("{} mode has not been implemented yet.".format(mode))     
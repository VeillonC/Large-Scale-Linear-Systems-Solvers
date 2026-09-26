import numpy as np
import scipy.sparse as sp

def jacobi_dense(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Jacobi iterative solver for dense matrices.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    errors = []
    for k in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        err = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(err)
        
        if err < tol:
            return x_new, k + 1, errors
        x = x_new
        
    return x, max_iter, errors


def jacobi_sparse(A_csr, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Jacobi iterative solver using SciPy CSR sparse matrix representation.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    diag_A = A_csr.diagonal()
    D_inv = 1.0 / diag_A
    
    R = A_csr.copy()
    R.setdiag(0)
    R.eliminate_zeros()
    
    errors = []
    for k in range(max_iter):
        x_new = D_inv * (b - R.dot(x))
        err = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(err)
        
        if err < tol:
            return x_new, k + 1, errors
        x = x_new
        
    return x, max_iter, errors


def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Gauss-Seidel iterative solver.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    errors = []
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
            
        err = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(err)
        
        if err < tol:
            return x_new, k + 1, errors
        x = x_new
        
    return x, max_iter, errors


def sor_solver(A, b, omega, x0=None, tol=1e-6, max_iter=1000):
    """
    Successive Over-Relaxation (SOR) iterative solver.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    errors = []
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_gauss = (b[i] - s1 - s2) / A[i, i]
            x_new[i] = (1 - omega) * x[i] + omega * x_gauss
            
        err = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(err)
        
        if err < tol:
            return x_new, k + 1, errors
        x = x_new
        
    return x, max_iter, errors

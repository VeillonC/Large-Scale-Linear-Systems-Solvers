import numpy as np

def generate_tridiagonal_matrix(n, a=-1, b=2, c=-1):
    """
    Generates a 1D Laplacian-like tridiagonal matrix of size n x n.
    """
    main_diag = b * np.ones(n)
    off_diag = a * np.ones(n - 1)
    A = np.diag(main_diag) + np.diag(off_diag, k=-1) + np.diag(off_diag, k=1)
    return A

def compute_spectral_radius(M):
    """
    Computes the spectral radius rho(M) = max(|lambda|).
    """
    eigenvalues = np.linalg.eigvals(M)
    return np.max(np.abs(eigenvalues))

def compute_optimal_omega(A):
    """
    Computes theoretical optimal relaxation parameter omega for SOR
    based on Jacobi spectral radius for tridiagonal matrices.
    """
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    D_inv = np.linalg.inv(D)
    J = np.dot(D_inv, -L_plus_U)
    
    rho_J = compute_spectral_radius(J)
    omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_J**2))
    return omega_opt, rho_J

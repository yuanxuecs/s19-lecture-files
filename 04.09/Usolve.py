import numpy as np
 
def Usolve(U, y):
    """Backward solve an upper triangular system Ux = y for x
    Parameters:
      U: the matrix, must be square, upper triangular, with nonzeros on the diagonal
      y: the right-hand side vector
    Output:
      x: the solution vector to U @ x == y
    """
    # Check the input
    m, n = L.shape
    assert m == n, "matrix L must be square"
    assert np.all(np.triu(U) == U), "matrix U must be lower triangular"
    assert np.all(np.diag(U) != 0), "matrix U must have ones on the diagonal"
    
    # Make a copy of b that we will transform into the solution
    x = y.astype(np.float64).copy()
    
    # Forward solve
    for col in range(n):
      x[:col] -= x[col] * U[:col, col]
        
    return x
    



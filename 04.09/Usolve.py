import numpy as np
 
def Usolve(U, y):
    """Backward solve an upper triangular system Ux = y for x
    Parameters:
      U: the matrix, must be square, upper triangular, with nonzeros on the diagonal
      y: the right-hand side vector
    Output:
      x: the solution vector to U @ x == y
    """
    m, n = U.shape
    assert m == n, "matrix U must be square"
    assert np.all(np.triu(U) == U), "matrix U must be Upper triangular"
    assert np.all(np.diag(U) != 0), "matrix U must have ones on the diagonal"
    yn, = y.shape
    assert yn == n,"must be same size as U"
    # Make a copy of y that we will transform into the solution
    x = y.astype(np.float64).copy()
    # back solve
    for col in reversed(range(n)):
        x[col] = x[col]/U[col,col]
        x[:col] = x[:col] - x[col]*U[:col,col]
    return x
   
    



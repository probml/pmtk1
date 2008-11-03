from cvxopt import matrix, spmatrix, sparse, mul
from cvxopt import blas, lapack, amd
import pychordal as pyc
from pylab import load           
from pylab import save
#from pickle import load
import sys

from math import sqrt

MAXITERS = 50
TOL = 1e-5

def mleq(C, A, b):
    '''Solves the maximum likelihood problem with equality constraints
    
    minimize    -logdet(X) + Tr(C*X)
    subject to   Tr(Ai*X) = bi,   i = 1,...,m
                 X \in S_V

    where V is a chordal sparsity pattern implicitly specified from
    the sparsity pattern of the lower triangular part of C.

    
    X = mleq(C, A, b)
    
    ARGUMENTS    
    C      a symmetric sparse matrix with dimensions n x n. Only
           the lower triangular part is used.

    A      a list of symmetric sparse matrices A1, ..., Am with 
           dimensions n x n, Only the lower triangular part is used.

    b      a matrix of dimensions n x 1

    RETURNS    
    X      a symmetric sparse matrix with dimensions n x n
    '''

    p = pyc.maxcardsearch(C)

    # sc is a vector that scales the diagonal elements of L(Ak)
    # note that sc must be reordered with the same permutation p
    # used for constructing the chordalpattern object
    sc = +C
    sc.V = 1.0
    sc -= spmatrix(1 - 1.0/sqrt(2), range(C.size[0]), range(C.size[0]))
    sc = pyc.tril(pyc.perm(pyc.symmetrize(sc),p))
    sc = sqrt(2)*sc.V

    C = pyc.cvxopt_to_pychordal(C, p)
    X = pyc.project(C, spmatrix(1.0, range(n), range(n)))
    A = [ pyc.project(C, Ak) for Ak in A ]

    m = len(A)
    H, y = matrix(0.0, (m,m)), matrix(0.0, (m,1))

    print "It  obj       ntdecr"
    for iters in xrange(MAXITERS):

        # Newton step:
        #   -P(X^{-1}*dX*X^{-1}) + sum_i Ai*yi  = -R
        #                              <Ai, dX> = -ri
        # R  = P(X^{-1}) - C
        # ri = <Ai, X> - bi
        #
        L = pyc.copy(X)
        pyc.cholesky(L)
        Y = pyc.copy(L)
        pyc.partial_inv(Y)

        R = pyc.copy(Y)
        pyc.axpy(C, R, -1.0)

        # H[j,k] = <Aj, dXk>
        # We evalutate H efficiently as   H = At'*At
        #
        # where At = [ sc.*(L^adj)^{-1}(A1), ... , sc.*(L^adj^){-1}(Am)]
        #
        # g[j]   = -rj - Tr(Aj*dX0)
        #        =  bj - Tr(Aj, X + dX0 )
        
        # P(X^{-1}*dX0*X^{-1}) = R
        dX0 = pyc.copy(R)
        pyc.hessian(L, Y, dX0, inv = True, adj=True)

        At = [ pyc.copy(Aj) for Aj in A ]
        for j in xrange(m): 
            pyc.hessian(L, Y, At[j], inv=True, adj=True)

        for k in xrange(m):      
            y[k] = b[k] - pyc.dot(A[k],X) - pyc.dot(At[k], dX0)
        
        At2 = matrix([[mul(sc,pyc.pychordal_to_cvxopt(Ak)[0].V)] for Ak in At])
        blas.syrk(At2, H, trans='T')
        lapack.posv(H, y)
        
        # dX = dX0 + sum_k yk*dXk
        dX = pyc.copy(dX0)
        for k in xrange(m): 
            pyc.axpy(At[k], dX, y[k])

        ntdecr = sqrt(pyc.dot(dX0, dX))
                        
        pyc.hessian(L, Y, dX, inv = True, adj=False)

        f = -pyc.logdet(L) + pyc.dot(C, X)
        print "%2d % .2e  %.1e" %(iters+1, f, ntdecr)

        if ntdecr < TOL: break
                
        #X += dX/(1+ntdecr)
        pyc.axpy(dX, X, 1.0/(1+ntdecr))
        
    X, p = pyc.pychordal_to_cvxopt(X)
    return pyc.perm(pyc.symmetrize(X),p)

def covselFast(C):
    '''Solves the covariance selection problem
    
    minimize    -logdet(X) + Tr(C*X)
    subject to   X_ij = 0,   (i,j) \in  Vt \ V
                 X \in S_Vt

    where V is a chordal sparsity pattern implicitly specified from
    the sparsity pattern of the lower triangular part of C, and Vt
    is the sparsity pattern of the correspondig chordal embedding.

    X = covsel(C)
    
    ARGUMENTS    
    C      a symmetric sparse matrix with dimensions n x n. Only
           the lower triangular part is used.

    RETURNS    
    X      a symmetric sparse matrix with dimensions n x n
    '''

    # V is the sparsity of C
    V = pyc.tril(C)
    V.V = 1.0
    
    # Vt is the sparsity pattern of the embedding
    Ce = pyc.embed(C)
    Vt = +Ce
    Vt.V = 1.0

    Vdiff = sparse(Vt-V)
    m = len(Vdiff)

    A = []
    for k in xrange(m):
        Ak = +Vdiff
        ek = matrix(0.0, (len(Ak),1))
        ek[k] = 1.0                    
        Ak.V = ek
        A.append(Ak)

    b = matrix(0.0, (m, 1))

    return mleq(Ce, A, b)


#if __name__ == '__main__':
#
#n = 7
#C = spmatrix(10.0, range(n), range(n))
#C[1:(n-1)*n:n+1] = 1.0
#C[n-1] = 1
#C[n-2] = 1

    # Comment out the line below to use a fill-reducing reordering 
    #C = pyc.perm(pyc.symmetrize(C), amd.order(C))

#X = covselFast(C)
#save('outtest.data',matrix(C))

#Y = sparse(matrix(load('outtest.data')))
#Y = load(open("covsel.bin","r"))
#covselFast(Y)

#n = 3;
#f = load('input.data')
#g = matrix(f)
#h = sparse(g)

n = int((sys.argv[3]));  
Y = sparse(matrix(load(sys.argv[1])))
Csel = covselFast(Y)
save(sys.argv[2],matrix(Csel))

import numpy as np

def forsub(L,bs):
    n = bs.size
    xs = np.zeros(n)
    for i in range(n):
        xs[i] = (bs[i] - L[i,:i]@xs[:i])/L[i,i]
    return xs

def backsub(U,bs):
    n = bs.size
    xs = np.zeros(n)
    for i in reversed(range(n)):
        xs[i] = (bs[i] - U[i,i+1:]@xs[i+1:])/U[i,i]
    return xs

def testcreate(n,val):
    A = np.arange(val,val+n*n).reshape(n,n)
    A = np.sqrt(A)
    return A

def ludec(A):
    n = A.shape[0]
    U = np.copy(A)
    L = np.identity(n)

    for j in range(n-1):
        for i in range(j+1,n):
            coeff = U[i,j]/U[j,j]
            U[i,j:] -= coeff*U[j,j:]
            L[i,j] = coeff

    return L, U

def mag(xs):
    return np.sqrt(np.sum(xs*xs))

def termcrit(xolds,xnews):
    errs = np.abs((xnews - xolds)/xnews)
    return np.sum(errs)
    
def power(A,kmax=6):
    zs = np.ones(A.shape[0])
    qs = zs/mag(zs)
    for k in range(1,kmax):
        zs = A@qs
        qs = zs/mag(zs)
        #print(k,qs)

    lam = qs@A@qs
    return lam, qs
    
def invpowershift(A,shift=20,kmax=200,tol=1.e-8):
    n = A.shape[0]
    znews = np.ones(n)
    qnews = znews/mag(znews)
    Astar = A - np.identity(n)*shift
    L, U = ludec(Astar)

    for k in range(1,kmax):
        qs = np.copy(qnews)
        ys = forsub(L,qs)
        znews = backsub(U,ys)
        qnews = znews/mag(znews)

        if qs@qnews<0:
            qnews = -qnews

        err = termcrit(qs,qnews)
        #print(k, qnews, err)

        if err < tol:
            lam = qnews@A@qnews
            break
    else:
        lam = qnews = None

    return lam, qnews


def qrdec(A):
    n = A.shape[0]
    Ap = np.copy(A)
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    for j in range(n):
        for i in range(j):
            R[i,j] = Q[:,i]@A[:,j]
            Ap[:,j] -= R[i,j]*Q[:,i]

        R[j,j] = mag(Ap[:,j])
        Q[:,j] = Ap[:,j]/R[j,j]
    return Q, R
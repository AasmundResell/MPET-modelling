from block import block_base
from block.object_pool import vec_pool
import petsc4py, sys
petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import numpy as np
print = PETSc.Sys.Print


class MyExpression(Expression):
    def eval(self, value, x):
        value[0] = 1.0
    def value_shape(self):
        return (1,)


def rm_basis(mesh):
    '''6 functions(Expressions) that are the rigid motions of the body.'''
    assert mesh.topology().dim() == 3
    
    x = SpatialCoordinate(mesh)
    dX = dx(domain=mesh)
    # Center of mass
    c = np.array([assemble(xi*dX) for xi in x])
    volume = assemble(Constant(1)*dX)
    c /= volume
    c_ = c
    c = Constant(c)

    # Gram matrix of rotations around canonical axis and center of mass
    R = np.zeros((3, 3))

    ei_vectors = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0, 1))]
    for i, ei in enumerate(ei_vectors):
        R[i, i] = assemble(inner(cross(x-c, ei), cross(x-c, ei))*dX)
        for j, ej in enumerate(ei_vectors[i+1:], i+1):
            R[i, j] = R[j, i] = assemble(inner(cross(x-c, ei), cross(x-c, ej))*dX)

    # Eigenpairs
    eigw, eigv = np.linalg.eigh(R)      
    if np.min(eigw) < 1E-8: warning('Small eigenvalues %g.' % np.min(eigw))
    eigv = eigv.T

    # Translations: ON basis of translation in direction of rot. axis
    translations = [Constant(v/sqrt(volume)) for v in eigv]

    # Rotations using the eigenpairs
    # C0, C1, C2 = c.values()
    C0, C1, C2 = c_

    def rot_axis_v(pair):
        '''cross((x-c), v)/sqrt(w) as an expression'''
        v, w = pair
        return Expression(('((x[1]-C1)*v2-(x[2]-C2)*v1)/A',
                           '((x[2]-C2)*v0-(x[0]-C0)*v2)/A',
                           '((x[0]-C0)*v1-(x[1]-C1)*v0)/A'),
                           C0=C0, C1=C1, C2=C2, 
                           v0=v[0], v1=v[1], v2=v[2], A=sqrt(w),
                           degree=1)

    # Roations are described as rot around v-axis centered in center of gravity 
    rotations = list(map(rot_axis_v, list(zip(eigv, eigw))))

    Z = translations + rotations
    return Z
      
def is_square(mat):
    '''Matrix is square'''
    return size(mat, 0) == size(mat, 1)


def orthogonalize_gs(vectors, A=None):
    '''Modified Gram-Schmidt orthogonalization'''
    assert A is None or is_square(A)
    mv = []
    for i, veci in enumerate(vectors):
        for j in range(i):
            vectors[i] -= veci.inner(mv[j])*vectors[j]

        if A is not None:
            hv = A*veci
        else:
            hv = veci.copy()
            
        norm = sqrt(veci.inner(hv))
        veci *= 1/norm
        hv *= 1/norm
        mv.append(hv)

    return vectors


"""

def first(x):
    '''First item in iterable'''
    return next(iter(x))


def identity_matrix(V, e=1.):
    '''Diagonal matrix'''
    # Avoiding assembly if this is Real space.
    diag = Function(V).vector()
    global_size = diag.size()
    local_size = diag.local_size()
    #mpi_comm = d.MPI.comm_world

    mpi_comm = V.mesh().mpi_comm()
    
    mat = PETSc.Mat().createAIJ(size=[[local_size, global_size],
                                      [local_size, global_size]], nnz=1, comm=mpi_comm)
    diag = as_backend_type(diag).vec()
    diag.set(e)
    mat.setDiagonal(diag)

    print(type(V.dofmap().tabulate_local_to_global_dofs()))
    lgmap = PETSc.LGMap().create(list(map(int, V.dofmap().tabulate_local_to_global_dofs())), comm=mpi_comm)
    mat.setLGMap(lgmap, lgmap)

    mat.assemblyBegin()
    mat.assemblyEnd()

    return d.PETScMatrix(mat)



class RMBasis(block_base):
    '''
    L2 basis of the space of rigid motions represented as a dim(V) x dim(Q)
    matrix such that each row holds coeffcients of interpolant of a rigid motion 
    basis function.
    '''
    def __init__(self, V, Q, Z=None):
        #assert V.element().value_rank() == 1, "Need vector valued function space"
        #print(V.dim())
        #assert V.dim() == 3 or V.dim() == 2, "Need vector valued function space"
        print(Q.dim())
        assert Q.dim() == 6 or Q.dim() == 3

        # Rigid motions as Functions, might be precomputed outside
        if Z is None: 
            Z = rm_basis(V.mesh())
        else:
            warning('Using precomputed basis.')
        # We keep the coefficents for computing the action
        self.basis = [interpolate(z, V).vector() for z in Z]
        self.V = V
        self.Q = Q

        self.indices = range(*Q.dofmap().ownership_range())
        self.comm = V.mesh().mpi_comm()

    def __iter__(self):
        '''Iterate over basis functions (in V)'''
        I = np.eye(6)
        for i in I: yield self.rigid_motion(i)

    def transpmult(self, x):
        '''Apply to V vectors'''
        values = np.array([z.inner(x) for z in self.basis])  # Lives on everybody
        local_values = values[self.indices]

        y = self.create_vec(1)
        # Fill in
        y.set_local(local_values)
        y.apply('insert')

        return y

    def matvec(self, y):
        '''Apply to Q vectors'''
        return self.rigid_motion(y).vector()

    @vec_pool
    def create_vec(self, dim):
        if dim == 1:
            return Function(self.Q).vector()
        else:
            return Function(self.V).vector()

    def rigid_motion(self, c):
        '''Rigid motion represented in the basis by c.'''
        if isinstance(c, np.ndarray):
            assert len(c) == 6, (c, type(c), len(c))

            x = self.create_vec(0)
            x.zero()
            for ck, zk in zip(c, self.basis): x.axpy(ck, zk)
            
            return d.Function(self.V, x)

        if isinstance(c, GenericVector):
            c = c.gather_on_zero()  # Only master, other have []
            # Communicate the coefficients to everybody
            if len(c) == 0:
                c = np.zeros(6)
            c = self.comm.bcast(c)

            return self.rigid_motion(c)

        assert False


class Projector(block_base):
    '''
    Orthognal projector to rigid motions complement. Actions:
        P*u \in {Z^{\perp}} and P'*b \in {Z^{\polar}} 
    '''
    def __init__(self, Zh):
        # We keep the coefficents for computing the action
        self.basis = [z.copy() for z in Zh.basis]

        V = Zh.V
        # Keep mass matrix for taking inner products
        u, v = TrialFunction(V), TestFunction(V)
        self.M = assemble(inner(u, v)*dx)
        # Auxiliary vector again for inner products
        self.aux = first(self.basis).copy()
        self.V = V
        # Measuring compononts of function in Z
        self.alphas = [0]*6
        # ... functional in Zpolar
        self.betas = [0]*6

    def transpmult(self, x):
        '''
        Orthogonalize vector representing functional in V. Output represents a 
        functional in the polar set of Z.
        '''
        M, aux = self.M, self.aux

        y = self.create_vec(dim=1)
        y.zero()
        y.axpy(1, x)  # y = x
        # Record alpha to get the Z content
        for i, z in enumerate(self.basis):  # y -= Mz*<y, z>
            beta = y.inner(z)  # l2
            M.mult(z, aux)
            y.axpy(-beta, aux)
            self.betas[i] = beta
        return y

    def matvec(self, y):
        '''
        Orthogonalize vector representing function in V. Output represents a
        function orthogonal complement of Z.
        '''
        M, aux = self.M, self.aux

        M.mult(y, aux)  
        x = self.create_vec(dim=0)
        x.zero()
        x.axpy(1, y)  # x = y
        # Record alpha to get the Z content
        for i, z in enumerate(self.basis):  # x -= (x, z)*z
            alpha = aux.inner(z)  # aux*x  is L2
            x.axpy(-alpha, z)
            self.alphas[i] = alpha
        return x

    @vec_pool
    def create_vec(self, dim):
        return Function(self.V).vector()


def test_Zh(n=4):
    '''L2 orthogonality'''
    mesh = UnitCubeMesh(n, n, n)

    V = VectorFunctionSpace(mesh, 'CG', 1)
    Q = VectorFunctionSpace(mesh, 'R', 0, 6)
    Zh = RMBasis(V, Q)

    for i, zi in enumerate(Zh):
        for j, zj in enumerate(Zh):
            if i == j:
                e = assemble(inner(zi, zj)*dx)
                assert near(e, 1.0, 1E-12), e
            else:
                e = assemble(inner(zi, zj)*dx)
                abs(e) < 1E-12, e

    return True


def test_P(n=4):
    '''Projections'''
    from block import block_transpose
    mesh = UnitCubeMesh(n, n, n)

    V = VectorFunctionSpace(mesh, 'CG', 1)
    Q = VectorFunctionSpace(mesh, 'R', 0, 6)
    Zh = RMBasis(V, Q)

    P = Projector(Zh)
    # Is a projector
    f = Function(V)
    x = f.vector()
    as_backend_type(x).vec().setRandom()

    Px = P*x
    assert any(abs(alpha) > 0 for alpha in P.alphas)
    PPx = P*Px
    tol = 1E-12
    assert all(abs(alpha) < tol for alpha in P.alphas)
    assert abs((Px - PPx).norm('l2')) < tol

    y = assemble(inner(f, TestFunction(V))*dx)
    Py = block_transpose(P)*y
    assert any(abs(beta) > 0 for beta in P.betas)
    PPy = block_transpose(P)*Py
    assert all(abs(beta) < tol for beta in P.betas)
    assert abs((Py - PPy).norm('l2')) < tol

    return True

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    assert all(test() for test in (test_I, test_Zh, test_P))
"""

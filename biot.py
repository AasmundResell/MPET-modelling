from dolfin import *
from rm_basis_L2 import rigid_motions

def elastic_stress(u, E, nu):
    "Define the standard linear elastic constitutive equation."
    d = u.geometric_dimension()
    I = Identity(d)
    mu = E/(2.0*((1.0 + nu)))
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    s = 2*mu*sym(grad(u)) + lmbda*div(u)*I
    return s

def problem(meshfile):
    print("reading mesh \n")
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), meshfile, "r")
    hdf.read(mesh, "/mesh", False)
    d = mesh.geometry().dim()  
    markers = MeshFunction("size_t", mesh, d-1)
    hdf.read(markers,"/boundaries")
    u_nullspace = True
    return mesh, markers, u_nullspace

def solve(mesh, u_nullspace,markers,T):

    t = 0.0;
    nu = 0.495; dt = 0.00675;
    E_WHITE = 1895.0
    E_GRAY = 1389.0
    E = 1642.0;
    c = 3.9e-4;
    K = 1.4e-14/8.9e-4*1.e6
    dx = Measure("dx", domain=mesh)
    dsm = Measure("ds", domain=mesh, subdomain_data=markers)
    d = mesh.geometry().dim()  
    if d == 2:
        PIAL_MARKER = 2
        VENTRICLES_MARKER = 1         
        prefix = "results/donut_"+"nu"+str(nu)
    elif d == 3:
        if False:
            PIAL_MARKER = 4
            VENTRICLES_MARKER = 8         
            prefix = "solution/brain_"+"nu"+str(nu)
        if True:
            # use this for Vegard mesh
            PIAL_MARKER = 1
            VENTRICLES_MARKER = 2
            BRAIN_STEM_MARKER = 3         
            prefix = "solution/brain_vegard_"+"nu"+str(nu)+"dt"+str(dt)

    time = Constant(t)
    dim = mesh.geometry().dim()
    f = Constant((0,)*dim)
    p1 = Expression("4.9 *sin(2*pi*t) + 13.7*sin(0.5*pi*t)", degree=1, t=time)
    n = FacetNormal(mesh)
    
    sigma = lambda u: elastic_stress(u, E, nu)

    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)    

    if u_nullspace:
        Z = rigid_motions(mesh)
        dimZ = len(Z)
        RU = VectorElement('R', mesh.ufl_cell(), 0, dimZ)
        M = MixedElement([V, Q, RU])
        VZ = FunctionSpace(mesh, M)
        U = Function(VZ)
        u, p, z = TrialFunctions(VZ)
        v, q, r = TestFunctions(VZ)
    else:
        M = MixedElement([V, Q])
        VZ = FunctionSpace(mesh, M)
        U = Function(VZ)
        u, p = TrialFunctions(VZ)
        v, q = TestFunctions(VZ)

    up_ = Function(VZ)
    u_ = split(up_)[0]
    p_ = split(up_)[1]

    F = inner(sigma(u), sym(grad(v)))*dx() - p*div(v)*dx() \
      + c/dt*p*q*dx() + 1/dt*div(u)*q*dx() + K*inner(grad(p),grad(q))*dx()

    if u_nullspace: 
        F += sum(z[i]*inner(v, Z[i])*dx() for i in range(dimZ)) \
             + sum(r[i]*inner(u, Z[i])*dx() for i in range(dimZ))
        bcs = []
    else: #Applies fixed displacements on SAS/Pial
        bcs = [DirichletBC(VZ.sub(0), Constant((0,)*d), markers, PIAL_MARKER)]

    L0 = inner(-p1*n, v)*dsm(PIAL_MARKER) +inner(-p1*n, v)*dsm(BRAIN_STEM_MARKER) + c/dt*p_*q*dx() + 1/dt*div(u_)*q*dx()

    print("assemble matrix\n")
    A = assemble(F)
    for bc in bcs:
        print("apply bc \n")
        bc.apply(A)

    solver = LUSolver(A, "mumps")

    file_u = File(prefix+"/u.pvd")
    file_u.write(U.sub(0), t)
    fileu_hdf5 = HDF5File(mesh.mpi_comm(), prefix + "/u.h5", "w")
    fileu_hdf5.write(U.sub(0),"/u", 0.0)

    file_p = File(prefix+"/p.pvd")
    file_p.write(U.sub(1), t)
    filep_hdf5 = HDF5File(mesh.mpi_comm(), prefix + "/p.h5", "w")
    filep_hdf5.write(U.sub(1),"/p", 0.0)

    file_vol_pial = open(prefix+"/volume_change_pial.txt", "w")
    file_vol_pial.write("t,vol_change\n")

    file_vol_ventricles = open(prefix+"/volume_change_ventricles.txt", "w")
    file_vol_ventricles.write("t,vol_change\n")

    print("enter time loop\n")
    while(float(time)<T - 1e-9):
        t+=dt
        print("t = ", t)
        time.assign(t)
        b = assemble(L0)
        solver.solve(A,U.vector(),b)
        up_.assign(U)

        vol_change_pial = assemble(dot(U.sub(0),n)*(dsm(PIAL_MARKER)+dsm(BRAIN_STEM_MARKER)))
        print("vol_change_pial = ", vol_change_pial)
        vol_change_ventricles = assemble(dot(U.sub(0),n)*dsm(VENTRICLES_MARKER))
        print("vol_change_ventricles = ", vol_change_ventricles)

        file_u.write(U.sub(0), t)
        fileu_hdf5.write(U.sub(0),"/u", t)
        
        file_p.write(U.sub(1), t)
        filep_hdf5.write(U.sub(1),"/p", t)

        file_vol_ventricles.write(str(t)+","+str(vol_change_ventricles)+" \n")
        file_vol_pial.write(str(t)+","+str(vol_change_pial)+" \n")

    file_vol_pial.close()
    file_vol_ventricles.close()
    fileu_hdf5.close()
    filep_hdf5.close()

 

if __name__ == '__main__':

    #meshfile = "donut2D.h5"  
    meshfile = "/home/asmund/dev/MPET-modelling/meshes/parenchyma16_with_DTI.h5" 

    mesh, markers, u_nullspace = problem(meshfile)
    solve(mesh, u_nullspace, markers, 4)

from fenics import *
import rigid_motions
import ufl
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle
import os.path
import pylab
import matplotlib
import pandas
import glob
from pathlib import Path
from block import block_mat, block_vec, block_transpose, block_bc,block_assemble
#from block.iterative import *
from block.algebraic.petsc import AMG

import warnings
from block.dolfin_util import *

import petsc4py, sys
petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
print = PETSc.Sys.Print


matplotlib.rcParams["lines.linewidth"] = 3
matplotlib.rcParams["axes.linewidth"] = 3
matplotlib.rcParams["axes.labelsize"] = "xx-large"
matplotlib.rcParams["grid.linewidth"] = 1
matplotlib.rcParams["xtick.labelsize"] = "xx-large"
matplotlib.rcParams["ytick.labelsize"] = "xx-large"
matplotlib.rcParams["legend.fontsize"] = "xx-large"
matplotlib.rcParams["font.size"] = 14

class MPET:
    def __init__(self,
                 mesh,
                 boundary_markers,
                 boundary_conditionsU,
                 boundary_conditionsP,
                 **kwargs,
        ):
        self.mesh = mesh        
        self.boundary_markers = boundary_markers
        self.boundary_conditionsU = boundary_conditionsU
        self.boundary_conditionsP = boundary_conditionsP
        self.filesave = kwargs.get("file_save")
        self.uNullspace = kwargs.get("uNullspace")
        
        #Number of boundaries
        if kwargs.get("num_boundaries"):
            self.boundaryNum=int(kwargs.get("num_boundaries"))
        else:
            self.boundaryNum=3

        print("Number of boundaries:",self.boundaryNum)
        self.numPnetworks = kwargs.get("num_networks") 
        


        self.plot_from = kwargs.get("plot_from")
        self.plot_to = kwargs.get("plot_to")
    
        self.T = kwargs.get("T")
        self.numTsteps = kwargs.get("num_T_steps")
        self.t = np.linspace(0,float(self.T),int(self.numTsteps)+1)
        
        self.element_type = kwargs.get("element_type")
        self.solverType = kwargs.get("solver")
        self.preconditioner = kwargs.get("preconditioner")
        

        self.f_val = kwargs.get("f")
        self.rho = kwargs.get("rho")
        self.nu = kwargs.get("nu")
        self.E =  kwargs.get("E")
        self.mu_f = kwargs.get("mu_f")
        self.kappa = kwargs.get("kappa")
        self.alpha_val = kwargs.get("alpha")
        self.c_val = kwargs.get("c")
        self.p_initial = kwargs.get("p_initial")
        p_initial0 =  sum([-x*y for x,y in zip(self.alpha_val,self.p_initial)])
        self.p_initial.insert(0,p_initial0)
        self.gamma = np.reshape(kwargs.get("gamma"),(self.numPnetworks,self.numPnetworks))
        self.K_val = []

        for i in range(self.numPnetworks):
            self.gamma[i,i] = sum(self.gamma[i,:])

        #Ensure lists
        if not isinstance(self.alpha_val,list): self.alpha_val = [self.alpha_val]
        if not isinstance(self.c_val,list): self.c_val = [self.c_val]
        if not isinstance(self.p_initial,list): self.p_initial = [self.p_initial]
        if not isinstance(self.kappa,list): self.kappa= [self.kappa] 
        if not isinstance(self.mu_f,list): self.mu_f = [self.mu_f]

        for i in range(self.numPnetworks):
            self.kappa[i] = self.kappa[i]*1e6 #m² to mm²
            self.K_val.append(self.kappa[i]/self.mu_f[i])
            
        
        self.sourceFile = kwargs.get("source_file")
        self.scaleMean = kwargs.get("scale_mean")
        self.g = [self.GenerateNumpySeries(),None,None]
        self.Lambda = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))        
        self.conversionP = 133.32 #Pressure conversion: mmHg to Pa
        self.dim = self.mesh.topology().dim()

        
        #Boundary parameters
        if kwargs.get("Compliance_sas"):
            self.C_SAS =  kwargs.get("Compliance_sas")
        if kwargs.get("Compliance_ven"):
            self.C_VEN =  kwargs.get("Compliance_ven") 
        if kwargs.get("Compliance_spine"):
            self.C_SP =  kwargs.get("Compliance_spine") 

        if kwargs.get("ScalePressure"): #For scaling CSF pressure on boundaries
            self.Pscale =  kwargs.get("ScalePressure")
        else:
            self.Pscale = 1.0

        #For alternative model for ventricles
        self.L = kwargs.get("length")
        self.d = kwargs.get("diameter")

        self.beta_VEN = kwargs.get("beta_ven")
        self.beta_SAS = kwargs.get("beta_sas")

        self.p_BC_initial = [kwargs.get("p_ven_initial"),kwargs.get("p_sas_initial")]
        self.p_BC_initial.append(kwargs.get("p_spine_initial"))

        print("\nSetting up problem...\n")

        print("Generating UFL expressions\n")
        self.generateUFLexpressions()
        
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        self.n = FacetNormal(self.mesh)  # normal vector on the boundary
        self.dt = self.T / self.numTsteps

        
    def AMG_testing(self):
        """ Used for testing the settings of the AMG preconditions on a simplified
        linear elasticity problem
        Solves 
        -div(sym grad u) = f in  Omega
            sigma.n = h on  boundary
        where sigma(u) = 2*mu*eps(u) + lambda*div(u)*I. The problem is reformulated by
        Lagrange multiplier nu to inforce orthogonality with the space of rigid
        motions. To get robustnes in lmbda solid pressure p = lambda*div u is introduced. 
        The system to be solved with MinRes is 
        P*[A C  B; *[u, = P*[L,
           C' D 0;   p,      0,
           B' 0 0]   nu]     0]
        with P a precondtioner. We run on series of meshes to show mesh independence
        of the solver.
        """

        progress = Progress("Time-stepping", self.numTsteps)
        N = 20
        self.mesh = BoxMesh(Point(1, 1, 1), Point(2, 1.5, 1.25), N, N, N)

        t = 0.0
        #self.f = Expression(('(1-cos(2*pi*t))*A*sin(2*x[0]*100/5)', '(1-cos(2*pi*t)*A*cos(3*100*(x[0]+x[1]+x[2])/5)', '(1-cos(2*pi*t)*A*sin(100*x[2]/5)'), degree=3, A=0.01,t = t)
        
        self.f = Expression(('A*sin(2*x[0]/5)', 'A*cos(3*(x[0]+x[1]+x[2])/5)', 'A*sin(x[2]/5)'), degree=3, A=0.01)
    
        h = Constant((0, 0, 0))
        self.mu = 1.0
        self.Lambda = 1E3
        self.alpha[1] = 1
        self.c[0] = Constant(1e-2)
        self.K[0] = Constant(0.02)

        generateExact = True #Generating exact solution
        #self.CreateForceTerm(generateExact) #Generates f

        V = VectorFunctionSpace(self.mesh, 'CG', 2)
        Q0 = FunctionSpace(self.mesh, 'CG', 1)
        Q1 = FunctionSpace(self.mesh, 'CG', 1)
        
        u, v = TrialFunction(V), TestFunction(V)
        p0, q0 = TrialFunction(Q0), TestFunction(Q0)
        p1, q1 = TrialFunction(Q1), TestFunction(Q1)
        
        a = 2*self.mu*inner(sym(grad(u)), sym(grad(v)))*dx
        
        A = assemble(a)
        
        c = inner(div(v), p0)*dx
        C = assemble(c)
        
        d = -(inner(p0, q0)/Constant(self.Lambda))*dx
        D = assemble(d)

        e = -self.alpha[1]*(inner(p1, q0)/Constant(self.Lambda))*dx
        E = assemble(e)

        f1 = -(self.c[0]+self.alpha[1]**2/Constant(self.Lambda))*inner(p1,q1)*dx
        f2 = self.K[0] * dot(grad(p1), grad(q1)) * dx
        
        F = assemble(f1+f2)

        
        m = inner(u, v)*dx

        AA = block_assemble([[A,                  C,                  0], 
                             [block_transpose(C), D,                  E],
                             [0,                  block_transpose(E), F],
                             ])
        
        # Right hand side
        L = inner(self.f, v)*dx + inner(h, v)*ds
        

        b0 = assemble(L)
        b1 = assemble(inner(Constant(0), q0)*dx)
        b2 = assemble(inner(Constant(0), q1)*dx)

        boundary = BoxBoundary(self.mesh)

        rhs_bc = block_bc([[DirichletBC(V,Constant((0,0,0)),boundary.bottom)],None,None],False)
        #rhs_bc = block_bc([[DirichletBC(V,Constant((0,0,0)),self.boundary_markers,1)],None,None],False)
        rhs_bc.apply(AA)

        # Block diagonal preconditioner
        dummy_rhs = inner(Constant((0, )*len(v)), v)*dx
        IV, _ = assemble_system(a+m, dummy_rhs, bcs=rhs_bc[0])
        IQ = assemble(inner(p0, q0)*dx)
        IM = assemble(f1+f2)
        
        BB = block_mat([[AMG(IV), 0,            0],
                        [0,       AMG(IQ),      0],
                        [0,       0,            AMG(IM)],
                        ])
        
        # Solve, using random initial guess
        x0 = AA.create_vec()
        [as_backend_type(xi).vec().setRandom() for xi in x0]
        
        AAinv = MinRes2(AA,
                       precond=BB,
                       initial_guess=x0,
                      # iter = 2000,
                       maxiter=200,
                       tolerance=1E-8,
                       show=3,
                       relativeconv=True)

        xdmfU = XDMFFile(self.filesave + "/FEM_results/U.xdmf")
        xdmfU.parameters["flush_output"]=True
        xdmfP0 = XDMFFile(self.filesave + "/FEM_results/P0.xdmf")
        xdmfP0.parameters["flush_output"]=True
        xdmfP1 = XDMFFile(self.filesave + "/FEM_results/P1.xdmf")
        xdmfP1.parameters["flush_output"]=True

        x = None

        
        u = Function(V)
        p0 = Function(Q0)
        p1 = Function(Q1)
        
        
        #while t < self.T:
        bb = block_vec([b0, b1, b2])
        rhs_bc.apply(AA).apply(bb)

        x = AAinv*bb

        U,P0,P1 = x

        u.vector()[:] = U[:]
        p0.vector()[:] = P0[:]
        p1.vector()[:] = P1[:]

        xdmfU.write(u, t)
        xdmfP0.write(p0, t)
        xdmfP1.write(p1, t)

        t += self.dt
        #self.f.t = t
        
        progress +=1

    def CreateForceTerm(self,generateExact = False):

        import sympy as sym

        x, y, z = sym.symbols("x[0], x[1], x[2]")
        mu = self.mu
        Lambda = self.Lambda

        t = sym.symbols("t")
        u =  (
        sym.sin(2 * sym.pi * y /100) * (-1 + sym.cos(2 * sym.pi * x/100))
        + 1 / (mu + Lambda) * sym.sin(sym.pi * x/100) * sym.sin(sym.pi * y/100)
        )
        v = (
            sym.sin(2 * sym.pi * x) * (1 - sym.cos(2 * sym.pi * y))
            + 1 / (mu + Lambda) * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
        )
        w = 0
        p = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1) + sym.diff(w, z, 1))

        epsilonxx = sym.diff(u, x, 1)
        epsilonyy = sym.diff(v, y, 1)
        epsilonzz = sym.diff(w, z, 1)

        epsilonxy = 1 / 2 * (sym.diff(u, y, 1) + sym.diff(v, x, 1))
        epsilonyz = 1 / 2 * (sym.diff(v, z, 1) + sym.diff(w, y, 1))
        epsilonzx = 1 / 2 * (sym.diff(w, x, 1) + sym.diff(u, z, 1))

        fx = 2 * mu * (sym.diff(epsilonxx, x, 1) + sym.diff(epsilonxy, y, 1)+ sym.diff(epsilonzx, z, 1)) - sym.diff(
            p, x, 1
        )  # calculate force term x
        fy = 2 * mu * (sym.diff(epsilonxy, x, 1) + sym.diff(epsilonyy, y, 1) + sym.diff(epsilonyz, z, 1)) - sym.diff(
            p, y, 1
        )  # calculate force term y
        fz = 2 * mu * (sym.diff(epsilonzx, x, 1) + sym.diff(epsilonyz, y, 1) + sym.diff(epsilonzz, z, 1)) - sym.diff(
            p, z, 1
        )  # calculate force term z
        
        variables = [
            u,
            v,
            w,
            p,
            mu,
            Lambda,
            fx,
            fy,
            fz,
        ]
        
        variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code
        
        UFLvariables = [Expression(var, degree=2, t=0) for var in variables]

        self.f = Expression((variables[-3],variables[-2],variables[-1]), degree=3, t=0)
        
        (
            u,
            v,
            w,
            p,
            mu,
            Lambda,
            fx,
            fy,
            fz,
        ) = UFLvariables

        
        U = as_vector((u, v, w))
        #p_initial = [p0,p1,p2]
        if generateExact:
            xdmfUE = XDMFFile(self.filesave + "/FEM_results/UE.xdmf")
            xdmfUE.parameters["flush_output"]=True
            xdmfPE = XDMFFile(self.filesave + "/FEM_results/PE.xdmf")
            xdmfPE.parameters["flush_output"]=True
            t = 0.0
            V_e = VectorFunctionSpace(self.mesh, "CG", 2)
            Q_e = FunctionSpace(self.mesh, "CG", 1)
            
            u_e = Expression((variables[0], variables[1],variables[2]), degree=2, t=t)
            p_e = Expression(variables[3], degree=2, t=t)
            
            u_es = Function(V_e)
            p_es = Function(Q_e)
            
            #while t < self.T:
            print("t: ",t)
            u_es.vector()[:] = project(u_e, V_e).vector()[:]
            p_es.vector()[:] = project(p_e, Q_e).vector()[:]
            
            xdmfUE.write(u_es, t)
            xdmfPE.write(p_es, t)
            t += self.dt
            u_e.t = t
            p_e.t = t

 
        
    def blockSolve(self):
            
        """
        Solves the MPET problem as a block system
        """

        N = 20
        self.mesh = BoxMesh(Point(1, 1, 1), Point(2, 1.5, 1.25), N, N, N)

        
        # Add progress bar
        progress = Progress("Time-stepping", self.numTsteps)
        set_log_level(LogLevel.PROGRESS)
        
        xdmfU = XDMFFile(self.filesave + "/FEM_results/u.xdmf")
        xdmfU.parameters["flush_output"]=True

        xdmfP = []
        
        for i in range(self.numPnetworks+1):
            xdmfP.append(XDMFFile(self.filesave + "/FEM_results/p" + str(i) + ".xdmf"))        
            xdmfP[i].parameters["flush_output"]=True

        """
        #f for float value, not dolfin expression
        p_VEN_f = self.p_BC_initial[0]
        p_SAS_f = self.p_BC_initial[1]
        p_SP_f = self.p_BC_initial[2]
        

        print("P_VEN =,", p_VEN_f)
        print("P_SAS =,", p_SAS_f)

        if self.dim == 3:
            dimZ = 6
        elif self.dim == 2:
            dimZ = 3

        # Class representing the intial conditions for pressures
        class InitialConditions(UserExpression):
            def __init__(self,p_initial):
                super().__init__(degree=1)
                self.P_init = p_initial

            def eval(self, values, x):
                values = self.P_init

            def value_shape(self):
                return (1,)

        p0_init = InitialConditions(self.p_initial[0])
        p1_init = InitialConditions(self.p_initial[1])
        p2_init = InitialConditions(self.p_initial[2])
        p3_init = InitialConditions(self.p_initial[3])
        
        # variational formulation
        sources = []  # Contains the source term for each network
        transfer = [] # Contains the transfer terms for each network
        innerProdP = []
        # Contains the inner product of the gradient of p_j for each network
        dotProdP = []  # Contains the dot product of alpha_j & p_j,
        timeD_ = []  # Time derivative for the current step
        timeD_n = []  # Time derivative for the previous step

        self.bcs_D = []  # Contains the terms for the Dirichlet boundaries
        self.integrals_N = []  # Contains the integrals for the Neumann boundaries
        self.time_expr = []  # Terms that needs to be updated at each timestep
        # Terms that contains the windkessel bc that is updated each timestep
        self.windkessel_terms = []

        # Contains the integrals for the Robin boundaries, LHS
        self.integrals_R_L = []
        # Contains the integrals for the Robin boundaries, RHS
        self.integrals_R_R = []
        """
        
        h = Constant((0, 0, 0))
        self.mu = 1.0
        self.Lambda = 1E3
        self.alpha[1] = 1
        self.c[0] = Constant(1e-2)
        self.K[0] = Constant(0.02)

        generateExact = True #Generating exact solution
        #self.CreateForceTerm(generateExact) #Generates f

        # For cube
        V = VectorFunctionSpace(self.mesh, 'CG', 2)
        Q = FunctionSpace(self.mesh, 'CG', 1)


        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)


        a = 2*self.mu*inner(sym(grad(u)), sym(grad(v)))*dx
        
        A = assemble(a)
        
        c = inner(div(v), p)*dx
        C = assemble(c)
        
        d = -(inner(p, q)/Constant(self.Lambda))*dx
        D = assemble(d)

        e = -self.alpha[1]*(inner(p, q)/Constant(self.Lambda))*dx
        E = assemble(e)

        f1 = -(self.c[0]+self.alpha[1]**2/Constant(self.Lambda))*inner(p,q)*dx
        f2 = self.K[0] * dot(grad(p), grad(q)) * dx
        
        F = assemble(f1+f2)

        m = inner(u, v)*dx
        M = assemble(m)

        dimZ = 6


        X = VectorFunctionSpace(self.mesh, 'R', 0, dim=6)
        Z = None

        Zh = rigid_motions.rm_basis(self.mesh)

        
        z,r = TrialFunction(X),TestFunction(X)

        B = sum(z[i]*inner(v, Zh[i])*dx() for i in range(dimZ)) 
        BT = sum(r[i]*inner(u, Zh[i])*dx() for i in range(dimZ))

        # System operator
        AA = block_assemble([[A,                  C,                  0, B], 
                             [block_transpose(C), D,                  E, 0],
                             [0,                  block_transpose(E), F, 0],
                             [BT,                 0,                  0, 0],
                             ])
        
        # Right hand side
        rhs_bc = block_bc([None,None,[DirichletBC(Q,Constant(0),boundary.bottom)]],False)
        rhs_bc.apply(AA)

        print("Len v:",len(v))

        # Block diagonal preconditioner
        IV = assemble(a+m)
        IQ = assemble(inner(p, q)*dx)
        dummy_rhs = inner(Constant((0, )*len(q)), q)*dx
        IM, _ = assemble_system(f1+f2, dummy_rhs, bcs=rhs_bc[2])
        IX = rigid_motions.identity_matrix(X)

        BB = block_mat([[AMG(IV), 0,           0,         0],
                        [0,       AMG(IQ),     0,         0],
                        [0,       0,           AMG(IM),   0],
                        [0,       0,           0,         IX],
                        ])


        self.f = Expression(('A*sin(2*x[0]/5)', 'A*cos(3*(x[0]+x[1]+x[2])/5)', 'A*sin(x[2]/5)'), degree=3, A=0.01)
        h = Constant((0, 0, 0))
        
        L = inner(self.f, v)*dx + inner(h, v)*ds
        
        b0 = assemble(L)
        b1 = assemble(inner(Constant(0), q)*dx)
        # Equivalent to assemble(inner(Constant((0, )*6), q)*dx) but cheaper
        b2 = assemble(inner(Constant(0), q)*dx)
        b3 = Function(X).vector()
        # Solve, using random initial guess
        x0 = AA.create_vec()
        [as_backend_type(xi).vec().setRandom() for xi in x0]
        
        AAinv = MinRes2(AA,
                        precond=BB,
                        initial_guess=x0,
                        #iter = 2000,
                        maxiter=200,
                        tolerance=1E-8,
                        show=3,
                        relativeconv=True)

        x = None

        t=0.0

        u_v = Function(V)
        p0_v = Function(Q)
        p1_v = Function(Q)
        
        
        #while t < self.T:
        bb = block_vec([b0, b1, b2, b3])

        r1 = bb - AA*x0
        y = BB*r1
        beta1 = block_vec.inner(r1,y)
        
        
        #  Test for an indefinite preconditioner.
        #  If b = 0 exactly, stop with x = 0.
        if beta1 < 0:
            raise ValueError('Preconditioner is negative-definite')
        if beta1 == 0:
            warnings.warn("Preconditioner vec-mat-vec product is zero")
        else:
            print('Preconditioner is positive-definite')


        x = AAinv*bb

        U,P0,P1,nu = x

        u_v.vector()[:] = U[:]
        p0_v.vector()[:] = P0[:]
        p1_v.vector()[:] = P1[:]

        xdmfU.write(u_v, t)
        xdmfP[0].write(p0_v, t)
        xdmfP[1].write(p1_v, t)

        t += self.dt
        #self.f.t = t
        """
        #Apply initial conditions
        #p0_prev.vector()[:] = self.p_initial[0]
        #p1_prev.vector()[:] = self.p_initial[1]
        #p2_prev.vector()[:] = self.p_initial[2]
        #p3_prev.vector()[:] = self.p_initial[3]
        
        #self.applyPressureBC_BLOCK(Q,p,q)
        self.applyDisplacementBC(V,v)
        self.alpha[0] = Constant(1.0)
        self.alpha[1] = Constant(1.0)
        self.Lambda = Constant(1e5)
        self.mu    = Constant(1e5)

        def a_u(u,v):
            return self.mu * (inner(grad(u), grad(v)) + inner(grad(u), nabla_grad(v))) * dx

        def a_p(K,p,q):
            return self.dt*K * dot(grad(p), grad(q)) * dx
        
        def b_u(p,v):
            return inner(div(v),p)* dx
    
        def b_0(alpha,p,q):
            return alpha / self.Lambda * inner(p, q) * dx

        def d(p,q,eq,numP): #Time derivatives
            d_p = 0
            if (eq==numP):
                print("Adding storage term for equation {}".format(eq))
                d_p  = self.c[eq-1] * p
            d_eps  = self.alpha[eq] / self.Lambda * self.alpha[numP]*p 
            return (d_p + d_eps) * q * dx

        def f(g, q):
            return inner(g, q) * dx



        # Terms for total pressure equation
        #b2 = b_0(self.alpha[2])
        #b3 = b_0(self.alpha[3])

        # Time derivatives, current step
        #d12 = d(p,q,1,2)
        #d13 = d(p,q,1,3)

        #d21 = d(p,q,2,1)
        #d22 = d(p,q,2,2)
        #d23 = d(p,q,2,3)

        #d31 = d(p,q,2,1)
        #d32 = d(p,q,2,2)
        #d33 = d(p,q,2,3)

        # Time derivatives, previous step

        #d12_prev = d(p2_prev,q,1,2)
        #d13_prev = d(p3_prev,q,1,3)

        #d20_prev = d(p0_prev,q,2,0)
        #d21_prev = d(p1_prev,q,2,1)
        #d22_prev = d(p2_prev,q,2,2)
        #d23_prev = d(p3_prev,q,2,3)

        #d30_prev = d(p0_prev,q,3,0)
        #d31_prev = d(p1_prev,q,3,1)
        #d32_prev = d(p2_prev,q,3,2)
        #d33_prev = d(p3_prev,q,3,3)


        #Transfer terms
        #s11 = self.dt*self.gamma[0,0]*dot(p,q)*dx
        #s12 = self.dt*self.gamma[0,1]*dot(p,q)*dx
        #s13 = self.dt*self.gamma[0,2]*dot(p,q)*dx
        #s21 = self.dt*self.gamma[1,0]*dot(p,q)*dx
        #s22 = self.dt*self.gamma[1,1]*dot(p,q)*dx
        #s23 = self.dt*self.gamma[1,2]*dot(p,q)*dx
        #s31 = self.dt*self.gamma[2,0]*dot(p,q)*dx
        #s32 = self.dt*self.gamma[2,1]*dot(p,q)*dx
        #33 = self.dt*self.gamma[2,2]*dot(p,q)*dx

        # Source term for p1
        g_space = FunctionSpace(self.mesh, "CG",1)
        g_1 = Function(g_space)
        self.time_expr.append((self.g[0], g_1))

        ##BLOCK_SYSTEM##

        ##BLOCK_MATRIX##

        ##MOMENTUM_AND_RIGID_MOTION##
        epsilon = lambda u: sym(grad(u))
        #sigma = lambda u: 2*self.mu*epsilon(u) + self.Lambda*tr(epsilon(u))*Identity(self.dim)

        au = 2*self.mu*inner(epsilon(u), epsilon(v))*dx
        bu = b_u(p0,v)
        buT = b_u(q0,u)

        a0 = b_0(self.alpha[0],p0,q0)
        b0 = b_0(self.alpha[1],p1,q0)
        b0T = b_0(self.alpha[1],p0,q1)

        d11 = (self.c[0]+self.alpha[1]**2 / self.Lambda)*p1*q1*dx
        c = d11 + self.dt*a_p(self.K[0],p1,q1)

        d10_prev = self.alpha[1]*self.alpha[0] / self.Lambda*p0_prev*q1*dx
        d11_prev = (self.c[0]+self.alpha[1]**2 / self.Lambda)*p1_prev*q1*dx

        Au = assemble(au)
        Bu = assemble(bu)
        BuT = assemble(buT)


        ##TOTAL_PRESSURE##
        A0 = assemble(-a0)
        B0 = assemble(-b0)

        ##FLUID_PRESSURES##
        B0T =assemble(-b0T)
        C = assemble(-c)
        

        pv = Expression("A*(1-cos(2*pi*t))",degree=3,t=t,A=200.0)
        ppv = Expression("A*(1-cos(2*pi*t))",degree=3,t=t,A=2000.0)
        ud = Expression(("A*(1-cos(2*pi*t))","0.0","0.0"),degree=3,t=t,A=2000.0)


        #rhs_bc = block_bc([self.bcs_D, None,None], False)

        rhs_bc = block_bc([DirichletBC(V, Constant((0,0,0)),self.boundary_markers,1), None,DirichletBC(Q1, Constant((0.0)),self.boundary_markers,1)], False)

        rhs_bc.apply(AA)

        [[Au, Bu, Na],
         [BuT,A0, B0],
         [Nc,B0T, C],
         ] = AA

        IV = AMG(Au)
        I0 = AMG(assemble(inner(p0, q0)*dx))

        IP = AMG(C)

        #IX = rigid_motions.identity_matrix(X)

        BB = block_mat([[IV, 0,  0],
                        [0,  I0, 0],
                        [0,  0,  IP],
                        ]) 


        x0 =  AA.create_vec() #Initial guess       


        # Solve, using random initial guess
        [as_backend_type(xi).vec().setRandom() for xi in x0]
        #BBu = assemble(inner(-self.n*pv, v)*self.ds(2)) #Pressure on surface is negative (compression)
     
        BBu = assemble(inner(Constant((0,0,0)), v)*dx)
        BB0 = assemble(inner(Constant(0), q0)*dx)
        BBC = assemble(- d10_prev - d11_prev - self.dt*f(g_1, q1))# - self.dt* dot(ppv, q1) * ds)
        
        bb = block_vec([BBu,BB0,BBC])
        rhs_bc.apply(AA).apply(bb)

        r1 = bb - AA*x0
        y = BB*r1
        beta1 = block_vec.inner(r1,y)
        
        
        #  Test for an indefinite preconditioner.
        #  If b = 0 exactly, stop with x = 0.
        if beta1 < 0:
            raise ValueError('Preconditioner is negative-definite')
        if beta1 == 0:
            warnings.warn("Preconditioner vec-mat-vec product is zero")
        else:
            print('Preconditioner is positive-definite')

        #Define solver type
        AAinv = MinRes(AA,
                       precond=BB,
                       initial_guess=x0,
                       maxiter=200,
                       tolerance=1E-8,
                       show=2,
                       relativeconv=True,
                       )



        tvec = np.arange(0,self.T,self.dt)
        DV_Vec = np.zeros(len(tvec))
        g_1.vector()[:] = self.g[0][0]
        self.m = assemble(g_1*dx(self.mesh))

        
        u = Function(V)
        p0 = Function(Q0)
        p1 = Function(Q1)
        
        x = None
        i=0

        #BBu = assemble(inner(-self.n*pv, v)*self.ds(2)) #Pressure on surface is negative (compression)
        while t < self.T:

            #BBu = assemble(inner(Constant((0,0,0)), v)*dx)
            #BB0 = assemble(inner(Constant(0), q0)*dx)
            BBC = assemble(-d10_prev - d11_prev - self.dt*inner(g_1, q1)*dx)

            bb = block_vec([BBu,BB0,BBC])
            rhs_bc.apply(AA).apply(bb)

            start = timeit.timeit()
            print("Timeit started")
            x = AAinv * bb
            end = timeit.timeit()
            print("Elaspsed time: ",end - start)

            U,P0,P1 = x
            u.vector()[:] = U[:]
            p0.vector()[:] = P0[:]
            p1.vector()[:] = P1[:]
            results = self.generate_diagnostics(u,p0,p1)

            #Write solution at time t
            xdmfU.write(u, t)
            xdmfP0.write(p0, t)
            xdmfP[0].write(p1, t)
            t +=float(self.dt)
            pv.t = t
            ppv.t = t
            ud.t = t

            g_1.vector()[:] = self.g[0][i+1]
            
            u_prev.vector()[:] = U
            p0_prev.vector()[:] = P0
            p1_prev.vector()[:] = P1

            self.m = assemble(g_1*dx(self.mesh))
            print("Arterial inflow:",self.m)

            progress += 1
            
            results["t"] = t

            pickle.dump(results, open("%s/data_set/qois_%d.pickle" % (self.filesave, i), "wb"))
            i +=1
            

        """


        
        """
        ##MATRIX_ASSEMBLY##
        AA = block_mat([[Au,                  Bu, 0,  0,  0, L],
                        [block_transpose(Bu),A0, B1, B2, B3, 0],
                        [0,                  C0, A1, C2, C3, 0],
                        [0,                  D0, D1, A2, D3, 0],
                        [0,                  E0, E1, E2, A3, 0],
                        [block_transpose(L), 0,  0,  0,  0,  0],
                        ])

        # Block diagonal preconditioner
        IV = assemble(a + m)
        IQ = assemble(inner(p, q)*dx)
        IX = rigid_motions.identity_matrix(X)
        P1 = assemble(d11 + a_p(self.K[0]) + s11)
        P2 = assemble(d22 + a_p(self.K[1]) + s22)
        P3 = assemble(d33 + a_p(self.K[2]) + s33)

        BB = block_mat([[AMG(IV), 0,       0,       0,       0,       0],
                        [0,       AMG(IQ), 0,       0,       0,       0],
                        [0,       0,       P1,      0,       0,       0],
                        [0,       0,       0,       P2,      0,       0],
                        [0,       0,       0,       0,       P3,      0],
                        [0,       0,       0,       0,       0,       IX],
                        ])
        x0 =  AA.create_vec() #Initial guess
        [as_backend_type(xi).vec().setRandom() for xi in x0]

        AAinv = MinRes(AA, precond=BB, initial_guess=x0, maxiter=120, tolerance=1E-8,
                   show=2, relativeconv=True)
        

        ##BLOCK_VECTOR##

        ##MOMENTUM##
        #bu = assemble(sum(self.integrals_N)) #Only applies for this specific set of BC
        bu = assemble(inner(self.n*Constant(0), v)*ds)
        
        ##TOTAL_PRESSURE##
        b0 = assemble(inner(Constant(0), q)*dx)

        ##FLUID_PRESSURE_1##
        b1 = assemble(d10_prev+ d11_prev+ d12_prev+ d13_prev + self.dt*f(g_1, q))

        ##FLUID_PRESSURE_2##
        b2 = assemble(d20_prev+ d21_prev+ d22_prev+ d23_prev)


        ##FLUID_PRESSURE_3##
        beta1,P_r1 =  self.boundary_conditionsP[(3,1)]["RobinWK"]
        n31 = self.dt*inner(P_r1, q) * self.ds(1)
        beta2,P_r2 =  self.boundary_conditionsP[(3,2)]["RobinWK"]
        n32 = self.dt*inner(P_r2, q) * self.ds(2)
        b3 = assemble(d30_prev + d31_prev + d32_prev + d33_prev + n31 + n32)

        ##RIGID_MOTION##
        # Equivalent to assemble(inner(Constant((0, )*6), q)*dx) but cheaper
        bz = Function(X).vector()

        #bcs = block_bc([None, None,None,[self.bcs_D[0]],None], False)
        DBC = DirichletBC(Q,Constant(0)*self.Pscale,self.boundary_markers,1,)
        bcs = block_bc([None, None,None,[DBC],None], False)
        rhs_bc = bcs.apply(AA)

        """

        """
        ##TESTING
        bb = block_assemble([bu, b0, b1, b2, b3, bz])
        rhs_bc.apply(bb)
        
        x = None
        x = AAinv * bb

        U,P0,P1,P2,P3,lam = x

        u = Function(V, U)
        p0 = Function(Q, P0)
        p1 = Function(Q, P1)
        p2 = Function(Q, P2)
        p3 = Function(Q, P3)
        
        pj = [p1,p2,p3]
        
        results = self.generate_diagnostics(u,p0,p1,p2,p3)

        dV_PREV_SAS = 0.0
        dV_PREV_VEN = 0.0

        x = None

        for self.i,t in enumerate(self.t):    #range(0,self.numTsteps+1): #Time loop
            
            self.update_time_expr(t)# Update all time dependent terms
            print("t:",t)

            ##VECTOR_ASSEMBLY
            bb = block_assemble([bu, b0, b1, b2, b3, bz])
            rhs_bc.apply(bb)

            x = AAinv * bb

            U,P0,P1,P2,P3,lam = x
            u = Function(V, U)
            p0 = Function(Q, P0)
            p1 = Function(Q, P1)
            p2 = Function(Q, P2)
            p3 = Function(Q, P3)

            pj = [p1,p2,p3]

            results = self.generate_diagnostics(u,p0,p1,p2,p3)

            #Write solution at time t
            xdmfU.write(u, t)
            xdmfP0.write(p0, t)
            for j in range(self.numPnetworks):
                xdmfP[j].write(pj[j], t)


            #For calculating volume change in Windkessel model
            results["dV_SAS_PREV"] = dV_PREV_SAS
            results["dV_VEN_PREV"] = dV_PREV_VEN

            results["total_inflow"] = float(self.m)
            
            p_SAS_f, p_VEN_f,p_SP_f,Vv_dot,Vs_dot,Q_AQ,Q_FM = self.coupled_3P_model(p_SAS_f,p_VEN_f,p_SP_f,results) #calculates windkessel pressure @ t
            #p_SAS_f, p_VEN_f,p_SP_f,Vv_dot,Vs_dot,Q_AQ,Q_FM = self.coupled_3P_model_MK_constrained(p_SAS_f,p_VEN_f,p_SP_f,results) #calculates windkessel pressure @ t
            

            self.update_windkessel_expr(p_SAS_f,p_VEN_f) # Update all terms dependent on the windkessel pressures


            results["p_SAS"] = p_SAS_f
            results["p_VEN"] = p_VEN_f
            results["p_SP"] = p_SP_f

            results["Q_AQ"] = Q_AQ
            results["Q_FM"] = Q_FM
            results["Vv_dot"] = Vv_dot
            results["Vs_dot"] =  Vs_dot
            results["t"] = t

            dV_PREV_SAS = results["dV_SAS"]
            dV_PREV_VEN = results["dV_VEN"]

            pickle.dump(results, open("%s/data_set/qois_%d.pickle" % (self.filesave, i), "wb"))
            

            u_prev.vector()[:] = U
            p0_prev.vector()[:] = P0
            p1_prev.vector()[:] = P1
            p2_prev.vector()[:] = P2
            p3_prev.vector()[:] = P3

            progress += 1

         
        res = []
        res = split(up)
        u = project(res[0], W.sub(0).collapse())
        p = []
        
        self.u_sol = u
        self.p_sol = p

        """


    def solve(self):

        print("\nSetting up problem...\n")

        print("Generating UFL expressions\n")
        self.generateUFLexpressions()
        
        
        
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        self.n = FacetNormal(self.mesh)  # normal vector on the boundary

        self.dt = self.T / self.numTsteps

        # Add progress bar
        progress = Progress("Time-stepping", self.numTsteps)
        set_log_level(LogLevel.PROGRESS)
        
        xdmfU = XDMFFile(self.filesave + "/FEM_results/u.xdmf")
        xdmfU.parameters["flush_output"]=True
        


        xdmfP0 = XDMFFile(self.filesave + "/FEM_results/p0.xdmf")
        xdmfP0.parameters["flush_output"]=True

        xdmfP = []
        for i in range(self.numPnetworks):
            xdmfP.append(XDMFFile(self.filesave + "/FEM_results/p" + str(i+1) + ".xdmf"))        
            xdmfP[i].parameters["flush_output"]=True

        #f for float value, not dolfin expression
        p_VEN_f = self.p_BC_initial[0]
        p_SAS_f = self.p_BC_initial[1]
        p_SP_f = self.p_BC_initial[2]


        print("P_VEN =,", p_VEN_f)
        print("P_SAS =,", p_SAS_f)

                
        # Generate function space
        V = VectorElement(self.element_type, self.mesh.ufl_cell(), 2, self.dim)  # Displacements

        Q_0 = FiniteElement(self.element_type, self.mesh.ufl_cell(), 1)  # Total pressure
        mixedElement = []
        mixedElement.append(V)
        mixedElement.append(Q_0)
        for i in range(  self.numPnetworks):
            Q = FiniteElement(self.element_type, self.mesh.ufl_cell(), 1)
            mixedElement.append(Q)
        
        if self.uNullspace:
            
            Z = rigid_motions.rm_basis(self.mesh)
            dimZ = len(Z)
            print("LengthZ:", dimZ)
            RU = VectorElement('R', self.mesh.ufl_cell(), 0, dimZ)
            mixedElement.append(RU)
            W_element = MixedElement(mixedElement)
            W = FunctionSpace(self.mesh, W_element)

            
            test = TestFunction(W)
            q = split(test)[0:self.numPnetworks+2]  # q[0] = v, q[1],q[2],... = q_0,q_1,...
            r = split(test)[-1]

            trial = TrialFunction(W)
            p_ = split(trial)[0:self.numPnetworks+2]  # p_[0] = u_, p_[1],p_[2],... = p_0,p_1,...
            z = split(trial)[-1]
            up_n = Function(W)
        
            p_n = split(up_n)[0:self.numPnetworks+2] # p_n[0] = u_n, p_n[1],p_n[2],... = p0_n,p1_n,...

        else:
            dimZ = 0
            W_element = MixedElement(mixedElement)
            W = FunctionSpace(self.mesh, W_element)
            
            test = TestFunction(W)
            q = split(test)  # q[0] = v, q[1],q[2],... = q_0,q_1,...
            
            trial = TrialFunction(W)
            p_ = split(trial)  # p_[0] = u_, p_[1],p_[2],... = p_0,p_1,...
            
            up_n = Function(W)
            
            p_n = split(up_n)  # p_n[0] = u_n, p_n[1],p_n[2],... = p0_n,p1_n,...

        # variational formulation
        sources = []  # Contains the source term for each network
        transfer = [] # Contains the transfer terms for each network
        innerProdP = []
        # Contains the inner product of the gradient of p_j for each network
        dotProdP = []  # Contains the dot product of alpha_j & p_j,
        timeD_ = []  # Time derivative for the current step
        timeD_n = []  # Time derivative for the previous step

        self.bcs_D = []  # Contains the terms for the Dirichlet boundaries
        self.integrals_N = []  # Contains the integrals for the Neumann boundaries
        self.time_expr = []  # Terms that needs to be updated at each timestep
        # Terms that contains the windkessel bc that is updated each timestep
        self.windkessel_terms = []

        # Contains the integrals for the Robin boundaries, LHS
        self.integrals_R_L = []
        # Contains the integrals for the Robin boundaries, RHS
        self.integrals_R_R = []

        #sigmoid = "1/(1+exp(-t + 4))"
        #self.RampSource = Expression(sigmoid,t=0.0,degree=2)
        
        def a_u(u, v):
            return self.mu * (inner(grad(u), grad(v)) + inner(grad(u), nabla_grad(v))) * dx

        def a_p(K, p, q):
            return K * dot(grad(p), grad(q)) * dx
        
        def b(s, v):
            return s * div(v) * dx
    
        def c(alpha, p, q):
            return alpha / self.Lambda * dot(p, q) * dx

        def d(p,q,numP): #Time derivatives
            d_p  = self.c[numP] * p[numP + 2]
            d_eps  = self.alpha[numP+1] / self.Lambda * sum(a * b for a, b in zip(self.alpha, p[1:])) 
            return (1 / self.dt)*(d_p + d_eps) * q[numP + 2] * dx
                
        def f(f, v):
            return dot(f, v) * dx(self.mesh)

        
        #Apply terms for each fluid network
        for i in range(self.numPnetworks):  # apply for each network
            if isinstance(
                self.g[i], TimeSeries
            ):  # If the source term is a TimeSeries object
                print("Adding timeseries for source term")
                g_space = FunctionSpace(self.mesh, "CG",1)
                g_i = Function(g_space)
                sources.append(f(g_i, q[i + 2]))  # Applying source term
                self.time_expr.append((self.g[i], g_i))
            elif isinstance(self.g[i], np.ndarray):  # If the timeseries term is a numpy array
                print("Adding numpy timeseries for source term")
                g_space = FunctionSpace(self.mesh, "CG",1)
                g_i = Function(g_space)
                sources.append(f(g_i, q[i + 2]))  # Applying source term
                self.time_expr.append((self.g[i], g_i))
            elif self.g[i] is not None:
                print("Adding expression for source term")
                sources.append(f(self.g[i], q[i + 2]))  # Applying source term
                self.time_expr.append(self.g[i])

            innerProdP.append(a_p(self.K[i], p_[i + 2], q[i + 2]))  # Applying diffusive term

            # Applying time derivatives
            timeD_.append(d(p_,q,i)) #lhs
            timeD_n.append(d(p_n,q,i)) #rhs

                
        if self.gamma.any(): #Add transfer terms
            print("Adding transfer terms")
            for i in range(self.numPnetworks): 
                for j in range(self.numPnetworks):

                    if self.gamma[i,j] and i != j:
                        transfer.append(self.gamma[i,j]*(p_[i+2]-p_[j+2])*q[i+2]*dx)
        dotProdP = [c(alpha,p, q[1]) for alpha,p in zip(self.alpha, p_[1:])]

        DIMCLASS = self.dim + self.numPnetworks + 1 + dimZ  #For InitialConditions class

        # Class representing the intial conditions
        class InitialConditions(UserExpression):
            def __init__(self,networks,dimensions,p_initial):
                super().__init__(degree=networks + dimensions + 1)
                self.N = networks
                self.DIM = dimensions
                self.P_init = p_initial

            def eval(self, values, x):
                for d in range(self.DIM): #For each dimension
                    values[d] = 0.0

                values[self.DIM + 1] = self.P_init[0] #Total pressure

                for n in range(self.N): #For each network
                    values[self.DIM + 1 + n] = self.P_init[n+1]
            def value_shape(self):
                return (DIMCLASS,)
 



        u_init = InitialConditions(self.numPnetworks,self.dim,self.p_initial)
        up_n.interpolate(u_init)

        
        self.applyPressureBC(W,p_,q)
        self.applyDisplacementBC(W.sub(0),q[0])
 
        #self.time_expr.append(self.RampSource)

        #########lhs###########
        F = (
            a_u(p_[0], q[0])
            + b(p_[1], q[0])
            + b(q[1], p_[0])
            - sum(dotProdP)
            + sum(innerProdP)
            + sum(transfer)
            + sum(self.integrals_R_L)
            + sum(timeD_)
        )

        if self.uNullspace:
                F += sum(z[i]*inner(q[0], Z[i])*dx() for i in range(dimZ)) \
                  + sum(r[i]*inner(p_[0], Z[i])*dx() for i in range(dimZ))

        #########rhs##########

        F  -= (
            f(self.f, q[0])
            + sum(sources)
            + sum(timeD_n)
            + sum(self.integrals_N)
            + sum(self.integrals_R_R)
        )

        [self.time_expr.append(self.f[i]) for i in range(self.dim)]
        A = assemble(lhs(F))

        up = Function(W)
       
        dV_PREV_SAS = 0.0
        dV_PREV_VEN = 0.0
      
        for self.i,t in enumerate(self.t):    #range(0,self.numTsteps+1): #Time loop
            
            self.update_time_expr(t)# Update all time dependent terms
            print("t:",t)
            b = assemble(rhs(F))

            [bc.apply(A) for bc in self.bcs_D]
            [bc.apply(b) for bc in self.bcs_D]
            
            solve(A, up.vector(), b)#, self.solverType, self.preconditioner ) #Solve system

            #Write solution at time t
            up_split = up.split(deepcopy = True)
            results = self.generate_diagnostics(*up_split)

            xdmfU.write(up.sub(0), t)
            xdmfP0.write(up.sub(1), t)
            for j in range(self.numPnetworks):
                xdmfP[j].write(up.sub(j+2), t)


            #For calculating volume change in Windkessel model
            results["dV_SAS_PREV"] = dV_PREV_SAS
            results["dV_VEN_PREV"] = dV_PREV_VEN

            results["total_inflow"] = float(self.m)
            
            #p_SAS_f, p_VEN_f,p_SP_f,Vv_dot,Vs_dot,Q_AQ,Q_FM = self.coupled_3P_model(p_SAS_f,p_VEN_f,p_SP_f,results) #calculates windkessel pressure @ t
            #p_SAS_f, p_VEN_f,p_SP_f,Vv_dot,Vs_dot,Q_AQ,Q_FM = self.coupled_3P_model_MK_constrained(p_SAS_f,p_VEN_f,p_SP_f,results) #calculates windkessel pressure @ t
            

            #self.update_windkessel_expr(p_SAS_f,p_VEN_f) # Update all terms dependent on the windkessel pressures

            #results["p_SAS"] = p_SAS_f
            #results["p_VEN"] = p_VEN_f
            #results["p_SP"] = p_SP_f

            #results["Q_AQ"] = Q_AQ
            #results["Q_FM"] = Q_FM
            #results["Vv_dot"] = Vv_dot
            #results["Vs_dot"] =  Vs_dot
            results["t"] = t

            #dV_PREV_SAS = results["dV_SAS"]
            #dV_PREV_VEN = results["dV_VEN"]

            pickle.dump(results, open("%s/data_set/qois_%d.pickle" % (self.filesave, i), "wb"))
            

            up_n.assign(up)
            progress += 1

            #self.t += self.dt
         
        res = []
        res = split(up)
        u = project(res[0], W.sub(0).collapse())
        p = []
        
        self.u_sol = u
        self.p_sol = p

    def plotResults(self,plotCycle = 0):

        plotDir = "%s/plots/" %self.filesave
        initPlot = int(self.numTsteps/self.T*self.plot_from)
        endPlot = int(self.numTsteps/self.T*self.plot_to)
        print("Init from: ",initPlot)
        print("End at: ",endPlot)

        self.plot_from = int(self.plot_from)
        self.plot_to = int(self.plot_to)

        print("Plot from: ",self.plot_from)
        print("Plot to: ",self.plot_to)

        V_fig, V_ax = pylab.subplots(figsize=(12, 8))
        dV_dot_fig, dV_dot_ax = pylab.subplots(figsize=(12, 8))
        PW_figs, PW_axs  = pylab.subplots(figsize=(12, 8)) #Pressure Windkessel
        BV_figs, BV_axs  = pylab.subplots(figsize=(12, 8)) #Outflow venous blood
        ABP_figs,ABP_axs  = pylab.subplots(figsize=(12, 8)) #Outflow venous blood
        Qv_figs, Qv_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to ventricles
        Qs_figs, Qs_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to SAS
        pmax_figs, pmax_axs = pylab.subplots(figsize=(12, 8))
        pmin_figs, pmin_axs = pylab.subplots(figsize=(12, 8))
        pmean_figs, pmean_axs = pylab.subplots(figsize=(12, 8))
        v_fig, v_ax = pylab.subplots(figsize=(12, 8))
        t_fig, t_ax = pylab.subplots(figsize=(12, 8))
        
        # Color code the pressures: red, purple and blue
        colors = ["crimson", "navy", "cornflowerblue"]
        markers = [".-", ".-", ".-"]

        x_ticks = [self.plot_from + int(0.5*i) for i in range(int(self.plot_to/0.5)+1-self.plot_from*2)]
        
        print("x_ticks:",x_ticks)
        
        df = self.load_data()
        names = df.columns
        times = (df["t"].to_numpy())
         # Plot volume vs time
        V_ax.plot(times[initPlot:endPlot], df["dV"][initPlot:endPlot], markers[0], color="seagreen",label="div(u)dx")
        V_ax.plot(times[initPlot:endPlot], df["dV_SAS"][initPlot:endPlot], markers[1], color="darkmagenta",label="(u * n)ds_{SAS}")
        V_ax.plot(times[initPlot:endPlot], df["dV_VEN"][initPlot:endPlot], markers[2], color="royalblue",label="(u * n)ds_{VEN}")
        V_ax.set_xlabel("time (s)")
        V_ax.set_xticks(x_ticks)
        V_ax.set_ylabel("V (mm$^3$)")
        V_ax.grid(True)
        V_ax.legend()
        V_fig.savefig(plotDir + "brain-Vol.png")
        
        if "Vv_dot" in df.keys():
            
            # Plot volume derivative
            dV_dot_ax.plot(times[initPlot:endPlot], df["Vv_dot"][initPlot:endPlot], markers[0], color="seagreen",label="$dV_{VEN}$/dt")
            dV_dot_ax.plot(times[initPlot:endPlot], df["Vs_dot"][initPlot:endPlot], markers[0], color="darkmagenta",label="$dV_{SAS}$/dt")
            
            dV_dot_ax.set_xlabel("time (s)")
            dV_dot_ax.set_xticks(x_ticks)
            dV_dot_ax.set_ylabel("V_dot (mm$^3$/s)")
            dV_dot_ax.grid(True)
            dV_dot_ax.legend()
            dV_dot_fig.savefig(plotDir + "brain-V_dot.png")

        
        # Plot max/min of the pressures
        for i in range(1,self.numPnetworks+1):
            pmax_axs.plot(times[initPlot:endPlot], df["max_p_%d" % i][initPlot:endPlot], markers[0],
                       color=colors[i-1], label="$p_%d$" % i,)

            pmax_axs.set_xlabel("time (s)")
            pmax_axs.set_xticks(x_ticks)
            pmax_axs.set_ylabel("$\max \, p$ (Pa)")
            pmax_axs.grid(True)
            pmax_axs.legend()

            pmin_axs.plot(times[initPlot:endPlot], df["min_p_%d" % i][initPlot:endPlot], markers[0],
                       color=colors[i-1], label="$p_%d$" % i,)

            pmin_axs.set_xlabel("time (s)")
            pmin_axs.set_xticks(x_ticks)
            pmin_axs.set_ylabel("$\min \, p$ (Pa)")
            pmin_axs.grid(True)
            pmin_axs.legend()

            pmean_axs.plot(times[initPlot:endPlot], df["mean_p_%d" % i][initPlot:endPlot], markers[0],
                       color=colors[i-1], label="$p_%d$" % i,)

            pmean_axs.set_xlabel("time (s)")
            pmean_axs.set_xticks(x_ticks)
            pmean_axs.set_ylabel("average P (Pa)")
            pmean_axs.grid(True)
            pmean_axs.legend()

        pmax_figs.savefig(plotDir + "brain-p_max.png")
        pmin_figs.savefig(plotDir + "brain-p_min.png")
        pmean_figs.savefig(plotDir + "brain-p_avg.png")


        
        # Plot average compartment velocity (avg v_i)
        for i in range(1,self.numPnetworks+1):
                v_ax.plot(times[initPlot:endPlot], df["v%d_avg" % i][initPlot:endPlot], markers[0], color=colors[i-1],
                          label="$v_%d$" % i)
        v_ax.set_xlabel("time (s)")
        v_ax.set_xticks(x_ticks)
        v_ax.set_ylabel("Average velocity $v$ (mm/s)")
        v_ax.grid(True)
        v_ax.legend()
        v_fig.savefig(plotDir + "brain-vs.png")

        if "Q_SAS_N3" in df.keys():

            Q_SAS =  df["Q_SAS_N3"] 
            

            
            # Plot outflow of CSF
            Qs_axs.plot(times[initPlot:endPlot], Q_SAS[initPlot:endPlot], markers[0], color="seagreen",label="$Q_{SAS}$")
            
            Qs_axs.set_xlabel("time (s)")
            Qs_axs.set_xticks(x_ticks)
            Qs_axs.set_ylabel("Q (mm$^3$/s)")
            Qs_axs.grid(True)
            Qs_axs.legend()
            Qs_figs.savefig(plotDir + "brain-Q_sas.png")
            
            
        if "Q_VEN_N3" in df.keys():
            Q_VEN =  df["Q_VEN_N3"]
            Qv_axs.plot(times[initPlot:endPlot], Q_VEN[initPlot:endPlot], markers[0], color="darkmagenta",label="$Q_{VEN}$")
            Qv_axs.set_xlabel("time (s)")
            Qv_axs.set_xticks(x_ticks)
            Qv_axs.set_ylabel("Q (mm$^3$/s)")
            Qv_axs.grid(True)
            Qv_axs.legend()
            Qv_figs.savefig(plotDir + "brain-Q_ven.png")

        
        BA = df["G_a"]
        BV_axs.plot(times[initPlot:endPlot], BA[initPlot:endPlot], markers[0], color="darkmagenta",label="$B_{a}$")
        if "Q_SAS_N2" in df.keys():
            BV = df["Q_SAS_N2"] + df["Q_VEN_N2"]
                
            # Plot outflow of venous blood
            BV_axs.plot(times[initPlot:endPlot], BV[initPlot:endPlot], markers[0], color="seagreen",label="$B_{v}$")

        BV_axs.set_xlabel("time (s)")
        BV_axs.set_xticks(x_ticks)
        BV_axs.set_ylabel("Absolute blood flow (mm$^3$/s)")
        BV_axs.grid(True)
        BV_axs.legend()
        BV_figs.savefig(plotDir + "brain-BV.png")


        # Plot Windkessel pressure
        if "p_SP" in df.keys():
            PW_axs.plot(times[initPlot:endPlot], df["p_SP"][initPlot:endPlot], markers[1], color="cornflowerblue",label="$p_{SP}$")

        if "p_SAS" in df.keys():

            PW_axs.plot(times[initPlot:endPlot], df["p_SAS"][initPlot:endPlot], markers[0], color="seagreen",label="$p_{SAS}$")
        if "p_VEN" in df.keys():

            PW_axs.plot(times[initPlot:endPlot], df["p_VEN"][initPlot:endPlot], markers[1], color="darkmagenta",label="$p_{VEN}$")

            PW_axs.set_xlabel("time (s)")
            PW_axs.set_xticks(x_ticks)
            PW_axs.set_ylabel("P ($Pa$)")
            PW_axs.grid(True)
            PW_axs.legend()
            PW_figs.savefig(plotDir + "brain-WK.png")

        
        if "T12" in df.keys():
            # Plot transfer rates (avg v_i)
            t_ax.plot(times[initPlot:endPlot], df["T12"][initPlot:endPlot], markers[0], color="darkmagenta", label="$T_{12}$")
            t_ax.plot(times[initPlot:endPlot], df["T13"][initPlot:endPlot], markers[0], color="royalblue", label="$T_{13}$")
            t_ax.set_xlabel("time (s)")
            t_ax.set_xticks(x_ticks)
            t_ax.set_ylabel("Transfer rate ($L^2$-norm)")
            t_ax.grid(True)
            t_ax.legend()
            t_fig.savefig(plotDir + "brain-Ts.png")
    
        if "Q_AQ" in df.keys():

            Qaq_figs, Qaq_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to SAS
            Q_AQ = df["Q_AQ"]

            Qfm_figs, Qfm_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to SAS


  
            Qaq_axs.plot(times[initPlot:endPlot], Q_AQ[initPlot:endPlot], markers[0], color="darkmagenta",label="$Q_{AQ}$")
            Qaq_axs.set_xlabel("time (s)")
            Qaq_axs.set_xticks(x_ticks)
            Qaq_axs.set_ylabel("Q (mL/s)")
            Qaq_axs.grid(True)
            Qaq_axs.legend()
            Qaq_figs.savefig(plotDir + "brain-Q_aq.png")

        if "Q_FM" in df.keys():

            Qfm_figs, Qfm_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to SAS
            Q_FM = df["Q_FM"]


            Qfm_axs.plot(times[initPlot:endPlot], Q_FM[initPlot:endPlot], markers[0], color="darkmagenta",label="$Q_{FM}$")
            Qfm_axs.set_xlabel("time (s)")
            Qfm_axs.set_xticks(x_ticks)
            Qfm_axs.set_ylabel("Q (mL/s)")
            Qfm_axs.grid(True)
            Qfm_axs.legend()
            Qfm_figs.savefig(plotDir + "brain-Q_fm.png")



    def generate_diagnostics(self,*args):
        results = {}
        u = args[0] 
        p_list = []
        for arg in args[1:self.numPnetworks+2]:
            p_list.append(arg)

        # Volume displacement:
        dV = assemble(div(u)*dx)
        results["dV"] = dV
        print("div(u)*dx (mm^3) = ", dV)
        
        V = VectorFunctionSpace(self.mesh, 'CG' ,1)

        # Pressures
        Vol = assemble(1*dx(self.mesh))
        A = np.sqrt(Vol)
        for (i, p) in enumerate(p_list):
            results["max_p_%d" % (i)] = max(p.vector())
            results["min_p_%d" % (i)] = min(p.vector())
            results["mean_p_%d" % (i)] = assemble(p*dx)/Vol
            
            if i > 0: # Darcy velocities
                v = project(self.K[i-1]*grad(p),
                            V,
                            solver_type = self.solverType,
                            preconditioner_type = self.preconditioner,
                            )
                v_avg = norm(v, "L2")/A
                results["v%d_avg" % (i)] = v_avg
        
                #Calculate outflow for each network
                results["Q_SAS_N%d" %(i)] = assemble(-self.K[i-1] * dot(grad(p), self.n) * self.ds(1))
                results["Q_VEN_N%d" %(i)] = assemble(-self.K[i-1] * dot(grad(p), self.n) * self.ds(2)) + assemble(
                -self.K[i-1] * dot(grad(p),self.n) * self.ds(3))

        
        results["G_a"] = self.m
        results["dV_SAS"] = assemble(dot(u,self.n)*self.ds(1))
        results["dV_VEN"] = assemble(dot(u,self.n)*(self.ds(2) + self.ds(3)))
        

        # Transfer rates
        """
        S = FunctionSpace(self.mesh, "CG", 1)
        t12 = project(self.gamma[0,1]*(p_list[1]-p_list[2]),S)
        t13 = project(self.gamma[0,2]*(p_list[1]-p_list[3]),S)
        results["T12"] = norm(t12,"L2")
        results["T13"] = norm(t13,"L2")
        """
        return results
 

    def simple_mass_WK_model(self,p_SAS,p_VEN,results):
        """
        This model couples the ventricles and SAS. Ventricles are modeled with a mass
        conservation expression and the SAS uses a Windkessel model.
        """

        #Assume only flow from ECS flow out to the CSF filled cavities

        scale = 10**(0)

        print("Pressure for SAS: ", p_SAS)
        print("Pressure for VEN: ", p_VEN)
        #P_SAS is determined from Windkessel parameters
        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS:",Q_SAS)

        #P_VEN is determined from volume change of the ventricles
        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN:",Q_VEN)

        #Volume change of ventricles
        V_dot = 1/self.dt*(results["dV_VEN"]-results["dV_VEN_PREV"])
        print("V_dot:",V_dot)

        K = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant

        p_SAS_next = p_SAS  + self.dt/self.C_SAS *( Q_SAS + V_dot + Q_VEN)*scale
 
        p_VEN_next = p_SAS_next + 1/K*(Q_VEN + V_dot)

        return p_SAS_next, p_VEN_next,V_dot


    def coupled_2P_model(self,p_SAS,p_VEN,results):
        """
        This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
        with Windkessel models.

        Solves using implicit (backward) Euler

        dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS))
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN + G_aq(p_SAS - p_VEN))
        
        """

        #Assume only flow from ECS flow out to the CSF filled cavities

        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS:",Q_SAS)

        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN:",Q_VEN)

        #Scale to avoid instabilities
        V_dotScale = 1/100

        #Volume change of ventricles
        Vv_dot = V_dotScale/self.dt*(results["dV_VEN"]-results["dV_VEN_PREV"])
        

        #Volume change of SAS
        Vs_dot = V_dotScale/self.dt*(results["dV_SAS"]-results["dV_SAS_PREV"])

        print("Rate of volume change, ventricles:",Vv_dot/V_dotScale)
        print("Rate of volume change, SAS",Vs_dot/V_dotScale)

        #Conductance Aqueduct
        G_aq = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant
        
        Q_AQ = G_aq*(p_SAS - p_VEN)
        print("Q_AQ:",Q_AQ)
        b_SAS = p_SAS + self.dt/self.C_SAS * (Q_SAS + Vs_dot)
        b_VEN = p_VEN + self.dt/self.C_VEN * (Q_VEN + Vv_dot)

        A_11 = 1 + self.dt*G_aq/self.C_SAS
        A_12 = -self.dt*G_aq/self.C_SAS
        A_21 = -self.dt*G_aq/self.C_VEN
        A_22 = 1 + self.dt*G_aq/self.C_VEN
        

        b = np.array([b_SAS,b_VEN])
        A = np.array([[A_11, A_12],[A_21, A_22]])

        x = np.linalg.solve(A,b) #x_0 = p_SAS, x_1 = p_VEN

        print("Pressure for SAS: ", x[0])
        print("Pressure for VEN: ", x[1])

        return x[0], x[1],Vv_dot/V_dotScale,Vs_dot/V_dotScale,Q_AQ


    def coupled_3P_model(self,p_SAS,p_VEN,p_SP,results):
        """
        This model calculates a 3-pressure lumped model for the SAS, ventricles and spinal-SAS compartments

        Solves using implicit (backward) Euler

        Equations:
        dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS) + G_fm(p_SP-p_SAS)
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEn + G_aq(p_SAS - p_VEN))
        dp_sp/dt = 1/C_sp(G_fm(p_SAS-p_SP))
        
        """

        if (self.t[self.i] < 4.0):
            VolScale = 1/10000 #mm³ to mL   
        else:
            VolScale = 1/1000 #mm³ to mL


        #P_SAS is determined from Windkessel parameters
        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS[mm³] :",Q_SAS)

        #P_VEN is determined from volume change of the ventricles
        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN[mm³] :",Q_VEN)

        #Volume change of ventricles
        Vv_dot = 1/self.dt*(results["dV_VEN"]-results["dV_VEN_PREV"])
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-results["dV_SAS_PREV"])
        
        print("Volume change VEN[mm³] :",Vv_dot)
        print("Volume change SAS[mm³] :",Vs_dot)

        #Conductance
        G_aq = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant
        #G_aq = 5/133 #mL/mmHg to mL/Pa, from Ambarki2007
        G_aq = G_aq*1/1000 #mm³/Pa to mL/Pa
        G_fm = G_aq*10 #from Ambarki2007

        # "Positive" direction upwards, same as baledent article
        Q_AQ = G_aq*(p_SAS - p_VEN)
        Q_FM = G_fm*(p_SP - p_SAS)

        print("Q_AQ[mL]:",Q_AQ)
        print("Q_FM[mL]:",Q_FM)


        b_SAS = p_SAS + self.dt/self.C_SAS * (Q_SAS  + Vs_dot)* VolScale
        b_VEN = p_VEN + self.dt/self.C_VEN * (Q_VEN + Vv_dot)  * VolScale
        b_SP = p_SP
        A_11 = 1 + self.dt*G_aq/self.C_SAS + self.dt*G_fm/self.C_SAS 
        A_12 = -self.dt*G_aq/self.C_SAS
        A_13 = -self.dt*G_fm/self.C_SAS
        A_21 = -self.dt*G_aq/self.C_VEN
        A_22 = 1 + self.dt*G_aq/self.C_VEN
        A_23 = 0
        A_31 = -self.dt*G_fm/self.C_SP
        A_32 = 0
        A_33 = 1 + self.dt*G_fm/self.C_SP


        b = np.array([b_SAS, b_VEN, b_SP])
        A = np.array([[A_11, A_12, A_13],[A_21, A_22, A_23],[A_31, A_32, A_33]])
        x = np.linalg.solve(A,b) #x_0 = p_SAS, x_1 = p_VEN, x_2 = p_SP

        print("Pressure for SAS: ", x[0])
        print("Pressure for ventricles: ", x[1])
        print("Pressure in spinal-SAS:", x[2])

        return x[0], x[1],x[2],Vv_dot,Vs_dot,Q_AQ,Q_FM


    def coupled_3P_model_MK_constrained(self,p_SAS,p_VEN,p_SP,results):
        """
        This model calculates a 3-pressure lumped model for the SAS, ventricles and spinal-SAS compartments

        Also adds a Monroe-Kellie (MK) constrain to the equation through a multiplier

        Solves using implicit (backward) Euler

        Equations:
        dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS) + G_fm(p_SP-p_SAS))
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEn + G_aq(p_SAS - p_VEN))
        G_fm(p_SAS-p_SP) = Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS))
        
        """

        if (self.t[self.i] < 4.0):
            VolScale = 1/10000 #mm³ to mL   
        else:
            VolScale = 1/1000 #mm³ to mL


        #P_SAS is determined from Windkessel parameters
        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS[mm³] :",Q_SAS)

        #P_VEN is determined from volume change of the ventricles
        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN[mm³] :",Q_VEN)

        #Volume change of ventricles
        Vv_dot = 1/self.dt*(results["dV_VEN"]-results["dV_VEN_PREV"])
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-results["dV_SAS_PREV"])
        
        
        print("Volume change ventricles[mm³] :",Vv_dot)
        print("Volume change SAS[mm³] :",Vs_dot)

        #Conductance
        G_aq = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant
        #G_aq = 5/133 #mL/mmHg to mL/Pa, from Ambarki2007
        G_aq = G_aq*1/1000 #mm³/Pa to mL/Pa
        G_fm = G_aq*10 #from Ambarki2007

        # "Positive" direction upwards, same as baledent article
        Q_AQ = G_aq*(p_SAS - p_VEN)
        Q_FM = G_aq*(p_SP - p_SAS)
       
        print("Q_AQ[mL]:",Q_AQ)
        print("Q_FM[mL]:",Q_FM)


        b_SAS = p_SAS + self.dt/self.C_SAS * (Q_SAS  + Vs_dot)* VolScale
        b_VEN = p_VEN + self.dt/self.C_VEN * (Q_VEN + Vv_dot)  * VolScale
        b_SP = (Vs_dot + Q_SAS) * VolScale
        A_11 = 1 + self.dt*G_aq/self.C_SAS + self.dt*G_fm/self.C_SAS 
        A_12 = -self.dt*G_aq/self.C_SAS
        A_13 = - self.dt*G_fm/self.C_SAS
        A_21 = -self.dt*G_aq/self.C_VEN
        A_22 = 1 + self.dt*G_aq/self.C_VEN
        A_23 = 0
        A_31 = G_aq + G_fm 
        A_32 = -G_aq
        A_33 = -G_fm


        b = np.array([b_SAS, b_VEN, b_SP])
        A = np.array([[A_11, A_12, A_13],[A_21, A_22, A_23],[A_31, A_32, A_33]])
        x = np.linalg.solve(A,b) #x_0 = p_SAS, x_1 = p_VEN, x_2 = p_SAS

        print("Pressure for SAS: ", x[0])
        print("Pressure for Ventricles: ", x[1])
        print("Pressure for Spinal Cord: ", x[2])
       
        return x[0], x[1],x[2],Vv_dot,Vs_dot,Q_AQ,Q_FM


    def coupled_3P_nonlinear_model(self,p_SAS,p_VEN,p_SP,results):
        """
        This model calculates a 3-pressure lumped model for the SAS, ventricles and spinal-SAS compartments

        Solves using explicit (forward) Euler

        Equations:
        dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS) + G_fm(p_SP-p_SAS)
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN + G_aq(p_SAS - p_VEN))
        dp_sp/dt = 1/C_sp(G_fm(p_SAS-p_SP))
        
        """

        if (self.t[self.i] < 4.0):
            VolScale = 1/10000 #mm³ to mL   
        else:
            VolScale = 1/1000 #mm³ to mL


            #P_SAS is determined from Windkessel parameters
        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS[mm³] :",Q_SAS)

        #P_VEN is determined from volume change of the ventricles
        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN[mm³] :",Q_VEN)

        #Volume change of ventricles
        Vv_dot = 1/self.dt*(results["dV_VEN"]-results["dV_VEN_PREV"])
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-results["dV_SAS_PREV"])
        
        
        print("Vv_dot[mm³] :",Vv_dot)
        print("Vs_dot[mm³] :",Vs_dot)

        #Conductance
        G_aq = 5/133 #mL/mmHg to mL/Pa, from Ambarki2007
        G_fm = G_aq*10 #from Ambarki2007

        # "Positive" direction upwards, same as baledent article
        Q_AQ = G_aq*(p_SAS - p_VEN)
        Q_FM = G_fm*(p_SP - p_SAS)

        print("Q_AQ[mL]:",Q_AQ)
        print("Q_FM[mL]:",Q_FM)

        E_1 = 1#[1/mL]
        E_2 = 0.25


        p_r = 0.0 #referance pressure, mmHg
        
        C_SAS = 1/(E_1*(p_SP-p_r))
        C_VEN = 1/(E_1*(p_VEN-p_r))
        C_SP = 1/(E_2*(p_SP-p_r))

        print("C_SAS/C_VEN:",C_SAS)
        print("C_SP:",C_SP)

        
        p_SAS_nn = p_SAS + self.dt/C_SAS * ((Q_SAS  + Vs_dot)* VolScale -Q_AQ + Q_FM)
        p_VEN_nn = p_VEN + self.dt/C_VEN * ((Q_VEN + Vv_dot)  * VolScale + Q_AQ)
        p_SP_nn = p_SP + self.dt/C_SP*(-Q_FM)

        x = np.array([p_SAS_nn,p_VEN_nn,p_SP_nn])
        
        print("Pressure for SAS: ", x[0])
        print("Pressure for ventricles: ", x[1])
        print("Pressure in spinal-SAS:", x[2])

        return x[0], x[1],x[2],Vv_dot,Vs_dot,Q_AQ,Q_FM


    def applyPressureBC_BLOCK(self,Q,p,q):
        
        for i in range(1,self.numPnetworks+1):  # apply for each network
            print("Network: ", i)    
            for j in range(1, self.boundaryNum + 1):  # for each boundary
                print("Boundary: ", j)
                if "Dirichlet" in self.boundary_conditionsP[(i, j)]:
                    print(
                        "Applying Dirichlet BC for pressure network: %i and boundary surface %i "
                        % (i, j)
                    )
                    expr = self.boundary_conditionsP[(i, j)]["Dirichlet"]
                    bcp = DirichletBC(
                        Q,
                        expr,
                        self.boundary_markers,
                        j,
                    )
                    self.bcs_D.append(bcp)
                    self.time_expr.append(expr)
                elif "DirichletWK" in self.boundary_conditionsP[(i, j)]:
                    print(
                        "Applying Dirichlet Windkessel BC for pressure network: %i and boundary surface %i "
                        % (i, j)
                    )
                    expr = self.boundary_conditionsP[(i, j)]["DirichletWK"]
                    bcp = DirichletBC(
                        Q,
                        expr*self.Pscale,
                        self.boundary_markers,
                        j,
                    )
                    self.bcs_D.append(bcp)
                    self.windkessel_terms.append(expr)
                elif "Robin" in self.boundary_conditionsP[(i, j)]:
                    print("Applying Robin BC for pressure")
                    print("Applying Robin LHS")
                    beta, P_r = self.boundary_conditionsP[(i, j)]["Robin"]
                    self.integrals_R_L.append(inner(beta * p, q) * self.ds(j))
                    if P_r:
                        print("Applying Robin RHS")
                        self.integrals_R_R.append(inner(beta * P_r, q) * self.ds(j))
                        self.time_expr.append(P_r)
                elif "RobinWK" in self.boundary_conditionsP[(i, j)]:
                    print("Applying Robin BC with Windkessel referance pressure")
                    beta, P_r = self.boundary_conditionsP[(i, j)]["RobinWK"]
                        
                    print("Applying Robin LHS")
                    self.integrals_R_L.append(inner(beta * p * self.Pscale, q) * self.ds(j))

                    print("Applying Robin RHS")
                    self.integrals_R_R.append(inner(beta * P_r * self.Pscale, q) * self.ds(j))
                    self.windkessel_terms.append(P_r)
                elif "Neumann" in self.boundary_conditionsP[(i,j)]:
                   if self.boundary_conditionsP[(i,j)]["Neumann"] != 0:
                        print("Applying Neumann BC.")
                        N = self.boundary_conditionsP[(i,j)]["Neumann"]
                        self.integrals_N.append(inner(N * self.Pscale, q) * self.ds(j))
                        self.time_expr.append(N)





    def applyPressureBC(self,W,p_,q):
        
        for i in range(1,self.numPnetworks+1):  # apply for each network
            print("Network: ", i)    
            for j in range(1, self.boundaryNum + 1):  # for each boundary
                print("Boundary: ", j)
                if "Dirichlet" in self.boundary_conditionsP[(i, j)]:
                    print(
                        "Applying Dirichlet BC for pressure network: %i and boundary surface %i "
                        % (i, j)
                    )
                    expr = self.boundary_conditionsP[(i, j)]["Dirichlet"]
                    bcp = DirichletBC(
                        W.sub(i + 1),
                        expr,
                        self.boundary_markers,
                        j,
                    )
                    self.bcs_D.append(bcp)
                    self.time_expr.append(expr)
                elif "DirichletWK" in self.boundary_conditionsP[(i, j)]:
                    print(
                        "Applying Dirichlet Windkessel BC for pressure network: %i and boundary surface %i "
                        % (i, j)
                    )
                    expr = self.boundary_conditionsP[(i, j)]["DirichletWK"]
                    bcp = DirichletBC(
                        W.sub(i + 1),
                        expr*self.Pscale,
                        self.boundary_markers,
                        j,
                    )
                    self.bcs_D.append(bcp)
                    self.windkessel_terms.append(expr)
                elif "Robin" in self.boundary_conditionsP[(i, j)]:
                    print("Applying Robin BC for pressure")
                    print("Applying Robin LHS")
                    beta, P_r = self.boundary_conditionsP[(i, j)]["Robin"]
                    self.integrals_R_L.append(inner(beta * p_[i + 1], q[i + 1]) * self.ds(j))
                    if P_r:
                        print("Applying Robin RHS")
                        self.integrals_R_R.append(inner(beta * P_r, q[i + 1]) * self.ds(j))
                        self.time_expr.append(P_r)
                elif "RobinWK" in self.boundary_conditionsP[(i, j)]:
                    print("Applying Robin BC with Windkessel referance pressure")
                    beta, P_r = self.boundary_conditionsP[(i, j)]["RobinWK"]
                        
                    print("Applying Robin LHS")
                    self.integrals_R_L.append(inner(beta * p_[i + 1] * self.Pscale, q[i + 1]) * self.ds(j))

                    print("Applying Robin RHS")
                    self.integrals_R_R.append(inner(beta * P_r * self.Pscale, q[i + 1]) * self.ds(j))
                    self.windkessel_terms.append(P_r)
                elif "Neumann" in self.boundary_conditionsP[(i,j)]:
                    if self.boundary_conditionsP[(i,j)]["Neumann"] != 0:
                        print("Applying Neumann BC.")
                        N = self.boundary_conditionsP[(i,j)]["Neumann"]
                        self.integrals_N.append(inner(N * self.Pscale, q[i+1]) * self.ds(j))
                        self.time_expr.append(N)


    def applyDisplacementBC(self,V,v):
        # Defining boundary conditions for displacements
        for i in self.boundary_conditionsU:
            print("i = ", i)
            if "Dirichlet" in self.boundary_conditionsU[i]:
                print("Applying Dirichlet BC.")
                for j in range(self.dim): #For each dimension
                    exprU =self.boundary_conditionsU[i]["Dirichlet"][j]
                    self.bcs_D.append(
                        DirichletBC(
                            V.sub(j),
                            exprU,
                            self.boundary_markers,
                            i,
                        )
                    )
                    self.time_expr.append(exprU)
            elif "Neumann" in self.boundary_conditionsU[i]:
                if self.boundary_conditionsU[i]["Neumann"] != 0:
                    print("Applying Neumann BC.")
                    N = self.boundary_conditionsU[i]["Neumann"]
                    self.integrals_N.append(inner(-self.n * N, v) * self.ds(i))
                    self.time_expr.append(N)
            elif "NeumannWK" in self.boundary_conditionsU[i]:
                if self.boundary_conditionsU[i]["NeumannWK"] != 0:
                    print("Applying Neumann BC with windkessel term.")
                    N = self.boundary_conditionsU[i]["NeumannWK"]
                    self.integrals_N.append(inner(-self.n * N * self.Pscale, v) * self.ds(i))
                    self.windkessel_terms.append(N)

                 
    def generateUFLexpressions(self):
        import sympy as sym
            
        t = sym.symbols("t")

        fx = 0.0 #self.f_val  # force term y-direction
        fy = 0.0 #self.f_val  # force term y
        fz = 0.0 
        #p_initial0 =  sum([-x*y for x,y in zip(self.alpha_val,self.p_initial_val)])

        RampSource = 0
        variables = [
            self.mu,
            self.Lambda,
            fx,
            fy,
            fz,
            RampSource,
        ]

        variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

        UFLvariables = [
            Expression(var, degree=0, t=0.0 ) for var in variables
        ]  # Generate ufl varibles
 

        (
            self.mu_UFL,
            self.Lambda_UFL,
            fx_UFL,
            fy_UFL,
            fz_UFL,
            RS_UFL,
        ) = UFLvariables

        if self.dim == 2:
            self.f = as_vector((fx, fy))
        elif self.dim == 3:
            self.f = as_vector((fx, fy, fz))

        self.RampSource = RS_UFL 
        self.alpha = []
        self.c = []
        self.K = []

        
        self.alpha.append(Constant(1.0)) 

        #For each network
        for i in range(self.numPnetworks):
            variables = [
                self.alpha_val[i],
                self.c_val[i],
                self.K_val[i],
            ]

            variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

            UFLvariables = [
                Expression(var, degree=0, t=0.0 ) for var in variables
            ]  # Generate ufl varibles
            
            (
                alpha_UFL,
                c_UFL,
                K_UFL,
            ) = UFLvariables

            self.c.append(c_UFL)
            self.K.append(K_UFL)
            self.alpha.append(alpha_UFL)
        
         
 
    def printSetup(self):

        print("\n SOLVER SETUP\n")
        print(tabulate([['Problem Dimension', self.dim],
                        ['Number of networks', self.numPnetworks],
                        ['Total time',self.T],
                        ['Number of timesteps',self.numTsteps],
                        ['Element type',self.element_type]],

                       headers=['Setting', 'Value']))

    
        print("\n DOMAIN PARAMETERS\n")
        print(tabulate([['Young Modulus', self.E, 'Pa'],
                        ['nu', self.nu, "--"],
                        ['lambda', self.Lambda, 'Pa'],
                        ['mu', self.mu, 'Pa'],
                        ['rho', self.rho, "kg/m³"],
                        ['c', self.c_val, "1/Pa"],
                        ['kappa', self.kappa, 'mm^2'],
                        ['mu_f', self.mu_f, 'Pa*s'],
                        ['alpha', self.alpha_val],
                        ['gamma', self.gamma, "1/Pa"]],
                       headers=['Parameter', 'Value', 'Unit']))


        print("\n BOUNDARY PARAMETERS\n")
        print(tabulate([['Compliance SAS', self.C_SAS, 'mm^3/mmHg'],
                        ['Compliance Ventricles', self.C_VEN, 'mm^3/mmHg'],
                        ['beta SAS', self.beta_SAS, '--' ],
                        ['beta ventricles', self.beta_VEN, '--']],
                        headers=['Parameter', 'Value', 'Unit']))

    def update_time_expr(self,t):
        for expr in self.time_expr:
            if isinstance(expr, ufl.tensors.ComponentTensor):
                for dimexpr in expr.ufl_operands:
                    for op in dimexpr.ufl_operands:
                        try:
                            op.t = t
                        except:
                            print("passing for: ", expr)
                            pass
            elif isinstance(expr, tuple):
                if isinstance(expr[0], TimeSeries):  
                    expr[0].retrieve(expr[1].vector(), t,interpolate=False)
                    self.m = assemble(expr[1]*dx)
                elif isinstance(expr[0], np.ndarray):  
                    expr[1].vector()[:] = expr[0][self.i]    
                    self.m = assemble(expr[1]*dx)                
                else:
                    self.operand_update(expr, t)
                    
            
    def operand_update(self,expr, t):
        if isinstance(expr, ufl.algebra.Operator):
            for op in expr.ufl_operands:
                update_operator(expr.ufl_operands, t)
        elif isinstance(expr, ufl.Coefficient):
            expr.t = t

    
    def update_windkessel_expr(self, p_SAS_next,p_VEN_next):
        for expr in self.windkessel_terms:
            try:
                expr.p_SAS = p_SAS_next
                expr.p_VEN = p_VEN_next 
            except:
                try:
                    expr.p_SAS = p_SAS_next
                except:
                    expr.p_VEN = p_VEN_next


    def load_data(self): 
        it = None
        # Get number of results files
        directory = os.path.join("%s/data_set" % self.filesave)
        files = glob.glob("%s/qois_*.pickle" % directory)
        N = len(files)
        
        # Read all results files into list of results, and stuff into
        # pandas DataFrame
        results = []
        for n in range(N):
            res_n = pickle.load(open("%s/qois_%d.pickle" % (directory, n), "rb"))
            results.append(res_n)
        df = pandas.DataFrame(results)
        
        return df

    def GenerateNumpySeries(self):
        source_scale = 1/1173670.5408281302 #1/mm³


        Q = FunctionSpace(self.mesh,"CG",1)
        
        time_period = 1.0
        data = np.loadtxt(self.sourceFile, delimiter = ",")
        t = data[:,0]
        source = data[:,1]
        g = np.interp(self.t,t,source,period = 1.0)*source_scale
        if self.scaleMean:
            g -= np.mean(g)

        return g



    def get_system(self,n):
        '''MPET biot with 3 networks. Return system to be solved with PETSc'''
        # For simplicity we consider a stationary problem and displacement
        # and network pressures are fixed to 0 on the entire boundary
        mesh = UnitCubeMesh(n, n, n)
        cell = mesh.ufl_cell()
        
        Velm = VectorElement('Lagrange', cell, 2)
        Qi_elm = FiniteElement('Lagrange', cell, 1)  # For one pressure
        Qelm = MixedElement([Qi_elm]*4)
        Welm = MixedElement([Velm, Qelm]) 
        
        W = FunctionSpace(mesh, Welm)
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)
        
        print(W.dim(), "<<<<<", mesh.num_entities_global(mesh.topology().dim()))
        
        bcs = [DirichletBC(W.sub(0), Constant((0, )*len(u)), 'on_boundary')]
        # Add the network ones
        bcs.extend([DirichletBC(W.sub(1).sub(net), Constant(0), 'on_boundary')
                    for net in range(1, 4)])
        
        mu, lmbda = Constant(self.mu), Constant(self.Lambda)
        alphas = Constant((self.alpha_val[0], self.alpha_val[1],self.alpha_val[2]))
        cs = Constant((self.c_val[0], self.c_val[1], self.c_val[2]))
        Ks = Constant((self.K_val[0], self.K_val[1], self.K_val[2]))
        
        # Exchange matrix; NOTE: I am not sure about the sign here
        Ts = Constant(((0, self.gamma[0,1], self.gamma[0,2]),
                       (self.gamma[1,0], 0, self.gamma[1,2]),
                       (self.gamma[2,0], self.gamma[2,1], 0)))

        # The first of the pressure is the total pressure, the rest ones
        # are networks
        pT, *ps = split(p)
        qT, *qs = split(q)

        nnets = len(ps)
        
        a = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx + inner(pT, div(v))*dx
             + inner(qT, div(u))*dx
             - (1/lmbda)*inner(pT, qT)*dx
             - (1/lmbda)*sum(inner(alphas[i]*ps[i], qT)*dx for i in range(nnets)))

        # Add the eq tested with networks
        for j in range(nnets):
            a = a - (1/lmbda)*inner(alphas[j]*qs[j], pT)*dx
            # The diagonal part
            a = a - (inner(cs[j]*ps[j], qs[j])*dx +
                     inner(Ks[j]*grad(ps[j]), grad(qs[j]))*dx +
                     (1/lmbda)*sum(inner(alphas[i]*ps[i], qs[j])*dx for i in range(nnets)) +
                    sum(inner(Ts[j, i]*(ps[j] - ps[i]), qs[j])*dx for i in range(nnets) if i != j))

        # Now the preconditioner operator
        a_prec = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
                  + (1/lmbda + 1/(2*mu))*inner(pT, qT)*dx)
        # Add the eq tested with networks
        for j in range(nnets):
            a_prec =  a_prec + (1/lmbda)*inner(alphas[j]*qs[j], pT)*dx
            # The diagonal part
            a_prec = a_prec + (inner(cs[j]*ps[j], qs[j])*dx +
                               inner(Ks[j]*grad(ps[j]), grad(qs[j]))*dx +
                               (1/lmbda)*sum(inner(alphas[i]*ps[i], qs[j])*dx for i in range(nnets)) +
                               sum(inner(Ts[j, i]*(ps[j] - ps[i]), qs[j])*dx for i in range(nnets) if i != j))

        r = SpatialCoordinate(mesh)
        # Soma fake rhs
        L = inner(r, v)*dx
            
        B, _ = assemble_system(a_prec, L, bcs)
        A, b = assemble_system(a, L, bcs)

        return A, b, W, B



    def SolvePETSC(self):

        xdmfU = XDMFFile(self.filesave + "/FEM_results/u.xdmf")
        xdmfU.parameters["flush_output"]=True

        xdmfP = []
        
        for i in range(self.numPnetworks+1):
            xdmfP.append(XDMFFile(self.filesave + "/FEM_results/p" + str(i) + ".xdmf"))        
            xdmfP[i].parameters["flush_output"]=True

        A, b, W, B = self.get_system(8)
    
        solver = PETScKrylovSolver()
        solver.parameters['error_on_nonconvergence'] = False
        ksp = solver.ksp()
        
        solver.set_operators(A, B)
        OptDB = PETSc.Options()    
        OptDB.setValue('ksp_type', 'minres')
        OptDB.setValue('pc_type', 'fieldsplit')
        OptDB.setValue('pc_fieldsplit_type', 'additive')  # schur
        OptDB.setValue('pc_fieldsplit_schur_fact_type', 'diag')   # diag,lower,upper,full
        
        # Only apply preconditioner
        OptDB.setValue('fieldsplit_0_ksp_type', 'preonly')
        OptDB.setValue('fieldsplit_1_ksp_type', 'preonly')
        # Set the splits
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        splits = tuple((str(i), PETSc.IS().createGeneral(W.sub(i).dofmap().dofs()))
                       for i in range(W.num_sub_spaces()))
        pc.setFieldSplitIS(*splits)        
        assert len(splits) == 2
        
        OptDB.setValue('fieldsplit_0_pc_type', 'lu')  
        OptDB.setValue('fieldsplit_1_pc_type', 'hypre')  # AMG in cbc block
        
        OptDB.setValue('ksp_norm_type', 'preconditioned')
        # Some generics
        OptDB.setValue('ksp_view', None)
        OptDB.setValue('ksp_monitor_true_residual', None)    
        OptDB.setValue('ksp_converged_reason', None)
        # NOTE: minres does not support unpreconditioned
        OptDB.setValue('ksp_rtol', 1E-10)
        # Use them!
        ksp.setFromOptions()
        
        wh = Function(W)
        solver.solve(wh.vector(), b)
        print(W.dim())
        
        u,p = wh.split(deepcopy = True)
        t = 0
        xdmfU.write(u, t)
        for j in range(self.numPnetworks+1):
            xdmfP[j].write(p.sub(j), t)




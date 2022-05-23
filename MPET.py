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
import timeit
from pathlib import Path
from block import block_mat, block_vec, block_transpose, block_bc,block_assemble
from mshr import *

import warnings

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
        self.filesave = "results/{}".format(kwargs.get("file_save"))
        self.uNullspace = kwargs.get("uNullspace")
        self.v_transfer_term = kwargs.get("vein_transfer")
        
        self.mesh_type = kwargs.get("GEOM")
        self.fileStats = open("{}/stats.txt".format(self.filesave),"w")

        if kwargs.get("description"):
            self.description = kwargs.get("description")
            self.fileStats.write(self.description)
            self.fileStats.write('\n')
            
        
        
        self.Vol = assemble(1*dx(self.mesh))
        self.dim = self.mesh.topology().dim()
 
        
        
        
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
        self.dt = self.T / self.numTsteps
        self.t = np.linspace(0,float(self.T),int(self.numTsteps)+1)
        
        self.element_type = kwargs.get("element_type")
        self.solverType = kwargs.get("solver")
        self.preconditioner = kwargs.get("preconditioner")
        self.uConditioner = kwargs.get("uConditioner")

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
        self.g = [self.GenerateNumpySeries(),None, None]
        self.Lambda = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))        
        
        
        #Boundary parameters
        if kwargs.get("Compliance_sas"):
            self.C_SAS =  kwargs.get("Compliance_sas")
        if kwargs.get("Compliance_ven"):
            self.C_VEN =  kwargs.get("Compliance_ven") 
        if kwargs.get("Compliance_spine"):
            self.C_SP =  kwargs.get("Compliance_spine") 
        self.Pscale = 1.0

        #For alternative model for ventricles
        self.L = kwargs.get("length")
        self.d = kwargs.get("diameter")

        self.beta_VEN = kwargs.get("beta_ven")
        self.beta_SAS = kwargs.get("beta_sas")

        self.p_BC_initial = [kwargs.get("p_ven_initial"),kwargs.get("p_sas_initial")]
        self.p_BC_initial.append(kwargs.get("p_spine_initial"))

        if kwargs.get("ICP"):
            self.ICP = kwargs.get("ICP")
        if kwargs.get("PVI"):
            self.PVI = kwargs.get("PVI")
        
        self.p_vein = kwargs.get("p_vein")
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

 
        
  
    def solve(self):

        print("\nSetting up problem...\n")

        print("Generating UFL expressions\n")
        self.generateUFLexpressions()
        
        
        
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        self.n = FacetNormal(self.mesh)  # normal vector on the boundary


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

        if self.v_transfer_term:
            p_vein = Constant(1117)
            gamma_v = Constant(9.77*10**(-5))
            transfer.append(gamma_v*(p_[3]-p_vein)*q[3]*dx)

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


            results["total_inflow"] = float(self.m)

            results["p_SAS"] = p_SAS_f #NOTE THESE ARE EQUIVALENT FROM PREVIOUS TIMESTEP, NOT CURRENT 
            results["p_VEN"] = p_VEN_f 


            #ICP, p_VEN_f,Vv_dot,Vs_dot,Q_AQ = self.P2_nonlinear_model(dV_PREV_SAS,dV_PREV_VEN,results) 
            ICP, p_VEN_f,Vv_dot,Vs_dot,Q_AQ = self.P2_nonlinear_noncommunicating_model(dV_PREV_SAS,dV_PREV_VEN,results)

            self.update_windkessel_expr(ICP + p_SAS_f ,ICP + p_VEN_f ) # Update all terms dependent on the windkessel pressures
            print("Relative Pressure SAS:",p_SAS_f)
            print("Relative Pressure VEN:",p_VEN_f)

            results["Q_AQ"] = Q_AQ

            results["ICP"] = ICP

            results["Vv_dot"] = Vv_dot
            results["Vs_dot"] = Vs_dot
            results["t"] = t
            
            dV_PREV_SAS = results["dV_SAS"]
            dV_PREV_VEN = results["dV_VEN"]

            pickle.dump(results, open("%s/data_set/qois_%d.pickle" % (self.filesave, self.i), "wb"))

            
            up_n.assign(up)
            progress += 1



    def printStatistics(self):

        
        initStats = int(self.numTsteps/self.T*self.plot_from)
        endStats = int(self.numTsteps/self.T*self.plot_to)
        print("Init from: ",initStats)
        print("End at: ",endStats)

        self.plot_from = int(self.plot_from)
        self.plot_to = int(self.plot_to)

        statCycles = int(self.plot_to - self.plot_from) #Assuming 1 second cycles

        df = self.load_data()
        names = df.columns
        times = (df["t"].to_numpy())

        Q_A = np.mean(df["G_a"][initStats:endStats] - df["TOTAL_OUTFLOW_N1"])

        BV = df["TOTAL_OUTFLOW_N2"]
        Q_V = np.mean(df["TOTAL_OUTFLOW_N2"][initStats:endStats])
        T_av = np.mean(df["T12"][initStats:endStats])
        T_ap = np.mean(df["T13"][initStats:endStats])
        T_va = np.mean(df["T21"][initStats:endStats])
        T_pa = np.mean(df["T31"][initStats:endStats])
        if "T23" in df.keys():
            T_vp = np.mean(df["T23"][initStats:endStats])
            T_pv = np.mean(df["T32"][initStats:endStats])
        else:
            T_vp,T_pv = 0,0
        Q_CSF = np.mean(df["TOTAL_OUTFLOW_N3"][initStats:endStats])

        Q_VEN = np.mean(df["Q_VEN_N3"][initStats:endStats])
        Q_SAS = np.mean(df["Q_SAS_N3"][initStats:endStats])

        dt = statCycles/(endStats-initStats)

        print("dt:", dt)

        #ONLY WORKS FOR STEADY STATE
        SV_AQ = sum(np.abs(df["Q_AQ"][initStats:endStats])*dt)/(statCycles*2)
        if "Q_FM" in df.keys():
            SV_FM = sum(np.abs(df["Q_FM"][initStats:endStats])*dt)/(statCycles*2)
        else:
            SV_FM = 0
            Q_FM = -np.mean(df["Vs_dot"][initStats:endStats] + df["Q_SAS_N3"][initStats:endStats] - df["Q_AQ"][initStats:endStats])
        q1_AVG = np.mean(df["q_avg_1"][initStats:endStats])
        q2_AVG = np.mean(df["q_avg_2"][initStats:endStats])
        q3_AVG = np.mean(df["q_avg_3"][initStats:endStats])

        
        flowStatsH = '\nFLOW STATISTICS\n'
        flowStats = tabulate([['Total arterial inflow', Q_A,"mm^3/s"],
                        ['Total venous outflow', Q_V,"mm^3/s"],
                        ['Net CSF outflow', Q_CSF,"mm^3/s"],
                        ['Net VENTRICULAR CSF outflow', Q_VEN,"mm^3/s"],
                        ['Net SAS CSF outflow', Q_SAS,"mm^3/s"],
                        ['Mean arterio-venous transfer',T_av,"mm^3/s"],
                        ['Mean arterio-perivascular transfer',T_ap,"mm^3/s"],
                        ['Mean venous-arterio transfer',T_va,"mm^3/s"],
                        ['Mean perivascular-arterio transfer',T_pa,"mm^3/s"],
                        ['Mean venous-perivascular transfer',T_vp,"mm^3/s"],
                        ['Mean perivascular-venous transfer',T_pv,"mm^3/s"],
                        ['Aqueductal Stroke Volume', SV_AQ,"mL/s"],
                        ['Spinal Stroke Volume', SV_FM,"mL/s"],
                        ['Mean bulk arteriole velocity', q1_AVG,"mm/s"],
                        ['Mean bulk venous velocity', q2_AVG,"mm/s"],
                        ['Mean bulk perivascular velocity', q3_AVG,"mm/s"]],                        
                       headers=['Quantity', 'Value','Unit'])

        print(flowStatsH)
        print(flowStats)
        self.fileStats.write('\n')
        self.fileStats.write(flowStatsH)
        self.fileStats.write('\n')
        self.fileStats.write(flowStats)
        

        dV_max = max(df["dV"][initStats:endStats])
        dV_min = min(df["dV"][initStats:endStats])

        dV_peak = dV_max - dV_min

        dispStatsH = "\nDISPLACEMENTS STATISTICS\n"
        dispStats =tabulate([['Peak brain expansion', dV_max,"mm^3"],
                        ['Peak brain contraction', dV_min,"mm^3"],
                        ['Peak brain stroke volume', dV_peak,"mm^3"],
                        ['Max tissue displacement', 0,"mm"]],
                        headers=['Quantity', 'Value','Unit'])
        print(dispStatsH)
        print(dispStats)
        self.fileStats.write('\n')
        self.fileStats.write(dispStatsH)
        self.fileStats.write('\n')
        self.fileStats.write(dispStats)
        self.fileStats.write('\n')
        

        p1_max = max(df["max_p_1"][initStats:endStats])
        p1_min = min(df["min_p_1"][initStats:endStats])
        p2_max = max(df["max_p_2"][initStats:endStats])
        p2_min = min(df["max_p_2"][initStats:endStats])
        p3_max = max(df["max_p_3"][initStats:endStats])
        p3_min = min(df["min_p_3"][initStats:endStats])

        ICP = np.mean(df["ICP"][initStats:endStats])
        ICP_pulse = max(df["ICP"][initStats:endStats]) - min(df["ICP"][initStats:endStats])
        p_VEN = np.mean(df["p_VEN"][initStats:endStats])
    
        
        pressureStatsH = "\nPRESSURE STATISTICS\n"
        pressureStats = tabulate([['Max arteriole pressure (temporal)',p1_max ,"Pa"],
                        ['Min arteriole pressure (temporal)', p1_min,"Pa"],
                        ['Mean arteriole pressure (temporal)', np.mean(df["mean_p_1"][initStats:endStats]),"Pa"],
                        ['Arteriole pulse pressure (temporal)',p1_max - p1_min,"Pa"],
                        ['Max venous pressure (temporal)', p2_max,"Pa"],
                        ['Min venous pressure (temporal)', p2_min,"Pa"],
                        ['Mean venous pressure (temporal)', np.mean(df["mean_p_2"][initStats:endStats]),"Pa"],
                        ['Venous pulse pressure (temporal)',p2_max - p2_min,"Pa"],
                        ['Max perivascular pressure (temporal)', p3_max,"Pa"],
                        ['Min perivascular pressure (temporal)', p3_min,"Pa"],
                        ['Mean perivascular pressure (temporal)', np.mean(df["mean_p_3"][initStats:endStats]),"Pa"],
                        ['Perivascular pulse pressure (temporal)',p3_max - p3_min,"Pa"],
                        ['Mean ICP (temporal)',ICP,"Pa"],
                        ['Intracranial pulse pressure ',ICP_pulse,"Pa"],
                        ['Mean relative ventricular pressure (temporal)',p_VEN,"Pa"]],
                        headers=['Quantity', 'Value','Unit'])

        print(pressureStatsH)
        print(pressureStats)
        self.fileStats.write('\n')
        self.fileStats.write(pressureStatsH)
        self.fileStats.write('\n')
        self.fileStats.write(pressureStats)
        

    def plotResults(self):

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
        Vven_fig, Vven_ax = pylab.subplots(figsize=(12, 8))
        dV_dot_fig, dV_dot_ax = pylab.subplots(figsize=(12, 8))
        PW_figs, PW_axs  = pylab.subplots(figsize=(12, 8)) #Pressure Windkessel
        BV_figs, BV_axs  = pylab.subplots(figsize=(12, 8)) #Outflow venous blood
        ABP_figs,ABP_axs  = pylab.subplots(figsize=(12, 8)) #Outflow venous blood
        Qv_figs, Qv_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to ventricles
        ICV_fig, ICV_ax  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to ventricles
        Qs_figs, Qs_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to SAS
        pmax_figs, pmax_axs = pylab.subplots(figsize=(12, 8))
        pmin_figs, pmin_axs = pylab.subplots(figsize=(12, 8))
        pmean_figs, pmean_axs = pylab.subplots(figsize=(12, 8))
        OUTFLOW_figs, OUTFLOW_axs = pylab.subplots(figsize=(12, 8))
        v_fig, v_ax = pylab.subplots(figsize=(12, 8))
        q_fig, q_ax = pylab.subplots(figsize=(12, 8))
        t_fig, t_ax = pylab.subplots(figsize=(12, 8))
        ICP_figs, ICP_axs = pylab.subplots(figsize=(12, 8))
        
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
        V_ax.plot(times[initPlot:endPlot], df["dV_SAS"][initPlot:endPlot] + df["dV_VEN"][initPlot:endPlot], markers[1], color="darkmagenta",label="$(u \cdot n)ds_{SAS}$")
        V_ax.set_xlabel("time (s)")
        V_ax.set_xticks(x_ticks)
        V_ax.set_ylabel("V (mm$^3$)")
        V_ax.grid(True)
        V_ax.legend()
        V_fig.savefig(plotDir + "brain-Vol.png")


        Vven_ax.plot(times[initPlot:endPlot], df["dV_VEN"][initPlot:endPlot], markers[2], color="royalblue",label="$(u \cdot n)ds_{VEN}$")
        Vven_ax.set_xlabel("time (s)")
        Vven_ax.set_xticks(x_ticks)
        Vven_ax.set_ylabel("Volume ventricles (mm$^3$)")
        Vven_ax.grid(True)
        Vven_ax.legend()
        Vven_fig.savefig(plotDir + "brain-Vol_ven.png")

        Q_A = df["G_a"][initPlot:endPlot] - df["TOTAL_OUTFLOW_N1"][initPlot:endPlot]
        Q_V = df["TOTAL_OUTFLOW_N2"][initPlot:endPlot]

        scaleBV = np.mean(Q_A)/np.mean(Q_V) #Scales vein outflow to conserve BV over each CC 
        
        Q_CSF = df["TOTAL_OUTFLOW_N3"][initPlot:endPlot] - df["Q_AQ"][initPlot:endPlot]*1000 
        dVsdt = df["Vs_dot"][initPlot:endPlot] 
        Q_BV = Q_A - Q_V*scaleBV #Net BV going into the cranium 
        

        ICV =  Q_CSF + dVsdt - Q_BV

        ICV_ax.plot(times[initPlot:endPlot], df["dV_VEN"][initPlot:endPlot], markers[2], color="royalblue",label="$(u \cdot n)ds_{VEN}$")
        ICV_ax.set_xlabel("time (s)")
        ICV_ax.set_xticks(x_ticks)
        ICV_ax.set_ylabel("Intracranial volume (mm$^3$)")
        ICV_ax.grid(True)
        ICV_ax.legend()
        ICV_fig.savefig(plotDir + "brain-ICV.png")
        

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

        OUTFLOW_axs.plot(times[initPlot:endPlot], df["TOTAL_OUTFLOW_N2"][initPlot:endPlot], markers[0],
                       color=colors[i-1], label="$Q_2$",)

        OUTFLOW_axs.set_xlabel("time (s)")
        OUTFLOW_axs.set_xticks(x_ticks)
        OUTFLOW_axs.set_ylabel("Total outflow (mm$^3$/s)")
        OUTFLOW_axs.grid(True)
        OUTFLOW_axs.legend()



        pmax_figs.savefig(plotDir + "brain-p_max.png")
        pmin_figs.savefig(plotDir + "brain-p_min.png")
        pmean_figs.savefig(plotDir + "brain-p_avg.png")
        pmean_figs.savefig(plotDir + "brain-p_avg.png")
        OUTFLOW_figs.savefig(plotDir + "brain-outflow.png")


        
        # Plot average compartment velocity (avg v_i)
        for i in range(1,self.numPnetworks+1):
                q_ax.plot(times[initPlot:endPlot], df["q_avg_%d" % i][initPlot:endPlot], markers[0], color=colors[i-1],
                          label="$q_%d$" % i)
        q_ax.set_xlabel("time (s)")
        q_ax.set_xticks(x_ticks)
        q_ax.set_ylabel("Average bulk velocity $v$ (mm/s)")
        q_ax.grid(True)
        q_ax.legend()
        q_fig.savefig(plotDir + "brain-q_avg.png")

        for i in range(1,self.numPnetworks+1):
                v_ax.plot(times[initPlot:endPlot], df["v_avg_%d" % i][initPlot:endPlot], markers[0], color=colors[i-1],
                          label="$v_%d$" % i)
        v_ax.set_xlabel("time (s)")
        v_ax.set_xticks(x_ticks)
        v_ax.set_ylabel("Average norm?? velocity $v$ (mm/s)")
        v_ax.grid(True)
        v_ax.legend()
        v_fig.savefig(plotDir + "brain-v_avg.png")


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

        
        BA = df["G_a"] - df["TOTAL_OUTFLOW_N1"]
        BV_axs.plot(times[initPlot:endPlot], BA[initPlot:endPlot], markers[0], color="darkmagenta",label="$B_{a}$")
        if "TOTAL_OUTFLOW_N2" in df.keys():
            BV = df["TOTAL_OUTFLOW_N2"]
                
            # Plot outflow of venous blood
            BV_axs.plot(times[initPlot:endPlot], BV[initPlot:endPlot], markers[0], color="seagreen",label="$B_{v}$")

            BV_axs.plot(times[initPlot:endPlot], Q_V*scaleBV , markers[0], color="cornflowerblue",label="$B_{vC}$")


        BV_axs.set_xlabel("time (s)")
        BV_axs.set_xticks(x_ticks)
        BV_axs.set_ylabel("Absolute blood flow (mm$^3$/s)")
        BV_axs.grid(True)
        BV_axs.legend()
        BV_figs.savefig(plotDir + "brain-BV.png")


        # Plot Windkessel pressure
        if "p_SP" in df.keys():
            PW_axs.plot(times[initPlot:endPlot],  df["p_SP"][initPlot:endPlot], markers[1], color="cornflowerblue",label="$p_{SP}$")

        if "p_SAS" in df.keys():

            PW_axs.plot(times[initPlot:endPlot], df["p_SAS"][initPlot:endPlot], markers[0], color="seagreen",label="$p_{SAS}$")
        if "p_VEN" in df.keys():

            PW_axs.plot(times[initPlot:endPlot], df["p_VEN"][initPlot:endPlot], markers[1], color="darkmagenta",label="$p_{VEN}$")

            PW_axs.set_xlabel("time (s)")
            PW_axs.set_xticks(x_ticks)
            PW_axs.set_ylabel("Relative Pressure ($Pa$)")
            PW_axs.grid(True)
            PW_axs.legend()
            PW_figs.savefig(plotDir + "brain-WK.png")

        if "ICP" in df.keys():
            ICP_axs.plot(times[initPlot:endPlot], df["ICP"][initPlot:endPlot], markers[1], color="darkmagenta",label = "$ICP_s$")
            ICP_axs.plot(times[initPlot:endPlot], df["ICP"][initPlot:endPlot] + df["p_VEN"][initPlot:endPlot], markers[1], color="seagreen" ,label = "$ICP_v$")

            ICP_axs.set_xlabel("time (s)")
            ICP_axs.set_xticks(x_ticks)
            ICP_axs.set_ylabel("ICP ($Pa$)")
            ICP_axs.grid(True)
            ICP_axs.legend()
            ICP_figs.savefig(plotDir + "brain-ICP.png")

        if "T12" in df.keys():
            # Plot transfer rates (avg v_i)
            t_ax.plot(times[initPlot:endPlot], df["T12"][initPlot:endPlot], markers[0], color="darkmagenta", label="$T_{12}$")
            t_ax.plot(times[initPlot:endPlot], df["T13"][initPlot:endPlot], markers[0], color="royalblue", label="$T_{13}$")
            if "T32" in df.keys():
                t_ax.plot(times[initPlot:endPlot], df["T32"][initPlot:endPlot], markers[0], color="seagreen", label="$T_{32}$")
            if self.v_transfer_term:
                t_ax.plot(times[initPlot:endPlot], df["Tv"][initPlot:endPlot], markers[0], color="seagreen", label="$T_{vein}$")
            t_ax.set_xlabel("time (s)")
            t_ax.set_xticks(x_ticks)
            t_ax.set_ylabel("Transfer rate (mm$^3$/s)")
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
        Vds = Measure("ds", domain=V.mesh(), subdomain_data=self.boundary_markers)

        A = np.sqrt(self.Vol)
        for (i, p) in enumerate(p_list):
            results["max_p_%d" % (i)] = p.vector().max()
            print("Max pressure in network {}".format(i))
            print(results["max_p_%d" % (i)])
            results["min_p_%d" % (i)] = p.vector().min()
            print("Min pressure in network {}".format(i))
            print(results["min_p_%d" % (i)])
            results["mean_p_%d" % (i)] = assemble(p*dx)/self.Vol
            
            if i > 0: # Darcy velocities
                v = project(-self.K[i-1]*grad(p),
                            V,
                            solver_type = self.solverType,
                            preconditioner_type = self.preconditioner,
                            )
                v_avg = norm(v, "L2")/A
                results["v_avg_%d" % (i)] = v_avg
                #results["max_v_%d" % (i)] = norm(v,'linf')
                #results["min_v_%d" % (i)] = v.vector().min()
                
                results["q_avg_%d" % (i)] = assemble((v[0]**2+v[1]**2+v[2]**2)**(1/2)*dx)/self.Vol 
                
                results["TOTAL_OUTFLOW_N%d" %(i)] = assemble(div(v)*dx(V.mesh()))
                if self.v_transfer_term and i == 2:
                    p_vein = Constant(1117)
                    gamma_v = Constant(9.77*10**(-5))
                    results["Tv"] = assemble(gamma_v*(p_list[2]-p_vein)*dx)
                    results["TOTAL_OUTFLOW_N2"] = results["TOTAL_OUTFLOW_N2"] + results["Tv"] 
                print("Total outflow in network {}".format(i))
                print(results["TOTAL_OUTFLOW_N%d" %(i)])
                
                #Calculate outflow for each network
                results["Q_SAS_N%d" %(i)] = assemble(-self.K[i-1]*dot(grad(p),self.n) * self.ds(1))
                print("Total outflow in network {} from SAS ".format(i))
                print(results["Q_SAS_N%d" % (i)])

                results["Q_VEN_N%d" %(i)] = assemble(-self.K[i-1]*dot(grad(p),self.n) * self.ds(2)) + assemble(-self.K[i-1]*dot(grad(p),self.n) * self.ds(3))
                print("Total outflow in network {} from ventricles".format(i))
                print(results["Q_VEN_N%d" % (i)])



        results["G_a"] = self.m
        results["dV_SAS"] = assemble(dot(u,self.n)*self.ds(1))
        results["dV_VEN"] = assemble(dot(u,self.n)*(self.ds(2) + self.ds(3)))
        

        results["T12"] = assemble(self.gamma[0,1]*(p_list[1]-p_list[2])*dx)
        results["T13"] = assemble(self.gamma[0,2]*(p_list[1]-p_list[3])*dx)
        results["T21"] = assemble(self.gamma[1,0]*(p_list[1]-p_list[2])*dx)
        results["T31"] = assemble(self.gamma[2,0]*(p_list[1]-p_list[3])*dx)

        if self.gamma[1,2] > 0.0:
            results["T23"] = assemble(self.gamma[1,2]*(p_list[2]-p_list[3])*dx)
            results["T32"] = assemble(self.gamma[2,1]*(p_list[3]-p_list[2])*dx)
        if self.v_transfer_term:
            p_vein = Constant(1117)
            gamma_v = Constant(9.77*10**(-5))
            results["Tv"] = assemble(gamma_v*(p_list[2]-p_vein)*dx)
            results["TOTAL_OUTFLOW_N2"] = results["TOTAL_OUTFLOW_N2"] + results["Tv"] 
            
        return results
 
    def generate_diagnosticsPETSc(self,u,p):
        results = {}
        
        # Volume displacement:
        dV = assemble(div(u)*dx)
        results["dV"] = dV
        print("div(u)*dx (mm^3) = ", dV)
        
        V = VectorFunctionSpace(self.mesh, 'CG' ,1)
        Vds = Measure("ds", domain=V.mesh(), subdomain_data=self.boundary_markers)

        ps_v = p.split(deepcopy=True)
        
        Vol = assemble(1*dx(self.mesh))
        A = np.sqrt(Vol)

        for i in range(len(ps_v)):
            results["max_p_%d" % (i)] = ps_v[i].vector().max()
            print("Max pressure in network {}".format(i))
            print(results["max_p_%d" % (i)])
            results["min_p_%d" % (i)] = ps_v[i].vector().min()
            print("Min pressure in network {}".format(i))
            print(results["min_p_%d" % (i)])
            results["mean_p_%d" % (i)] = assemble(ps_v[i]*dx)/Vol
            
            if i > 0: # Darcy velocities
                v = project(self.K[i-1]*grad(ps_v[i]),
                            V,
                            #solver_type = self.solverType,
                            #preconditioner_type = self.preconditioner,
                            )
                v_avg = norm(v, "L2")/A
                results["v%d_avg" % (i)] = v_avg

                results["q_avg_%d" % (i)] = assemble((v[0]**2+v[1]**2+v[2]**2)**(1/2)*dx)/self.Vol
                
                results["TOTAL_OUTFLOW_N%d" %(i)] = assemble(div(v)*dx(V.mesh()))
                print("Total outflow in network {}".format(i))
                print(results["TOTAL_OUTFLOW_N%d" %(i)])
                
                #Calculate outflow for each network
                results["Q_SAS_N%d" %(i)] = assemble(-self.K[i-1]*dot(grad(p),self.n) * self.ds(1))
                print("Total outflow in network {} from SAS ".format(i))
                print(results["Q_SAS_N%d" % (i)])

                results["Q_VEN_N%d" %(i)] = assemble(-self.K[i-1]*dot(grad(p),self.n) * self.ds(2)) + assemble(-self.K[i-1]*dot(grad(p),self.n) * self.ds(3))
                print("Total outflow in network {} from ventricles".format(i))
                print(results["Q_VEN_N%d" % (i)])


        results["G_a"] = self.m
        results["dV_SAS"] = assemble(dot(u,self.n)*self.ds(1))
        results["dV_VEN"] = assemble(dot(u,self.n)*(self.ds(2) + self.ds(3)))
        

        # Transfer rates
        S = FunctionSpace(self.mesh, "CG", 1)
        t12 = project(self.gamma[0,1]*(ps_v[1]-ps_v[2]),
                        S,
                        #solver_type = self.solverType,
                        #preconditioner_type = self.preconditioner,
                        )
        
        t13 = project(self.gamma[0,2]*(ps_v[1]-ps_v[3]),
                                      S,
                        #solver_type = self.solverType,
                        #preconditioner_type = self.preconditioner,
                        )
        results["T12"] = norm(t12,"L2")
        results["T13"] = norm(t13,"L2")

        results["G_a"] = self.m
        results["dV_SAS"] = assemble(dot(u,self.n)*self.ds(1))
        results["dV_VEN"] = assemble(dot(u,self.n)*(self.ds(2) + self.ds(3)))
        

        results["T12"] = assemble(self.gamma[0,1]*(ps_v[1]-ps_v[2])*dx)
        results["T13"] = assemble(self.gamma[0,2]*(ps_v[1]-ps_v[3])*dx)
        results["T21"] = assemble(self.gamma[1,0]*(ps_v[1]-ps_v[2])*dx)
        results["T31"] = assemble(self.gamma[2,0]*(ps_v[1]-ps_v[3])*dx)
        results["T23"] = assemble(self.gamma[1,2]*(ps_v[2]-ps_v[3])*dx)
        results["T32"] = assemble(self.gamma[2,1]*(ps_v[3]-ps_v[2])*dx)
        
        return results
  

    def coupled_2P_model(self,dV_SAS_prev,dV_VEN_prev,results):
        """
        This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
        with Windkessel models.

        Solves using implicit (backward) Euler

        dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS))
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN + G_aq(p_SAS - p_VEN))
        
        """

        p_SAS = results["p_SAS"]
        p_VEN = results["p_VEN"]
        
        if (self.t[self.i] < 2.0):
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
        Vv_dot = 1/self.dt*(results["dV_VEN"]-dV_VEN_prev)
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-dV_SAS_prev)
        
        print("Volume change VEN[mm³] :",Vv_dot)
        print("Volume change SAS[mm³] :",Vs_dot)

        #Conductance
        G_aq = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant
        #G_aq = 5/133 #mL/mmHg to mL/Pa, from Ambarki2007
        G_aq = G_aq*1/1000 #mm³/Pa to mL/Pa

        b_SAS = p_SAS + self.dt/self.C_SAS * (Q_SAS + Vs_dot)* VolScale
        b_VEN = p_VEN + self.dt/self.C_VEN * (Q_VEN + Vv_dot)* VolScale

        A_11 = 1 + self.dt*G_aq/self.C_SAS
        A_12 = -self.dt*G_aq/self.C_SAS
        A_21 = -self.dt*G_aq/self.C_VEN
        A_22 = 1 + self.dt*G_aq/self.C_VEN
        
        
        b = np.array([b_SAS,b_VEN])
        A = np.array([[A_11, A_12],[A_21, A_22]])

        x = np.linalg.solve(A,b) #x_0 = p_SAS, x_1 = p_VEN

        print("Pressure for SAS: ", x[0])
        print("Pressure for VEN: ", x[1])

        # "Positive" direction upwards, same as baledent article
        Q_AQ = G_aq*(p_SAS - p_VEN)

        print("Q_AQ[mL]:",Q_AQ)


        print("Pressure for SAS: ", x[0])
        print("Pressure for ventricles: ", x[1])

        deltaV = (Q_SAS + Vs_dot + Q_AQ)*VolScale

        print("DeltaV: ",deltaV)

        if (self.t[self.i] <= 1.0):
            PVI = self.PVI*10 #mL   
        else: 
            PVI = self.PVI #mL

        ICP = self.ICP*10**(deltaV*self.dt/PVI)

        return x[0], x[1],Vv_dot ,Vs_dot ,Q_AQ, ICP

    def P2_nonlinear_model(self,dV_SAS_prev,dV_VEN_prev,results):
        """
        This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
        with Windkessel models.

        Solves using implicit (backward) Euler

        ICP = p0*10^((Vs_dot + Q_SAS + G_aq * p_VEN)*dt/PVI)
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN - G_aq * p_VEN)
        
        """

        p_VEN = results["p_VEN"]
        
        if (self.t[self.i] < 5.0):
            VolScale = 1/10000 #mm³ to mL   
        elif (self.t[self.i] >= 5.0 and self.t[self.i] <= 6.0): 
            VolScale = 1/(10000-9000*(self.t[self.i]-5.0))
        else:
            VolScale = 1/1000 #mm³ to mL

        print("Volume scale:",(int)(1/VolScale))
        #P_SAS is determined from Windkessel parameters
        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS[mm³] :",Q_SAS)

        #P_VEN is determined from volume change of the ventricles
        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN[mm³] :",Q_VEN)

        Q_CSF = results["TOTAL_OUTFLOW_N3"] #Outflow over boundary
        #Volume change of ventricles
        Vv_dot = 1/self.dt*(results["dV_VEN"]-dV_VEN_prev)
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-dV_SAS_prev)
        
        print("dV/dt VEN[mm³/s] :",Vv_dot)
        print("dV/dt SAS[mm³/s] :",Vs_dot)

        #Conductance
        G_aq = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant
        G_aq = G_aq*1/1000 #mm³/Pa to mL/Pa

        
        if (self.t[self.i] <= 3.0):
            PVI = self.PVI*2 #mL
        elif (self.t[self.i] > 3.0 and self.t[self.i] <= 4.0): 
            PVI = self.PVI*(2-1*(self.t[self.i]-3.0))
        else:
            PVI = self.PVI


        if (self.t[self.i] < 8.0):
            p_VEN = 0
        else:
            p_VEN = (VolScale*Vv_dot + self.C_VEN/self.dt*p_VEN)/(self.C_VEN/self.dt+G_aq)

        Q_AQ = -G_aq*p_VEN
        # "Positive" direction upwards, same as baledent article
        print("Q_AQ[mL]:",Q_AQ)

      
        if (self.t[self.i] < 2.0):
            ICP = self.ICP
            self.Q_CSF_T = 0.0
            self.Q_AQ_T = 0.0
            self.dVs = 0.0
        else:
            #if (self.i % 25 ==  0 ):
                #self.dVs = 0.0
            
            self.Q_CSF_T = self.Q_CSF_T + Q_CSF*self.dt  #Total outflow
            self.dVs = self.dVs + Vs_dot*self.dt
            if (self.t[self.i] > 8.0):
                self.Q_AQ_T = self.Q_AQ_T + Q_AQ*self.dt

            dVrel = True
            dPVI = 500
            if dVrel:
                dVs = ( self.dVs + self.Q_CSF_T + self.Q_AQ_T )*VolScale
                ICP = self.ICP*10**(dVs/PVI)
            else:
                dVs = (self.Q_CSF_T + self.Q_AQ_T )*VolScale
                ICP = self.ICP*10**(dVs/PVI + Vs_dot*VolScale/dPVI)

            print("dV[ml]: ",dVs)
            print("$Q_aq [mL]$",self.Q_AQ_T)
  


        print("Pressure for SAS: ", ICP)
        print("Pressure for VEN: ", ICP + p_VEN)

        return ICP, p_VEN,Vv_dot ,Vs_dot ,Q_AQ



    def P2_nonlinear_noncommunicating_model(self,dV_SAS_prev,dV_VEN_prev,results):
        """
        This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
        with Windkessel models.

        Solves using implicit (backward) Euler

        ICP = p0*10^((Vs_dot + Q_SAS + G_aq * p_VEN)*dt/PVI)
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN - G_aq * p_VEN)
        
        """

        p_VEN = results["p_VEN"]
        
        if (self.t[self.i] < 5.0):
            VolScale = 1/10000 #mm³ to mL   
        elif (self.t[self.i] >= 5.0 and self.t[self.i] <= 6.0): 
            VolScale = 1/(10000-9000*(self.t[self.i]-5.0))
        else:
            VolScale = 1/1000 #mm³ to mL

        print("Volume scale:",(int)(1/VolScale))
        #P_SAS is determined from Windkessel parameters
        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS[mm³] :",Q_SAS)

        #P_VEN is determined from volume change of the ventricles
        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN[mm³] :",Q_VEN)

        Q_CSF = results["TOTAL_OUTFLOW_N3"] #Outflow over boundary
        #Volume change of ventricles
        Vv_dot = 1/self.dt*(results["dV_VEN"]-dV_VEN_prev)
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-dV_SAS_prev)
        
        print("dV/dt VEN[mm³/s] :",Vv_dot)
        print("dV/dt SAS[mm³/s] :",Vs_dot)


        
        if (self.t[self.i] <= 3.0):
            PVI = self.PVI*2 #mL
        elif (self.t[self.i] > 3.0 and self.t[self.i] <= 4.0): 
            PVI = self.PVI*(2-1*(self.t[self.i]-3.0))
        else:
            PVI = self.PVI


        
        if (self.t[self.i] < 2.0):
            ICP = self.ICP
            self.Q_CSF_T = 0.0
            self.dVs = 0.0
        else:
            #if (self.i % 25 ==  0 ):
                #self.dVs = 0.0
                
            self.Q_CSF_T = self.Q_CSF_T + Q_SAS*self.dt  #Total outflow
            self.dVs = self.dVs + Vs_dot*self.dt
            
            if (self.t[self.i] < 8.0):
                self.dVv = 0
                self.Q_VEN_T = 0
            else:
                
                self.Q_VEN_T = self.Q_VEN_T + Q_VEN*self.dt  #Total outflow
                self.dVv = self.dVv + Vv_dot*self.dt
            
                PVIv = 50    
                dVv = ( self.dVv + self.Q_VEN_T)*VolScale
                p_VEN = self.ICP*(10**(dVv/PVIv)-1)
            
                print("dVv[ml]: ",dVv)
  
            dVrel = True
            dPVI = 500
            if dVrel:
                
                dVs = ( self.dVs + self.Q_CSF_T)*VolScale
                ICP = self.ICP*10**(dVs/PVI)
            else:
                dVs = (self.Q_CSF_T)*VolScale
                ICP = self.ICP*10**(dVs/PVI + Vs_dot*VolScale/dPVI)

            print("dV[ml]: ",dVs)
  


        print("Pressure for SAS: ", ICP)
        print("Pressure for VEN: ", ICP + p_VEN)

        return ICP, p_VEN,Vv_dot ,Vs_dot ,0.0

    def coupled_2P_nonnonlinear_model(self,dV_SAS_prev,dV_VEN_prev,results):
        """
        This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
        with Windkessel models.

        Solves using implicit (backward) Euler

        ICP = p0*10^((Vs_dot + Q_SAS + G_aq * p_VEN)*dt/PVI)
        p_VEN = pv(10^((Vv_dot + Q_VEN + G_aq * p_VEN)*dt/PVI_v)-1)
        
        """

        
        from scipy.optimize import fsolve

        
        if (self.t[self.i] < 2.0):
            VolScale = 1/10000 #mm³ to mL   
        elif (self.t[self.i] > 2.0 and self.t[self.i] <= 3.0): 
            VolScale = 1/(10000-9000*(self.t[self.i]-2.0))
        else:
            VolScale = 1/1000 #mm³ to mL

        p_VEN = results["p_VEN"]
        #P_SAS is determined from Windkessel parameters
        Q_SAS = results["Q_SAS_N3"]
        print("Q_SAS[mm³] :",Q_SAS)

        #P_VEN is determined from volume change of the ventricles
        Q_VEN = results["Q_VEN_N3"]
        print("Q_VEN[mm³] :",Q_VEN)

        #Volume change of ventricles
        Vv_dot = 1/self.dt*(results["dV_VEN"]-dV_VEN_prev)
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-dV_SAS_prev)
        
        print("Volume change VEN[mm³] :",Vv_dot)
        print("Volume change SAS[mm³] :",Vs_dot)

        #Conductance
        G_aq = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant
        #G_aq = 5/133 #mL/mmHg to mL/Pa, from Ambarki2007
        G_aq = G_aq*1/1000 #mm³/Pa to mL/Pa

        deltaVs = (Q_SAS + Vs_dot )*VolScale
        print("dVs/dt[ml/s]: ",deltaVs)

        deltaVv = (Q_VEN + Vv_dot )*VolScale
        print("dVv/dt[ml/s]: ",deltaVv)
 
        if (self.t[self.i] <= 3.0):
            PVI = self.PVI*10 #mL
        #elif (self.t[self.i] > 3.0 and self.t[self.i] <= 4.0): 
        #    PVI = self.PVI*10*(1-(self.t[self.i]-3.0))
        else:
            PVI = self.PVI*10
        
        
        if (self.i % 25 ==  0 ):
                self.ref_volume_sas = results["dV_SAS"] 
                self.ref_volume_ven = results["dV_VEN"]
                print("Reference volume for SAS: ", self.ref_volume_sas)
                print("Reference volume for VEN: ", self.ref_volume_ven)

        ICP_v = None

        if (self.t[self.i] < 1.0):
            ICP = 1330
            self.Q_SAS_T = 0.0
            self.Q_VEN_T = 0.0
            self.Q_AQ_T = 0.0
        else:
            self.Q_SAS_T = self.Q_SAS_T + Q_SAS*self.dt #Total outflow
            self.Q_VEN_T = self.Q_VEN_T + Q_VEN*self.dt #Total outflow

            
            dVols = results["dV_SAS"] - self.ref_volume_sas 
            dVolv = results["dV_VEN"] - self.ref_volume_ven 

            dVs = ( dVols + self.Q_SAS_T+ self.Q_VEN_T - self.Q_AQ_T)*VolScale
            dVv = ( dVolv + self.Q_AQ_T)*VolScale
            print("dVs[ml]: ",dVs)
            print("dVv[ml]: ",dVv)
            dVv_dot = VolScale*(Vv_dot + Q_VEN)
            
            self.ICP = 1330
            def funclog(p):
                ICP_s,ICP_v = p
                return [ICP_s - self.ICP*10**((dVs + G_aq*(ICP_v - ICP_s))/PVI),
                        ICP_v - ICP_s - self.ICP*10**((dVv + G_aq*(ICP_s - ICP_v))/(PVI*100))]
            def funclinV1(p):
                ICP,p_VEN = p 
                return [ICP - self.ICP*10**((dVs + G_aq*p_VEN)/PVI),
                        p_VEN - (dVv_dot + 0.4343*PVI/ICP/self.dt*p_VEN)/(0.4343*PVI/ICP/self.dt+G_aq)]

            def funclinV2(p):
                ICP,p_VEN = p               
                return [ICP - self.ICP*10**((dVs + G_aq*p_VEN)/PVI),
                        p_VEN - (dVv_dot + 0.4343*PVI/(ICP + p_VEN)/self.dt*p_VEN)/(0.4343*PVI/(ICP + p_VEN)/self.dt+G_aq)]

            if (self.t[self.i] < 5.0):
                ICP = self.ICP*10**(dVs/PVI)
                ICP_v = ICP
            else:
                ICP = self.ICP*10**(dVs/PVI)
                ICP_v = ICP + self.ICP*(10**(dVv/(PVI))-1)
                #ICP, ICP_v = fsolve(funclog, [self.ICP, self.ICP])

            #ICP, p_VEN = fsolve(funclinV1, [self.ICP, 5])
            #ICP, p_VEN = fsolve(funclinV2, [self.ICP, 0])


        if ICP_v == None:
            Q_AQ = -G_aq*p_VEN
        else:
            Q_AQ = G_aq*(ICP - ICP_v)
            p_VEN = ICP_v - ICP
        self.Q_VEN_T = self.Q_VEN_T
        self.Q_SAS_T = self.Q_SAS_T
        self.Q_AQ_T =self.Q_AQ_T + Q_AQ*1/1000 #previos

        # "Positive" direction upwards, same as baledent article
        print("Q_AQ[mL]:",Q_AQ)

        print("Pressure for SAS: ", ICP)
        print("Pressure for VEN: ", ICP + p_VEN)

        return ICP, p_VEN, Vv_dot, Vs_dot ,Q_AQ




    def coupled_3P_model(self,dV_SAS_prev,dV_VEN_prev,results):
        """
        This model calculates a 3-pressure lumped model for the SAS, ventricles and spinal-SAS compartments

        Solves using implicit (backward) Euler

        Equations:
        dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS) + G_fm(p_SP-p_SAS)
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEn + G_aq(p_SAS - p_VEN))
        dp_sp/dt = 1/C_sp(G_fm(p_SAS-p_SP))
        
        """
        p_SAS = results["p_SAS"]
        p_VEN = results["p_VEN"]
        p_SP = results["p_SP"]

        if (self.t[self.i] < 2.0):
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
        Vv_dot = 1/self.dt*(results["dV_VEN"]-dV_VEN_prev)
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-dV_SAS_prev)
        
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

        deltaV = (Q_SAS + Vs_dot + Q_AQ)*VolScale

        print("DeltaV: ",deltaV)

        if (self.t[self.i] < 1.0):
            PVI = self.PVI*10 #mL   
        else:
            PVI = self.PVI #mL

        ICP = 1330*10**(deltaV*self.dt/PVI)


        return x[0], x[1],x[2],Vv_dot,Vs_dot,Q_AQ,Q_FM,ICP


    def coupled_3P_nonlinear_model(self,dV_SAS_prev,dV_VEN_prev,results):
        """
        This model couples the two pressures between the ventricles and SAS through the aqueduct, both compartments are modeled
        with Windkessel models. The ICP is modelled by an exponential compliance

        Solves using implicit (backward) Euler

        dp_sas/dt = 1/C_sas(Vs_dot + Q_SAS + G_aq(p_VEN - p_SAS))
        dp_ven/dt = 1/C_ven(Vv_dot + Q_VEN + G_aq(p_SAS - p_VEN))

        ICP = p0*10^(dV*dt/PVI)
        
        """

        p_SAS = results["p_SAS"]
        p_VEN = results["p_VEN"]
        
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
        Vv_dot = 1/self.dt*(results["dV_VEN"]-dV_VEN_prev)
        
        #Volume change of SAS
        Vs_dot = 1/self.dt*(results["dV_SAS"]-dV_SAS_prev)
        
        print("Volume change VEN[mm³] :",Vv_dot)
        print("Volume change SAS[mm³] :",Vs_dot)

        #Conductance
        G_aq = np.pi*self.d**4/(128*self.L*self.mu_f[2]) #Poiseuille flow constant
        G_aq = G_aq*1/1000 #mm³/Pa to mL/Pa

        b_SAS = p_SAS #+ self.dt/self.C_SAS * (Q_SAS)* VolScale
        b_VEN = p_VEN + self.dt/self.C_VEN * (Q_VEN + Vv_dot)* VolScale

        A_11 = 1 + self.dt*G_aq/self.C_SAS
        A_12 = -self.dt*G_aq/self.C_SAS
        A_21 = -self.dt*G_aq/self.C_VEN
        A_22 = 1 + self.dt*G_aq/self.C_VEN
        
        
        b = np.array([b_SAS,b_VEN])
        A = np.array([[A_11, A_12],[A_21, A_22]])

        if (self.t[self.i] < 3.0):
            x = np.array([0.0,0.0])
        else:
            x = np.linalg.solve(A,b) #x_0 = p_SAS, x_1 = p_VEN

        # "Positive" direction upwards, same as baledent article
        Q_AQ = G_aq*(x[0] - x[1])

        print("Q_AQ[mL]:",Q_AQ)


        deltaV = (Vs_dot + Q_SAS)*VolScale

        print("DeltaV: ",deltaV)

        if (self.t[self.i] <= 1.0):
            PVI = self.PVI*10 #mL   
        else: 
            PVI = self.PVI

        ICP = self.ICP*10**(deltaV*self.dt/PVI)
        print("Absolute pressure in SAS: ", x[0] + ICP)
        print("Absolute pressure in ventricles: ", x[1] + ICP)


        return x[0], x[1],Vv_dot ,Vs_dot ,Q_AQ, ICP

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
        
        if self.dim == 2:
            self.f = Constant((0,0))
        elif self.dim == 3:
            self.f = Constant((0,0,0))
            
        self.mu_UFL = Constant(self.mu)
        self.Lambda_UFL  = Constant(self.Lambda)
            
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

        
        setupH = "\nSOLVER SETUP\n"

        setup = tabulate([['Problem Dimension', self.dim],
                          ['Number of networks', self.numPnetworks],
                          ['Number of boundaries', self.boundaryNum],
                          ['Solver type', self.solverType],
                          ['Total time',self.T],
                          ['Number of timesteps',self.numTsteps],
                          ['Element type',self.element_type],
                          ['Mesh type',self.mesh_type],
                          ['Volume of mesh', self.Vol]],
                         headers=['Setting', 'Value']
                        )
        print(setupH)
        print(setup)
        self.fileStats.write(setupH)
        self.fileStats.write('\n')
        self.fileStats.write(setup)
                    
        paramH = "\nDOMAIN PARAMETERS\n"
        param = tabulate([['Young Modulus', self.E, 'Pa'],
                        ['nu', self.nu, "--"],
                        ['lambda', self.Lambda, 'Pa'],
                        ['mu', self.mu, 'Pa'],
                        ['c', self.c_val, "1/Pa"],
                        ['kappa', self.kappa, "mm^2"],
                        ['mu_f', self.mu_f, 'Pa*s'],
                        ['alpha', self.alpha_val],
                        ['gamma', self.gamma, "1/Pa"]],
                       headers=['Parameter', 'Value', 'Unit'])

        print(paramH)
        print(param)
        self.fileStats.write('\n')
        self.fileStats.write(paramH)
        self.fileStats.write('\n')
        self.fileStats.write(param)
        self.fileStats.write('\n')
        

        bParamH = "\nBOUNDARY PARAMETERS\n"
        bParam = tabulate([['Compliance SAS', self.C_SAS, 'mm^3/mmHg'],
                        ['Compliance Ventricles', self.C_VEN, 'mm^3/mmHg'],
                        ['ICP baseline (p0)', self.ICP, 'Pa'],
                        ['PVI', self.PVI, 'mL'],
                        ['Venous back pressure', self.p_vein, 'Pa'],
                        ['beta SAS', self.beta_SAS, '--' ],
                        ['beta ventricles', self.beta_VEN, '--']],
                        headers=['Parameter', 'Value', 'Unit'])

        print(bParamH)
        print(bParam)
        self.fileStats.write('\n')
        self.fileStats.write(bParamH)
        self.fileStats.write('\n')
        self.fileStats.write(bParam)
        self.fileStats.write('\n')
        
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
                    if isinstance(expr[1],type(expr[1])):
                        expr[1].assign(Constant(expr[0][self.i]))
                    else:
                        expr[1].vector()[:] = expr[0][self.i]    
                    self.m = assemble(expr[1]*dx(self.mesh))                
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

        
        source_scale = 1/self.Vol #1/mm³

        source_scale = 1/1173887
        
        #self.g_mean = 13011.4*source_scale
        
        time_period = 1.0
        data = np.loadtxt(self.sourceFile, delimiter = ",")
        t = data[:,0]
        source = data[:,1]
        g = np.interp(self.t,t,source,period = 1.0)*source_scale


        #plt.plot(self.t,g)
        #plt.show()

        return g

    def get_system_fixed(self):
        '''MPET biot with 3 networks. Return system to be solved with PETSc'''
        # For simplicity we consider a stationary problem and displacement
        # and network pressures are fixed to 0 on the entire boundary
            
        mesh = self.mesh

        cell = mesh.ufl_cell()
        n = FacetNormal(mesh)

        Velm = VectorElement('Lagrange', cell, 2)

        Qi_elm = FiniteElement('Lagrange', cell, 1)  # For one pressure
        Qelm = MixedElement([Qi_elm]*4)

        Welm = MixedElement([Velm, Qelm]) 
        
        W = FunctionSpace(mesh, Welm)
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        print(W.dim(), "<<<<<", mesh.num_entities_global(mesh.topology().dim()))

        self.bcs_D = []
       
        p_VENOUS = self.boundary_conditionsP[(2, 1)]["Dirichlet"]
        p_SAS = Constant(self.p_BC_initial[0])
        p_VEN = Constant(self.p_BC_initial[1])
        
        self.bcs_D.extend([DirichletBC(W.sub(0), Constant((0, )*len(u)), self.boundary_markers, 2,)])

        ##P2
        self.bcs_D.extend([DirichletBC(W.sub(1).sub(2), p_VENOUS, self.boundary_markers, 1,)])
        self.bcs_D.extend([DirichletBC(W.sub(1).sub(2), p_VENOUS, self.boundary_markers, 2,)])
        #self.bcs_D.extend([DirichletBC(W.sub(1).sub(2), p_VENOUS, self.boundary_markers, 3,)])
        
        ##P3 BCs FOR PARAVASCULAR NETWORK
        self.bcs_D.extend([DirichletBC(W.sub(1).sub(3), p_SAS, self.boundary_markers, 1,)])
        self.bcs_D.extend([DirichletBC(W.sub(1).sub(3), p_VEN, self.boundary_markers, 2,)])
        #self.bcs_D.extend([DirichletBC(W.sub(1).sub(3), p_VEN, self.boundary_markers, 3,)])
        
        
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

        tau = Constant(self.dt)

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
                     tau*inner(Ks[j]*grad(ps[j]), grad(qs[j]))*dx +
                     (1/lmbda)*sum(inner(alphas[i]*ps[i], qs[j])*dx for i in range(nnets)) +
                    sum(tau*inner(Ts[j, i]*(ps[j] - ps[i]), qs[j])*dx for i in range(nnets) if i != j))

        beta_SAS, _ = self.boundary_conditionsP[(3, 1)]["RobinWK"]
        beta_VEN, _ = self.boundary_conditionsP[(3, 2)]["RobinWK"]
        
        #a = a 
        #- inner(tau*beta_SAS*ps[2],qs[2])*self.ds(1)
        #- inner(tau*beta_VEN*ps[2],qs[2])*self.ds(2)
        #- inner(tau*beta_VEN*ps[2],qs[2])*self.ds(3)


        # Now the preconditioner operator
        a_prec = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx  
                  + (1/lmbda + 1/(2*mu))*inner(pT, qT)*dx)

        # Add the eq tested with networks
        for j in range(nnets):
            a_prec =  a_prec + (1/lmbda)*inner(alphas[j]*qs[j], pT)*dx
            # The diagonal part
            a_prec = a_prec + (inner(cs[j]*ps[j], qs[j])*dx +
                               tau*inner(Ks[j]*grad(ps[j]), grad(qs[j]))*dx +
                               (1/lmbda)*sum(inner(alphas[i]*ps[i], qs[j])*dx for i in range(nnets)) +
                               sum(tau*inner(Ts[j, i]*(ps[j] - ps[i]), qs[j])*dx for i in range(nnets) if i != j))

      #  a_prec = a_prec
      #  + inner(tau*beta_SAS*ps[2],qs[2])*self.ds(1)
      #  + inner(tau*beta_VEN*ps[2],qs[2])*self.ds(2)
      #  + inner(tau*beta_VEN*ps[2],qs[2])*self.ds(3)

        #Initial RHS 
        g = Constant(self.g[0][0]) #Source term

 
        L = (
            inner(-n*p_SAS,v)*self.ds(1) 
        + inner(-n*p_VEN,v)*self.ds(2)  
        - inner(tau*g,qs[0])*dx 
        #- inner(tau*beta_SAS*p_SAS,qs[2])*self.ds(1)
        #- inner(tau*beta_VEN*p_VEN,qs[2])*self.ds(2)
        #- inner(tau*beta_VEN*p_VEN,qs[2])*self.ds(3)
        )
        
        ic = Constant(sum([(0, 0, 0),  
                           tuple(self.p_initial[net] for net in range(1+self.numPnetworks))], ()))
        print(ic.ufl_shape)
        wh_ = interpolate(ic, W)

        pT_, *ps_ = split(wh_.sub(1))       


        #Add terms from the previous timestep
        for j in range(nnets): 
            L = L - (1/lmbda)*inner(alphas[j]*qs[j], pT_)*dx
            # The diagonal part
            L = L - (inner(cs[j]*ps_[j], qs[j])*dx +
                     (1/lmbda)*sum(inner(alphas[i]*ps_[i], qs[j])*dx for i in range(nnets)))

        
        B, _ = assemble_system(a_prec, L, self.bcs_D)
        
        # For dealing with the need to reassemble b in a time loop we do
        # things a bit differently 
        assembler = SystemAssembler(a, L, self.bcs_D)

        return assembler, W, B, wh_, g, p_SAS, p_VEN



    def get_system_pureNeumann(self):
        '''MPET biot with 3 networks. Return system to be solved with PETSc'''
        # For simplicity we consider a stationary problem and displacement
        # and network pressures are fixed to 0 on the entire boundary

        mesh = self.mesh
        
        cell = mesh.ufl_cell()
        n = FacetNormal(mesh)

        Velm = VectorElement('Lagrange', cell, 2)

        Qi_elm = FiniteElement('Lagrange', cell, 1)  # For one pressure
        Qelm = MixedElement([Qi_elm]*4)

        if self.LM:
            Zelm = VectorElement('Real', cell, 0, dim=6)
            Welm = MixedElement([Velm, Qelm, Zelm]) 

            W = FunctionSpace(mesh, Welm)
            u, p, phis = TrialFunctions(W)
            v, q, psis = TestFunctions(W)
        else:
            Welm = MixedElement([Velm, Qelm]) 

            W = FunctionSpace(mesh, Welm)
            u, p = TrialFunctions(W)
            v, q = TestFunctions(W)

        print(W.dim(), "<<<<<", mesh.num_entities_global(mesh.topology().dim()))

        self.bcs_D = []
       
        p_VENOUS = self.boundary_conditionsP[(2, 1)]["Dirichlet"]

        self.bcs_D.extend([DirichletBC(W.sub(1).sub(2), p_VENOUS, self.boundary_markers, 1,)])
        #self.bcs_D.extend([DirichletBC(W.sub(1).sub(net), Constant(0), "on_boundary") for net in range(1,4)])
        
        
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

        self.dt = float(self.T/self.numTsteps)
        tau = Constant(self.dt)

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
                     tau*inner(Ks[j]*grad(ps[j]), grad(qs[j]))*dx +
                     (1/lmbda)*sum(inner(alphas[i]*ps[i], qs[j])*dx for i in range(nnets)) +
                    sum(tau*inner(Ts[j, i]*(ps[j] - ps[i]), qs[j])*dx for i in range(nnets) if i != j))

        if self.LM:
            # Orthogonality constraints
            basis = rigid_motions.rm_basis(mesh)
            for i, zi in enumerate(basis):
                a += phis[i]*inner(v, zi)*dx
                a += psis[i]*inner(u, zi)*dx


        # Now the preconditioner operator
        a_prec = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx  
                  + (1/lmbda + 1/(2*mu))*inner(pT, qT)*dx)
        if self.LM: #Add mass matrix
            a_prec = a_prec + inner(2*mu*u, v)*dx

        # Add the eq tested with networks
        for j in range(nnets):
            a_prec =  a_prec + (1/lmbda)*inner(alphas[j]*qs[j], pT)*dx
            # The diagonal part
            a_prec = a_prec + (inner(cs[j]*ps[j], qs[j])*dx +
                               tau*inner(Ks[j]*grad(ps[j]), grad(qs[j]))*dx +
                               (1/lmbda)*sum(inner(alphas[i]*ps[i], qs[j])*dx for i in range(nnets)) +
                               sum(tau*inner(Ts[j, i]*(ps[j] - ps[i]), qs[j])*dx for i in range(nnets) if i != j))

        if self.LM:
        
            for i, zi in enumerate(basis):
                a_prec += inner(phis[i]*zi, psis[i]*zi)*dx
                for j, zj in enumerate(basis[i+1:], i+1):
                    a_prec += inner(phis[i]*zi, psis[j]*zj)*dx
                    a_prec += inner(psis[i]*zi, phis[j]*zj)*dx            
            
        #Initial RHS 
        g = Constant(self.g[0][0]) #Source term

        p_SAS = Constant(self.p_BC_initial[0])
        p_VEN = Constant(self.p_BC_initial[1])

        L =  - tau*inner(g,qs[0])*dx + inner(-n*p_SAS,v)*self.ds(1) + inner(-n*p_VEN,v)*self.ds(2)

        ic = Constant(sum([(0, 0, 0),  
                           tuple(self.p_initial[net] for net in range(1+self.numPnetworks))], ()))

        wh_ = interpolate(ic, W)

        pT_, *ps_ = split(wh_.sub(1))       


        #Add terms from the previous timestep
        for j in range(nnets): 
            L = L - (1/lmbda)*inner(alphas[j]*qs[j], pT_)*dx
            # The diagonal part
            L = L - (inner(cs[j]*ps_[j], qs[j])*dx +
                     (1/lmbda)*sum(inner(alphas[i]*ps_[i], qs[j])*dx for i in range(nnets)))

        
        B, _ = assemble_system(a_prec, L, self.bcs_D)
        
        # For dealing with the need to reassemble b in a time loop we do
        # things a bit differently 
        assembler = SystemAssembler(a, L, self.bcs_D)

        return assembler, W, B, wh_, g, p_SAS, p_VEN


    def SolvePETSC(self):

        # Add progress bar
        progress = Progress("Time-stepping", self.numTsteps)
        set_log_level(LogLevel.PROGRESS)
       
        xdmfU = XDMFFile(self.filesave + "/FEM_results/u.xdmf")
        xdmfU.parameters["flush_output"]=True

        xdmfP = []
        
        for i in range(self.numPnetworks+1):
            xdmfP.append(XDMFFile(self.filesave + "/FEM_results/p" + str(i) + ".xdmf"))
            xdmfP[i].parameters["flush_output"]=True

        self.LM = False #True if using Lagrange Multipliers for RM removal

        assert self.uNullspace or self.uNullspace is False and self.LM is False, "Dont need LM if system is non-singular"

        if self.uNullspace:
            assembler, W, B, wh_, g, p_SAS, p_VEN = self.get_system_pureNeumann()
        else: 
            assembler, W, B, wh_, g, p_SAS, p_VEN = self.get_system_fixed()
        mesh = W.mesh()

        if self.uNullspace:
        
            # RM basis as expressions
            basis = rigid_motions.rm_basis(mesh)
            # we want to represent RM modes in W
            U = W.sub(0).collapse()

            basis_W = []
            for z in basis:
                zU = interpolate(z, U)  # Get them in the displacement space
                zW = Function(W)        # In W the modes that the form (z, 0)
                assign(zW.sub(0), zU)   # where 0 is for all the pressure spaces
                basis_W.append(zW.vector())

            # Nullspace, here we want l^2 orthogonality
            basis_W = rigid_motions.orthogonalize_gs(basis_W, A=None)

            # Remove RM from the RHS
            Z = VectorSpaceBasis(basis_W)

        comm = W.mesh().mpi_comm()
        # Before the time loop we can get the A once and for all
        A = PETScMatrix(comm)
        assembler.assemble(A)

        b = PETScVector(comm)
        # In the time loop suppose we do changes that modify bc values etc
        # so we need to reassemble
        assembler.assemble(b)
        
        if self.uNullspace and not self.LM:
            # Orthogonalize the newly filled vector
            Z.orthogonalize(b)
            

        solver = PETScKrylovSolver()
        solver.parameters['error_on_nonconvergence'] = True
        ksp = solver.ksp()

        if self.uNullspace:
            # Attach the nullspace to the system operators
            print(self.T)
            # NOTE: I put it also to the preconditioner in cases it might help
            # AMG later
            Z_ = PETSc.NullSpace().create([as_backend_type(z).vec() for z in basis_W])
            A_, B_ = (as_backend_type(mat).mat() for mat in (A, B))
            [mat.setNearNullSpace(Z_) for mat in (A_, B_)]
 
        solver.set_operators(A, B)

        OptDB = PETSc.Options()    
        OptDB.setValue('ksp_type', 'minres')
        OptDB.setValue('pc_type', 'fieldsplit')
        OptDB.setValue('pc_fieldsplit_type', 'additive')  # schur
        OptDB.setValue('pc_fieldsplit_schur_fact_type', 'diag')   # diag,lower,upper,full
        
        # Only apply preconditioner
        OptDB.setValue('fieldsplit_0_ksp_type', 'preonly')
        OptDB.setValue('fieldsplit_1_ksp_type', 'preonly')
        if self.LM:
            OptDB.setValue('fieldsplit_2_ksp_type', 'preonly')
        
        # Set the splits
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        splits = tuple((str(i), PETSc.IS().createGeneral(W.sub(i).dofmap().dofs()))
                       for i in range(W.num_sub_spaces()))
        pc.setFieldSplitIS(*splits)
        
        
        if self.LM:
            assert len(splits) == 3
            OptDB.setValue('fieldsplit_0_pc_type', 'gamg')
            OptDB.setValue('fieldsplit_2_pc_type', 'jacobi')    
        else:
            assert len(splits) == 2
        
            if self.uConditioner == 'lu':
                OptDB.setValue('fieldsplit_0_pc_type', 'lu')
            elif self.uConditioner == 'hypre':
                OptDB.setValue('fieldsplit_0_pc_type', 'hypre')
                OptDB.setValue('fieldsplit_0_pc_hypre_boomeramg_strong_threshold', '0.65')
                OptDB.setValue('fieldsplit_0_pc_hypre_boomeramg_coarsen_type','HMIS')
                OptDB.setValue('fieldsplit_0_pc_hypre_boomeramg_interp_type','ext+i')
            else:
                return exit(1)
        
        
        OptDB.setValue('ksp_norm_type', 'preconditioned')
        # Some generics
        OptDB.setValue('ksp_view', None)
        
        OptDB.setValue('ksp_monitor_true_residual', None)    
        OptDB.setValue('ksp_converged_reason', None)
        # NOTE: minres does not support unpreconditioned
        OptDB.setValue('ksp_atol', 1E-08)
        OptDB.setValue('ksp_rtol', 1E-20)
        OptDB.setValue("ksp_max_it", 2500)
        OptDB.setValue('ksp_initial_guess_nonzero', '1')
        
        # Use them!
        ksp.setFromOptions()

        t = 0.0

        p_VEN_f = self.p_BC_initial[0]
        p_SAS_f = self.p_BC_initial[1]
        p_SP_f = self.p_BC_initial[2]
        
        wh = Function(W)
        
        self.m = assemble(g*dx(self.mesh))
        
        self.i = 0
        print(self.numTsteps)

        dV_PREV_SAS = 0.0
        dV_PREV_VEN = 0.0

                
        while t < self.T:
            
            starttime = timeit.default_timer()
            solver.solve(wh.vector(), b)
            print("System solved in {} seconds".format(timeit.default_timer() - starttime))

            if self.LM:
                u,p,phis = wh.split(deepcopy = True)
            else:
                u,p = wh.split(deepcopy = True)

            if self.uNullspace and not self.LM:
                for i in range(6):
                    print("Rigid motion, before:", assemble(inner(u, basis[i])*dx))
                    
                Z.orthogonalize(wh.vector())
                u,p = wh.split(deepcopy = True)
                
                for i in range(6):
                    print("Rigid motion, after:", assemble(inner(u, basis[i])*dx))
        
            xdmfU.write(u, t)
            for j in range(self.numPnetworks+1):
                xdmfP[j].write(p.sub(j), t)

            results = self.generate_diagnosticsPETSc(u,p)

            self.m = assemble(g*dx(self.mesh))
            results["total_inflow"] = float(self.m)

            results["p_SAS"] = p_SAS_f
            results["p_VEN"] = p_VEN_f
            results["p_SP"] = p_SP_f

            p_SAS_f, p_VEN_f,p_SP_f,Vv_dot,Vs_dot,Q_AQ,Q_FM = self.coupled_3P_model(dV_PREV_SAS,dV_PREV_VEN,results) 
            
            p_SAS.assign(Constant(p_SAS_f))
            p_VEN.assign(Constant(p_VEN_f))


            print("Pressure SAS:",p_SAS_f)
            print("Pressure VEN:",p_VEN_f)

            results["Q_AQ"] = Q_AQ
            results["Q_FM"] = Q_FM

            results["Vv_dot"] = Vv_dot
            results["Vs_dot"] = Vs_dot
            results["t"] = t
            
            dV_PREV_SAS = results["dV_SAS"]
            dV_PREV_VEN = results["dV_VEN"]

            pickle.dump(results, open("%s/data_set/qois_%d.pickle" % (self.filesave, self.i), "wb"))

            t += self.dt
            self.i = self.i + 1

            if self.i > self.numTsteps:
                break
            
            g.assign(Constant(self.g[0][self.i]))

            wh_.vector()[:] = wh.vector()[:] 

            progress += 1

            assembler.assemble(b)
            
            print("Norm l2 b:",norm(b, 'l2'))

            #if self.uNullspace and not self.LM:
            #    for i in range(6):
            #        print("Rigid motion, b:", assemble(inner(b, basis[i])*dx))
           
            
            if self.uNullspace and not self.LM:
                # Orthogonalize the newly filled vector
                Z.orthogonalize(b)

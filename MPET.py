from fenics import *
from mshr import *
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
        self.boundaryNum=3 #Number of boundaries
        self.filesave = kwargs.get("file_save")
        # Create simulation directory if not existing
        info("Simulation directory: %s" % self.filesave)
        if not os.path.isdir(self.filesave):
            info("Did not find %s/, creating directory" % self.filesave)
            os.mkdir(self.filesave)
    
        self.numPnetworks = kwargs.get("num_networks") 
        self.T = kwargs.get("T")
        self.numTsteps = kwargs.get("num_T_steps")
        self.E =  kwargs.get("E")
        self.nu = kwargs.get("nu")
        self.element_type = kwargs.get("element_type")
        self.f_val = kwargs.get("f")
        self.rho = kwargs.get("rho")

        #For each network
        self.mu_f = kwargs.get("mu_f")
        print("mu_f:",self.mu_f)
        self.kappa = kwargs.get("kappa")
        self.alpha_val = kwargs.get("alpha")
        self.c_val = kwargs.get("c")
        self.p_initial = kwargs.get("p_initial")
        p_initial0 =  sum([-x*y for x,y in zip(self.alpha_val,self.p_initial)])
        self.p_initial.insert(0,p_initial0)
        self.gamma = np.reshape(kwargs.get("gamma"),(self.numPnetworks,self.numPnetworks))
        print("gamma:",self.gamma)
        self.K_val = []

        

        
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
        self.g = [ReadSourceTerm(self.mesh,self.sourceFile,self.T,inflow = "Netto"),None,None]

        self.Lambda = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))        
        self.conversionP = 133.32 #Pressure conversion: mmHg to Pa
        self.dim = self.mesh.topology().dim()

        #Boundary parameters
        self.C_SAS =  kwargs.get("Compliance_sas")/self.conversionP # [microL/mmHg] to [mm^3/Pa]
        self.C_VEN =  kwargs.get("Compliance_ven")/self.conversionP # [microL/mmHg] to [mm^3/Pa]
        self.R =  kwargs.get("Resistance")*self.conversionP*60e-3 # [mmHg*min/mL] to [Pa*/mm^3]

        self.beta_VEN = kwargs.get("beta_ven")
        self.beta_SAS = kwargs.get("beta_sas")
        self.p_BC_initial = [kwargs.get("p_ven_initial"),kwargs.get("p_sas_initial")]


    def solve(self):
        """
        Called from init?

        """
        print("\nSetting up problem...\n")

        print("Generating UFL expressions\n")
        self.generateUFLexpressions()
        
        
        
        DIMCLASS = self.dim + self.numPnetworks +1 #For InitialConditions class
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


        p_VEN = self.p_BC_initial[0]
        p_SAS = self.p_BC_initial[1]


        print("P_VEN =,", p_VEN)
        print("P_SAS =,", p_SAS)

        
        # Generate function space
        V = VectorElement(self.element_type, self.mesh.ufl_cell(), 2, self.dim)  # Displacements

        Q_0 = FiniteElement(self.element_type, self.mesh.ufl_cell(), 1)  # Total pressure
        mixedElement = []
        mixedElement.append(V)
        mixedElement.append(Q_0)
        for i in range(  self.numPnetworks):
            Q = FiniteElement(self.element_type, self.mesh.ufl_cell(), 1)
            mixedElement.append(Q)

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
                
        def F(f, v):
            return dot(f, v) * dx(self.mesh)

        
        #Apply terms for each fluid network
        for i in range(self.numPnetworks):  # apply for each network
            if isinstance(
                self.g[i], dolfin.cpp.adaptivity.TimeSeries
            ):  # If the source term is a time series instead of a an expression
                print("Adding timeseries for source term")
                g_space = FunctionSpace(self.mesh, mixedElement[i + 2])
                g_i = Function(g_space)
                sources.append(F(g_i, q[i + 2]))  # Applying source term
                self.time_expr.append((self.g[i], g_i))
            elif self.g[i] is not None:
                print("Adding expression for source term")
                sources.append(F(self.g[i], q[i + 2]))  # Applying source term
                self.time_expr.append(self.g[i])

            innerProdP.append(a_p(self.K[i], p_[i + 2], q[i + 2]))  # Applying diffusive termlg

            # Applying time derivatives
            timeD_.append(d(p_,q,i)) #lhs
            timeD_n.append(d(p_n,q,i)) #rhs

                
        if self.gamma.any(): #Add transfer terms
            print("Adding transfer terms")
            for i in range(self.numPnetworks): 
                for j in range(self.numPnetworks):
                    if self.gamma[i,j]:
                        transfer.append(self.gamma[i,j]*(p_[i+2]-p_[j+2])*q[i+2]*dx)
        dotProdP = [c(alpha,p, q[1]) for alpha,p in zip(self.alpha, p_[1:])]



        u_init = InitialConditions(self.numPnetworks,self.dim,self.p_initial)
        up_n.interpolate(u_init)

        
        self.applyPressureBC(W,p_,q)
        self.applyDisplacementBC(W,q)
 
        lhs = (
            a_u(p_[0], q[0])
            + b(p_[1], q[0])
            + b(q[1], p_[0])
            - sum(dotProdP)
            + sum(innerProdP)
            + sum(transfer)
            + sum(self.integrals_R_L)
            + sum(timeD_)
        ) 

        rhs = (
            F(self.f, q[0])
            + sum(sources)
            + sum(timeD_n)
            + sum(self.integrals_N)
            + sum(self.integrals_R_R)
        )

        [self.time_expr.append(self.f[i]) for i in range(self.dim)]
        A = assemble(lhs)
        [bc.apply(A) for bc in self.bcs_D]

        up = Function(W)
        t = 0.0


        for i in range(0,self.numTsteps+1): #Time loop
            
            self.update_time_expr(t)# Update all time dependent terms
            
            b = assemble(rhs)
            for bc in self.bcs_D:
                #            update_t(bc, t)
                bc.apply(b)

            solve(A, up.vector(), b) #Solve system

            #Write solution at time t
            up_split = up.split(deepcopy = True)
            results = self.generate_diagnostics(*up_split)

            xdmfU.write(up.sub(0), t)
            xdmfP0.write(up.sub(1), t)
            for j in range(self.numPnetworks):
                xdmfP[j].write(up.sub(j+2), t)

                        
            p_SAS, p_VEN = self.windkessel_model(p_SAS,p_VEN,results) #calculates windkessel pressure @ t

            self.update_windkessel_expr(p_SAS,p_VEN) # Update all terms dependent on the windkessel pressures

            results["p_SAS"] = p_SAS
            results["p_VEN"] = p_VEN
            results["t"] = t
            pickle.dump(results, open("%s/data_set/qois_%d.pickle" % (self.filesave, i), "wb"))
            
            up_n.assign(up)
            progress += 1

            t += self.dt
         
        res = []
        res = split(up)
        u = project(res[0], W.sub(0).collapse())
        p = []

        self.u_sol = u
        self.p_sol = p

    def plotResults(self):

        plotDir = "%s/plots/" %self.filesave

        its = None
        dV_fig, dV_ax = pylab.subplots(figsize=(12, 8))
        PW_figs, PW_axs  = pylab.subplots(figsize=(12, 8)) #Pressure Windkessel
        Q_figs, Q_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF
        BV_figs, BV_axs  = pylab.subplots(figsize=(12, 8)) #Outflow venous blood
        p_figs, p_axs = pylab.subplots(figsize=(12, 8))
        v_fig, v_ax = pylab.subplots(figsize=(12, 8))
        t_fig, t_ax = pylab.subplots(figsize=(12, 8))
        
        # Color code the pressures: red, purple and blue
        colors = ["crimson", "navy", "cornflowerblue"]
        markers = [".-", ".-", ".-"]

        x_ticks = [0.5*i for i in range(int(self.T/0.5)+1)]

        print("x_ticks:",x_ticks)

        df = self.load_data()
        names = df.columns
        times = (df["t"].to_numpy())
        
         # Plot volume change
        dV_ax.plot(times, df["dV"], markers[0], color="seagreen",label="div(u)dx")
        dV_ax.plot(times, df["dV_SAS"], markers[1], color="darkmagenta",label="(u * n)ds_{SAS}")
        dV_ax.plot(times, df["dV_VEN"], markers[2], color="royalblue",label="(u * n)ds_{VEN}")
        dV_ax.set_xlabel("time (s)")
        dV_ax.set_xticks(x_ticks)
        dV_ax.set_ylabel("dV (mm$^3$)")
        dV_ax.grid(True)
        dV_ax.legend()
        dV_fig.savefig(plotDir + "brain-dV.png")

        
        # Plot max/min of the pressures
        for i in range(1,self.numPnetworks+1):
            p_axs.plot(times, df["max_p_%d" % i], markers[0],
                       color=colors[i-1], label="$p_%d$" % i,)

            p_axs.set_xlabel("time (s)")
            p_axs.set_xticks(x_ticks)
            p_axs.set_ylabel("$\max \, p$ (Pa)")
            p_axs.grid(True)
            p_axs.legend()
        p_figs.savefig(plotDir + "brain-ps.png")


        
        # Plot average compartment velocity (avg v_i)
        for i in range(1,self.numPnetworks+1):
                v_ax.plot(times, df["v%d_avg" % i], markers[0], color=colors[i-1],
                          label="$v_%d$" % i)
        v_ax.set_xlabel("time (s)")
        v_ax.set_xticks(x_ticks)
        v_ax.set_ylabel("Average velocity $v$ (mm/s)")
        v_ax.grid(True)
        v_ax.legend()
        v_fig.savefig(plotDir + "brain-vs.png")


        Q_SAS = df["Q_SAS_N1"] + df["Q_SAS_N3"] 
        Q_VEN = df["Q_VEN_N1"] + df["Q_VEN_N3"]


        # Plot outflow of CSF
        Q_axs.plot(times, Q_SAS, markers[0], color="seagreen",label="$Q_{SAS}$")
        Q_axs.plot(times, Q_VEN, markers[0], color="darkmagenta",label="$Q_{VEN}$")
        Q_axs.set_xlabel("time (s)")
        Q_axs.set_xticks(x_ticks)
        Q_axs.set_ylabel("Q (mm$^3$/s)")
        Q_axs.grid(True)
        Q_axs.legend()
        Q_figs.savefig(plotDir + "brain-Q.png")

        BV = df["Q_SAS_N2"] + df["Q_VEN_N2"] 
        
        # Plot outflow of venous blood
        BV_axs.plot(times, BV, markers[0], color="seagreen")
        BV_axs.set_xlabel("time (s)")
        BV_axs.set_xticks(x_ticks)
        BV_axs.set_ylabel("Venous outflow (mm$^3$/s)")
        BV_axs.grid(True)
        BV_figs.savefig(plotDir + "brain-BV.png")


        # Plot Windkessel pressure
        PW_axs.plot(times, df["p_SAS"], markers[0], color="seagreen",label="$p_{SAS}$")
        PW_axs.plot(times, df["p_VEN"], markers[1], color="darkmagenta",label="$p_{VEN}$")
        PW_axs.set_xlabel("time (s)")
        PW_axs.set_xticks(x_ticks)
        PW_axs.set_ylabel("P ($Pa$)")
        PW_axs.grid(True)
        PW_axs.legend()
        PW_figs.savefig(plotDir + "brain-WK.png")


        # Plot transfer rates (avg v_i)
        t_ax.plot(times, df["T12"], markers[0], color="darkmagenta", label="$T_{12}$")
        t_ax.plot(times, df["T13"], markers[0], color="royalblue", label="$T_{13}$")
        t_ax.set_xlabel("time (s)")
        t_ax.set_xticks(x_ticks)
        t_ax.set_ylabel("Transfer rate ($L^2$-norm)")
        t_ax.grid(True)
        t_ax.legend()
        t_fig.savefig(plotDir + "brain-Ts.png")
    
        pylab.show()
        
    def printResults():

        return 0


    def generate_diagnostics(self,*args):
        results = {}
        u = args[0]
        p_list = []
        for arg in args[1:]:
            p_list.append(arg)

        # Change of volume:
        dV = assemble(div(u)*dx)
        results["dV"] = dV
        print("div(u)*dx (mm^3) = ", dV)

        V = VectorFunctionSpace(self.mesh, "DG", 0)

        print("len p_list:",len(p_list))
        # Pressures
        for (i, p) in enumerate(p_list):
            print("max(p_%d) (Pa) = " % (i), max(p.vector()))
            print("min(p_%d) (Pa) = " % (i), min(p.vector()))
            results["max_p_%d" % (i)] = max(p.vector())
            results["min_p_%d" % (i)] = min(p.vector())

            A = np.sqrt(assemble(1*dx(self.mesh)))
            if i > 0: # Darcy velocities
                v = project(self.K[i-1]*grad(p), V)
                v_avg = norm(v, "L2")/A
                print("avg_v%d(mm/s) = " % (i), v_avg)
                results["v%d_avg" % (i)] = v_avg

                #Calculate outflow for each network
                results["Q_SAS_N%d" %(i)] = assemble(-self.K[i-1] * dot(grad(p), self.n) * self.ds(1))
                results["Q_VEN_N%d" %(i)] = assemble(-self.K[i-1] * dot(grad(p), self.n) * self.ds(2)) + assemble(
                -self.K[i-1] * dot(grad(p),self.n) * self.ds(3))
                print("Q_SAS_N%d:" %(i),results["Q_SAS_N%d" %(i)])
                print("Q_VEN_N%d:" %(i),results["Q_VEN_N%d" %(i)])

            
        

        results["dV_SAS"] = assemble(dot(u,self.n)*self.ds(1))
        results["dV_VEN"] = assemble(dot(u,self.n)*self.ds(2)) + assemble(dot(u,self.n)*self.ds(3))

        # Transfer rates
        S = FunctionSpace(self.mesh, "CG", 1)
        
        t12 = project(self.gamma[0,1]*(p_list[1]-p_list[2]),S)
        t13 = project(self.gamma[0,2]*(p_list[1]-p_list[3]),S)
        results["T12"] = norm(t12,"L2")
        results["T13"] = norm(t13,"L2")
        return results
 


    def windkessel_model(self,p_SAS,p_VEN,results):

        #Assume both flow from capillaries and ECS flow out to the CSF filled cavities
        Q_SAS = results["Q_SAS_N3"] #+ results["Q_SAS_N1"] + results["dV_SAS"]
        Q_VEN = results["Q_VEN_N3"] #+ results["Q_VEN_N1"] + results["dV_VEN"]
        
        print("Q_SAS:",Q_SAS)
        print("Q_VEN:",Q_VEN)
        
            
        p_SAS_next = 1 / self.C_SAS * (self.dt * Q_SAS + (self.C_SAS - self.dt / self.R) * p_SAS)
        p_VEN_next = 1 / self.C_VEN * (self.dt * Q_VEN + (self.C_VEN - self.dt / self.R) * p_VEN)

        print("Pressure for SAS: ", p_SAS_next)
        print("Pressure for VEN: ", p_VEN_next)

        return p_SAS_next, p_VEN_next

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
                        expr,
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
                    self.integrals_R_L.append(inner(beta * p_[i + 1], q[i + 1]) * self.ds(j))

                    print("Applying Robin RHS")
                    self.integrals_R_R.append(inner(beta * P_r, q[i + 1]) * self.ds(j))
                    self.windkessel_terms.append(P_r)
                elif "Neumann" in self.boundary_conditionsP[(i,j)]:
                    if self.boundary_conditionsP[(i,j)]["Neumann"] != 0:
                        print("Applying Neumann BC.")
                        N = self.boundary_conditionsP[(i,j)]["Neumann"]
                        self.integrals_N.append(inner(-self.n * N, q[i+1]) * self.ds(j))
                        self.time_expr.append(N)


    def applyDisplacementBC(self,W,q):
        # Defining boundary conditions for displacements
        for i in self.boundary_conditionsU:
            print("i = ", i)
            if "Dirichlet" in self.boundary_conditionsU[i]:
                print("Applying Dirichlet BC.")
                for j in range(self.dim): #For each dimension
                    exprU =self.boundary_conditionsU[i]["Dirichlet"][j]
                    self.bcs_D.append(
                        DirichletBC(
                            W.sub(0).sub(j),
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
                    self.integrals_N.append(inner(-self.n * N, q[0]) * self.ds(i))
                    self.time_expr.append(N)
            elif "NeumannWK" in self.boundary_conditionsU[i]:
                if self.boundary_conditionsU[i]["NeumannWK"] != 0:
                    print("Applying Neumann BC with windkessel term.")
                    N = self.boundary_conditionsU[i]["NeumannWK"]
                    self.integrals_N.append(inner(-self.n * N, q[0]) * self.ds(i))
                    self.windkessel_terms.append(N)

                 
    def generateUFLexpressions(self):
        import sympy as sym
            
        t = sym.symbols("t")

        fx = 0.0 #self.f_val  # force term y-direction
        fy = 0.0 #self.f_val  # force term y-
        #p_initial0 =  sum([-x*y for x,y in zip(self.alpha_val,self.p_initial_val)])

 
        variables = [
            self.mu,
            self.Lambda,
            fx,
            fy,
            #p_initial0,
        ]

        variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

        UFLvariables = [
            Expression(var, degree=1, t=0.0 ) for var in variables
        ]  # Generate ufl varibles
 

        (
            self.my_UFL,
            self.Lambda_UFL,
            fx_UFL,
            fy_UFL,
            #p_initial0_UFL,
        ) = UFLvariables

        self.f = as_vector((fx, fy))

        #self.p_initial = []
        self.alpha = []
        self.c = []
        self.K = []

        #For total pressure
        #self.p_initial.append(p_initial0_UFL)
        self.alpha.append(Constant(1.0)) 

        #For each network
        for i in range(self.numPnetworks):
            variables = [
                self.alpha_val[i],
                self.c_val[i],
                self.K_val[i],
                #self.p_initial_val[i]
            ]

            variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

            UFLvariables = [
                Expression(var, degree=1, t=0.0 ) for var in variables
            ]  # Generate ufl varibles
            
            (
                alpha_UFL,
                c_UFL,
                K_UFL,
                #p_initial_UFL,
            ) = UFLvariables

            self.c.append(c_UFL)
            self.K.append(K_UFL)
            self.alpha.append(alpha_UFL)
            #self.p_initial.append(p_initial_UFL)
        
         
 
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
                        ['rho', self.rho, "kg/m³"],
                        ['c', self.c_val, "1/Pa"],
                        ['kappa', self.kappa, 'mm^2'],
                        ['mu_f', self.mu_f],
                        ['alpha', self.alpha_val],
                        ['gamma', self.gamma, "1/Pa"]],
                       headers=['Parameter', 'Value', 'Unit']))


        print("\n BOUNDARY PARAMETERS\n")
        print(tabulate([['Compliance SAS', self.C_SAS, 'mm^3/Pa'],
                        ['Compliance ventricles', self.C_VEN, 'mm^3/Pa'],
                        ['Resistance', self.R, 'Pa/s/mm^3'],
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
                if isinstance(expr[0], dolfin.cpp.adaptivity.TimeSeries):
                    expr[0].retrieve(expr[1].vector(), t)
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
        directory = os.path.join("%s/data_set/" % self.filesave)
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

def ReadSourceTerm(mesh,sourceData,periods,inflow = "Netto"):
    import csv
    g = TimeSeries("source_term")
    source_scale = 1/1173670.5408281302

    if inflow == "Brutto":
        offset = 10000
    else:
        offset = 0.0
    Q = FunctionSpace(mesh,"CG",1)
    time_period = 0.0
    for i in range(int(periods)):
        #Read in the source term data
        with open(sourceData) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                   #print(f'Column names are {", ".join(row)}')
                   line_count += 1
                else:
                   #print(float(row[1]))
                   source = Constant((float(row[1])+offset)*source_scale) #Uniform source on the domain
                   source = project(source, Q)
                   g.store(source.vector(),float(row[0]) + i*time_period)
                   #print(f"\t timestep {row[0]} adding {row[1]} to the source.")
                   line_count += 1
            if i == 0:
                time_period = float(row[0])
                print("t =", time_period)
            print(f'Processed {line_count} lines.')
    return g


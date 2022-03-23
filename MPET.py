from fenics import *
from mshr import *
from rm_basis_L2 import rigid_motions
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
        #Number of boundaries

        if kwargs.get("num_boundaries"):
            self.boundaryNum=int(kwargs.get("num_boundaries"))
        else:
            self.boundaryNum=3

        print("Number of boundaries:",self.boundaryNum)
        self.filesave = kwargs.get("file_save")
        self.uNullspace = kwargs.get("uNullspace")

        # Create simulation director if not existing
        info("Simulation directory: %s" % self.filesave)
        if not os.path.isdir(self.filesave):
            info("Did not find %s/, creating directory" % self.filesave)
            os.mkdir(self.filesave)

            path.mkdir(parents=True,exist_ok=True)
            path = Path("/home/asmund/dev/MPET-modelling/%s/data_set/" %self.filesave)
            path.mkdir(parents=True,exist_ok=True)
            path = Path("/home/asmund/dev/MPET-modelling/%s/plots/" %self.filesave)
            path.mkdir(parents=True,exist_ok=True)
            path = Path("/home/asmund/dev/MPET-modelling/%s/FEM_results/" %self.filesave)
            path.mkdir(parents=True,exist_ok=True)
    
        self.numPnetworks = kwargs.get("num_networks") 
        self.T = kwargs.get("T")
        self.numTsteps = kwargs.get("num_T_steps")
        self.t = np.linspace(0,float(self.T),int(self.numTsteps)+1)
        self.E =  kwargs.get("E")
        self.nu = kwargs.get("nu")
        self.element_type = kwargs.get("element_type")
        self.f_val = kwargs.get("f")
        self.rho = kwargs.get("rho")

        #For each network
        self.mu_f = kwargs.get("mu_f")
        self.kappa = kwargs.get("kappa")
        self.alpha_val = kwargs.get("alpha")
        self.c_val = kwargs.get("c")
        self.p_initial = kwargs.get("p_initial")
        p_initial0 =  sum([-x*y for x,y in zip(self.alpha_val,self.p_initial)])
        self.p_initial.insert(0,p_initial0)
        self.gamma = np.reshape(kwargs.get("gamma"),(self.numPnetworks,self.numPnetworks))
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
        self.scaleMean = kwargs.get("scale_mean")
        self.g = [self.ReadSourceTerm(),None,None]
        
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

        #if kwargs.get("p_spine_initial"):

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
            
            Z = rigid_motions(self.mesh)
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

        sigmoid = "1/(1+exp(-t + 4))"
        self.RampSource = Expression(sigmoid,t=0.0,degree=2)
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
                g_space = FunctionSpace(self.mesh, "CG",1)
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
        self.applyDisplacementBC(W,q)
 
        #self.time_expr.append(self.RampSource)

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

        if self.uNullspace:
                lhs += sum(z[i]*inner(q[0], Z[i])*dx() for i in range(dimZ)) \
                    + sum(r[i]*inner(p_[0], Z[i])*dx() for i in range(dimZ))
        
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
        self.t = 0.0

 
        dV_PREV_SAS = 0.0
        dV_PREV_VEN = 0.0
      
        for i in range(0,self.numTsteps+1): #Time loop
            
            self.update_time_expr(self.t)# Update all time dependent terms
            self.RampSource.t = self.t
            b = assemble(rhs)
            for bc in self.bcs_D:
                #            update_t(bc, t)
                bc.apply(b)

            solve(A, up.vector(), b) #Solve system

            #Write solution at time t
            up_split = up.split(deepcopy = True)
            results = self.generate_diagnostics(*up_split)

            xdmfU.write(up.sub(0), self.t)
            xdmfP0.write(up.sub(1), self.t)
            for j in range(self.numPnetworks):
                xdmfP[j].write(up.sub(j+2), self.t)


            #For calculating volume change in Windkessel model
            results["dV_SAS_PREV"] = dV_PREV_SAS
            results["dV_VEN_PREV"] = dV_PREV_VEN

            results["total_inflow"] = float(self.m)
            
            #p_SAS_f, p_VEN_f,Vv_dot,Vs_dot,Q_AQ = self.coupled_2P_model(p_SAS_f,p_VEN_f,results) #calculates windkessel pressure @ t
            p_SAS_f, p_VEN_f,p_SP_f,Vv_dot,Vs_dot,Q_AQ,Q_FM = self.coupled_3P_model(p_SAS_f,p_VEN_f,p_SP_f,results) #calculates windkessel pressure @ t
            self.update_windkessel_expr(p_SAS_f,p_VEN_f) # Update all terms dependent on the windkessel pressures


            results["p_SAS"] = p_SAS_f
            results["p_VEN"] = p_VEN_f
            results["p_SP"] = p_SP_f

            results["Q_AQ"] = Q_AQ
            results["Q_FM"] = Q_FM
            results["Vv_dot"] = Vv_dot
            results["Vs_dot"] =  Vs_dot
            results["t"] = self.t

            dV_PREV_SAS = results["dV_SAS"]
            dV_PREV_VEN = results["dV_VEN"]

            pickle.dump(results, open("%s/data_set/qois_%d.pickle" % (self.filesave, i), "wb"))
            

            up_n.assign(up)
            progress += 1

            self.t += self.dt
         
        res = []
        res = split(up)
        u = project(res[0], W.sub(0).collapse())
        p = []
        
        self.u_sol = u
        self.p_sol = p

    def plotResults(self,plotCycle = 0.0):

        plotDir = "%s/plots/" %self.filesave
        initPlot = int(self.numTsteps/self.T*plotCycle) 

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

        x_ticks = [plotCycle + 0.5*i for i in range(int(self.T/0.5)+1-plotCycle*2)]
        
        print("x_ticks:",x_ticks)
        
        df = self.load_data()
        names = df.columns
        times = (df["t"].to_numpy())
         # Plot volume vs time
        V_ax.plot(times[initPlot:-1], df["dV"][initPlot:-1], markers[0], color="seagreen",label="div(u)dx")
        V_ax.plot(times[initPlot:-1], df["dV_SAS"][initPlot:-1], markers[1], color="darkmagenta",label="(u * n)ds_{SAS}")
        V_ax.plot(times[initPlot:-1], df["dV_VEN"][initPlot:-1], markers[2], color="royalblue",label="(u * n)ds_{VEN}")
        V_ax.set_xlabel("time (s)")
        V_ax.set_xticks(x_ticks)
        V_ax.set_ylabel("V (mm$^3$)")
        V_ax.grid(True)
        V_ax.legend()
        V_fig.savefig(plotDir + "brain-Vol.png")

        # Plot volume derivative
        dV_dot_ax.plot(times[initPlot:-1], df["Vv_dot"][initPlot:-1], markers[0], color="seagreen",label="$dV_{VEN}$/dt")
        dV_dot_ax.plot(times[initPlot:-1], df["Vs_dot"][initPlot:-1], markers[0], color="darkmagenta",label="$dV_{SAS}$/dt")
        
        dV_dot_ax.set_xlabel("time (s)")
        dV_dot_ax.set_xticks(x_ticks)
        dV_dot_ax.set_ylabel("V_dot (mm$^3$/s)")
        dV_dot_ax.grid(True)
        dV_dot_ax.legend()
        dV_dot_fig.savefig(plotDir + "brain-V_dot.png")

        
        # Plot max/min of the pressures
        for i in range(1,self.numPnetworks+1):
            pmax_axs.plot(times[initPlot:-1], df["max_p_%d" % i][initPlot:-1], markers[0],
                       color=colors[i-1], label="$p_%d$" % i,)

            pmax_axs.set_xlabel("time (s)")
            pmax_axs.set_xticks(x_ticks)
            pmax_axs.set_ylabel("$\max \, p$ (Pa)")
            pmax_axs.grid(True)
            pmax_axs.legend()

            pmin_axs.plot(times[initPlot:-1], df["min_p_%d" % i][initPlot:-1], markers[0],
                       color=colors[i-1], label="$p_%d$" % i,)

            pmin_axs.set_xlabel("time (s)")
            pmin_axs.set_xticks(x_ticks)
            pmin_axs.set_ylabel("$\min \, p$ (Pa)")
            pmin_axs.grid(True)
            pmin_axs.legend()

            pmean_axs.plot(times[initPlot:-1], df["mean_p_%d" % i][initPlot:-1], markers[0],
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
                v_ax.plot(times[initPlot:-1], df["v%d_avg" % i][initPlot:-1], markers[0], color=colors[i-1],
                          label="$v_%d$" % i)
        v_ax.set_xlabel("time (s)")
        v_ax.set_xticks(x_ticks)
        v_ax.set_ylabel("Average velocity $v$ (mm/s)")
        v_ax.grid(True)
        v_ax.legend()
        v_fig.savefig(plotDir + "brain-vs.png")


        Q_SAS =  df["Q_SAS_N3"] 
        Q_VEN =  df["Q_VEN_N3"]
        

        
        # Plot outflow of CSF
        Qs_axs.plot(times[initPlot:-1], Q_SAS[initPlot:-1], markers[0], color="seagreen",label="$Q_{SAS}$")

        Qs_axs.set_xlabel("time (s)")
        Qs_axs.set_xticks(x_ticks)
        Qs_axs.set_ylabel("Q (mm$^3$/s)")
        Qs_axs.grid(True)
        Qs_axs.legend()
        Qs_figs.savefig(plotDir + "brain-Q_sas.png")


        Qv_axs.plot(times[initPlot:-1], Q_VEN[initPlot:-1], markers[0], color="darkmagenta",label="$Q_{VEN}$")
        Qv_axs.set_xlabel("time (s)")
        Qv_axs.set_xticks(x_ticks)
        Qv_axs.set_ylabel("Q (mm$^3$/s)")
        Qv_axs.grid(True)
        Qv_axs.legend()
        Qv_figs.savefig(plotDir + "brain-Q_ven.png")

        

        BV = df["Q_SAS_N2"] + df["Q_VEN_N2"] 
        
        # Plot outflow of venous blood
        BV_axs.plot(times[initPlot:-1], BV[initPlot:-1], markers[0], color="seagreen")
        BV_axs.set_xlabel("time (s)")
        BV_axs.set_xticks(x_ticks)
        BV_axs.set_ylabel("Venous outflow (mm$^3$/s)")
        BV_axs.grid(True)
        BV_figs.savefig(plotDir + "brain-BV.png")


        # Plot Windkessel pressure
        PW_axs.plot(times[initPlot:-1], df["p_SAS"][initPlot:-1], markers[0], color="seagreen",label="$p_{SAS}$")
        PW_axs.plot(times[initPlot:-1], df["p_VEN"][initPlot:-1], markers[1], color="darkmagenta",label="$p_{VEN}$")

        if "p_SP" in df.keys():
            PW_axs.plot(times[initPlot:-1], df["p_SP"][initPlot:-1], markers[1], color="cornflowerblue",label="$p_{SP}$")


        PW_axs.set_xlabel("time (s)")
        PW_axs.set_xticks(x_ticks)
        PW_axs.set_ylabel("P ($mmHg$)")
        PW_axs.grid(True)
        PW_axs.legend()
        PW_figs.savefig(plotDir + "brain-WK.png")


        # Plot transfer rates (avg v_i)
        t_ax.plot(times[initPlot:-1], df["T12"][initPlot:-1], markers[0], color="darkmagenta", label="$T_{12}$")
        t_ax.plot(times[initPlot:-1], df["T13"][initPlot:-1], markers[0], color="royalblue", label="$T_{13}$")
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


  
            Qaq_axs.plot(times[initPlot:-1], Q_AQ[initPlot:-1], markers[0], color="darkmagenta",label="$Q_{AQ}$")
            Qaq_axs.set_xlabel("time (s)")
            Qaq_axs.set_xticks(x_ticks)
            Qaq_axs.set_ylabel("Q (mL/s)")
            Qaq_axs.grid(True)
            Qaq_axs.legend()
            Qaq_figs.savefig(plotDir + "brain-Q_aq.png")

        if "Q_FM" in df.keys():

            Qfm_figs, Qfm_axs  = pylab.subplots(figsize=(12, 8)) #Outflow CSF to SAS
            Q_FM = df["Q_FM"]


            Qfm_axs.plot(times[initPlot:-1], Q_FM[initPlot:-1], markers[0], color="darkmagenta",label="$Q_{FM}$")
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
        #print("div(u)*dx (mm^3) = ", dV)

        V = VectorFunctionSpace(self.mesh, "DG", 0)

        # Pressures
        for (i, p) in enumerate(p_list):
            results["max_p_%d" % (i)] = max(p.vector())
            results["min_p_%d" % (i)] = min(p.vector())
            results["mean_p_%d" % (i)] = np.mean(p.vector())
            
            A = np.sqrt(assemble(1*dx(self.mesh)))
            if i > 0: # Darcy velocities
                v = project(self.K[i-1]*grad(p), V)
                v_avg = norm(v, "L2")/A
                results["v%d_avg" % (i)] = v_avg

                #Calculate outflow for each network
                results["Q_SAS_N%d" %(i)] = assemble(-self.K[i-1] * dot(grad(p), self.n) * self.ds(1))
                results["Q_VEN_N%d" %(i)] = assemble(-self.K[i-1] * dot(grad(p), self.n) * self.ds(2)) + assemble(
                -self.K[i-1] * dot(grad(p),self.n) * self.ds(3))


        results["dV_SAS"] = assemble(dot(u,self.n)*self.ds(1))
        results["dV_VEN"] = assemble(dot(u,self.n)*(self.ds(2) + self.ds(3)))
        

        # Transfer rates
        S = FunctionSpace(self.mesh, "CG", 1)
        
        t12 = project(self.gamma[0,1]*(p_list[1]-p_list[2]),S)
        t13 = project(self.gamma[0,2]*(p_list[1]-p_list[3]),S)
        results["T12"] = norm(t12,"L2")
        results["T13"] = norm(t13,"L2")
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
        G_aq = 5/133 #mL/mmHg to mL/Pa, from Ambarki2007
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
                        self.integrals_N.append(inner(-self.n * N * self.Pscale, q[i+1]) * self.ds(j))
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
                    self.integrals_N.append(inner(-self.n * N * self.Pscale, q[0]) * self.ds(i))
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
            Expression(var, degree=2, t=0.0 ) for var in variables
        ]  # Generate ufl varibles
 

        (
            self.my_UFL,
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
                Expression(var, degree=1, t=0.0 ) for var in variables
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
                if isinstance(expr[0], dolfin.cpp.adaptivity.TimeSeries):
                    expr[0].retrieve(expr[1].vector(), t,interpolate=False)
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

    def ReadSourceTerm(self):
        FileName = self.sourceFile + "series"
        if os.path.exists(FileName + ".h5"):
            print("Removing old timeseries")
            os.remove(FileName + ".h5")
            
        g = TimeSeries(FileName)
        
        source_scale = 1/1173670.5408281302 #1/mm³
        
        Q = FunctionSpace(self.mesh,"CG",1)
        time_period = 1.0
        data = np.loadtxt(self.sourceFile, delimiter = ",")
        t = data[:,0]
        source = data[:,1]
        dataInterp = np.interp(self.t,t,source,period = 1.0)

        if self.scaleMean:
            dataInterp -= np.mean(dataInterp)
        source_fig, source_ax = pylab.subplots(figsize=(16, 8))
        source_ax.plot(self.t,dataInterp)

        for j,data in enumerate(dataInterp):
            source = Constant((float(data))*source_scale) #Uniform source on the domain
            source = project(source, Q)
            g.store(source.vector(),self.t[j])
        return g


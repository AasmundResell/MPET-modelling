from fenics import *
from mshr import *
import ufl
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


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
        
        self.numPnetworks = kwargs.get("num_networks") 
        self.T = kwargs.get("T")
        self.numTsteps = kwargs.get("num_T_steps")
        self.E =  kwargs.get("E")
        self.nu = kwargs.get("nu")
        self.element_type = kwargs.get("element_type")
        self.f_val = kwargs.get("f")
        self.rho = kwargs.get("rho")
        self.gamma = kwargs.get("gamma")
        print("gamma:",self.gamma)
        #For each network
        self.mu_f = kwargs.get("mu_f")
        print("mu_f:",self.mu_f)
        self.kappa = kwargs.get("kappa")
        self.alpha_val = kwargs.get("alpha")
        self.c_val = kwargs.get("c")
        self.p_initial_val = kwargs.get("p_initial")
        
        self.K_val = []

        #Ensure lists
        if not isinstance(self.alpha_val,list): self.alpha_val = [self.alpha_val]
        if not isinstance(self.c_val,list): self.c_val = [self.c_val]
        if not isinstance(self.p_initial_val,list): self.p_initial_val = [self.p_initial_val]
        if not isinstance(self.kappa,list): self.kappa= [self.kappa] 
        if not isinstance(self.mu_f,list): self.mu_f = [self.mu_f]

        for i in range(self.numPnetworks):
            self.kappa[i] = self.kappa[i]*1e6 #m² to mm²
            self.K_val.append(self.kappa[i]/self.mu_f[i])
            
        
        self.sourceFile = kwargs.get("source_file")
        self.g = [ReadSourceTerm(self.mesh,self.sourceFile,inflow = "Brutto"),None,None]

        self.Lambda = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))        
        self.conversionP = 133.32 #Pressure conversion: mmHg to Pa
        self.dim = self.mesh.topology().dim()

        #Boundary parameters
        self.C =  kwargs.get("Compliance")/self.conversionP # [microL/mmHg] to [mm^3/Pa]
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
        
        filesave = "solution_test"
        
        

        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        n = FacetNormal(self.mesh)  # normal vector on the boundary

        dt = self.T / self.numTsteps

        # Add progress bar
        progress = Progress("Time-stepping", self.numTsteps)
        set_log_level(LogLevel.PROGRESS)

        xdmfU = XDMFFile(filesave + "/u.xdmf")
        xdmfP0 = XDMFFile(filesave + "/p0.xdmf")
        xdmfP = []
        for i in range(self.numPnetworks):
            xdmfP.append(XDMFFile(filesave + "/p" + str(i+1) + ".xdmf"))        

        p_VEN = self.p_BC_initial[0]
        p_SAS = self.p_BC_initial[1]


        print("P_VEN =,", p_VEN)
        print("P_SAS =,", p_SAS)
        # Storing values for plotting
        t_vec = np.zeros(self.numTsteps)
        p_SAS_vec = np.zeros(self.numTsteps)
        p_VEN_vec = np.zeros(self.numTsteps)
        Q_SAS_vec = np.zeros(self.numTsteps)
        Q_VEN_vec = np.zeros(self.numTsteps)
        
        
        # Generate function space
        V = VectorElement(self.element_type, self.mesh.ufl_cell(), 2, self.dim)  # Displacements
        V_test = FunctionSpace(self.mesh, V)

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

            # Applying time derivative
            timeD_.append(
                (
                    (1 / dt)
                    * (
                        self.c[i] * p_[i + 2]
                        + self.alpha[i] / self.Lambda * sum(a * b for a, b in zip(self.alpha, p_[1:]))
                        )
                )
                * q[i + 2]
                * dx
            )
            timeD_n.append(
                (
                    (1 / dt)
                    * (
                        self.c[i] * p_n[i + 2]
                        + self.alpha[i] / self.Lambda * sum(a * b for a, b in zip(self.alpha, p_n[1:]))
                    )
                )
                * q[i + 2]
                * dx
            )

        #Add transfer terms
        for i in range(self.numPnetworks): 
            for j in range(self.numPnetworks):
                transfer.append(self.gamma[i][j]*(p_[i+2]-p_[j+2])*q[i+2])
    
        dotProdP = [c(alpha,p, q[1]) for alpha,p in zip(self.alpha, p_[1:])]


        for i in range(2, self.numPnetworks):
            # Apply inital pressure
            up_n.sub(i).assign(interpolate(self.p_initial[i - 1], W.sub(i).collapse()))
        up_n.sub(1).assign(
            interpolate(self.p_initial[0], W.sub(1).collapse())
        )  # Apply intial total pressure

        
        self.applyPressureBC(n,ds,p_,q)
        self.applyDisplacementBC(n,W,ds,q)

        
        

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
            F(self.f, q[0]) + sum(sources) + sum(timeD_n) + sum(self.integrals_N) + sum(self.integrals_R_R)
        )

        [self.time_expr.append(self.f[i]) for i in range(self.dim)]
        A = assemble(lhs)
        [bc.apply(A) for bc in self.bcs_D]

        up = Function(W)
        t = 0
        

        for i in range(self.numTsteps):
            t += dt
            for expr in self.time_expr:  # Update all time dependent terms
                self.update_t(expr, t)

            b = assemble(rhs)
            for bc in self.bcs_D:
                #            update_t(bc, t)
                bc.apply(b)

            solve(A, up.vector(), b)
            

            #K is chosen based on the network that is in direct contact with the CSF compartments
            Q_SAS = assemble(-self.K[0] * dot(grad(up.sub(2)), n) * ds(1))
            Q_VEN = assemble(-self.K[0] * dot(grad(up.sub(2)), n) * ds(2)) + assemble(
                -self.K[0] * dot(grad(up.sub(2)), n) * ds(3)
            )
            p_SAS_vec[i] = p_SAS
            p_VEN_vec[i] = p_VEN
            t_vec[i] = t
            Q_SAS_vec[i] = Q_SAS
            Q_VEN_vec[i] = Q_VEN
                
            print("Outflow SAS at t=%.2f is Q_SAS=%.10f " % (t, Q_SAS_vec[i]))
            print("Outflow VEN at t=%.2f is Q_VEN=%.10f " % (t, Q_VEN_vec[i]))
            p_SAS_next = 1 / self.C * (dt * Q_SAS + (self.C - dt / self.R) * p_SAS)
            p_VEN_next = 1 / self.C * (dt * Q_VEN + (self.C - dt / self.R) * p_VEN)
            
            print("Pressure for SAS: ", p_SAS_next)
            print("Pressure for VEN: ", p_VEN_next)
                
            for expr in self.windkessel_terms:
                try:
                    expr.p_SAS = p_SAS_next
                    expr.p_VEN = p_VEN_next
                except:
                    try:
                        expr.p_SAS = p_SAS_next
                    except:
                        expr.p_VEN = p_VEN_next
            p_SAS = p_SAS_next
            p_VEN = p_VEN_next

            xdmfU.write(up.sub(0), t)
            xdmfP0.write(up.sub(1), t)
            for i in range(self.numPnetworks):
                xdmfP[i].write(up.sub(i+2), t)

            up_n.assign(up)
            progress += 1

        
        res = []
        res = split(up)
        u = project(res[0], W.sub(0).collapse())
        p = []

        for i in range(1, self.numPnetworks + 2):
            p.append(project(res[i], W.sub(i).collapse()))

        fig1 = plt.figure(1)
        plt.plot(t_vec, p_SAS_vec)
        plt.title("Pressure in the subarachnoidal space")
        fig2 = plt.figure(2)
        plt.plot(t_vec, p_VEN_vec)
        plt.title("Pressure in the brain ventricles")
        fig3 = plt.figure(3)
        plt.plot(t_vec, Q_SAS_vec)
        plt.plot(t_vec, Q_VEN_vec)
        plt.title("Fluid outflow of the brain parenchyma")
        plt.show()

        self.u_sol = u
        self.p_sol = p


    def applyPressureBC(self,n,ds,p_,q):
        
        for i in range(self.numPnetworks):  # apply for each network
            print("Network: ", i)    
            for j in range(1, self.boundaryNum + 1):  # for each boundary
                print("Boundary: ", j)
                if "Dirichlet" in self.boundary_conditionsP[(i + 1, j)]:
                    print(
                        "Applying Dirichlet BC for pressure network: %i and boundary surface %i "
                        % (i + 1, j)
                    )
                    expr = self.boundary_conditionsP[(i + 1, j)]["Dirichlet"]
                    bcp = DirichletBC(
                        W.sub(i + 2),
                        expr,
                        self.boundary_markers,
                        j,
                    )
                    self.bcs_D.append(bcp)
                    self.time_expr.append(expr)
                elif "DirichletWK" in self.boundary_conditionsP[(i + 1, j)]:
                    print(
                        "Applying Dirichlet Windkessel BC for pressure network: %i and boundary surface %i "
                        % (i + 1, j)
                    )
                    expr = self.boundary_conditionsP[(i + 1, j)]["DirichletWK"]
                    bcp = DirichletBC(
                        W.sub(i + 2),
                        expr,
                        self.boundary_markers,
                        j,
                    )
                    self.bcs_D.append(bcp)
                    self.windkessel_terms.append(expr)
                elif "Robin" in self.boundary_conditionsP[(i + 1, j)]:
                    print("Applying Robin BC for pressure")
                    print("Applying Robin LHS")
                    beta, P_r = self.boundary_conditionsP[(i + 1, j)]["Robin"]
                    self.integrals_R_L.append(inner(beta * p_[i + 2], q[i + 2]) * ds(j))
                    if P_r:
                        print("Applying Robin RHS")
                        self.integrals_R_R.append(inner(beta * P_r, q[i + 2]) * ds(j))
                        self.time_expr.append(P_r)
                elif "RobinWK" in self.boundary_conditionsP[(i + 1, j)]:
                    print("Applying Robin BC with Windkessel referance pressure")
                    beta, P_r = self.boundary_conditionsP[(i + 1, j)]["RobinWK"]
                        
                    print("Applying Robin LHS")
                    self.integrals_R_L.append(inner(beta * p_[i + 2], q[i + 2]) * ds(j))

                    print("Applying Robin RHS")
                    self.integrals_R_R.append(inner(beta * P_r, q[i + 2]) * ds(j))
                    self.windkessel_terms.append(P_r)
                elif "Neumann" in self.boundary_conditionsP[(i+1,j)]:
                    if self.boundary_conditionsP[i]["Neumann"] != 0:
                        print("Applying Neumann BC.")
                        N = self.boundary_conditionsP[(i+1,j)]["Neumann"]
                        self.integrals_N.append(inner(-n * N, q[i+2]) * ds(j))
                        self.time_expr.append(N)


    def applyDisplacementBC(self,n,W,ds,q):
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
                    self.integrals_N.append(inner(-n * N, q[0]) * ds(i))
                    self.time_expr.append(N)
            elif "NeumannWK" in self.boundary_conditionsU[i]:
                if self.boundary_conditionsU[i]["NeumannWK"] != 0:
                    print("Applying Neumann BC with windkessel term.")
                    N = self.boundary_conditionsU[i]["NeumannWK"]
                    self.integrals_N.append(inner(-n * N, q[0]) * ds(i))
                    self.windkessel_terms.append(N)


    def generateUFLexpressions(self):
        import sympy as sym
            
        t = sym.symbols("t")

        fx = 0.0 #self.f_val  # force term y-direction
        fy = 0.0 #self.f_val  # force term y-
        p_initial0 =  sum([-x*y for x,y in zip(self.alpha_val,self.p_initial_val)])

 
        variables = [
            self.mu,
            self.Lambda,
            fx,
            fy,
            p_initial0,
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
            p_initial0_UFL,
        ) = UFLvariables

        self.f = as_vector((fx, fy))

        self.p_initial = []
        self.alpha = []
        self.c = []
        self.K = []

        #For total pressure
        self.p_initial.append(p_initial0_UFL)
        self.alpha.append(1) 

        #For each network
        for i in range(self.numPnetworks):
            variables = [
                self.alpha_val[i],
                self.c_val[i],
                self.K_val[i],
                self.p_initial_val[i]
            ]

            variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

            UFLvariables = [
                Expression(var, degree=1, t=0.0 ) for var in variables
            ]  # Generate ufl varibles
            
            (
                alpha_UFL,
                c_UFL,
                K_UFL,
                p_initial_UFL,
            ) = UFLvariables

            self.c.append(c_UFL)
            self.K.append(K_UFL)
            self.alpha.append(alpha_UFL)
            self.p_initial.append(p_initial_UFL)
        
         
 
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
        print(tabulate([['Compliance', self.C, 'mm^3/Pa'],
                        ['Resistance', self.R, 'Pa/s/mm^3'],
                        ['beta SAS', self.beta_SAS, '--' ],
                        ['beta ventricles', self.beta_VEN, '--']],
                        headers=['Parameter', 'Value', 'Unit']))



    def update_t(self,expr, t):
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
            

def ReadSourceTerm(mesh,filestr,periods = 3,inflow = "Netto"):
    import csv
    g = TimeSeries("source_term")
    source_scale = 1/1173670.5408281302

    if inflow == "Brutto":
        offset = 10000
    else:
        offset = 0.0
    Q = FunctionSpace(mesh,"CG",1)
    time_period = 0.0
    for i in range(periods):
        #Read in the source term data
        with open(filestr) as csv_file:
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


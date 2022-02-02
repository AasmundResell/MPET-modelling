from fenics import *
from mshr import *
import ufl
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


class MPET:
    def __init__(self,
                 nDim,
                 mesh,
                 boundary_markers,
                 boundary_conditionsU,
                 boundary_conditionsP,
                 **kwargs,
        ):
        self.dim = nDim 
        self.mesh = mesh        
        self.boundary_markers = boundary_markers
        self.boundary_conditionsU = boundary_conditionsU
        self.boundary_conditionsP = boundary_conditionsP


        self.n_networks = kwargs.get("num_networks") 
        self.T = kwargs.get("T")
        self.numTsteps = kwargs.get("num_T_steps")
        self.E =  kwargs.get("E")
        self.nu = kwargs.get("nu")
        self.c = kwargs.get("c")
        self.element_type = kwargs.get("element_type")
        self.f = kwargs.get("f")

        #For each network
        self.rho = kwargs.get("rho")
        self.mu = kwargs.get("mu")
        self.kappa = kwargs.get("kappa")
        self.alpha = kwargs.get("alpha")
        self.K = self.kappa/self.mu

        self.Lambda = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.my = self.E/(2*(1+self.nu))        
        self.conversionP = 133.32 #Pressure conversion: mmHg to Pa

        

    def generateUFLexpressions(self):
        import sympy as sym
            
        t = sym.symbols("t")
        
        p_VEN = sym.symbols("p_VEN")
        p_SAS = sym.symbols("p_SAS")
        fx = 0  # force term x-direction
        fy = 0  # force term y-direction
        p_initial1 = 0.0
        p_initial0 = -self.alpha*p_initial1
        pSkull = p_SAS
        pVentricles = p_VEN


        #TODO:SET IN CONFIG FILE 
        Compliance = 2.5e3/conversionP # [microL/mmHg] to [mm^3/Pa]
        Resistance = 10.81*conversionP*60e-3 # [mmHg*min/microL] to [Pa*s/mm^3]
        beta_VEN = 0.1
        beta_SAS = 0.1
        
        print("C=",Compliance)
        print("R=",Resistance)
    
        variables = [
            self.my,
            self.Lambda,
            self.alpha,
            self.c,
            self.K,
            fx,
            fy,
            pSkull,
            pVentricles,
            p_initial0,
            p_initial1,
        ]

        variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

        UFLvariables = [
            Expression(var, degree=2, t=0.0, p_VEN= 0.0,p_SAS = 0.0) for var in variables
        ]  # Generate ufl varibles

        (
            self.my_UFL,
            self.Lambda_UFL,
            alpha1_UFL,
            c1_UFL,
            K1_UFL,
            fx_UFL,
            fy_UFL,
            self.pSkull_UFL,
            self.pVentricles_UFL,
            p_initial0_UFL,
            p_initial1_UFL,
        
        ) = UFLvariables
        self.alpha_list = [1,alpha1_UFL]
        self.c_list = [1,alpha1_UFL]
        self.K_list = [1,alpha1_UFL]
        self.p_init = [p_initial0_UFL, p_initial1_UFL]
        
        self.f_UFL = as_vector((fx, fy))
        self.U_UFL = as_vector((Expression("0.0", degree=2), Expression("0.0", degree=2)))
        source_scale = 1/1173670.5408281302 
        self.g = [ReadSourceTerm(self.mesh,source_scale)]
 
    def printSetup(self):

        print("\n SOLVER SETUP\n")
        print(tabulate([['Problem Dimension', self.dim],
                        ['Number of networks', self.networks],
                        ['Total time',self.T],
                        ['Number of timesteps',self.numTsteps],
                        ['Element type',self.element_type]],

                       headers=['Setting', 'Value']))

    
        print("\n MATERIAL PARAMETERS\n")
        print(tabulate([['Young Modulus', self.E, 'Pa'],
                        ['nu', self.nu],
                        ['c', self.c],
                        ['kappa', self.kappa, 'mm^2'],
                        ['rho', self.rho],
                        ['mu', self.mu],
                        ['alpha', self.alpha]],
                       headers=['Parameter', 'Value', 'Unit']))

    def problemSetup(self):
        """
        Called from init?

        """
        print("Setting up problem...\n")

        print("Generating UFL expressions\n")
        self.generateUFLexpressions()
        boundaryNum=1,
        p_initial=None,
        filesave = "solution_test",


        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        self.n = FacetNormal(self.mesh)  # normal vector on the boundarylg
        if not transient:
            dt = 0
            numTsteps = 1
            progress =  0
        elif transient:  # Will need to be saved during calculation
            self.dt = self.T / self.numTsteps
            # Add progress bar
            progress = Progress("Time-stepping", numTsteps)
            set_log_level(LogLevel.PROGRESS)

        xdmfU = XDMFFile(filesave + "/u.xdmf")
        xdmfP0 = XDMFFile(filesave + "/p0.xdmf")
        xdmfP1 = XDMFFile(filesave + "/p1.xdmf")        
        
        if WindkesselBC:
            p_VEN = p_initial[1].p_VEN
            p_SAS = p_initial[1].p_SAS
            p_initial[1].p
            print("P_VEN =,", p_VEN)
            print("P_SAS =,", p_SAS)
            # Storing values for plotting
            t_vec = np.zeros(numTsteps)
            p_SAS_vec = np.zeros(numTsteps)
            p_VEN_vec = np.zeros(numTsteps)
            Q_SAS_vec = np.zeros(numTsteps)
            Q_VEN_vec = np.zeros(numTsteps)
            R = Resistance
            C = Compliance

        
        # Generate function space
        V = VectorElement("CG", mesh.ufl_cell(), 1, dim=mesh.topology().dim())  # Displacements
        V_test = FunctionSpace(mesh, V)

        Q_0 = FiniteElement("CG", mesh.ufl_cell(), 1)  # Total pressure
        mixedElement = []
        mixedElement.append(V)
        mixedElement.append(Q_0)
        for i in range(numPnetworks):
            Q = FiniteElement("CG", mesh.ufl_cell(), 1)
            mixedElement.append(Q)

        W_element = MixedElement(mixedElement)
        W = FunctionSpace(mesh, W_element)
        
        test = TestFunction(W)
        q = split(test)  # q[0] = v, q[1],q[2],... = q_0,q_1,...
        
        trial = TrialFunction(W)
        p_ = split(trial)  # p_[0] = u_, p_[1],p_[2],... = p_0,p_1,...
        
        up_n = Function(W)
        
        # variational formulation
        sources = []  # Contains the source term for each network
        innerProdP = []
        # Contains the inner product of the gradient of p_j for each network

        dotProdP = []  # Contains the dot product of alpha_j & p_j,
        timeD_ = []  # Time derivative for the current step
        timeD_n = []  # Time derivative for the previous step
        bcs_D = []  # Contains the terms for the Dirichlet boundaries
        integrals_N = []  # Contains the integrals for the Neumann boundaries
        time_expr = []  # Terms that needs to be updated at each timestep

        # Terms that contains the windkessel bc that is updated each timestep
        windkessel_terms = []

        # Contains the integrals for the Robin boundaries, LHS
        integrals_R_L = []
        # Contains the integrals for the Robin boundaries, RHS
        integrals_R_R = []

        def a_u(u, v):
            return my * (inner(grad(u), grad(v)) + inner(grad(u), nabla_grad(v))) * dx

        def a_p(K, p, q):
            return K * dot(grad(p), grad(q)) * dx

        def b(s, v):
            return s * div(v) * dx

        def c(alpha, p, q):
            return alpha / Lambda * dot(p, q) * dx
    
        def F(f, v):
            return dot(f, v) * dx
    
        if transient:  # Apply initial conditions
            # Assume displacements to be zero intially
            for i in range(2, numPnetworks):
                # Apply inital pressure
                up_n.sub(i).assign(interpolate(p_initial[i - 1], W.sub(i).collapse()))

            up_n.sub(1).assign(
                interpolate(p_initial[0], W.sub(1).collapse())
            )  # Apply intial total pressure
            p_n = split(up_n)  # p_n[0] = u_n, p_n[1],p_n[2],... = p0_n,p1_n,...

        for i in range(self.num_networks):  # apply for each network
            print("Network: ", i)    
            for j in range(1, boundaryNum + 1):  # for each boundary
                print("Boundary: ", j)
                if "Dirichlet" in boundary_conditionsP[(i + 1, j)]:
                    print(
                        "Applying Dirichlet BC for pressure network: %i and boundary surface %i "
                        % (i + 1, j)
                    )
                    expr = boundary_conditionsP[(i + 1, j)]["Dirichlet"]
                    bcp = DirichletBC(
                        W.sub(i + 2),
                        expr,
                        boundary_markers,
                        j,
                    )
                    bcs_D.append(bcp)
                    time_expr.append(expr)
                elif "DirichletWK" in boundary_conditionsP[(i + 1, j)]:
                    print(
                        "Applying Dirichlet Windkessel BC for pressure network: %i and boundary surface %i "
                        % (i + 1, j)
                    )
                    expr = boundary_conditionsP[(i + 1, j)]["DirichletWK"]
                    bcp = DirichletBC(
                        W.sub(i + 2),
                        expr,
                        boundary_markers,
                        j,
                    )
                    bcs_D.append(bcp)
                    windkessel_terms.append(expr)
                elif "Robin" in boundary_conditionsP[(i + 1, j)]:
                    print("Applying Robin BC for pressure")
                    print("Applying Robin LHS")
                    beta, P_r = boundary_conditionsP[(i + 1, j)]["Robin"]
                    integrals_R_L.append(inner(beta * p_[i + 2], q[i + 2]) * ds(j))
                    if P_r:
                        print("Applying Robin RHS")
                        integrals_R_R.append(inner(beta * P_r, q[i + 2]) * ds(j))
                        time_expr.append(P_r)
                    elif "RobinWK" in boundary_conditionsP[(i + 1, j)]:
                        print("Applying Robin BC with Windkessel referance pressure")
                        print("Applying Robin LHS")
                        beta, P_r = boundary_conditionsP[(i + 1, j)]["RobinWK"]
                        integrals_R_L.append(inner(beta * p_[i + 2], q[i + 2]) * ds(j))
                        print("Applying Robin RHS")
                        integrals_R_R.append(inner(beta * P_r, q[i + 2]) * ds(j))
                        windkessel_terms.append(P_r)

            if isinstance(
                g[i], dolfin.cpp.adaptivity.TimeSeries
            ):  # If the source term is a time series instead of a an expression
                print("Adding timeseries for source term")
                g_space = FunctionSpace(mesh, mixedElement[i + 2])
                g_i = Function(g_space)
                sources.append(F(g_i, q[i + 2]))  # Applying source term
                time_expr.append((g[i], g_i))
            elif g[i] is not None:
                print("Adding expression for source term")
                sources.append(F(g[i], q[i + 2]))  # Applying source term
                time_expr.append(g[i])

            dotProdP.append(c(alpha[i + 1], p_[i + 2], q[1]))  # Applying time derivative
            innerProdP.append(a_p(K[i], p_[i + 2], q[i + 2]))  # Applying diffusive termlg
            if transient:  # implicit euler
                timeD_.append(
                    (
                        (1 / dt)
                        * (
                            cj[i] * p_[i + 2]
                            + alpha[i] / Lambda * sum(a * b for a, b in zip(alpha, p_[1:]))
                        )
                    )
                    * q[i + 2]
                    * dx
                )
                timeD_n.append(
                    (
                        (1 / dt)
                        * (
                            cj[i] * p_n[i + 2]
                            + alpha[i] / Lambda * sum(a * b for a, b in zip(alpha, p_n[1:]))
                        )
                    )
                    * q[i + 2]
                    * dx
                )

        dotProdP.append(c(alpha[0], p_[1], q[1]))

        # Defining boundary conditions for displacements
        for i in boundary_conditionsU:
            print("i = ", i)
            if "Dirichlet" in boundary_conditionsU[i]:
                print("Applying Dirichlet BC.")
                exprU = boundary_conditionsU[i]["Dirichlet"][0]
                exprV = boundary_conditionsU[i]["Dirichlet"][1]
                bcs_D.append(
                    DirichletBC(
                        W.sub(0).sub(0),
                        exprU,
                        boundary_markers,
                        i,
                    )
                )
                bcs_D.append(
                    DirichletBC(
                        W.sub(0).sub(1),
                        exprV,
                        boundary_markers,
                        i,
                    )
                )
                time_expr.append(exprU)
                time_expr.append(exprV)
                
                if mesh.topology().dim() == 3:
                    exprW = boundary_conditionsU[i]["Dirichlet"][2]
                    bcs_D.append(
                        DirichletBC(
                            W.sub(0).sub(2),
                            exprW,
                            boundary_markers,
                            i,
                        )
                    )
                    time_expr.append(exprW)

            elif "Neumann" in boundary_conditionsU[i]:
                if boundary_conditionsU[i]["Neumann"] != 0:
                    print("Applying Neumann BC.")
                    N = boundary_conditionsU[i]["Neumann"]
                    integrals_N.append(inner(-n * N, q[0]) * ds(i))
                    time_expr.append(N)
            elif "NeumannWK" in boundary_conditionsU[i]:
                if boundary_conditionsU[i]["NeumannWK"] != 0:
                    print("Applying Neumann BC with windkessel term.")
                    N = boundary_conditionsU[i]["NeumannWK"]
                    print("N = ", N)
                    integrals_N.append(inner(-n * N, q[0]) * ds(i))
                    windkessel_terms.append(N)

        lhs = (
            a_u(p_[0], q[0])
            + b(p_[1], q[0])
            + b(q[1], p_[0])
            - sum(dotProdP)
            + sum(innerProdP)
            + sum(integrals_R_L)
            + sum(timeD_)
        )

        rhs = (
            F(f, q[0]) + sum(sources) + sum(timeD_n) + sum(integrals_N) + sum(integrals_R_R)
        )

        time_expr.append(f[0])
        time_expr.append(f[1])
        A = assemble(lhs)
        [bc.apply(A) for bc in bcs_D]

        up = Function(W)
        t = 0

        for i in range(numTsteps):
            t += dt
            for expr in time_expr:  # Update all time dependent terms
                update_t(expr, t)

            b = assemble(rhs)
            for bc in bcs_D:
                #            update_t(bc, t)
                bc.apply(b)

            solve(A, up.vector(), b)
            
            if WindkesselBC:

                Q_SAS = assemble(-K[0] * dot(grad(up.sub(2)), n) * ds(1))
                Q_VEN = assemble(-K[0] * dot(grad(up.sub(2)), n) * ds(2)) + assemble(
                    -K[0] * dot(grad(up.sub(2)), n) * ds(3)
                )
                p_SAS_vec[i] = p_SAS
                p_VEN_vec[i] = p_VEN
                t_vec[i] = t
                Q_SAS_vec[i] = Q_SAS
                Q_VEN_vec[i] = Q_VEN
                
                print("Outflow SAS at t=%.2f is Q_SAS=%.10f " % (t, Q_SAS_vec[i]))
                print("Outflow VEN at t=%.2f is Q_VEN=%.10f " % (t, Q_VEN_vec[i]))
                p_SAS_next = 1 / C * (dt * Q_SAS + (C - dt / R) * p_SAS)
                p_VEN_next = 1 / C * (dt * Q_VEN + (C - dt / R) * p_VEN)
                
                print("Pressure for SAS: ", p_SAS_next)
                print("Pressure for VEN: ", p_VEN_next)
                
                for expr in windkessel_terms:
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

            #vtkU << (up.sub(0), t)
            #vtkP0 << (up.sub(1), t)
            #vtkP1 << (up.sub(2), t)
            xdmfU.write(up.sub(0), t)
            xdmfP0.write(up.sub(1), t)
            xdmfP1.write(up.sub(2), t)

            up_n.assign(up)
            progress += 1
        res = []
        res = split(up)
        u = project(res[0], W.sub(0).collapse())
        p = []

        for i in range(1, numPnetworks + 2):
            p.append(project(res[i], W.sub(i).collapse()))

        if WindkesselBC:
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

        return u, p


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
            operand_update(expr, t)

            
    def operand_update(self,expr, t):
        if isinstance(expr, ufl.algebra.Operator):
            for op in expr.ufl_operands:
                update_operator(expr.ufl_operands, t)
        elif isinstance(expr, ufl.Coefficient):
            expr.t = t


def ReadSourceTerm(mesh,source_scale=1.0,periods = 3):
    filestr = 'data/baladont_tot_inflow_series_shifted.csv'
    g = TimeSeries("source_term")
    Q = FunctionSpace(mesh,"CG",1)
    time_period = 0.0
    for i in range(periods):
        #Read in the source term data
        with open(filestr) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                   print(f'Column names are {", ".join(row)}')
                   line_count += 1
                else:
                   print(float(row[1]))
                   source = Constant(float(row[1])*source_scale) #Uniform source on the domain
                   source = project(source, Q)
                   g.store(source.vector(),float(row[0]) + i*time_period)
                   print(f"\t timestep {row[0]} adding {row[1]} to the source.")
                   line_count += 1
            if i == 0:
                time_period = float(row[0])
                print("t =", time_period)
            print(f'Processed {line_count} lines.')
    return g


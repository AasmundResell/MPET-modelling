import os
print(os.environ['PATH'])
import sys
print(sys.executable)
from meshes.brain_mesh_2D import generate_2D_brain_mesh_mm, generate_2D_brain_mesh_m
#from meshes.read_brain_mesh_3D import read_brain_mesh_3D,read_brain_scale
from ufl import FacetNormal, as_vector
from dolfin import *
import yaml


from MPET import MPET


class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class BoundaryInner(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and  (x[0] ** 2 + x[1] ** 2) ** (1 / 2) <= 32

class BoundaryChannel(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and (near(x[0], -1, tol) or near(x[0], 1, tol))

class BoundarySpine(SubDomain):
    def inside(self, x, on_boundary):
        return 10

def run_MPET_2D_fixedOuter():

    ymlFile = open("Test_3Network_fixed_outer.yml") 
    parsedValues = yaml.load(ymlFile, Loader=yaml.FullLoader)
    materialParameters = parsedValues['material_parameters']
    settings = parsedValues['solver_settings']
    sourceParameters = parsedValues['source_data']
    boundaryParameters = parsedValues['boundary_parameters']
    
 
    meshN = settings["mesh_resolution"]
    mesh = generate_2D_brain_mesh_mm(meshN)
    n = FacetNormal(mesh)  # normal vector on the boundary
    
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)
    
    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries
    bx1 = BoundaryInner() 
    bx1.mark(boundary_markers, 2)  # Overwrites the inner ventricles boundary
    bx2 = BoundaryChannel()
    bx2.mark(boundary_markers, 3)  # Overwrites the channel ventricles boundary

    U,pVentricles,pSkull = generateUFL_BCexpressions()
    beta_VEN = boundaryParameters["beta_ven"]
    beta_SAS = boundaryParameters["beta_sas"]
    p_BP = boundaryParameters["p_vein"] #Back pressure, veins

    #Generate boundary conditions for the displacements
    #The integer keys represents a boundary (marker)
    boundary_conditionsU = {
        1: {"Dirichlet": U},
        2: {"NeumannWK": pVentricles},
        3: {"NeumannWK": pVentricles},
    }
    
    #Generate boundary conditions for the fluid pressures
    #Indice 1 in the touple key represents the fluid network
    #Indice 2 in the touple key represents a boundary (marker)
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"Neumann": 0}, 
        (1, 2): {"Neumann": 0},
        (1, 3): {"Neumann": 0},
        (2, 1): {"Dirichlet": Constant(p_BP)},
        (2, 2): {"Dirichlet": Constant(p_BP)}, #A lot of large veins on the lower part of the brain, any better way of implementing this?? (robin cond?)
        (2, 3): {"Neumann": 0},
        (3, 1): {"RobinWK": (beta_SAS,pSkull)},
        (3, 2): {"RobinWK": (beta_VEN,pVentricles)},
        (3, 3): {"RobinWK": (beta_VEN,pVentricles)},
    }


    Solver2D = MPET(
        mesh,
        boundary_markers,
        boundary_conditionsU,
        boundary_conditionsP,
        **settings,
        **materialParameters,
        **sourceParameters,
        **boundaryParameters,
    )
    
    Solver2D.printSetup()

    Solver2D.solve()

    Solver2D.plotResults()
 
def run_MPET_2D_fixedChannel():

    ymlFile = open("Test_3Network_fixed_channel.yml") 
    parsedValues = yaml.load(ymlFile, Loader=yaml.FullLoader)
    materialParameters = parsedValues['material_parameters']
    settings = parsedValues['solver_settings']
    sourceParameters = parsedValues['source_data']
    boundaryParameters = parsedValues['boundary_parameters']
    
 
    meshN = settings["mesh_resolution"]
    mesh = generate_2D_brain_mesh_mm(meshN)
    n = FacetNormal(mesh)  # normal vector on the boundary
    
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)
    
    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries
    bx1 = BoundaryInner() 
    bx1.mark(boundary_markers, 2)  # Overwrites the inner ventricles boundary
    bx2 = BoundaryChannel()
    bx2.mark(boundary_markers, 3)  # Overwrites the channel ventricles boundary

    U,pVentricles,pSkull = generateUFL_BCexpressions()
    beta_VEN = boundaryParameters["beta_ven"]
    beta_SAS = boundaryParameters["beta_sas"]
    p_BP = boundaryParameters["p_vein"] #Back pressure, veins

    #Generate boundary conditions for the displacements
    #The integer keys represents a boundary (marker)
    boundary_conditionsU = {
        1: {"NeumannWK": pSkull},
        2: {"NeumannWK": pVentricles},
        3: {"Dirichlet": U},
    }
    
    #Generate boundary conditions for the fluid pressures
    #Indice 1 in the touple key represents the fluid network
    #Indice 2 in the touple key represents a boundary (marker)
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"Neumann": 0}, 
        (1, 2): {"Neumann": 0},
        (1, 3): {"Neumann": 0},
        (2, 1): {"Dirichlet": Constant(p_BP)},
        (2, 2): {"Neumann": 0},
        (2, 3): {"Neumann": 0},
        (3, 1): {"RobinWK": (beta_SAS,pSkull)},
        (3, 2): {"RobinWK": (beta_VEN,pVentricles)},
        (3, 3): {"RobinWK": (beta_VEN,pVentricles)},
    }


    Solver2D = MPET(
        mesh,
        boundary_markers,
        boundary_conditionsU,
        boundary_conditionsP,
        **settings,
        **materialParameters,
        **sourceParameters,
        **boundaryParameters,
    )
    
    Solver2D.printSetup()

    Solver2D.solve()

    Solver2D.plotResults()
    
def generateUFL_BCexpressions():
    import sympy as sym
    
    t = sym.symbols("t")
    
    p_VEN = sym.symbols("p_VEN")
    p_SAS = sym.symbols("p_SAS")
    
    pSkull = p_SAS
    pVentricles = p_VEN
    
    symbols = [
        pSkull,
        pVentricles,
    ]

     
    variables = [sym.printing.ccode(var) for var in symbols]  # Generate C++ code

    
    
    UFLvariables = [
        Expression(var, degree=2, t=0.0, p_VEN= 0.0,p_SAS = 0.0) for var in variables
    ]  # Generate ufl varibles


    (
        pSkull_UFL,
        pVentricles_UFL,
    ) = UFLvariables
    U_UFL = as_vector((Expression("0.0", degree=2), Expression("0.0", degree=2)))
    return U_UFL, pSkull_UFL, pVentricles_UFL


if __name__ == "__main__":
    run_MPET_2D_fixedOuter()
    

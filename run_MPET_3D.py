import os
print(os.environ['PATH'])
import sys
print(sys.executable)
from ufl import FacetNormal, as_vector
from mshr import Sphere,Box, generate_mesh
from dolfin import *
import yaml
import matplotlib.pyplot as plt

from MPET import MPET


class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class BoundaryInner(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and  (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2) <= 32

def run_MPET_3D_TestSphere():

    ymlFile = open("configurations/3D_TEST_RigidMotion_3PWK.yml") 
    parsedValues = yaml.load(ymlFile, Loader=yaml.FullLoader)
    materialParameters = parsedValues['material_parameters']
    settings = parsedValues['solver_settings']
    sourceParameters = parsedValues['source_data']
    boundaryParameters = parsedValues['boundary_parameters']

    mesh = Mesh("meshes/sphere_mesh/sphere_hollow.xml")
    
    info(mesh)
    #plot(mesh, "3D mesh")
    
    #plt.show()
    
    n = FacetNormal(mesh)  # normal vector on the boundary
    

    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 2)
    #boundary_markers.set_all(9999)
    
    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries
    bx1 = BoundaryInner()
    bx1.mark(boundary_markers, 2)  # Overwrites the inner ventricles boundary

    File('bnd.pvd')<<boundary_markers

    U,pVentricles,pSkull = generateUFL_BCexpressions()
    beta_VEN = boundaryParameters["beta_ven"]
    beta_SAS = boundaryParameters["beta_sas"]
    p_BP = boundaryParameters["p_vein"] #Back pressure, veins

    #Generate boundary conditions for the displacements
    #The integer keys represents a boundary (marker)
    boundary_conditionsU = {
        1: {"NeumannWK": pSkull},
        2: {"NeumannWK": pVentricles},
    }
    
    #Generate boundary conditions for the fluid pressures
    #Indice 1 in the touple key represents the fluid network
    #Indice 2 in the touple key represents a boundary (marker)
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"Neumann": 0}, 
        (1, 2): {"Neumann": 0},
        (2, 1): {"Dirichlet": Constant(p_BP)},
        (2, 2): {"Dirichlet": Constant(p_BP)}, #Better to not cause assymteric pressure distribution?
        (3, 1): {"RobinWK": (beta_SAS,pSkull)},
        (3, 2): {"RobinWK": (beta_VEN,pVentricles)},
    }


    Solver3D = MPET(
        mesh,
        boundary_markers,
        boundary_conditionsU,
        boundary_conditionsP,
        **settings,
        **materialParameters,
        **sourceParameters,
        **boundaryParameters,
    )
    
    Solver3D.printSetup()

    Solver3D.solve()

    Solver3D.plotResults()
 
 
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
    return U_UFL,pSkull_UFL, pVentricles_UFL


if __name__ == "__main__":
    run_MPET_3D_TestSphere()
    """
    origin = Point(0.0, 0.0, 0.0)

    meshN = settings["mesh_resolution"]



    r1 = 100  # Outer radius (mm)
    r2 = 30  # Inner radius  (mm)

    parenchyma = Sphere(origin, r1)
    ventricles = Sphere(origin, r2)
    channel = Box(Point(0.0, -50.0, -50.0),Point(100, 50.0, 50.0))

    import numpy as np

    g3d = parenchyma - ventricles #- channel

    # Test printing
    info("\nCompact output of 3D geometry:")
    info(g3d)
    info("\nVerbose output of 3D geometry:")
    info(g3d, True)


    mesh = generate_mesh(g3d, meshN)
    info(mesh)
    plot(mesh, "3D mesh")
    """

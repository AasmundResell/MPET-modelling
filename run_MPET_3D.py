import os
print(os.environ['PATH'])
import sys
print(sys.executable)
from ufl import FacetNormal, as_vector
from dolfin import *
import yaml
import matplotlib.pyplot as plt
from MPET import MPET
from meshes.read_brain_mesh_3D import read_brain_mesh_3D, create_sphere_mesh

class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class BoundaryInner(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and  (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2) <= 32

def run_MPET_3D():

    ymlFile = open("configurations/TEST_CBC_BLOCK.yml") 
    
    parsedValues = yaml.load(ymlFile, Loader=yaml.FullLoader)
    materialParameters = parsedValues['material_parameters']
    settings = parsedValues['solver_settings']
    sourceParameters = parsedValues['source_data']
    boundaryParameters = parsedValues['boundary_parameters']
    meshN = settings["mesh_resolution"]


    mesh,subdomains,facet_f = read_brain_mesh_3D()
    
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
        (2, 2): {"Neumann":0},
        (2, 3): {"Dirichlet": Constant(p_BP)}, #Cerebellum close to jugular veins
        (3, 1): {"RobinWK": (beta_SAS,pSkull)},
        (3, 2): {"RobinWK": (beta_VEN,pVentricles)},
        (3, 3): {"RobinWK": (beta_SAS,pSkull)},
    }

    Solver3D = MPET(
        mesh,
        facet_f,
        boundary_conditionsU,
        boundary_conditionsP,
        **settings,
        **materialParameters,
        **sourceParameters,
        **boundaryParameters,
    )
    
    Solver3D.printSetup()

    #Solver3D.blockSolve()
    #Solver3D.AMG_testing()
    Solver3D.SolvePETSC()
    #Solver3D.plotResults()

    Solver3D.fileStats.close() #close file

def run_MPET_3D_TestSphere():

    ymlFile = open("configurations/SPHERE_paraV_TEST2.yml") 
    
    parsedValues = yaml.load(ymlFile, Loader=yaml.FullLoader)
    materialParameters = parsedValues['material_parameters']
    settings = parsedValues['solver_settings']
    sourceParameters = parsedValues['source_data']
    boundaryParameters = parsedValues['boundary_parameters']

    meshN = settings["mesh_resolution"]


    mesh = Mesh("meshes/sphere_mesh/sphere_hollow.xml")
    facet_f = MeshFunction("size_t", mesh, 2)

    bx0 = BoundaryOuter()
    bx0.mark(facet_f,1)
    bx1 = BoundaryInner()
    bx1.mark(facet_f,2)

    """
    
    path = "/home/asmund/dev/MPET-modelling/sphereScaledN13.h5"
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(),path, "r")
    hdf.read(mesh, "/mesh", False)

    facet_f = MeshFunction("size_t", mesh,mesh.topology().dim()-1)

    hdf.read(facet_f, "/facet")

    File('meshes/bnd.pvd')<<facet_f

    #plot(mesh, "3D mesh")
    #plt.show()

    info(mesh)
    info(facet_f)

    """

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
    #iIndice 1 in the touple key represents the fluid network
    #Indice 2 in the touple key represents a boundary (marker)
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"Neumann": 0}, 
        (1, 2): {"Neumann": 0},
        (2, 1): {"Dirichlet": Constant(p_BP)},
        (2, 2): {"Dirichlet": Constant(p_BP)},
        (3, 1): {"DirichletWK": pSkull},
        (3, 2): {"DirichletWK": pVentricles},
    }
        #(2, 1): {"Neumann": 0},
        #(2, 2): {"Neumann": 0},
        #(3, 1): {"RobinWK": (beta_SAS,pSkull)},
        #(3, 2): {"RobinWK": (beta_VEN,pVentricles)},

    Solver3D = MPET(
        mesh,
        facet_f,
        boundary_conditionsU,
        boundary_conditionsP,
        **settings,
        **materialParameters,
        **sourceParameters,
        **boundaryParameters,
    )
    
    Solver3D.printSetup()
    #Solver3D.solve()

    Solver3D.plotResults()
    Solver3D.printStatistics()


    Solver3D.fileStats.close() #close file
    

 
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
    U_UFL = as_vector((Constant(0), Constant(0), Constant(0)))
    return U_UFL,pSkull_UFL, pVentricles_UFL




if __name__ == "__main__":
    #run_MPET_3D()
    run_MPET_3D_TestSphere()
 

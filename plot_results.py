import os
print(os.environ['PATH'])
import sys
print(sys.executable)
from ufl import FacetNormal, as_vector
from dolfin import *
import yaml
import matplotlib.pyplot as plt
from MPET import MPET
from run_MPET_3D import generateUFL_BCexpressions
from meshes.read_brain_mesh_3D import read_brain_mesh_3D, create_sphere_mesh

class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class BoundaryInner(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and  (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2) <= 32

def plot_results_sphere():

    plot_from = int(sys.argv[1])
    plot_to = int(sys.argv[2])
    ymlFile = open("configurations/SPHERE_N15_paraV_TEST1.yml") 
    
    parsedValues = yaml.load(ymlFile, Loader=yaml.FullLoader)
    materialParameters = parsedValues['material_parameters']
    settings = parsedValues['solver_settings']
    sourceParameters = parsedValues['source_data']
    boundaryParameters = parsedValues['boundary_parameters']

    meshN = settings["mesh_resolution"]


    """
    mesh = Mesh("meshes/sphere_mesh/sphere_hollow.xml")
    facet_f = MeshFunction("size_t", mesh, 2)

    bx0 = BoundaryOuter()
    bx0.mark(facet_f,1)
    bx1 = BoundaryInner()
    bx1.mark(facet_f,2)

    """
    
    path = "/home/asmund/dev/MPET-modelling/sphereScaledN15.h5"
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
    
    Solver3D.plotResults(plot_from,plot_to)
    Solver3D.printStatistics(plot_from,plot_to)


    Solver3D.fileStats.close() #close file
    
if __name__ == "__main__":
    plot_results_sphere()
 

from fenics import *
from matplotlib.pyplot import show

import numpy as np
from mshr import Sphere,Box, generate_mesh

def read_brain_mesh_3D():

    path = "/home/asmund/dev/MPET-modelling/meshes/parenchyma16_with_DTI.h5"
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(),path , "r")
    hdf.read(mesh, "/mesh", False)
    SD = MeshFunction("size_t", mesh,mesh.topology().dim())
    hdf.read(SD, "/subdomains")
    bnd = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
    hdf.read(bnd, "/boundaries")
    #lookup_table = MeshFunction("size_t", mesh, mesh.topology().dim())
    #hdf.read(lookup_table, '/lookup_table')
    #TensorSpace = TensorFunctionSpace(mesh, 'DG', 0)
    #MDSpace = FunctionSpace(mesh, 'DG', 0)
    #MD = Function(MDSpace)
    #Kt = Function(TensorSpace)
    #hdf.read(MD, '/MD')
    #hdf.read(Kt, '/DTI')
    
    File('meshes/subdomains.pvd')<<SD
    File('meshes/bnd.pvd')<<bnd 

    return mesh,SD,bnd

def create_sphere_mesh(N):

    mesh,facet_f = GenerateSphereMesh(N)

    hdf = HDF5File(mesh.mpi_comm(), "sphere.h5" , "w")
    hdf.write(mesh,"/mesh")
    hdf.write(facet_f,"/facet")


def read_brain_scale(mesh):
    dx = Measure("dx", domain=mesh)
    tot_parenchyma_vol = assemble(1*dx)
    vol_scale = 1.0/tot_parenchyma_vol
    print("Volume of parenchyma in mm³: ",tot_parenchyma_vol)
    return vol_scale

def GenerateSphereMesh(N):
    # Radius of outer and inner sphere
    oradius, iradius = 100., 30.
    
    # Geometry
    outer_sphere = Sphere(Point(0., 0., 0.), oradius)
    inner_sphere = Sphere(Point(0., 0., 0.), iradius)
    g3d = outer_sphere - inner_sphere
    mesh = generate_mesh(g3d, N)
    
    facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    # Mark all external facets
    DomainBoundary().mark(facet_f, 1)

    # Now mark the inner boundary by checking that radius is smaller than radius
    # of mid sphere
    center = Point(0., 0., 0.)
    mid = 0.5*(oradius - iradius)
    for facet in SubsetIterator(facet_f, 1):
        if facet.midpoint().distance(center) < mid:
            facet_f[facet] = 2

    # See that both values are now present
    print("LEN outer boundary: ",len(np.where(facet_f.array() == 1)[0]))
    print("LEN inner boundary: ",len(np.where(facet_f.array() == 2)[0]))
    return mesh,facet_f

if __name__ == "__main__":
    mesh = read_brain_mesh_3D()
    #scale = read_brain_scale(mesh)
       

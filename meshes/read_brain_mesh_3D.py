from fenics import *
from matplotlib.pyplot import show

def read_brain_mesh_3D():

    path = "/home/asmund/dev/FEniCS-Brain-Flow/meshes/parenchyma16_with_DTI.h5"
    mesh = Mesh()
    #hdf = HDF5File(mesh.mpi_comm(),path , "r")
    #hdf.read(mesh, "/mesh", False)
    SD = MeshFunction("size_t", mesh,mesh.topology().dim())
    #hdf.read(SD, "/subdomains")
    bnd = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
    #hdf.read(bnd, "/boundaries")
    #lookup_table = MeshFunction("size_t", mesh, mesh.topology().dim())
    #hdf.read(lookup_table, '/lookup_table')
    #TensorSpace = TensorFunctionSpace(mesh, 'DG', 0)
    #MDSpace = FunctionSpace(mesh, 'DG', 0)
    #MD = Function(MDSpace)
    #Kt = Function(TensorSpace)
    #hdf.read(MD, '/MD')
    #hdf.read(Kt, '/DTI')
    
    #File('subdomains.pvd')<<SD
    #File('bnd.pvd')<<bnd

    return mesh,SD,bnd

def read_brain_scale(mesh):
    dx = Measure("dx", domain=mesh)
    tot_parenchyma_vol = assemble(1*dx)
    vol_scale = 1.0/tot_parenchyma_vol
    print("Volume of parenchyma in mm³: ",tot_parenchyma_vol)
    return vol_scale
if __name__ == "__main__":
    mesh = read_brain_mesh_3D()
    scale = read_brain_scale(mesh)
       

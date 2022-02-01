import os
print(os.environ['PATH'])


from fenics import *
from mshr import Circle, Rectangle, generate_mesh
import matplotlib.pyplot as plt
import numpy as np



def generate_2D_brain_mesh_mm(n=16):

    class Border(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    origin = Point(0.0, 0.0)

    channelCoord1 = Point(-1.0, 0.0)
    channelCoord2 = Point(1.0, -100)

    r1 = 100  # Outer radius (mm)
    r2 = 30  # Inner radius  (mm)

    parenchyma = Circle(origin, r1)
    ventricles = Circle(origin, r2)
    aqueduct = Rectangle(channelCoord1, channelCoord2)
    import numpy as np

    geometry = parenchyma - ventricles - aqueduct
    
    mesh = generate_mesh(geometry, n)
    Border = Border()
    for j in range(3):
        mesh.init(1, 2) # Initialise facet to cell connectivity
        markers = MeshFunction("bool", mesh,mesh.topology().dim()-1,False)
        Border.mark(markers,True)
        
        mesh = refine(mesh, markers)
                

    plot(mesh)
    #plt.show()
    
    return mesh

def generate_2D_brain_mesh_m(n=16):

    class Border(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    
    origin = Point(0.0, 0.0)

    channelCoord1 = Point(-1.0e-3, 0.0e-3)
    channelCoord2 = Point(1.0e-3, -100e-3)

    r1 = 100e-3  # Outer radius (m)
    r2 = 30e-3  # Inner radius  (m)

    parenchyma = Circle(origin, r1)
    ventricles = Circle(origin, r2)
    aqueduct = Rectangle(channelCoord1, channelCoord2)

    geometry = parenchyma - ventricles - aqueduct
    
    mesh = generate_mesh(geometry, n)
    Border = Border()
    for j in range(3):
        mesh.init(1, 2) # Initialise facet to cell connectivity
        markers = MeshFunction("bool", mesh,mesh.topology().dim()-1,False)
        Border.mark(markers,True)
        
        mesh = refine(mesh, markers)

        
    plot(mesh)
    plt.show()
    return mesh

if __name__ == "__main__":
    print("hello")
    mesh_mm = generate_2D_brain_mesh_mm(16)
    mesh_m = generate_2D_brain_mesh_m(16)

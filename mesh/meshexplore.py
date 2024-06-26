# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mesh

# %%
from pathlib import Path
from mpi4py import MPI
import adios2.bindings
import dolfinx
import numpy as np

import pyvista
import h5py
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# %%
comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(comm, 2, 2,)

# %%
pyvista.start_xvfb()

tdim = domain.topology.dim
num_cells_local = domain.topology.index_map(tdim).size_local
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, tdim, np.arange(num_cells_local, dtype=np.int32))

p = pyvista.Plotter(window_size=[800, 800])
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
# grid = pyvista.Unstructured(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
plotter.show()

# %%
domain.name

# %%
domain.geometry.dim

# %%
domain.geometry.input_global_indices

# %%
domain.geometry.x

# %%
domain.geometry.dofmap

# %%
domain.geometry.index_map().size_local

# %%
domain.geometry.index_map().local_range

# %%
domain.geometry.index_map().ghosts

# %%
domain.geometry.index_map().owners

# %%
domain.topology.cell_name()

# %%
domain.topology.cell_type

# %%
domain.topology.entity_types

# %%
domain.topology.original_cell_index

# %%
domain.topology.index_maps(0)

# %%
domain.topology.index_maps(0)[0].local_range

# %%
domain.topology.index_maps(1)

# %% [raw]
# domain.topology.index_maps(1)[0].local_range

# %%
domain.topology.index_maps(2)

# %%
domain.topology.index_maps(2)[0].local_range

# %%
domain.topology.index_maps(2)[0].size_local

# %%
domain.topology.index_map(0)

# %%
domain.topology.index_map(0).local_range

# %%
domain.topology.index_map(1)

# %%
domain.topology.index_map(2)

# %%
domain.topology.index_map(2).local_range

# %%
domain.topology.connectivity(2,0)

# %%
domain.topology.connectivity(0,2)

# %%
domain.topology.create_connectivity(0,2)

# %%
domain.topology.connectivity(1,0)

# %%
domain.topology.connectivity(2,1)

# %%
domain.topology.connectivity(1,0)

# %%
domain.topology.create_connectivity(2,0)

# %%
domain.topology.create_connectivity(1,0)

# %%
domain.topology.create_connectivity(2,1)

# %%
domain.topology.connectivity(2,0)

# %%
domain.topology.connectivity(2,1)

# %%
cn_10 = domain.topology.connectivity(1,0)
cn_10

# %%
cn_10.array

# %%
cn_10.offsets

# %%
domain.topology.create_entities(1)
num_edges = domain.topology.index_map(1).size_local + domain.topology.index_map(1).num_ghosts
domain.topology.create_connectivity(1,domain.geometry.dim)
edges =dolfinx.mesh.entities_to_geometry(domain, 1, np.arange(num_edges, dtype=np.int32), False)
edges

# %%
local_indices = range(len(domain.geometry.input_global_indices))
global_indices = np.array(domain.geometry.input_global_indices)
nodes = dict(zip(local_indices, global_indices))



# %%
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# %%
nx.draw(G, with_labels=True)

# %%
pos = dict(zip(local_indices, domain.geometry.x[:,:-1]))
pos

# %%
nx.draw(G, with_labels=True, pos=pos)

# %%
nx.draw(G, pos=pos, labels=nodes, with_labels=True, font_color="w")

# %%
domain.geometry.index_map().local_range

# %%
cn_10.links(0)

# %%
cn_10.links(9)

# %%
cn_10.num_nodes

# %%
domain.basix_cell()

# %%
domain.ufl_cell()

# %%
domain.ufl_domain()

# %%
domain.geometry.cmap.degree

# %%
domain.geometry.cmap.dim

# %%
domain.geometry.cmap.dtype

# %%
domain.geometry.cmap.variant

# %% [markdown]
# ## Write using ADIOS2

# %%
comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(comm, 2, 2,)

gdim = domain.geometry.dim
filename = Path("test.h5")
engine = "HDF5"
adios = adios2.bindings.ADIOS(domain.comm)
io = adios.DeclareIO("test")
io.SetEngine(engine)
file = io.Open(str(filename), adios2.bindings.Mode.Write)
file.BeginStep()

# %%
geom = np.array(domain.geometry.x, dtype=np.float64)[:,:gdim].copy()
x_var = io.DefineVariable("x", geom,
                 shape=[domain.geometry.index_map().size_global, gdim],
                 start=[domain.geometry.index_map().local_range[0], 0],
                 count=[domain.geometry.index_map().local_range[1]-domain.geometry.index_map().local_range[0], gdim])
file.Put(x_var, geom)
file.EndStep()
file.Close()
adios.RemoveIO("test")

# %%
inf = h5py.File("test.h5", "r")

# %%
print(inf.keys())

# %%
inf["Step0"]

# %%
inf["Step0"].keys()

# %%
inf["Step0"]["x"]

# %%
np.array(inf["Step0"]["x"])

# %% [markdown]
# ## ipyparallel
#
# from adios4dolfinx docs

# %%
# # Introduction to IPython parallel
# The following demos heavily rely on IPython-parallel to illustrate how checkpointing works when
# using multiple MPI processes.
# We illustrate what happens in parallel by launching three MPI processes
# using [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/)

import logging

import ipyparallel as ipp


def hello_mpi():
    # We define all imports inside the function as they have to be launched on the remote engines
    from mpi4py import MPI

    print(f"Hello from rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}")


with ipp.Cluster(engines="mpi", n=3, log_level=logging.ERROR) as cluster:
    # We send the query to run the function `hello_mpi` on all engines
    query = cluster[:].apply_async(hello_mpi)
    # We wait for all engines to finish
    query.wait()
    # We check that all engines exited successfully
    assert query.successful(), query.error
    # We print the output from each engine
    print("".join(query.stdout))


# %%
def print_info(ns=2, mesh_type=1):
    from mpi4py import MPI
    import dolfinx
    import numpy as np
    import networkx as nx
    import pyvista
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    if mesh_type == 1:
        domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, ns, ns, dolfinx.mesh.CellType.triangle)
    elif mesh_type == 2:
        domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, ns, ns, dolfinx.mesh.CellType.quadrilateral)
    elif mesh_type == 3:
        domain = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, ns, ns, ns, dolfinx.mesh.CellType.tetrahedron)
    elif mesh_type == 4:
        domain = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, ns, ns, ns, dolfinx.mesh.CellType.hexahedron)

    print(f"Hello from rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}")
    name = domain.name
    dim = domain.geometry.dim
    geometry = {
        "input_global_indices" : domain.geometry.input_global_indices,
        "x" : domain.geometry.x,
        "dofmap" : domain.geometry.dofmap,
        "size_global" : domain.geometry.index_map().size_global,
        "size_local" : domain.geometry.index_map().size_local,
        "local_range" : domain.geometry.index_map().local_range,
        "ghosts" : domain.geometry.index_map().ghosts,
        "owners" : domain.geometry.index_map().owners,
    }
    domain.topology.create_connectivity(2,0)
    domain.topology.create_connectivity(1,0)
    domain.topology.create_connectivity(2,1)
    domain.topology.create_connectivity(0,2)
    
    topology = {
        "cell_name" : domain.topology.cell_name(),
        "cell_type" : domain.topology.cell_type,
        "entity_types" : domain.topology.entity_types,
        "orginal_cell_index" : domain.topology.original_cell_index,

        "index_map_0_local_range" : domain.topology.index_map(0).local_range,
        "index_map_0_ghosts" : domain.topology.index_map(0).ghosts,
        "index_map_0_owners" : domain.topology.index_map(0).owners,
        
        "index_map_1_local_range" : domain.topology.index_map(1).local_range,
        "index_map_1_ghosts" : domain.topology.index_map(1).ghosts,
        "index_map_1_owners" : domain.topology.index_map(1).owners,
        
        "index_map_2_local_range" : domain.topology.index_map(2).local_range,
        "index_map_2_ghosts" : domain.topology.index_map(2).ghosts,
        "index_map_2_owners" : domain.topology.index_map(2).owners,

        "index_maps_0_0_local_range" : domain.topology.index_maps(0)[0].local_range,
        "index_maps_0_0_size_local" : domain.topology.index_maps(0)[0].size_local,
        "index_maps_0_0_ghosts" : domain.topology.index_maps(0)[0].ghosts,
        "index_maps_0_0_owners" : domain.topology.index_maps(0)[0].owners,
        
        "index_maps_1_0_local_range" : domain.topology.index_maps(1)[0].local_range,
        "index_maps_1_0_size_local" : domain.topology.index_maps(1)[0].size_local,
        "index_maps_1_0_ghosts" : domain.topology.index_maps(1)[0].ghosts,
        "index_maps_1_0_owners" : domain.topology.index_maps(1)[0].owners,
        
        "index_maps_2_0_local_range" : domain.topology.index_maps(2)[0].local_range,
        "index_maps_2_0_size_local" : domain.topology.index_maps(2)[0].size_local,        
        "index_maps_2_0_ghosts" : domain.topology.index_maps(2)[0].ghosts,
        "index_maps_2_0_owners" : domain.topology.index_maps(2)[0].owners,

        "connectivity_2_0" : domain.topology.connectivity(2,0),
        "connectivity_1_0" : domain.topology.connectivity(1,0),
        "connectivity_2_1" : domain.topology.connectivity(2,1),
        "connectivity_0_2" : domain.topology.connectivity(0,2),

        "connectivity_1_0_array" : domain.topology.connectivity(1,0).array,
        "connectivity_1_0_offsets" : domain.topology.connectivity(1,0).offsets,

    }
    local_indices = range(len(domain.geometry.input_global_indices))
    global_indices = np.array(domain.geometry.input_global_indices)
    nodes = dict(zip(local_indices, global_indices))
    domain.topology.create_entities(1)
    num_edges = domain.topology.index_map(1).size_local + domain.topology.index_map(1).num_ghosts
    domain.topology.create_connectivity(1,dim)
    edges =dolfinx.mesh.entities_to_geometry(domain, 1, np.arange(num_edges, dtype=np.int32), False)

    # colors = {0:"b", 1:"r", 2:"g", 3:"y", 4:"m", 5:"k"}
    colors = dict(zip(range(10), ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']))

    color = colors[MPI.COMM_WORLD.rank]
    node_color = geometry["size_local"]*[color] + list(map(colors.get, geometry["owners"]))

    G = nx.Graph()
    G.add_nodes_from(local_indices)
    G.add_edges_from(edges)

    print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\tname:{name}")
    print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\tdim:{dim}")
    for key, value in geometry.items():
        print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\t{key}:{value}")

    for key, value in topology.items():
        print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\t{key}:{value}")

    # Visualization
    if mesh_type == 1 or mesh_type == 2:
        pos = dict(zip(local_indices, domain.geometry.x[:,:-1]))

        sub1 = plt.subplot(121)
        sub1.set_title(f"Rank {MPI.COMM_WORLD.rank} : Local node numbering")
        nx.draw(G, pos=pos, with_labels=True, node_color=node_color, font_color="w")

        sub2 = plt.subplot(122)
        sub2.set_title(f"Rank {MPI.COMM_WORLD.rank} : Global node numbering")
        nx.draw(G, pos=pos, labels=nodes, with_labels=True, node_color=node_color, font_color="w")

        return plt.gca()

    if mesh_type == 3 or mesh_type == 4:
        pos = dict(zip(local_indices, domain.geometry.x))
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        def plot_3D(ax, labels, node_color):

            # Plot the nodes - alpha is scaled by "depth" automatically
            ax.scatter(*node_xyz.T, s=400, ec="w", color=node_color)

            for i, txt in enumerate(labels):
                ax.text(*node_xyz[i], txt)

            # Plot the edges
            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color="tab:gray", alpha=0.1)

            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            return ax

        fig = plt.figure(figsize=(16,16))
        sub1 = plt.subplot(121, projection="3d")
        sub1.set_title(f"Rank {MPI.COMM_WORLD.rank} : Local node numbering")
        sub1 = plot_3D(sub1, local_indices, node_color)

        sub2 = plt.subplot(122, projection="3d")
        sub2.set_title(f"Rank {MPI.COMM_WORLD.rank} : Global node numbering")
        sub2 = plot_3D(sub2, global_indices, node_color)

        return plt.gca()



# %%
ns = 2
mesh_type = 1
n = 2
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 3
mesh_type = 1
n = 3
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 4
mesh_type = 1
n = 2
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 4
mesh_type = 1
n = 4
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 4
mesh_type = 2
n = 2
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 4
mesh_type = 2
n = 3
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 3
mesh_type = 4
n = 2
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 3
mesh_type = 4
n = 3
with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %% [markdown]
# END

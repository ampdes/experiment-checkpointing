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

import h5py
import networkx as nx

# %%
comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(comm, 2, 2,)

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
edges = cn_10.array.reshape(-1,2)
edges

# %%
G = nx.Graph()
G.add_nodes_from(domain.geometry.input_global_indices)
G.add_edges_from(edges)

# %%
nx.draw(G, with_labels=True)

# %%
pos = dict(zip(domain.geometry.input_global_indices, domain.geometry.x[:,:-1]))
pos

# %%
nx.draw(G, with_labels=True, pos=pos)

# %%
pos = dict(zip(range(9), domain.geometry.x[:,:-1]))
pos

# %%
nx.draw(G, with_labels=True, pos=pos)

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
# ## Quadrilaterals

# %%
domain = dolfinx.mesh.create_unit_square(comm, 2, 2, dolfinx.mesh.CellType.quadrilateral)


# %%
domain.name

# %%
domain.geometry.input_global_indices

# %%
domain.topology.create_connectivity(1,0)

# %%
cn_10 = domain.topology.connectivity(1,0)
cn_10

# %%
edges = cn_10.array.reshape(-1,2)
edges

# %%
G = nx.Graph()
G.add_nodes_from(domain.geometry.input_global_indices)
G.add_edges_from(edges)

# %%
nx.draw(G, with_labels=True)

# %%
nx.to_latex_raw(G)

# %%
nx.draw(G, with_labels=True, pos=nx.spring_layout(G))

# %%
nx.draw(G, with_labels=True, pos=nx.spring_layout(G))

# %%
pos = dict(zip(domain.geometry.input_global_indices, domain.geometry.x[:,:-1]))
pos

# %%
nx.draw(G, with_labels=True, pos=pos)

# %%
pos = dict(zip(range(len(domain.geometry.input_global_indices)), domain.geometry.x[:,:-1]))
pos

# %%
nx.draw(G, with_labels=True, pos=pos)

# %% [markdown]
# ## Cube

# %%
domain = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2, dolfinx.mesh.CellType.tetrahedron)


# %%
domain.name

# %%
domain.geometry.input_global_indices

# %%
domain.topology.create_connectivity(1,0)

# %%
cn_10 = domain.topology.connectivity(1,0)
cn_10

# %%
edges = cn_10.array.reshape(-1,2)
edges

# %%
G = nx.Graph()
G.add_nodes_from(domain.geometry.input_global_indices)
G.add_edges_from(edges)

# %%
nx.draw(G, with_labels=True)

# %% [raw]
# nx.to_latex_raw(G)

# %%
pos = dict(zip(domain.geometry.input_global_indices, domain.geometry.x[:,:-1]))
pos

# %%
nx.draw(G, with_labels=True, pos=pos)

# %%
pos = dict(zip(range(len(domain.geometry.input_global_indices)), domain.geometry.x[:,:-1]))
pos

# %%
nx.draw(G, with_labels=True, pos=pos)

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
    import networkx as nx
    import matplotlib.pyplot as plt

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
    cn_10 = domain.topology.connectivity(1,0)
    edges = cn_10.array.reshape(-1,2)
    nodes = dict(zip(range(len(domain.geometry.input_global_indices)),domain.geometry.input_global_indices))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    node_color = geometry["size_local"]*["blue"] + (geometry["size_global"] - geometry["size_local"] - 1)*["red"]

    G.add_edges_from(edges)
    # Find the local_to_global map of the indices for visualization
    pos = dict(zip(range(len(domain.geometry.input_global_indices)), domain.geometry.x[:,:-1]))    
    # pos = dict(zip(domain.geometry.input_global_indices, domain.geometry.x[:,:-1]))    
    
    print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\tname:{name}")
    print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\tdim:{dim}")
    for key, value in geometry.items():
        print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\t{key}:{value}")

    for key, value in topology.items():
        print(f"From rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}: \n\t{key}:{value}")

    # print(pos)
    sub1 = plt.subplot(121)
    sub1.set_title("Local node numbering")
    if MPI.COMM_WORLD.size < 3:
        nx.draw(G, pos=pos, with_labels=True, node_color=node_color)
    else:
        nx.draw(G, pos=pos, with_labels=True)

    sub2 = plt.subplot(122)
    sub2.set_title("Global node numbering")
    if MPI.COMM_WORLD.size < 3:
        nx.draw(G, pos=pos, labels=nodes, with_labels=True, node_color=node_color)
    else:
        nx.draw(G, pos=pos, labels=nodes, with_labels=True)

    return plt.gca()
    # plt.show()



# %%
ns = 2
mesh_type = 1
with ipp.Cluster(engines="mpi", n=2, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 3
mesh_type = 1
with ipp.Cluster(engines="mpi", n=3, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 4
mesh_type = 2
with ipp.Cluster(engines="mpi", n=2, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %%
ns = 3
mesh_type = 4
with ipp.Cluster(engines="mpi", n=2, log_level=logging.ERROR) as cluster:
    # Create a mesh and print info
    query = cluster[:].apply_async(print_info, ns, mesh_type)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# %% [markdown]
# END

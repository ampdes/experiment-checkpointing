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

# %% [markdown]
# ## TODO
#
#  - a function to query all data from each process
#  - write in parallel

# %% [markdown]
# END

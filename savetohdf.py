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
# # Save to HDF5 files
#
# Authors: Jørgen S. Dokken, Abdullah Mujahid
#
# SPDX-License-Identifier:    MIT

# %%
# Install h5py with mpi support
# HDF5_MPI="ON" HDF5_DIR="/usr/local" python3 -m pip install --break-system-packages --no-binary=h5py h5py -U --no-build-isolation

# %%
from pathlib import Path
from mpi4py import MPI
import h5py
import dolfinx
import basix
import numpy as np


# %%
def write_mesh(domain, hdf):

    g_imap = domain.geometry.index_map()
    num_nodes_local = g_imap.size_local
    geom_range = g_imap.local_range
    geom = domain.geometry.x[:num_nodes_local].copy()
    
    dt = h5py.string_dtype(encoding="ascii")
    hdf.attrs["Version"] = [2, 2]
    hdf.attrs["Type"] = np.string_("UnstructuredGrid")
    
    # Put local geometry
    global_shape = (g_imap.size_global, 3)
    geom_set = hdf.create_dataset(np.string_("Points"), global_shape, dtype=geom.dtype)
    geom_set[geom_range[0]:geom_range[1], :] = geom
    
    # Compute global topology
    g_imap = g_imap
    g_dmap = domain.geometry.dofmap
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
    cell_range = domain.topology.index_map(domain.topology.dim).local_range
    cmap = domain.geometry.cmap
    geom_layout = cmap.create_dof_layout()
    num_dofs_per_cell = geom_layout.num_entity_closure_dofs(domain.topology.dim)
    dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)
    assert g_dmap.shape[1] == num_dofs_per_cell
    dofs_out[:, :] = np.asarray(
    g_imap.local_to_global(g_dmap[:num_cells_local, :].reshape(-1))
    ).reshape(dofs_out.shape)
    cell_type = domain.topology.cell_type
    map_vtk = np.argsort(dolfinx.cpp.io.perm_vtk(cell_type, num_dofs_per_cell))
    dofs_out = dofs_out[:, map_vtk]
    
    # Put global topology
    geom_set = hdf.create_dataset("Connectivity", (num_cells_global*num_dofs_per_cell,), dtype=np.int64)
    geom_set[cell_range[0]*num_dofs_per_cell:num_dofs_per_cell*cell_range[1]] = dofs_out.reshape(-1)
    
    # Put cell type
    # Types
    type_set = hdf.create_dataset("Types", (num_cells_global,), dtype=np.uint8)
    cts = np.full(num_cells_local, dolfinx.cpp.io.get_vtk_cell_type(cell_type, domain.topology.dim))
    type_set[cell_range[0]:cell_range[1]] = cts
    
    # Geom dofmap offset
    con_part = hdf.create_dataset("NumberOfConnectivityIds", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
       con_part[domain.comm.rank] = num_cells_global*num_dofs_per_cell
    
    # Num cells
    num_cells = hdf.create_dataset("NumberOfCells", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
       num_cells[domain.comm.rank] = num_cells_global
    
    # num points
    num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
       num_points[domain.comm.rank] = g_imap.size_global
    
    # Offsets
    offsets = hdf.create_dataset("Offsets", shape=(num_cells_global+1,), dtype=np.int64)
    offsets[cell_range[0] + 1: cell_range[1] + 1] = np.arange(1, num_cells_local+1)*dofs_out.shape[1] + cell_range[0]*dofs_out.shape[1]


# %%
def write_cg(domain, V, u, hdf):

    hdf.attrs["Version"] = [2, 2]
    ascii_type = 'UnstructuredGrid'.encode('ascii')
    hdf.attrs.create('Type', ascii_type, dtype=h5py.string_dtype('ascii', len(ascii_type)))

    v_dmap = V.dofmap
    v_imap = V.dofmap.index_map

    # Local geometry
    num_nodes_local = v_imap.size_local
    local_range = v_imap.local_range
    geom = V.tabulate_dof_coordinates()[:num_nodes_local].copy()

    # Put local geometry
    imap = v_imap
    global_shape = (imap.size_global, 3)
    geom_set = hdf.create_dataset(np.string_("Points"), global_shape, dtype=geom.dtype)
    geom_set[local_range[0]:local_range[1], :] = geom
    
    
    # Compute global topology
    dmap = v_dmap
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
    cell_range = domain.topology.index_map(domain.topology.dim).local_range
    
    num_dofs_per_cell = dmap.list.shape[1]
    dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)
    
    dofs_out[:, :] = np.asarray(
    imap.local_to_global(dmap.list[:num_cells_local, :].reshape(-1))
    ).reshape(dofs_out.shape)
    cell_type = domain.topology.cell_type
    map_vtk = np.argsort(dolfinx.cpp.io.perm_vtk(cell_type, num_dofs_per_cell))
    dofs_out = dofs_out[:, map_vtk]
    
    
    # Put global topology
    geom_set = hdf.create_dataset("Connectivity", (num_cells_global*num_dofs_per_cell,), dtype=np.int64)
    geom_set[cell_range[0]*num_dofs_per_cell:num_dofs_per_cell*cell_range[1]] = dofs_out.reshape(-1)
    
    # Put cell type
    # Types
    type_set = hdf.create_dataset("Types", (num_cells_global,), dtype=np.uint8)
    cts = np.full(num_cells_local, dolfinx.cpp.io.get_vtk_cell_type(cell_type, domain.topology.dim))
    type_set[cell_range[0]:cell_range[1]] = cts
    
    # Geom dofmap offset
    con_part = hdf.create_dataset("NumberOfConnectivityIds", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
       con_part[domain.comm.rank] = num_cells_global*num_dofs_per_cell
    
    # Num cells
    num_cells = hdf.create_dataset("NumberOfCells", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
       num_cells[domain.comm.rank] = num_cells_global
    
    # num points
    num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
       num_points[domain.comm.rank] = imap.size_global
    
    # Offsets
    offsets = hdf.create_dataset("Offsets", shape=(num_cells_global+1,), dtype=np.int64)
    offsets[cell_range[0] + 1: cell_range[1] + 1] = np.arange(1, num_cells_local+1)*dofs_out.shape[1] + cell_range[0]*dofs_out.shape[1]
    # inf.close()
    
    # Put function
    u_size_global = u.x.index_map.size_global
    u_size_local = u.x.index_map.size_local
    u_range = u.x.index_map.local_range
    
    bs = u.x.block_size
    
    func = hdf.create_group('PointData')
    func_values = func.create_dataset(u.name, shape=(u_size_global, bs), dtype=u.dtype)
    func_values[u_range[0]:u_range[1], :] = u.x.array[:u_size_local*bs].reshape(-1,bs)
    
    # num points
    # num_points = func.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
    # if domain.comm.rank == 0:
    #    num_points[domain.comm.rank] = u_size_global
    
    # # Offsets for u  
    # u_offsets = func.create_dataset("Offsets", shape=(num_u_nodes+1,), dtype=np.int64)
    # u_offsets[u_range[0] + 1: u_range[1] + 1] = np.arange(1, u_size_local+1)*bs + u_range[0]*bs


# %% [markdown]
# ## Write mesh

# %%
comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(
   comm, 4, 4)
gdim = domain.geometry.dim

# %%
filename = Path("test.vtkhdf")
inf = h5py.File(filename, "w", driver="mpio", comm=comm)
hdf = inf.create_group(np.string_("VTKHDF"))

write_mesh(domain, hdf)
inf.close()


# %% [markdown]
# ## Function expressions

# %%
class u_scalar():
    def eval(self, x):
        return x[0]**2 + x[1]**2

u_s = u_scalar()

class u_vector():
    def eval(self, x):
        return (x[0]**2 + x[1]**2,
               10.0*(x[0]**2 + x[1]**2))

u_v = u_vector()

# %% [markdown]
# # Scalar function CG1

# %%
el = basix.ufl.element(
    "Lagrange",
    domain.ufl_cell().cellname(),
    1,
    basix.LagrangeVariant.gll_warped,
    shape=(domain.geometry.dim-1,),
    dtype=domain.geometry.x.dtype,
)

V = dolfinx.fem.functionspace(domain, el)

u = dolfinx.fem.Function(V)
u.interpolate(u_s.eval)

# %%
filename = "testscg1.vtkhdf"
inf = h5py.File(filename, "w", driver="mpio", comm=comm)
hdf = inf.create_group(np.string_("VTKHDF"))

write_cg(domain, V, u, hdf)

inf.close()

# %% [markdown]
# # Scalar function CG2

# %%
el = basix.ufl.element(
    "Lagrange",
    domain.ufl_cell().cellname(),
    2,
    basix.LagrangeVariant.gll_warped,
    shape=(domain.geometry.dim-1,),
    dtype=domain.geometry.x.dtype,
)

V = dolfinx.fem.functionspace(domain, el)

u = dolfinx.fem.Function(V)
u.interpolate(u_s.eval)

# %%
filename = "testscg2.vtkhdf"
inf = h5py.File(filename, "w", driver="mpio", comm=comm)
hdf = inf.create_group(np.string_("VTKHDF"))

write_cg(domain, V, u, hdf)

inf.close()

# %% [markdown]
# # Vector function CG1

# %%
el = basix.ufl.element(
    "Lagrange",
    domain.ufl_cell().cellname(),
    1,
    basix.LagrangeVariant.gll_warped,
    shape=(domain.geometry.dim,),
    dtype=domain.geometry.x.dtype,
)

V = dolfinx.fem.functionspace(domain, el)

u = dolfinx.fem.Function(V)
u.interpolate(u_v.eval)

# %%
filename = "testvcg1.vtkhdf"
inf = h5py.File(filename, "w", driver="mpio", comm=comm)
hdf = inf.create_group(np.string_("VTKHDF"))

write_cg(domain, V, u, hdf)

inf.close()

# %% [markdown]
# # Vector function CG2

# %%
el = basix.ufl.element(
    "Lagrange",
    domain.ufl_cell().cellname(),
    2,
    basix.LagrangeVariant.gll_warped,
    shape=(domain.geometry.dim,),
    dtype=domain.geometry.x.dtype,
)

V = dolfinx.fem.functionspace(domain, el)

u = dolfinx.fem.Function(V)
u.interpolate(u_v.eval)

# %%
filename = "testvcg2.vtkhdf"
inf = h5py.File(filename, "w", driver="mpio", comm=comm)
hdf = inf.create_group(np.string_("VTKHDF"))

write_cg(domain, V, u, hdf)

inf.close()

# %% [markdown]
# END

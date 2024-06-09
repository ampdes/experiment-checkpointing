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

from contextlib import contextmanager


# %%
@contextmanager
def HDF5File(filename, comm):
    file = h5py.File(filename, "w", driver="mpio", comm=comm)
    hdf = file.create_group(np.string_("VTKHDF"))

    hdf.attrs["Version"] = [2, 2]
    ascii_type = "UnstructuredGrid".encode("ascii")
    hdf.attrs.create(
        "Type", ascii_type, dtype=h5py.string_dtype("ascii", len(ascii_type))
    )

    try:
        yield hdf
    finally:
        file.close()


# %%
def write_grid(hdf, domain, dmap, imap, original=True, V=None):
    # Local geometry
    num_nodes_local = imap.size_local
    local_range = imap.local_range
    if original:
        geom = domain.geometry.x[:num_nodes_local].copy()
    else:
        geom = V.tabulate_dof_coordinates()[:num_nodes_local].copy()

    # Put local geometry
    global_shape = (imap.size_global, 3)
    geom_set = hdf.create_dataset(np.string_("Points"), global_shape, dtype=geom.dtype)
    geom_set[local_range[0] : local_range[1], :] = geom

    # Compute global topology
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
    cell_range = domain.topology.index_map(domain.topology.dim).local_range

    if original:
        cmap = domain.geometry.cmap
        geom_layout = cmap.create_dof_layout()
        num_dofs_per_cell = geom_layout.num_entity_closure_dofs(domain.topology.dim)
        assert dmap.shape[1] == num_dofs_per_cell
        dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)

        dofs_out[:, :] = np.asarray(
            imap.local_to_global(dmap[:num_cells_local, :].reshape(-1))
        ).reshape(dofs_out.shape)

    else:
        num_dofs_per_cell = dmap.list.shape[1]
        dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)

        dofs_out[:, :] = np.asarray(
            imap.local_to_global(dmap.list[:num_cells_local, :].reshape(-1))
        ).reshape(dofs_out.shape)

    cell_type = domain.topology.cell_type
    map_vtk = np.argsort(dolfinx.cpp.io.perm_vtk(cell_type, num_dofs_per_cell))
    dofs_out = dofs_out[:, map_vtk]

    # Put global topology
    geom_set = hdf.create_dataset(
        "Connectivity", (num_cells_global * num_dofs_per_cell,), dtype=np.int64
    )
    geom_set[
        cell_range[0] * num_dofs_per_cell : num_dofs_per_cell * cell_range[1]
    ] = dofs_out.reshape(-1)

    # Put cell type
    # Types
    type_set = hdf.create_dataset("Types", (num_cells_global,), dtype=np.uint8)
    cts = np.full(
        num_cells_local,
        dolfinx.cpp.io.get_vtk_cell_type(cell_type, domain.topology.dim),
    )
    type_set[cell_range[0] : cell_range[1]] = cts

    # Geom dofmap offset
    con_part = hdf.create_dataset("NumberOfConnectivityIds", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
        con_part[domain.comm.rank] = num_cells_global * num_dofs_per_cell

    # Num cells
    num_cells = hdf.create_dataset("NumberOfCells", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
        num_cells[domain.comm.rank] = num_cells_global

    # Num points
    num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
    if domain.comm.rank == 0:
        num_points[domain.comm.rank] = imap.size_global

    # Offsets
    offsets = hdf.create_dataset(
        "Offsets", shape=(num_cells_global + 1,), dtype=np.int64
    )
    offsets[cell_range[0] + 1 : cell_range[1] + 1] = (
        np.arange(1, num_cells_local + 1) * dofs_out.shape[1]
        + cell_range[0] * dofs_out.shape[1]
    )


# %%
def write_mesh(domain, filename):
    comm = domain.comm
    dmap = domain.geometry.dofmap
    imap = domain.geometry.index_map()
    with HDF5File(filename, comm) as hdf:
        write_grid(hdf, domain, dmap, imap, True)


# %%
def write_cg(domain, V, u, filename):
    comm = domain.comm
    dmap = V.dofmap
    imap = V.dofmap.index_map
    with HDF5File(filename, comm) as hdf:
        write_grid(hdf, domain, dmap, imap, False, V)

        # Put function
        u_size_global = u.x.index_map.size_global
        u_size_local = u.x.index_map.size_local
        u_range = u.x.index_map.local_range

        bs = u.x.block_size

        func = hdf.create_group("PointData")
        func_values = func.create_dataset(
            u.name, shape=(u_size_global, bs), dtype=u.dtype
        )
        func_values[u_range[0] : u_range[1], :] = u.x.array[
            : u_size_local * bs
        ].reshape(-1, bs)


# %% [markdown]
# ## Write mesh

# %%
comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(comm, 4, 4)
gdim = domain.geometry.dim

# %%
filename = Path("test.vtkhdf")
write_mesh(domain, filename)


# %% [markdown]
# ## Function expressions


# %%
class u_scalar:
    def eval(self, x):
        return x[0] ** 2 + x[1] ** 2


u_s = u_scalar()


class u_vector:
    def eval(self, x):
        return (x[0] ** 2 + x[1] ** 2, 10.0 * (x[0] ** 2 + x[1] ** 2))


u_v = u_vector()

# %% [markdown]
# ## Scalar function CG1

# %%
el = basix.ufl.element(
    "Lagrange",
    domain.ufl_cell().cellname(),
    1,
    basix.LagrangeVariant.gll_warped,
    shape=(domain.geometry.dim - 1,),
    dtype=domain.geometry.x.dtype,
)

V = dolfinx.fem.functionspace(domain, el)

u = dolfinx.fem.Function(V)
u.interpolate(u_s.eval)

# %%
filename = "testscg1.vtkhdf"
write_cg(domain, V, u, filename)

# %% [markdown]
# ## Scalar function CG2

# %%
el = basix.ufl.element(
    "Lagrange",
    domain.ufl_cell().cellname(),
    2,
    basix.LagrangeVariant.gll_warped,
    shape=(domain.geometry.dim - 1,),
    dtype=domain.geometry.x.dtype,
)

V = dolfinx.fem.functionspace(domain, el)

u = dolfinx.fem.Function(V)
u.interpolate(u_s.eval)

# %%
filename = "testscg2.vtkhdf"
write_cg(domain, V, u, filename)

# %% [markdown]
# ## Vector function CG1

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
write_cg(domain, V, u, filename)

# %% [markdown]
# ## Vector function CG2

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
write_cg(domain, V, u, filename)

# %% [markdown]
# ## DG spaces

# %%

# %% [markdown]
# ## Meshtags

# %%
meshtags = {}
for i in range(domain.topology.dim + 1):
    domain.topology.create_entities(i)
    e_map = domain.topology.index_map(i)

    entities = np.arange(e_map.size_local, dtype=np.int32)

    values = np.arange(e_map.size_local, dtype=np.int32) + e_map.local_range[0]
    meshtags[i] = dolfinx.mesh.meshtags(domain, i, entities, values)

# %%
dof_layout = domain.geometry.cmap.create_dof_layout()

connectivity = {}
for i in range(domain.topology.dim + 1):
    e_map = domain.topology.index_map(i)
    domain.topology.create_connectivity(i, domain.topology.dim)
    entities_to_geometry = dolfinx.mesh.entities_to_geometry(
        domain, i, meshtags[i].indices, False
    )
    
    connectivity[i] = domain.geometry.index_map().local_to_global(entities_to_geometry.reshape(-1))

# %%
meshtags[2].values

# %%
meshtags[2].indices

# %%
connectivity[2]

# %% [markdown]
# ## For writing Meshtags do we need PolyData instead of the UnstructuredGrid? 
#
# See https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/
#

# %% [markdown]
# END

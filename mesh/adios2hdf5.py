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
# # MWE by Jorgen Dokken
#
# Data in ADIOS2 HDF5 files organized in terms of subgroups `"Step0", ..., "StepN"`.
#
# The subgroups in VTKHDF is different than this.

# %%
from pathlib import Path
from mpi4py import MPI
import adios2.bindings
import dolfinx
import numpy as np

# %%
comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(
   comm, 5, 5,)

# %%
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

# %%
file.Put(x_var, geom)
file.EndStep()
file.Close()
adios.RemoveIO("test")

# %%
import h5py
inf = h5py.File("test.h5", "r")
print(inf.keys())

# %% [markdown]
# END

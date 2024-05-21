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
# # ADIOS2 read

# %%
import sys
from mpi4py import MPI

from adios2 import *
import adios2.bindings as bindings


# %% [markdown]
# ## Test from the adios2 repository

# %%
DATA_FILENAME = "hello-world-py.bp"

def write(ad, greeting):
    """write a string to a bp file"""
    io = ad.declare_io("hello-world-writer")
    var_greeting = io.define_variable("Greeting")
    w = io.open(DATA_FILENAME, bindings.Mode.Write)
    w.begin_step()
    w.put(var_greeting, greeting)
    w.end_step()
    w.close()
    return 0

def read(ad):
    """read a string from to a bp file"""
    io = ad.declare_io("hello-world-reader")
    r = io.open(DATA_FILENAME, bindings.Mode.Read)
    r.begin_step()
    var_greeting = io.inquire_variable("Greeting")
    message = r.get(var_greeting)
    r.end_step()
    r.close()
    return message

def test_simple_read_write():
    """driver function"""
    print("ADIOS2 version {0}".format(adios2.__version__))
    ad = adios2.Adios()
    greeting = "Hello World from ADIOS2"
    write(ad, greeting)
    message = read(ad)
    print("{}".format(message))



test_simple_read_write()

# %% [markdown]
# ## Testing read of the `mesh.bp`

# %%
filename = "build/mesh.bp"

# %%
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# %%
adios = adios2.Adios(comm)
io = adios.declare_io("mesh")
engine = io.open(filename, bindings.Mode.Read)

# %%
io.available_attributes()

# %%
io.available_variables()

# %%
io.inquire_variable("name")

# %%
io.inquire_variable("dim")

# %%
name = io.inquire_variable("name")
name

# %% [raw]
# name.Name()

# %%
from adios2 import FileReader

with FileReader(filename) as stream:
    # inspect variables
    variables = stream.available_variables()
    for name, info in variables.items():
        print("variable_name: " + name, end=" ")
        for key, value in info.items():
            print("\t" + key + ": " + value, end=" ")
        print()
    print()

# %% [markdown]
# END

// Mesh read

#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <unistd.h>
#include <mpi.h>

#include <adios2.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;


int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

//   {
//     int i=0;
//     while (i == 0)
//       sleep(5);
//   }

  {

    // ADIOS2
    const std::string fname("mesh");
    adios2::ADIOS adios(MPI_COMM_WORLD);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read
    {
    adios2::IO io = adios.DeclareIO(fname + "-read");
    adios2::Engine reader = io.Open(fname + ".bp", adios2::Mode::Read);

    reader.BeginStep();

    const std::map<std::string, adios2::Params> variables = io.AvailableVariables();
    for (const auto &variablePair : variables)
    {
        std::cout << "Name: " << variablePair.first;

        for (const auto &parameter : variablePair.second)
        {
            std::cout << "\t" << parameter.first << ": " << parameter.second << "\n";
        }
    }

    adios2::Variable<std::string> name = io.InquireVariable<std::string>("name");
    adios2::Variable<std::int16_t> dim = io.InquireVariable<std::int16_t>("dim");
    adios2::Variable<std::string> celltype = io.InquireVariable<std::string>("CellType");
    adios2::Variable<std::int32_t> degree = io.InquireVariable<std::int32_t>("Degree");
    adios2::Variable<std::int32_t> variant = io.InquireVariable<std::int32_t>("Variant");
    adios2::Variable<std::int64_t> n_nodes = io.InquireVariable<std::int64_t>("n_nodes");
    adios2::Variable<std::int64_t> n_cells = io.InquireVariable<std::int64_t>("n_cells");
    adios2::Variable<std::int32_t> n_dofs_per_cell = io.InquireVariable<std::int32_t>("n_dofs_per_cell");
    adios2::Variable<int64_t> input_global_indices = io.InquireVariable<int64_t>("input_global_indices");
    adios2::Variable<T> x = io.InquireVariable<T>("Points");
    adios2::Variable<int64_t> cell_indices = io.InquireVariable<int64_t>("cell_indices");
    adios2::Variable<int32_t> cell_indices_offsets = io.InquireVariable<int32_t>("cell_indices_offsets");

    std::string mesh_name;
    std::int16_t mesh_dim;
    std::string ecelltype;
    std::int32_t edegree;
    std::int32_t evariant;
    std::int64_t num_nodes_global;
    std::int64_t num_cells_global;
    std::int32_t num_dofs_per_cell;
    reader.Get(name, mesh_name);
    reader.Get(dim, mesh_dim);
    reader.Get(celltype, ecelltype);
    reader.Get(degree, edegree);
    reader.Get(variant, evariant);
    reader.Get(n_nodes, num_nodes_global);
    reader.Get(n_cells, num_cells_global);
    reader.Get(n_dofs_per_cell, num_dofs_per_cell);

    std::array<std::int64_t, 2> local_range = dolfinx::MPI::local_range(rank, num_nodes_global, size);
    int num_nodes_local = local_range[1] - local_range[0];

    std::array<std::int64_t, 2> cell_range = dolfinx::MPI::local_range(rank, num_cells_global, size);
    int num_cells_local = cell_range[1] - cell_range[0];

    std::vector<int64_t> mesh_input_global_indices;
    std::vector<T> mesh_x;
    std::vector<int64_t> topo_indices;
    std::vector<int32_t> topo_indices_offsets;

    if (input_global_indices)
    {
        input_global_indices.SetSelection({{local_range[0]}, {num_nodes_local}});
        reader.Get(input_global_indices, mesh_input_global_indices, adios2::Mode::Sync);
    }

    if (x)
    {
        x.SetSelection({{local_range[0],0}, {num_nodes_local,3}});
        reader.Get(x, mesh_x, adios2::Mode::Sync);
    }

    if (cell_indices)
    {
        cell_indices.SetSelection({{cell_range[0]*num_dofs_per_cell}, {cell_range[1]*num_dofs_per_cell}});
        reader.Get(cell_indices, topo_indices, adios2::Mode::Sync);
    }

    if (cell_indices_offsets)
    {
        cell_indices_offsets.SetSelection({{cell_range[0]}, {cell_range[1]}});
        reader.Get(cell_indices_offsets, topo_indices_offsets, adios2::Mode::Sync);
    }

    reader.EndStep();
    reader.Close();
    std::cout << mesh_name << "\n";
    std::cout << "Mesh dimensions: " << mesh_dim << "\n";
    }
  }

  // TODO: Construct mesh following https://github.com/FEniCS/dolfinx/blob/main/cpp/demo/mixed_topology/main.cpp
}

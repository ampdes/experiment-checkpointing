// Mesh read

#include <dolfinx.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/fem/petsc.h>
#include <unistd.h>
#include <mpi.h>

#include <bits/stdc++.h>
#include <adios2.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;


int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
//   MPI_Init(&argc, &argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

//   {
//     int i=0;
//     while (i == 0)
//       sleep(5);
//   }

    // ADIOS2
    const std::string fname("mesh");
    adios2::ADIOS adios(MPI_COMM_WORLD);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read
    adios2::IO io = adios.DeclareIO(fname + "-read");
    adios2::Engine reader = io.Open(fname + ".bp", adios2::Mode::Read);

    reader.BeginStep();

    const auto attributes = io.AvailableAttributes();

    for (const auto &attributepair : attributes)
    {
    std::cout << "Attribute: " << attributepair.first;
    for (const auto &attributepair : attributepair.second)
    {
        std::cout << "\tKey: " << attributepair.first << "\tValue: " << attributepair.second
                    << "\n";
    }
    std::cout << "\n";
    }

    const std::map<std::string, adios2::Params> variables = io.AvailableVariables();
    for (const auto &variablePair : variables)
    {
        std::cout << "Name: " << variablePair.first;

        for (const auto &parameter : variablePair.second)
        {
            std::cout << "\t" << parameter.first << ": " << parameter.second << "\n";
        }
    }

    adios2::Attribute<std::string> name = io.InquireAttribute<std::string>("name");
    adios2::Attribute<std::int16_t> dim = io.InquireAttribute<std::int16_t>("dim");
    adios2::Attribute<std::string> celltype = io.InquireAttribute<std::string>("CellType");
    adios2::Attribute<std::int32_t> degree = io.InquireAttribute<std::int32_t>("Degree");
    adios2::Attribute<std::string> variant = io.InquireAttribute<std::string>("LagrangeVariant");

    // adios2::Variable<std::string> name = io.InquireVariable<std::string>("name");
    // adios2::Variable<std::int16_t> dim = io.InquireVariable<std::int16_t>("dim");
    // adios2::Variable<std::string> celltype = io.InquireVariable<std::string>("CellType");
    // adios2::Variable<std::int32_t> degree = io.InquireVariable<std::int32_t>("Degree");
    // adios2::Variable<std::string> variant = io.InquireVariable<std::string>("LagrangeVariant");
    adios2::Variable<std::int64_t> n_nodes = io.InquireVariable<std::int64_t>("n_nodes");
    adios2::Variable<std::int64_t> n_cells = io.InquireVariable<std::int64_t>("n_cells");
    adios2::Variable<std::int32_t> n_dofs_per_cell = io.InquireVariable<std::int32_t>("n_dofs_per_cell");
    adios2::Variable<int64_t> input_global_indices = io.InquireVariable<int64_t>("input_global_indices");
    adios2::Variable<T> x = io.InquireVariable<T>("Points");
    adios2::Variable<int64_t> cell_indices = io.InquireVariable<int64_t>("cell_indices");
    adios2::Variable<int32_t> cell_indices_offsets = io.InquireVariable<int32_t>("cell_indices_offsets");

    // std::string mesh_name;
    // std::int16_t mesh_dim;
    // std::string ecelltype;
    // std::int32_t edegree;
    // std::string evariant;

    std::string mesh_name = name.Data()[0];
    std::int16_t mesh_dim = dim.Data()[0];
    std::string ecelltype = celltype.Data()[0];
    std::int32_t edegree = degree.Data()[0];
    std::string evariant = variant.Data()[0];

    std::int64_t num_nodes_global;
    std::int64_t num_cells_global;
    std::int32_t num_dofs_per_cell;
    // reader.Get(name, mesh_name);
    // reader.Get(dim, mesh_dim);
    // reader.Get(celltype, ecelltype);
    // reader.Get(degree, edegree);
    // reader.Get(variant, evariant);
    reader.Get(n_nodes, num_nodes_global);
    reader.Get(n_cells, num_cells_global);
    reader.Get(n_dofs_per_cell, num_dofs_per_cell);

    std::array<std::int64_t, 2> local_range = dolfinx::MPI::local_range(rank, num_nodes_global, size);
    int num_nodes_local = local_range[1] - local_range[0];

    std::array<std::int64_t, 2> cell_range = dolfinx::MPI::local_range(rank, num_cells_global, size);
    int num_cells_local = cell_range[1] - cell_range[0];

    std::vector<int64_t> mesh_input_global_indices(num_nodes_local);
    std::vector<T> mesh_x(num_nodes_local*3);
    std::vector<int64_t> topo_indices(num_cells_local*num_dofs_per_cell);
    std::vector<int32_t> topo_indices_offsets(num_cells_local+1);

    // std::vector<int64_t> mesh_input_global_indices;
    // std::vector<T> mesh_x;
    // std::vector<int64_t> topo_indices;
    // std::vector<int32_t> topo_indices_offsets;

    if (input_global_indices)
    {
        input_global_indices.SetSelection({{local_range[0]}, {num_nodes_local}});
        reader.Get(input_global_indices, mesh_input_global_indices.data(), adios2::Mode::Deferred);
    }

    if (x)
    {
        x.SetSelection({{local_range[0],0}, {num_nodes_local,3}});
        reader.Get(x, mesh_x.data(), adios2::Mode::Deferred);
    }

    if (cell_indices)
    {
        cell_indices.SetSelection({{cell_range[0]*num_dofs_per_cell}, {cell_range[1]*num_dofs_per_cell}});
        reader.Get(cell_indices, topo_indices.data(), adios2::Mode::Deferred);
    }

    if (cell_indices_offsets)
    {
        cell_indices_offsets.SetSelection({{cell_range[0]}, {cell_range[1]+1}});
        reader.Get(cell_indices_offsets, topo_indices_offsets.data(), adios2::Mode::Deferred);
    }

    reader.EndStep();
    reader.Close();

    std::cout << mesh_name << "\n";
    std::cout << "Mesh dimensions: " << mesh_dim << "\n";

    std::int32_t cell_offset = topo_indices_offsets[0];
    for (int i = 0; i < topo_indices_offsets.size(); ++i)
        topo_indices_offsets[i] -= cell_offset;

    // WIP: Construct mesh following https://github.com/FEniCS/dolfinx/blob/main/cpp/demo/mixed_topology/main.cpp
    graph::AdjacencyList<std::int64_t> cells_list(topo_indices, topo_indices_offsets);
    std::vector<std::int64_t> original_global_index(num_nodes_local);
    std::iota(original_global_index.begin(), original_global_index.end(), 0);
    std::vector<int> ghost_owners;
    std::vector<std::int64_t> boundary_vertices;

    mesh::CellType cell_type{mesh::to_type(ecelltype)};
    fem::CoordinateElement<double> element = fem::CoordinateElement<double>(cell_type, edegree);

    auto topo = std::make_shared<mesh::Topology>(mesh::create_topology(
            MPI_COMM_WORLD, topo_indices, original_global_index, ghost_owners,
            cell_type, boundary_vertices));

    auto topo_cells = topo->connectivity(2, 0);

    topo->create_connectivity(1, 0);

    if (rank == 0){
        std::cout << "cells\n------\n";
        for (int i = 0; i < topo_cells->num_nodes(); ++i)
            {
            std::cout << i << " [";
            for (auto q : topo_cells->links(i))
                std::cout << q << " ";
            std::cout << "]\n";
            }

        std::cout << "facets\n------\n";
        auto topo_facets = topo->connectivity(1, 0);
        for (int i = 0; i < topo_facets->num_nodes(); ++i)
            {
            std::cout << i << " [";
            for (auto q : topo_facets->links(i))
                std::cout << q << " ";
            std::cout << "]\n";
            }
    }

    std::vector<T> mesh_x_reduced(num_nodes_local*mesh_dim);
    for (int i = 0; i < num_nodes_local; ++i)
    {
        for (int j = 0; j < mesh_dim; ++j)
        mesh_x_reduced[i*mesh_dim + j] = mesh_x[i*3 + j];
    }

    std::sort(mesh_input_global_indices.begin(), mesh_input_global_indices.end());
    mesh::Geometry geometry = mesh::create_geometry(*topo, element,
                                                    mesh_input_global_indices, topo_indices, mesh_x_reduced, mesh_dim);

    mesh::Mesh<double> mesh(MPI_COMM_WORLD, topo, geometry);

//   MPI_Finalize();
  PetscFinalize();

  return 0;
}

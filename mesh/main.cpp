// Create a documentation of the DOLFINx data structures
// through a unit square mesh. And try out ADIOS2.

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

  {
    int i=0;
    while (i == 0)
      sleep(5);
  }

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                  {2, 2}, mesh::CellType::quadrilateral, part));

    const mesh::Geometry<U>& geometry = mesh->geometry();
    auto topology = mesh->topology();

    // ADIOS2
    const std::string fname("mesh");
    adios2::ADIOS adios(MPI_COMM_WORLD);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Write
    {
    adios2::IO io = adios.DeclareIO(fname + "-write");
    adios2::Engine writer = io.Open(fname + ".bp", adios2::Mode::Write);

    const std::string mesh_name = mesh->name;
    const std::int16_t mesh_dim = geometry.dim();
    const std::vector<int64_t> mesh_input_global_indices = geometry.input_global_indices();
    const std::span<const int64_t> mesh_input_global_indices_span(mesh_input_global_indices.begin(),
                                                            mesh_input_global_indices.end());
    const std::span<const T> mesh_x = geometry.x();

    auto imap = mesh->geometry().index_map();
    const std::int64_t num_nodes_global = imap->size_global();
    const std::int32_t num_nodes_local = imap->size_local();
    const std::int64_t offset = imap->local_range()[0];

    auto dmap = mesh->geometry().dofmap();

    // const std::shared_ptr<const dolfinx::common::IndexMap> topo0_im = topology->index_map(0);
    const std::shared_ptr<const dolfinx::common::IndexMap> topo_imap = topology->index_map(mesh_dim);
    const std::int64_t num_cells_global = topo_imap->size_global();
    const std::int32_t num_cells_local = topo_imap->size_local();
    const std::int64_t cell_offset = topo_imap->local_range()[0];

    adios2::Variable<std::string> name = io.DefineVariable<std::string>("name");
    adios2::Variable<std::int16_t> dim = io.DefineVariable<std::int16_t>("dim");
    adios2::Variable<std::int64_t> n_nodes = io.DefineVariable<std::int64_t>("n_nodes");
    adios2::Variable<std::int64_t> n_cells = io.DefineVariable<std::int64_t>("n_cells");
    adios2::Variable<std::int64_t> input_global_indices = io.DefineVariable<std::int64_t>("input_global_indices",
                                                                                          {num_nodes_global},
                                                                                          {offset},
                                                                                          {num_nodes_local},
                                                                                          adios2::ConstantDims);

    adios2::Variable<T> x = io.DefineVariable<T>("x", 
                                                 {num_nodes_global, 3},
                                                 {offset, 0},
                                                 {num_nodes_local, 3},
                                                 adios2::ConstantDims);

    // adios2::Variable<std::int64_t> original_cell_indices = io.DefineVariable<std::int64_t>("original_cell_indices",
    //                                                                                       {num_cells_global},
    //                                                                                       {cell_offset},
    //                                                                                       {num_cells_local},
    //                                                                                       adios2::ConstantDims);

    // WIP
    auto connectivity = topology->connectivity(mesh_dim, 0);
    // auto indices = connectivity->array();
    const std::vector<int32_t> indices = connectivity->array();
    const std::span<const int32_t> indices_span(indices.begin(),
                                                indices.size());

    // indices.end() is not 16 but a huge number!!

    // int32_t temp = connectivity->offsets()[num_cells_local];
    // const std::span<const int32_t> tempvec = indices_span.subspan(0, temp);
    // const std::vector<std::int64_t> connectivity_nodes_global = graph::build::compute_local_to_global(
    //                                                                  mesh_input_global_indices_span,
    //                                                                  tempvec
    //                                                                  );

    writer.BeginStep();
    writer.Put(name, mesh_name);
    writer.Put(dim, mesh_dim);
    writer.Put(n_nodes, num_nodes_global);
    writer.Put(n_cells, num_cells_global);
    writer.Put(input_global_indices, mesh_input_global_indices_span.subspan(0, num_nodes_local).data());
    writer.Put(x, mesh_x.subspan(0, num_nodes_local*3).data());
    writer.EndStep();
    writer.Close();
    }

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
    adios2::Variable<std::int64_t> n_nodes = io.InquireVariable<std::int64_t>("n_nodes");
    adios2::Variable<std::int64_t> n_cells = io.InquireVariable<std::int64_t>("n_cells");

    std::string mesh_name;
    std::int16_t mesh_dim;
    std::int64_t num_nodes_global;
    std::int64_t num_cells_global;
    reader.Get(name, mesh_name);
    reader.Get(dim, mesh_dim);
    reader.Get(n_nodes, num_nodes_global);
    reader.Get(n_cells, num_cells_global);

    adios2::Variable<int64_t> input_global_indices = io.InquireVariable<int64_t>("input_global_indices");
    adios2::Variable<T> x = io.InquireVariable<T>("x");

    const std::size_t Nindices = 9;
    if (input_global_indices) // means not found
    {
        std::vector<int64_t> mesh_input_global_indices;
        // read only the chunk corresponding to our rank
        input_global_indices.SetSelection({{Nindices * rank}, {Nindices}});

        reader.Get(input_global_indices, mesh_input_global_indices, adios2::Mode::Sync);

        if (rank == 0)
        {
            std::cout << "input_global_indices: \n";
            for (const auto number : mesh_input_global_indices)
            {
                std::cout << number << " ";
            }
            std::cout << "\n";
        }
    }
    reader.EndStep();
    reader.Close();
    std::cout << mesh_name << "\n";
    std::cout << "Mesh dimensions: " << mesh_dim << "\n";
    }
  }
}

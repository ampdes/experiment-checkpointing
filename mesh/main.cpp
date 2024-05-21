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
    const std::span<const T> _mesh_x = geometry.x();

    // std::mdspan<const int32_t, 
    //             std::dextents<std::size_t, 2UL>,
    //             std::layout_right, std::default_accessor<const int32_t>>
    //             mesh_dofmaps_x = geometry.dofmap();

    // Perhaps need to expose the currently private attribute
    // std::vector<std::vector<std::int32_t>> mesh_dofmaps_x = geometry._dofmaps;

    const std::shared_ptr<const dolfinx::common::IndexMap> index_map_ptr = geometry.index_map();
    // int size_global = index_map_ptr->size_global();
    // int size_local = index_map_ptr->size_local();

    std::vector<dolfinx::mesh::CellType> entities0 = topology->entity_types(0);
    std::vector<dolfinx::mesh::CellType> entities1 = topology->entity_types(1);
    std::vector<dolfinx::mesh::CellType> entities2 = topology->entity_types(2);

    const std::shared_ptr<const dolfinx::common::IndexMap> topo0_im = topology->index_map(0);
    const std::shared_ptr<const dolfinx::common::IndexMap> topo2_im = topology->index_map(2);

    // Need to compute the sizes correctly, by excluding the ghosts
    const std::size_t Nindices = mesh_input_global_indices.size();
    const std::size_t Nx = _mesh_x.size();

    adios2::Variable<std::string> name = io.DefineVariable<std::string>("name");
    adios2::Variable<std::int16_t> dim = io.DefineVariable<std::int16_t>("dim");
    adios2::Variable<std::int64_t> input_global_indices = io.DefineVariable<std::int64_t>("input_global_indices",
                                                                                          {size * Nindices},
                                                                                          {rank * Nindices},
                                                                                          {Nindices},
                                                                                          adios2::ConstantDims);
    adios2::Variable<std::float_t> x = io.DefineVariable<std::float_t>("x", 
                                                                       {size * Nx},
                                                                       {rank * Nx},
                                                                       {Nx}, adios2::ConstantDims);

    writer.BeginStep();
    writer.Put(name, mesh_name);
    writer.Put(dim, mesh_dim);
    writer.Put(input_global_indices, mesh_input_global_indices.data());
    // How to write PETScScalar?
    // writer.Put(x, _mesh_x.data());
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

    std::string mesh_name;
    std::int16_t mesh_dim;
    reader.Get(name, mesh_name);
    reader.Get(dim, mesh_dim);

    adios2::Variable<int64_t> input_global_indices = io.InquireVariable<int64_t>("input_global_indices");

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

// Create a documentation of the DOLFINx data structures
// through a unit square mesh. And try out ADIOS2.

#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <unistd.h>

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
    // Write
    const std::string fname("mesh");

    adios2::ADIOS adios(MPI_COMM_WORLD);
    adios2::IO io = adios.DeclareIO(fname);
    adios2::Engine writer = io.Open(fname + ".bp", adios2::Mode::Write);

    const std::string mesh_name = mesh->name;
    const std::int8_t mesh_dim = geometry.dim();

    adios2::Variable<std::string> name = io.DefineVariable<std::string>("Mesh name");
    adios2::Variable<std::int8_t> dim = io.DefineVariable<std::int8_t>("Mesh dim");

    writer.BeginStep();
    writer.Put(name, mesh_name);
    writer.Put(dim, mesh_dim);
    writer.EndStep();
    writer.Close();
    // Read

  }
}

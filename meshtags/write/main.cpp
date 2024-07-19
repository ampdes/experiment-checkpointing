// Mesh write

#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <mpi.h>
#include <unistd.h>

#include <adios2.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char *argv[]) {
  dolfinx::init_logging(argc, argv);
  // MPI_Init(&argc, &argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

    // {
    //   int i=0;
    //   while (i == 0)
    //     sleep(5);
    // }

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<U>>(
      mesh::create_box<U>(MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
                          {3, 4, 5}, mesh::CellType::tetrahedron, part));

  // auto mesh = std::make_shared<mesh::Mesh<U>>(
  //     mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
  //                         {2, 2}, mesh::CellType::quadrilateral, part));

  // ADIOS2
  const std::string fname("meshtags");
  adios2::ADIOS adios(MPI_COMM_WORLD);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Write
  adios2::IO io = adios.DeclareIO(fname + "-write");
  adios2::Engine writer = io.Open(fname + ".bp", adios2::Mode::Write);

  const dolfinx::mesh::Geometry<U> &geometry = mesh->geometry();
  auto topology = mesh->topology();

  // Cell meshtags
  int c_dim = geometry.dim();
  topology->create_entities(c_dim);
  const std::shared_ptr<const dolfinx::common::IndexMap> c_map =
      topology->index_map(c_dim);

  std::uint32_t num_entities = c_map->size_local();

  auto cmap = mesh->geometry().cmap();
  auto geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_entity =
      geom_layout.num_entity_closure_dofs(c_dim);

//   topology->create_connectivity(c_dim, c_dim);
  auto entities = topology->connectivity(c_dim, 0);
  std::vector<int32_t> entities_array(num_entities*num_dofs_per_entity);
  std::vector<int32_t> entities_offsets(num_entities+1);
//   std::vector<std::uint32_t> entities(num_entities);
  std::uint64_t c_offset = c_map->local_range()[0];
  std::vector<double> values(num_entities);

  for (int i = 0; i < (int)num_entities; ++i)
    {
    values[i] = (double)(i + c_offset);
    // entities[i] = i;
    }

  for (int i = 0; i < (int)num_entities+1; ++i)
    entities_offsets[i] = entities->offsets()[i];

  for (int i = 0; i < (int)(num_entities*num_dofs_per_entity); ++i)
    entities_array[i] = entities->array()[i];

  graph::AdjacencyList<std::int32_t> entities_local(entities_array, entities_offsets);

  std::vector<std::int32_t> indices_c = mesh::entities_to_index(*topology, c_dim, entities->array());
  std::cout << c_map->num_ghosts() << std::endl;
  std::cout << entities->array().size() << std::endl;
  std::cout << entities->offsets().size() << std::endl;
  std::cout << entities_local.array().size() << std::endl;
  std::cout << entities_local.offsets().size() << std::endl;
  std::cout << indices_c.size() << std::endl;
  std::cout << values.size() << std::endl;


  auto meshtags = std::make_shared<mesh::MeshTags<U>>(
      mesh::create_meshtags<U>(topology, c_dim, entities_local, values));
  
  // --------------------------------------------------------------------
  // meshtagsdata
  auto tag_entities = meshtags->indices();
  auto dim = meshtags->dim();
  std::uint32_t num_tag_entities_local = meshtags->topology()->index_map(dim)->size_local();

  int num_tag_entities = tag_entities.size();

  std::vector<std::int32_t> local_tag_entities;
  local_tag_entities.reserve(num_tag_entities);

  std::uint64_t num_saved_tag_entities = 0;
  for (int i = 0; i < num_tag_entities; ++i)
  {
    if (tag_entities[i] < (int)num_tag_entities_local)
    {
        num_saved_tag_entities += 1;
        local_tag_entities.push_back(tag_entities[i]);
        // local_tag_entities[num_local_tag_entities] = tag_entities[i];
    }
  }
  local_tag_entities.resize(num_saved_tag_entities);

  auto mtvalues = meshtags->values();
  const std::span<const double> local_values(mtvalues.begin(), mtvalues.end());

  // Compute the global offset for owned (local) vertex indices
  std::uint64_t local_start = 0;
  {
    MPI_Exscan(&num_saved_tag_entities, &local_start, 1, MPI_UINT64_T, MPI_SUM, mesh->comm());
  }
 
  std::uint64_t num_tag_entities_global = 0;
  MPI_Allreduce(&num_saved_tag_entities, &num_tag_entities_global, 1, MPI_UINT64_T, MPI_SUM, mesh->comm());

  std::vector<std::int32_t> entities_to_geometry =
      mesh::entities_to_geometry(*mesh, dim, tag_entities, false);
  
  auto imap = mesh->geometry().index_map();
  std::vector<std::int64_t> gindices(entities_to_geometry.size());

  std::iota(gindices.begin(), gindices.end(), 0);

  imap->local_to_global(entities_to_geometry, gindices);

  std::string name = meshtags->name;

  io.DefineAttribute<std::string>("name_meshtags", meshtags->name);

  adios2::Variable<std::int64_t> topology_var =
      io.DefineVariable<std::int64_t>(name + "_topology",
                                      {num_tag_entities_global, num_dofs_per_entity},
                                      {local_start, 0},
                                      {num_tag_entities_local, num_dofs_per_entity},
                                      adios2::ConstantDims);

  adios2::Variable<T> values_var =
      io.DefineVariable<T>(name + "_values",
                           {num_tag_entities_global},
                           {local_start},
                           {num_saved_tag_entities},
                           adios2::ConstantDims);


  writer.BeginStep();
  // meshtags
  writer.Put(topology_var, gindices.data());
  writer.Put(values_var, local_values.subspan(0, num_saved_tag_entities).data());

  writer.EndStep();
  writer.Close();

  // MPI_Finalize();
  PetscFinalize();

  return 0;
}

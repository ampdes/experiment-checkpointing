// Mesh write

#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <mpi.h>
#include <unistd.h>

#include <adios2.h>

std::map<basix::element::lagrange_variant, std::string> lagrange_variants{
    {basix::element::lagrange_variant::unset, "unset"},
    {basix::element::lagrange_variant::equispaced, "equispaced"},
    {basix::element::lagrange_variant::gll_warped, "gll_warped"},
};

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char *argv[]) {
  dolfinx::init_logging(argc, argv);
  // MPI_Init(&argc, &argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

    {
      int i=0;
      while (i == 0)
        sleep(5);
    }

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
//   auto mesh = std::make_shared<mesh::Mesh<U>>(
//       mesh::create_box<U>(MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
//                           {2, 2, 2}, mesh::CellType::tetrahedron, part));

  auto mesh = std::make_shared<mesh::Mesh<U>>(
      mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                          {2, 2}, mesh::CellType::quadrilateral, part));

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
  std::int16_t mesh_dim = geometry.dim();

  // Cell meshtags
  int c_dim = geometry.dim();
  topology->create_entities(c_dim);
  const std::shared_ptr<const dolfinx::common::IndexMap> c_map =
      topology->index_map(c_dim);

  std::uint32_t num_entities = c_map->size_local();

//   topology->create_connectivity(c_dim, c_dim);
  auto entities = topology->connectivity(c_dim, 0);
//   std::vector<std::uint32_t> entities(num_entities);
  std::uint64_t c_offset = c_map->local_range()[0];
  std::vector<double> values(num_entities);

  for (int i = 0; i < (int)num_entities; ++i)
    {
    values[i] = (double)(i + c_offset);
    // entities[i] = i;
    }

  auto meshtags = std::make_shared<mesh::MeshTags<U>>(
      mesh::create_meshtags<U>(topology, c_dim, *entities, values));
  
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
//   const std::span<const int32_t> local_values(mtvalues.begin(), mtvalues.end());

  // Compute the global offset for owned (local) vertex indices
  std::uint64_t local_start = 0;
  {
    MPI_Exscan(&num_saved_tag_entities, &local_start, 1, MPI_UINT64_T, MPI_SUM, mesh->comm());
  }
 
  std::uint64_t num_tag_entities_global = 0;
  MPI_Allreduce(&num_saved_tag_entities, &num_tag_entities_global, 1, MPI_UINT64_T, MPI_SUM, mesh->comm());

  auto cmap = mesh->geometry().cmap();
  auto geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_entity =
      geom_layout.num_entity_closure_dofs(dim);

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

//   // meshtdata
//   // auto imap = mesh->geometry().index_map();
//   std::uint64_t num_nodes_global = imap->size_global();
//   std::uint32_t num_nodes_local = imap->size_local();
//   std::uint64_t offset = imap->local_range()[0];

//   const std::shared_ptr<const dolfinx::common::IndexMap> topo_imap =
//       topology->index_map(mesh_dim);
//   std::uint64_t num_cells_global = topo_imap->size_global();
//   std::uint32_t num_cells_local = topo_imap->size_local();
//   std::uint64_t cell_offset = topo_imap->local_range()[0];

// //   auto cmap = mesh->geometry().cmap();
// //   auto geom_layout = cmap.create_dof_layout();
//   std::uint32_t num_dofs_per_cell =
//       geom_layout.num_entity_closure_dofs(mesh_dim);

//   const std::vector<int64_t> mesh_input_global_indices =
//       geometry.input_global_indices();
//   const std::span<const int64_t> mesh_input_global_indices_span(
//       mesh_input_global_indices.begin(), mesh_input_global_indices.end());
//   const std::span<const T> mesh_x = geometry.x();

//   auto connectivity = topology->connectivity(mesh_dim, 0);
//   auto indices = connectivity->array();
//   const std::span<const int32_t> indices_span(indices.begin(), indices.end());

//   auto indices_offsets = connectivity->offsets();

//   std::vector<std::int64_t> connectivity_nodes_global(
//       indices_offsets[num_cells_local]);

//   std::iota(connectivity_nodes_global.begin(), connectivity_nodes_global.end(),
//             0);

//   imap->local_to_global(
//       indices_span.subspan(0, indices_offsets[num_cells_local]),
//       connectivity_nodes_global);

//   for (std::size_t i = 0; i < indices_offsets.size(); ++i) {
//     indices_offsets[i] += cell_offset * num_dofs_per_cell;
//   }

//   const std::span<const int32_t> indices_offsets_span(indices_offsets.begin(),
//                                                       indices_offsets.end());

//   io.DefineAttribute<std::string>("name", mesh->name);
//   io.DefineAttribute<std::int16_t>("dim", geometry.dim());
//   io.DefineAttribute<std::string>("CellType",
//                                   dolfinx::mesh::to_string(cmap.cell_shape()));
//   io.DefineAttribute<std::int32_t>("Degree", cmap.degree());
//   io.DefineAttribute<std::string>("LagrangeVariant",
//                                   lagrange_variants[cmap.variant()]);

//   adios2::Variable<std::uint64_t> n_nodes =
//       io.DefineVariable<std::uint64_t>("n_nodes");
//   adios2::Variable<std::uint64_t> n_cells =
//       io.DefineVariable<std::uint64_t>("n_cells");
//   adios2::Variable<std::uint32_t> n_dofs_per_cell =
//       io.DefineVariable<std::uint32_t>("n_dofs_per_cell");

//   adios2::Variable<std::int64_t> input_global_indices =
//       io.DefineVariable<std::int64_t>("input_global_indices",
//                                       {num_nodes_global}, {offset},
//                                       {num_nodes_local}, adios2::ConstantDims);

//   adios2::Variable<T> x =
//       io.DefineVariable<T>("Points", {num_nodes_global, 3}, {offset, 0},
//                            {num_nodes_local, 3}, adios2::ConstantDims);

//   adios2::Variable<std::int64_t> cell_indices = io.DefineVariable<std::int64_t>(
//       "cell_indices", {num_cells_global * num_dofs_per_cell},
//       {cell_offset * num_dofs_per_cell}, {num_cells_local * num_dofs_per_cell},
//       adios2::ConstantDims);

//   adios2::Variable<std::int32_t> cell_indices_offsets =
//       io.DefineVariable<std::int32_t>(
//           "cell_indices_offsets", {num_cells_global + 1}, {cell_offset},
//           {num_cells_local + 1}, adios2::ConstantDims);

  writer.BeginStep();
//   writer.Put(n_nodes, num_nodes_global);
//   writer.Put(n_cells, num_cells_global);
//   writer.Put(n_dofs_per_cell, num_dofs_per_cell);
//   writer.Put(input_global_indices,
//              mesh_input_global_indices_span.subspan(0, num_nodes_local).data());
//   writer.Put(x, mesh_x.subspan(0, num_nodes_local * 3).data());
//   writer.Put(cell_indices, connectivity_nodes_global.data());
//   writer.Put(cell_indices_offsets,
//              indices_offsets_span.subspan(0, num_cells_local + 1).data());

  // meshtags
  writer.Put(topology_var, gindices.data());
//   writer.Put(values_var, local_values.subspan(0, num_saved_tag_entities).data());

  writer.EndStep();
  writer.Close();

  // MPI_Finalize();
  PetscFinalize();

  return 0;
}

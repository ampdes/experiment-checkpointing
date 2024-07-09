// Function write

#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <unistd.h>
#include <mpi.h>
#include <basix/finite-element.h>

#include <adios2.h>


std::map<basix::element::lagrange_variant, std::string> lagrange_variants {
                      {basix::element::lagrange_variant::unset, "unset"},
                      {basix::element::lagrange_variant::equispaced, "equispaced"},
                      {basix::element::lagrange_variant::gll_warped, "gll_warped"},
                    };

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;


int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  // MPI_Init(&argc, &argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

//   {
//     int i=0;
//     while (i == 0)
//       sleep(5);
//   }

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<U>>(
      mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                {2, 2}, mesh::CellType::quadrilateral, part));

  const mesh::Geometry<U>& geometry = mesh->geometry();
  auto topology = mesh->topology();

  auto element = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::quadrilateral, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace(mesh, element, {}));

  auto func = std::make_shared<fem::Function<T>>(V);

  // Interpolate sin(2 \pi x[0]) in the scalar Lagrange finite element
  // space
  func->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
          std::vector<T> f(x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
          f[p] = std::sin(2 * std::numbers::pi * x(0, p));
          return {f, {f.size()}};
      });

  // // Vector
  // auto W = std::make_shared<fem::FunctionSpace<U>>(
  //     fem::create_functionspace(mesh, element, {2}));

  // // Define solution function
  // auto vec_func = std::make_shared<fem::Function<T>>(W);

  // ADIOS2
  const std::string fname("function");
  adios2::ADIOS adios(MPI_COMM_WORLD);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Write
  const std::string mesh_name = mesh->name;
  const std::int16_t mesh_dim = geometry.dim();
  const std::vector<int64_t> mesh_input_global_indices = geometry.input_global_indices();
  const std::span<const int64_t> mesh_input_global_indices_span(mesh_input_global_indices.begin(),
                                                          mesh_input_global_indices.end());
  const std::span<const T> mesh_x = geometry.x();

  auto imap = mesh->geometry().index_map();
  std::uint64_t num_nodes_global = imap->size_global();
  std::uint32_t num_nodes_local = imap->size_local();
  std::uint64_t offset = imap->local_range()[0];

  const std::shared_ptr<const dolfinx::common::IndexMap> topo_imap = topology->index_map(mesh_dim);
  std::uint64_t num_cells_global = topo_imap->size_global();
  std::uint32_t num_cells_local = topo_imap->size_local();
  std::uint64_t cell_offset = topo_imap->local_range()[0];

  auto cmap = mesh->geometry().cmap();
  auto edegree = cmap.degree();
  auto ecelltype = cmap.cell_shape();
  auto elagrange_variant = cmap.variant();
  auto geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh_dim);

  auto connectivity = topology->connectivity(mesh_dim, 0);
  auto indices = connectivity->array();
  const std::span<const int32_t> indices_span(indices.begin(),
                                              indices.end());

  auto indices_offsets = connectivity->offsets();

  std::vector<std::int64_t> connectivity_nodes_global(indices_offsets[num_cells_local]);

  imap->local_to_global(indices_span.subspan(0, indices_offsets[num_cells_local]), connectivity_nodes_global);

  for (std::size_t i = 0; i < indices_offsets.size(); ++i)
  {
      indices_offsets[i] += cell_offset*num_dofs_per_cell;
  }

  const std::span<const int32_t> indices_offsets_span(indices_offsets.begin(),
                                                      indices_offsets.end());


  // function data
  std::string funcname = func->name();
  auto dofmap = func->function_space()->dofmap();
  auto values = func->x()->array();

  auto dmap = dofmap->map();
  auto dofmap_bs = dofmap->bs();
  auto num_dofs_per_cell_v = dmap.extent(1);
  auto num_dofs_local_dmap = num_cells_local * num_dofs_per_cell_v * dofmap_bs;
  auto index_map_bs = dofmap->index_map_bs();

  // std::vector<std::int32_t> dofs;
  // dofs.reserve(dmap.extent(0) * dmap.extent(1));
  // for (std::size_t i = 0; i < dmap.extent(0); ++i)
  //   for (std::size_t j = 0; j < dmap.extent(1); ++j)
  //     dofs.push_back(dmap(i, j));
  // for (auto ss : dofs)
  //   std::cout << ss << " ";

  std::vector<std::int32_t> dofs;
  dofs.reserve(num_dofs_local_dmap);

  // Use int16?
  std::vector<std::int32_t> rems;
  rems.reserve(num_dofs_local_dmap);
  int temp;

  for (std::size_t i = 0; i < num_cells_local; ++i)
    for (std::size_t j = 0; j < num_dofs_per_cell_v; ++j)
      for (std::size_t k = 0; k < dofmap_bs; ++k)
        temp = dmap(i, j) * dofmap_bs + k;
        dofs.push_back(std::floor(temp/index_map_bs));
        rems.push_back(temp % index_map_bs);

  for (auto ss : dofs)
    std::cout << ss << " ";

  auto dofmap_imap = dofmap.index_map();
  std::uint32_t dofmap_offset = dofmap_imap.local_range()[0];
  std::uint32_t num_dofmap_size = dofmap_imap.size_global();

  auto local_imap = dolfinx::common::IndexMap(mesh->comm, num_dofs_local_dmap);
  std::vector<std::int64_t> dofs_global(num_dofs_local_dmap);

  dofmap_imap->local_to_global(dofs, dofs_global);
  for (std::size_t i = 0; i < num_dofs_local_dmap; ++i)
    dofs_global[i] = dofs_global[i] * index_map_bs + rems[i];

  // Compute dofmap offsets
  std::uint32_t dofmap_offset = local_imap.local_range()[0];
  std::vector<std::int64_t> local_dofmap_offsets(num_cells_local + 1);
  for (std::size_t i = 0; i < num_dofs_local_dmap; ++i)
    local_dofmap_offsets[i] = i * num_dofs_per_cell * dofmap_bs + dofmap_offset;

  std::uint64_t num_dofs_global = local_imap.size_global() * index_map_bs;
  auto num_dofs_local = (local_imap.local_range()[1] - local_imap.local_range()[0]) * index_map_bs;

  //
  adios2::IO io = adios.DeclareIO(fname + "-write");
  adios2::Engine writer = io.Open(fname + ".bp", adios2::Mode::Write);
  
  io.DefineAttribute<std::string>("name", mesh_name);
  io.DefineAttribute<std::int16_t>("dim", mesh_dim);
  io.DefineAttribute<std::string>("CellType", mesh::to_string(cmap.cell_shape()));
  io.DefineAttribute<std::int32_t>("Degree", cmap.degree());
  io.DefineAttribute<std::string>("LagrangeVariant", lagrange_variants[elagrange_variant]);

  adios2::Variable<std::uint64_t> n_nodes = io.DefineVariable<std::uint64_t>("n_nodes");
  adios2::Variable<std::uint64_t> n_cells = io.DefineVariable<std::uint64_t>("n_cells");
  adios2::Variable<std::uint32_t> n_dofs_per_cell = io.DefineVariable<std::uint32_t>("n_dofs_per_cell");

  adios2::Variable<std::int64_t> input_global_indices = io.DefineVariable<std::int64_t>("input_global_indices",
                                                                                        {num_nodes_global},
                                                                                        {offset},
                                                                                        {num_nodes_local},
                                                                                        adios2::ConstantDims);

  adios2::Variable<T> x = io.DefineVariable<T>("Points",
                                                {num_nodes_global, 3},
                                                {offset, 0},
                                                {num_nodes_local, 3},
                                                adios2::ConstantDims);

  // To get the true size of the global array,
  // we can gather `indices_offsets[num_cells_local]`
  adios2::Variable<std::int64_t> cell_indices = io.DefineVariable<std::int64_t>("cell_indices",
                                                                                {num_cells_global*num_dofs_per_cell},
                                                                                {cell_offset*num_dofs_per_cell},
                                                                                {num_cells_local*num_dofs_per_cell},
                                                                                adios2::ConstantDims);

  adios2::Variable<std::int32_t> cell_indices_offsets = io.DefineVariable<std::int32_t>("cell_indices_offsets",
                                                                                {num_cells_global+1},
                                                                                {cell_offset},
                                                                                {num_cells_local+1},
                                                                                adios2::ConstantDims);

  // adios2::Variable<std::int64_t> original_cell_indices = io.DefineVariable<std::int64_t>("original_cell_indices",
  //                                                                                       {num_cells_global},
  //                                                                                       {cell_offset},
  //                                                                                       {num_cells_local},
  //                                                                                       adios2::ConstantDims);

  adios2::Variable<std::int32_t> pvar = io.DefineVariable<std::int32_t>("CellPermutations",
                                                                         {num_cells_global},
                                                                         {cell_offset},
                                                                         {num_cells_local},
                                                                         adios2::ConstantDims);

  adios2::Variable<std::int32_t> dofmapvar = io.DefineVariable<std::int32_t>(funcname + "_dofmap",
                                                                             {num_dofmap_size},
                                                                             {dofmap_offset},
                                                                             {num_dofs_local_dmap},
                                                                             adios2::ConstantDims);

  adios2::Variable<std::int32_t> xdofmap_var = io.DefineVariable<std::int32_t>(funcname + "_XDofmap",
                                                                              {num_cells_global+1},
                                                                              {cell_offset},
                                                                              {num_cells_local+1},
                                                                              adios2::ConstantDims);

  // WIP
  adios2::Variable<T> x = io.DefineVariable<T>(funcname + "_values",
                                                {num_dofs_global},
                                                {offset, 0},
                                                {num_nodes_local, 3},
                                                adios2::ConstantDims);


  writer.BeginStep();
  writer.Put(n_nodes, num_nodes_global);
  writer.Put(n_cells, num_cells_global);
  writer.Put(n_dofs_per_cell, num_dofs_per_cell);
  writer.Put(input_global_indices, mesh_input_global_indices_span.subspan(0, num_nodes_local).data());
  writer.Put(x, mesh_x.subspan(0, num_nodes_local*3).data());
  writer.Put(cell_indices, connectivity_nodes_global.data());
  writer.Put(cell_indices_offsets, indices_offsets_span.subspan(0, num_cells_local+1).data());
  writer.EndStep();
  writer.Close();

  // MPI_Finalize();
  PetscFinalize();

  return 0;
}

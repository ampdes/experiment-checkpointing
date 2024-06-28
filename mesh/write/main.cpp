// Mesh write

#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <unistd.h>
#include <mpi.h>

#include <adios2.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

// TODO:Basix function add: need enum to int function in basix
// #include <basix.h>
// std::map<basix::element::lagrange_variant, int> lagrange_variants {
//                       {basix::element::lagrange_variant::unset,0},
//                       {basix::element::lagrange_variant::equispaced,1},
//                       {basix::element::lagrange_variant::gll_warped,2},
//                     };

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

    auto cmap = mesh->geometry().cmap();
    auto edegree = cmap.degree();
    auto ecelltype = cmap.cell_shape();
    auto elagrange_variant = cmap.variant();
    auto geom_layout = cmap.create_dof_layout();
    int num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh_dim);
    
    adios2::Variable<std::string> name = io.DefineVariable<std::string>("name");
    adios2::Variable<std::int16_t> dim = io.DefineVariable<std::int16_t>("dim");
    adios2::Variable<std::string> celltype = io.DefineVariable<std::string>("CellType");
    adios2::Variable<std::int32_t> degree = io.DefineVariable<std::int32_t>("Degree");
    adios2::Variable<std::int32_t> variant = io.DefineVariable<std::int32_t>("Variant");
    adios2::Variable<std::int64_t> n_nodes = io.DefineVariable<std::int64_t>("n_nodes");
    adios2::Variable<std::int64_t> n_cells = io.DefineVariable<std::int64_t>("n_cells");
    adios2::Variable<std::int32_t> n_dofs_per_cell = io.DefineVariable<std::int32_t>("n_dofs_per_cell");
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

    auto connectivity = topology->connectivity(mesh_dim, 0);
    // std::vector<int32_t> indices = connectivity->array();
    auto indices = connectivity->array();
    const std::span<const int32_t> indices_span(indices.begin(),
                                                indices.end());

    // std::vector<int> indices_offsets = connectivity->offsets();
    auto indices_offsets = connectivity->offsets();
    for (std::size_t i = 0; i < indices_offsets.size(); ++i)
    {
        indices_offsets[i] += cell_offset*num_dofs_per_cell;
    }

    const std::span<const int32_t> indices_offsets_span(indices_offsets.begin(),
                                                        indices_offsets.end());

    std::vector<std::int64_t> connectivity_nodes_global(indices_offsets[num_cells_local]);

    imap->local_to_global(indices_span.subspan(0, indices_offsets[num_cells_local]), connectivity_nodes_global);

    // TODO:
    // In general, can't multiply cell_local_range*num_dofs_per_cell to get the offset,
    // Have to know in general the start of the offset

    writer.BeginStep();
    writer.Put(name, mesh_name);
    writer.Put(dim, mesh_dim);
    writer.Put(celltype, mesh::to_string(ecelltype));
    writer.Put(degree, edegree);
    // writer.Put(variant, lagrange_variants[elagrange_variant]);
    writer.Put(variant, 2);
    writer.Put(n_nodes, num_nodes_global);
    writer.Put(n_cells, num_cells_global);
    writer.Put(n_dofs_per_cell, num_dofs_per_cell);
    writer.Put(input_global_indices, mesh_input_global_indices_span.subspan(0, num_nodes_local).data());
    writer.Put(x, mesh_x.subspan(0, num_nodes_local*3).data());
    writer.Put(cell_indices, connectivity_nodes_global.data());
    writer.Put(cell_indices_offsets, indices_offsets_span.subspan(0, num_cells_local+1).data());
    writer.EndStep();
    writer.Close();
    }
  }
}

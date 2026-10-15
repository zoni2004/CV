from pathlib import Path
import pycolmap
import open3d as o3d

image_dir = Path("images")
work_dir = Path("colmap_output")
database = work_dir / "database.db"
sparse_dir = work_dir / "sparse"
work_dir.mkdir(exist_ok=True)
sparse_dir.mkdir(exist_ok=True)

pycolmap.extract_features(database, image_dir)
pycolmap.match_exhaustive(database)
reconstructions = pycolmap.incremental_mapping(database, image_dir, sparse_dir)

if not reconstructions:
    raise SystemExit("Reconstruction failed. Use more photos with more overlap.")

best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
print(best.summary())

cloud_path = work_dir / "point_cloud.ply"
best.export_PLY(str(cloud_path))

cloud = o3d.io.read_point_cloud(str(cloud_path))
cloud, kept = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
cloud.estimate_normals()
cloud.orient_normals_consistent_tangent_plane(30)

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8)
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(str(work_dir / "mesh.ply"), mesh)

print("Points:", len(cloud.points))
print("Triangles:", len(mesh.triangles))
o3d.visualization.draw_geometries([cloud], window_name="Point cloud")
o3d.visualization.draw_geometries([mesh], window_name="Mesh")

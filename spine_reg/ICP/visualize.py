import open3d as o3d
import numpy as np
import os

def visualize_point_cloud(pcd, window_name="Point Cloud"):
    """
    可视化 Open3D 点云对象。
    """
    if not pcd.has_points():
        print(f"Warning: Point cloud for '{window_name}' is empty or has no points.")
        return
    
    # 如果点云没有颜色，给它一个默认颜色（例如灰色）
    if not pcd.has_colors():
        print(f"Info: Point cloud for '{window_name}' has no colors. Assigning default gray.")
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def load_and_visualize_ply(file_path):
    """
    加载 PLY 文件并将其可视化为点云。
    """
    if not os.path.exists(file_path):
        print(f"Error: PLY file not found at {file_path}")
        return None
    try:
        print(f"Attempting to load PLY: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"\nLoaded PLY: {file_path}")
        print(f"Point cloud object: {pcd}") # 打印 pcd 对象本身
        if not pcd.has_points():
            print("PLY loaded BUT contains NO POINTS.")
        else:
            print(f"Number of points: {len(pcd.points)}")
            print(f"First 3 points: \n{np.asarray(pcd.points)[:3]}") # 打印前几个点
        
        if pcd.has_colors():
            print("PLY has color information.")
        else:
            print("PLY does not have color information.")
        
        visualize_point_cloud(pcd, window_name=f"PLY: {os.path.basename(file_path)}")
        return pcd
    except Exception as e:
        print(f"Error loading PLY file {file_path}: {e}")
        import traceback
        traceback.print_exc() # 打印完整的错误堆栈
        return None


def load_and_visualize_obj(file_path):
    """
    加载 OBJ 文件，将其顶点、法向量和面提取并可视化。
    """
    try:
        # 同时加载多个 OBJ 并合并
        if isinstance(file_path, (list, tuple)):
            meshes = []
            for fp in file_path:
                if not os.path.exists(fp):
                    print(f"Error: OBJ file not found at {fp}")
                    continue
                m = o3d.io.read_triangle_mesh(fp)
                print(f"\nLoaded OBJ: {fp}")
                meshes.append(m)
            if not meshes:
                return None
            mesh = meshes[0]
            for m in meshes[1:]:
                mesh += m
            print(f"Merged {len(meshes)} meshes, total vertices: {len(mesh.vertices)}")
        else:
            # 单文件加载
            mesh = o3d.io.read_triangle_mesh(file_path)
            print(f"\nLoaded OBJ: {file_path}")

        # 顶点检查
        if not mesh.has_vertices():
            print(f"OBJ file has no vertices.")
            return None
        print(f"Number of vertices: {len(mesh.vertices)}")

        # 面信息
        if mesh.has_triangles():
            print(f"Number of faces (triangles): {len(mesh.triangles)}")
        else:
            print("OBJ mesh does not contain face (triangle) information.")

        # 法向量：顶点法向量和面法向量
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            print("Computed vertex normals.")
        else:
            print("Vertex normals already present.")

        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
            print("Computed triangle normals.")
        else:
            print("Triangle normals already present.")

        # 颜色（可选）
        if mesh.has_vertex_colors():
            print("OBJ mesh has vertex color information.")
        else:
            print("OBJ mesh does not have vertex color information.")

        # 可视化网格（含法向量/面）
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="OBJ Mesh",
            mesh_show_back_face=True
        )
        return mesh

    except Exception as e:
        print(f"Error loading OBJ file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # --- 示例文件路径 ---
    # 请将这些路径替换为你的实际文件路径
    # 你提供的 OBJ 文件路径
    # obj_file_path = ["/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/脊柱/CT/CT1_left.obj",
    #                  "/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/脊柱/CT/CT1_right.obj",
    #                  "/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/脊柱/CT/CT1_top.obj",
    #                  "/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/脊柱/US/US1_left.obj",
    #                  "/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/脊柱/US/US1_right.obj",
    #                  "/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/脊柱/US/US1_top.obj",]
    
    
    obj_file_path = ["/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/bunny/bunny_left.obj",
                     "/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/bunny/bunny_right.obj",
                     "/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/bunny/bunny_middle.obj"
                     ]
    
    # 假设你有一个 PLY 文件，例如 "example.ply"
    ply_file_path = r"/home/taylor/Fast-Robust-ICP-master-VS2017_仿真/Fast-Robust-ICP-master-VS2017/data/US_002_1.ply"
  

    print("--- Visualizing PLY file ---")
    # ply_pcd = load_and_visualize_ply(ply_file_path)

    print("\n--- Visualizing OBJ file (vertices as point cloud) ---")
    obj_pcd_from_vertices = load_and_visualize_obj(obj_file_path)
    

    print("\n--- Analysis Complete ---")
import open3d as o3d
from downsample import *
from normal_estimation import *
from pc_skeletor import LBC
from pc_skeletor import SLBC
import mistree as mist
from pc_skeletor import Dataset

# np.bool = np.bool_

VISUALIZE = True

if __name__ == '__main__':

   # perform skeletonization
   #  downloader = Dataset()
   #  trunk_pcd_path, branch_pcd_path = downloader.download_semantic_tree_dataset()
   #
   #  pcd_trunk = o3d.io.read_point_cloud(trunk_pcd_path)
   #  pcd_branch = o3d.io.read_point_cloud(branch_pcd_path)
   #  pcd = pcd_trunk + pcd_branch
   #  s_lbc = SLBC(point_cloud={'trunk': pcd_trunk, 'branches': pcd_branch},
   #           semantic_weighting=30,
   #           down_sample=0.008,
   #           debug=True)
   #  s_lbc.extract_skeleton()
   #  s_lbc.extract_topology()
   #  s_lbc.visualize()
   #  s_lbc.show_graph(s_lbc.skeleton_graph)
   #  s_lbc.show_graph(s_lbc.topology_graph)
   #  s_lbc.save('./output')
   #  s_lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=300, output='./output')
   #  # Read point cloud:
    pcd = o3d.io.read_point_cloud("../data/maize/03-13_trimmed.ply")
    print(pcd.points[0])

    if VISUALIZE:
        o3d.visualization.draw_geometries([pcd])  # plot A and B
    pc_res = compute_pc_resolution(pcd)
    pcd_down = down_sample_pc(pcd, pc_res, 30.0)

    lbc = SLBC(point_cloud={'trunk':pcd}, down_sample= 100.0 * pc_res,debug=True)
    lbc.extract_skeleton()
    lbc.extract_topology()
    lbc.show_graph(lbc.skeleton_graph)
    lbc.show_graph(lbc.topology_graph)
    lbc.visualize()
    lbc.save('./output')
    lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=300, output='./output')
    print("down sampled point cloud size:", pcd2xyz(pcd_down).shape[0])

    if len(pcd.normals) == 0:
         estimate_normals(pcd_down,5.0,pc_res,30)
    if VISUALIZE:
        o3d.visualization.draw_geometries([pcd_down])  # plot A and B
else:
    pass

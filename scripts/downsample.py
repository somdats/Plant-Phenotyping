import numpy as np
import open3d as o3d
import os


def pcd2xyz(pcd):
    return np.asarray(pcd.points)


def compute_pc_resolution(pcd):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    sum = 0.0
    pcd_np = pcd2xyz((pcd))
    print("total no. of pts in point cloud:", pcd_np.shape[0])
    for vt in pcd_np:
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(vt, 2)
        sum = sum + dist[1]
    res = sum/pcd_np.shape[0]
    print("pt. cloud resolution:",res)
    return res


def down_sample_pc(pc_raw, voxel_size, mul):
    rad = mul * voxel_size
    pcd_down = pc_raw.voxel_down_sample(voxel_size = rad)
    return pcd_down


if __name__ == '__main__':
    # Read point cloud:
    pcd = o3d.io.read_point_cloud("../data/maize/03-13.ply")
    pc_res = compute_pc_resolution(pcd)
    pcd_down = down_sample_pc(pcd,pc_res,5.0)
    print("downsampled point cloud size:", pcd2xyz(pcd_down).shape[0])

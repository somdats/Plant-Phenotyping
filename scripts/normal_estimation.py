import open3d.cpu.pybind.geometry

from scripts.downsample import *


def estimate_normals(pcd, rad_mul, density, maxnn):
    radius_normal = density * rad_mul
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=maxnn))
        #o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=maxnn))
    o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(pcd, k=maxnn)

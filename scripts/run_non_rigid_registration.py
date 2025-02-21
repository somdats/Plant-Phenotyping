#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a demo for estimating the non-rigid registration parameters between a pair of skeletons.
"""
import time

import skeleton as skel
import numpy as np
import matplotlib.pyplot as plt
import skeleton_matching as skm
import non_rigid_registration as nrr
import visualize as vis
import open3d as o3d
import non_rigid_registration_lm as nrlm
import deform_point_cloud as dfm
import pathlib
import open3d
import sys
import os
from scipy.spatial import cKDTree
# %% load skeleton data and correpondences (matching results)
# species = 'maize'
# day1 = '03-13'
# day2 = '03-14'
# skel_path = '../data/{}/{}.graph.txt'
# corres_path = '../data/{}/{}-{}.corres.txt'
# input_path = "D:/Pheno4D/Pheno4D/Tomato01/with_normals"
# ply_file_name = "T01_0305.ply"
# pcd_ori = open3d.io.read_point_cloud(os.path.join(input_path,ply_file_name))
# pcd_ori.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in bluewe
# o3d.visualization.draw_geometries([pcd_ori])
write = True
interpolate = False
input_path = "D:/4d_plant_registration_data/maize/plant2/with_normals/"
file_path = input_path + "skeleton_matching.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# Create pairs of names
pairs = [(lines[i].strip(), lines[i+1].strip()) for i in range(0, len(lines)-1, 1)]

#write deformation values
file_name = input_path + "mse_maize_plant2_comp_weighted_with_pd_viterbi.txt"

if write:
    file = open(file_name, 'w')
    # Print the pairs
itr = 0
scores = []
for pair in pairs:
        #read skeleton
        print(pair)
        skeleton_A = pair[0]  #"M02_0316.ply_new_graph.txt"
        skeleton_B = pair[1]  #"M02_0317.ply_new_graph.txt"
        if interpolate and itr + 1 < len(pairs):
            skel_to_comp = pairs[itr+1][0]
        if interpolate and itr + 1 == len(pairs):
            skel_to_comp = pairs[itr-1][1]
        name_A = skeleton_A.split('.')[0]
        name_B = skeleton_B.split('.')[0]
        number_A = name_A.split('_')[-1][-2:]
        number_B = name_B.split('_')[-1][-2:]
        skel_path = input_path + skeleton_A

        skel_maize_3_A = skel.Skeleton.read_graph(skel_path)

        skel_path_2 = input_path + skeleton_B
        skel_maize_3_B = skel.Skeleton.read_graph(skel_path_2)

        #read original files

        ply_file_name = name_A + ".ply"
        ply2_file_name = name_B + ".ply"
        if interpolate:
           ply2_file_name = skel_to_comp.split('.')[0] + ".ply"
        #read correspondences
        corres_path = input_path + "corresp_pair_with_pd_" + number_A + "_" + number_B + ".txt"
        corres = np.loadtxt(corres_path, dtype = np.int32)

        # visualize input data
        # fh= plt.figure(figsize=(10,10))  # Create a figure and axis object
        # # Create a subplot with 3D projection
        # ax = fh.add_subplot(111, projection='3d')
        # vis.plot_skeleton(ax, skel_maize_3_A, 'b')
        # vis.plot_skeleton(ax, skel_maize_3_B, 'r')
        # vis.plot_skeleton_correspondences(ax, skel_maize_3_A, skel_maize_3_B, corres)
        # plt.title('Skeletons with correspondences.')
        # plt.show()

        # %% compute non-rigid registration params
        # set the paramst stat
        #iterative beam search
        params = {'num_iter': 15,
                  'w_rot' : 20, #10
                  'w_reg' : 20, #1
                  'w_corresp' : 10, #100
                  'w_fix' : 1,
                  'fix_idx' : [],
                  'R_fix' : [np.eye(3)],
                  't_fix' : [np.zeros((3,1))],
                  'use_robust_kernel' : True,
                  'robust_kernel_type' : 'cauchy',
                  'robust_kernel_param' : 20, #2
                  'debug' : True}

        # params = {'num_iter': 5,
        #           'w_rot' : 30,
        #           'w_reg' : 50,
        #           'w_corresp' : 10,
        #           'w_fix' : 1,
        #           'fix_idx' : [],
        #           'R_fix' : [np.eye(3)],
        #           't_fix' : [np.zeros((3,1))],
        #           'use_robust_kernel' : True,g
        #           'robust_kernel_type' : 'cauchy',
        #           'robust_kernel_param' : 15,
        #           'debug' : True}
        # tfs= []
        # pcd_src = open3d.io.read_point_cloud(os.path.join(input_path,ply_file_name))
        # src_pts_arr = np.asarray(pcd_src.points)
        # dfm.continuity_measure(skel_maize_3_A,skel_maize_3_B,corres,tfs,1.0,src_pts_arr)
        # call register function
        t_start = time.time()
        #T12 = nrr.register_skeleton(skel_maize_3_A, skel_maize_3_B, corres, params)

        T12 = nrlm.register_skeleton(skel_maize_3_A, skel_maize_3_B, corres, params)
        # Assuming `T` is your transformation matrix list
        # for i, transformation in enumerate(T12):
        #     print(f"Transformation matrix for node {i}:")
        #     for row in transformation:
        #         print(" ".join([f"{val:.3f}" for val in row]))
        time_e = time.time()
        print("total time for skeleton matching {:.2f}s".format(time_e - t_start))



        # %% Apply registration params to skeleton
        S2_hat = nrr.apply_registration_params_to_skeleton(skel_maize_3_A, T12)

        ##open3d visualization
        # pcd_A = o3d.geometry.PointCloud()
        # pcd_A.points = o3d.utility.Vector3dVector(S2_hat.XYZ)
        #
        # pcd_B = o3d.geometry.PointCloud()
        # pcd_B.points = o3d.utility.Vector3dVector(skel_maize_3_B.XYZ)
        #
        # pcd_A.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
        # pcd_B.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
        #vis = o3d.visualization.Visualizer()
        #o3d.visualization.draw_geometries([pcd_A])
        # time.sleep(5)
        # o3d.visualization.Visualizer.close()

        # %% visualize registration results
        # fh1 = plt.figure()
        # ax1 = fh1.add_subplot(111, projection='3d')
        #
        # #vis.plot_skeleton(ax1, skel_maize_3_20,'b');
        # vis.plot_skeleton(ax1, S2_hat,'b')
        # vis.plot_skeleton(ax1, skel_maize_3_B,'r')

        #vis.plot_skeleton_correspondences(ax1, S2_hat, skel_maize_3_21, corres)
        # plt.title("Skeleton registration results.")
        # plt.show()
        print("process done")
        # plt.pause(5)
        # plt.close()

        # main point cloud as the source path

        pcd_src = open3d.io.read_point_cloud(os.path.join(input_path,ply_file_name))
        src_pts_arr = np.asarray(pcd_src.points)
        pcd_tgt = open3d.io.read_point_cloud(os.path.join(input_path,ply2_file_name))
        #tgt_pts_arr = np.asarray(pcd_tgt.points)
        #transform source pt. cloud using the transformation obtained from spat. temporal registration
        deform_src = dfm.deform_pointcloud(src_pts_arr, T12, corres, skel_maize_3_A, skel_maize_3_B)
        #deform_src = dfm.ransac_weighted_line_projection(src_pts_arr, T12,  skel_maize_3_A)

        #open3d visualization
        pcd_A_dfm = o3d.geometry.PointCloud()
        pcd_A_dfm.points = o3d.utility.Vector3dVector(deform_src)
        print("no. of points after deformation:", len(pcd_A_dfm.points))


        avg_error, errors = dfm.mean_square_error(np.asarray(pcd_A_dfm.points), np.asarray(pcd_tgt.points))
        avg_pt_dist = dfm.compute_pc_resolution(pcd_src)
        avg_pt_dist_2 = dfm.compute_pc_resolution(pcd_tgt)
        if(avg_pt_dist < avg_pt_dist_2):
            avg_pt_dist = avg_pt_dist_2
        landmark_err = dfm.landmark_Accuracy(skel_maize_3_A,skel_maize_3_B,corres,T12,
                                             avg_pt_dist,pcd_src,pcd_tgt)
        scores.append(landmark_err)

        print("average error of deformation:", avg_error)
        if write:
            file.write(str(avg_error) + '\n')

        #color src points based on error
        pcd_dfm_color = dfm.color_point_cloud_based_on_error(pcd_A_dfm,errors, avg_error)
        #interpolation of point cloud
        # input_path = "D:/Pheno4D/Pheno4D/Tomato01/with_normals"
        # ply_file_name = "T01_0308.ply"
        # pcd_ori = open3d.io.read_point_cloud(os.path.join(input_path,ply_file_name))
        # inter_error = dfm.mean_square_error(np.asarray(pcd_A_dfm.points), np.asarray(pcd_ori.points))
        # print("interpolation error of deformation:", inter_error)

        pcd_A_dfm.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
        pcd_tgt.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
        # # pcd_src.paint_uniform_color([1.0, 1.0, 0.0]) # show B_pcd in red
        #o3d.visualization.draw_geometries([pcd_A_dfm,pcd_tgt])
        #dfm.display_vertical_color_bar(errors,avg_error)
        # time.sleep(3)
        # o3d.visualization.Visualizer.close()
        #o3d.visualization.draw_geometries([pcd_dfm_color])
        #o3d.visualization.draw_geometries([pcd_src,pcd_A_dfm])
        itr = itr + 1
if write:
    file.close()
scores_avg = 0.0
if len(scores) > 0:
    scores_avg = sum(scores)/len(scores)
print("average_landmark_score:", format(scores_avg,".2f"))


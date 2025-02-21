import skeleton as skel
import numpy as np
import time
import skeleton_matching as skm
import open3d as o3d

input_path = "D:/4d_plant_registration_data/maize/plant2/with_normals/"
out_path = input_path

# Read skeleton matching file
file_path = input_path + "skeleton_matching.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

pairs = [(lines[i].strip(), lines[i+1].strip()) for i in range(6, len(lines)-1, 1)]

def visualize_skeletons(pcd_A, pcd_B, corres, a_corres, b_corres):
    """ Function to visualize skeletons with correspondences using Open3D with thicker lines and larger dots. """
    # Create correspondence lines
    points = np.concatenate((a_corres, b_corres), axis=0)
    lines = [[i, i + len(a_corres)] for i in range(len(a_corres))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([0, 1, 0])  # Green color for correspondences

    # Increase Line Thickness (Hack)
    thick_lines = []
    for offset in np.linspace(-0.002, 0.002, 3):  # Small offset for thickness
        points_offset = points + np.array([offset, offset, 0])
        thick_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_offset),
            lines=o3d.utility.Vector2iVector(lines),
        )
        thick_line_set.paint_uniform_color([0, 0, 0])
        thick_lines.append(thick_line_set)

    # Increase Point Size
    pcd_A.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for A
    pcd_B.paint_uniform_color([1.0, 0.0, 0.0])  # Red for B

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Skeleton Correspondences")
    vis.add_geometry(pcd_A)
    vis.add_geometry(pcd_B)
    vis.add_geometry(line_set)
    for thick_line in thick_lines:
        vis.add_geometry(thick_line)  # Add extra layers to simulate thickness

    # Set Render Options
    opt = vis.get_render_option()
    opt.point_size = 8.0  # Increase point size (dots instead of squares)
    opt.line_width = 15  # Set thicker line width

    vis.run()
    vis.destroy_window()

# Process each skeleton pair
for pair in pairs:
    skeleton_A, skeleton_B = pair[0], pair[1]
    name_A, name_B = skeleton_A.split('.')[0], skeleton_B.split('.')[0]
    number_A, number_B = name_A.split('-')[-1][-2:], name_B.split('-')[-1][-2:]
    print("Matching skeleton:", name_A, name_B)

    output_name = out_path + f"corresp_pair_with_pd_{number_A}_{number_B}.txt"

    # Load skeletons
    skel_maize_3_A = skel.Skeleton.read_graph(input_path + skeleton_A)
    skel_maize_3_B = skel.Skeleton.read_graph(input_path + skeleton_B)

    # Convert to Open3D point cloud
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(skel_maize_3_A.XYZ)

    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(skel_maize_3_B.XYZ)

    # Perform skeleton matching
    params = {'weight_e': 0.01, 'match_ends_to_ends': True, 'use_labels': False, 'label_penalty': 1, 'debug': False}
    start_time = time.time()
    corres = skm.skeleton_matching(skel_maize_3_A, skel_maize_3_B, params)
    print("Total time for skeleton matching: {:.2f}s".format(time.time() - start_time))
    print("Estimated correspondences:", corres)

    # Extract corresponding points
    a_corres = skel_maize_3_A.XYZ[corres[:, 0], :]
    b_corres = skel_maize_3_B.XYZ[corres[:, 1], :]

    # Visualize using Open3D
    visualize_skeletons(pcd_A, pcd_B, corres, a_corres, b_corres)

    # Save correspondences
    np.savetxt(output_name, corres, fmt='%i')


    print("Done")

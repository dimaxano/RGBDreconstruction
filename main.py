import csv

import cv2
import numpy as np
import open3d as o3d

INTRINSIC_MATRIX = np.array([[613.688, 0.0, 323.035],
                            [0.0, 614.261, 242.229],
                            [0.0, 0.0, 1.0]])
DEPTH_TRUNC_THRESH = 80

if __name__ == "__main__":
    with open("./data/scanner.log", "r") as f:
        pose_data = csv.reader(f, delimiter=' ')
        pose_data = list(pose_data)
    
    mesh = []
    for i in range(194):
        print(f"Processing frame {i}")

        depth_path = f"./data/depth-{i:05}.png"
        rgb_path = f"./data/rgb-{i:05}.png"

        # prepare camera to world transform
        pose = np.array([float(x) for x in pose_data[i][2:]])
        X, Y, Z, r1, r2, r3 = pose
        P_wc = np.identity(4)
        
        R = cv2.Rodrigues(np.array([r1, r2, r3]))[0]
        P_wc[:3, :3] = R
        P_wc[:3, 3] = np.array([X, Y, Z])

        depth_img = cv2.imread(depth_path)
        rgb_img = cv2.imread(rgb_path)[:, :, ::-1]

        depth_img = depth_img[:, :, 0]
        depth_img[depth_img >= DEPTH_TRUNC_THRESH] = 0.0
        
        rows, cols = depth_img.shape

        point_cloud = []
        colors = []
        for v in range(rows):
            for u in range(cols):
                z = depth_img[v, u]
                if z == 0:
                    continue
                
                x = (u - INTRINSIC_MATRIX[0, 2]) * z / INTRINSIC_MATRIX[0, 0]
                y = (v - INTRINSIC_MATRIX[1, 2]) * z / INTRINSIC_MATRIX[1, 1]

                point_cw = P_wc.dot(np.array([x, y, z, 1.0]))
                point_cloud.append(point_cw[:3])
                colors.append(rgb_img[v, u] / 255.0)

        point_cloud = np.array(point_cloud)
        colors = np.array(colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        mesh.append(pcd)

    o3d.visualization.draw_geometries(mesh)
    # o3d.io.write_point_cloud("./pcd.ply", pcd)
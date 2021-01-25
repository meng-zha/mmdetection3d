# unit test code
import numpy as np
import open3d as o3d


def _write_ply(points, out_filename):
    """Write points into ``ply`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


if __name__ == "__main__":
    # points_1 = np.fromfile('../data/kitti_track/training/velodyne/0000_reduced/000000.bin',np.float32).reshape(-1,4)[:,:3]
    # points_2 = np.fromfile('../data/kitti_track/training/velodyne/0000_reduced/000015.bin',np.float32).reshape(-1,4)[:,:3]
    points_1 = np.asarray(o3d.io.read_triangle_mesh(
        '../data/kitti_track/outputs/3dssd_kitti_track_time_2_pretrained_20210122_104049/visualization/'\
            'data-kitti_track-training-velodyne-0000-000000/data-kitti_track-training-velodyne-0000-000000_hidden_points.obj'
    ).vertices)
    points_2 = np.asarray(o3d.io.read_triangle_mesh(
        '../data/kitti_track/outputs/3dssd_kitti_track_time_2_pretrained_20210122_104049/visualization/'\
            'data-kitti_track-training-velodyne-0000-000050/data-kitti_track-training-velodyne-0000-000050_hidden_points.obj'
    ).vertices)
    pose = np.loadtxt('../data/kitti_track/training/poses/0000.txt').reshape(
        -1, 4, 4)
    pose1 = pose[0]
    pose2 = pose[49]
    pad_1 = np.hstack([points_1, np.ones((points_1.shape[0], 1))])
    # points_1 = (pad_1@pose1.T)[:,:3]
    pad_2 = np.hstack([points_2, np.ones((points_2.shape[0], 1))])
    points_2 = (pad_2 @ pose2.T @ np.linalg.inv(pose1.T))[:, :3]
    # points_2 = (pad_2@pose2.T)[:,:3]
    _write_ply(points_1, '../work_dirs/hidden_points_1.obj')
    _write_ply(points_2, '../work_dirs/hidden_points_5.obj')

    points_1 = np.asarray(o3d.io.read_triangle_mesh(
        '../data/kitti_track/outputs/3dssd_kitti_track_time_2_pretrained_20210122_104049/visualization/'\
            'data-kitti_track-training-velodyne-0000-000000/data-kitti_track-training-velodyne-0000-000000_points.obj'
    ).vertices)
    points_2 = np.asarray(o3d.io.read_triangle_mesh(
        '../data/kitti_track/outputs/3dssd_kitti_track_time_2_pretrained_20210122_104049/visualization/'\
            'data-kitti_track-training-velodyne-0000-000050/data-kitti_track-training-velodyne-0000-000050_points.obj'
    ).vertices)
    pad_1 = np.hstack([points_1, np.ones((points_1.shape[0], 1))])
    # points_1 = (pad_1@pose1.T)[:,:3]
    pad_2 = np.hstack([points_2, np.ones((points_2.shape[0], 1))])
    points_2 = (pad_2 @ pose2.T @ np.linalg.inv(pose1.T))[:, :3]
    # points_2 = (pad_2@pose2.T)[:,:3]
    _write_ply(points_1, '../work_dirs/points_1.obj')
    _write_ply(points_2, '../work_dirs/points_5.obj')
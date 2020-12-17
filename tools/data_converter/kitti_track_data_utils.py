import pathlib
import numpy as np
from collections import OrderedDict
import os
import pandas as pd
from skimage import io

from tqdm import tqdm


def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_scene_index_str(scene_idx):
    return "{:04d}".format(scene_idx)


def get_kitti_info_path(
    scene_idx,
    idx,
    prefix,
    info_type="image_2",
    file_tail=".png",
    training=True,
    relative_path=True,
    exist_check=True,
):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    scene_idx_str = get_scene_index_str(scene_idx)
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path(
            "training") / info_type / scene_idx_str / img_idx_str
    else:
        file_path = pathlib.Path(
            "testing") / info_type / scene_idx_str / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        # velodyne lost in scene 0001/000177~000181.bin
        if scene_idx != 1 or idx not in [177, 178, 179, 180]:
            raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_scene_info_path(
    scene_idx,
    prefix,
    info_type="calib",
    file_tail=".txt",
    training=True,
    relative_path=True,
    exist_check=True,
):
    scene_idx_str = get_scene_index_str(scene_idx)
    scene_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path("training") / info_type / scene_idx_str
    else:
        file_path = pathlib.Path("testing") / info_type / scene_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_index(scene_idx, prefix, training=True):
    scene_idx_str = get_scene_index_str(scene_idx)
    prefix = pathlib.Path(prefix)

    if training:
        file_path = pathlib.Path("training") / "image_02" / scene_idx_str
    else:
        file_path = pathlib.Path("testing") / "image_02" / scene_idx_str
    dir_path = str(prefix / file_path)
    return len(os.listdir(dir_path))


def get_image_path(scene,
                   idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_kitti_info_path(scene, idx, prefix, "image_02", ".png",
                               training, relative_path, exist_check)


def get_velodyne_path(scene,
                      idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True):
    return get_kitti_info_path(scene, idx, prefix, "velodyne", ".bin",
                               training, relative_path, exist_check)


def get_label_path(scene,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_scene_info_path(scene, prefix, "label_02", ".txt", training,
                               relative_path, exist_check)


def get_calib_path(scene,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_scene_info_path(scene, prefix, "calib", ".txt", training,
                               relative_path, exist_check)


def get_kitti_track_image_info(path,
                               training=True,
                               label_info=True,
                               velodyne=False,
                               calib=False,
                               track=False,
                               scene_ids=10,
                               extend_matrix=True,
                               relative_path=True,
                               with_imageshape=True):
    """
    KITTI tracking annotation format version 2:
    {
        scenes:{
            image_ids: ids of frames in scene
        }
        frames: {
            [optional]points: [N, 3+] point cloud
            [optional, for kitti]image: {
                image_idx: ...
                image_path: ...
                image_shape: ...
            }
            point_cloud: {
                num_features: 4
                velodyne_path: ...
            }
            [optional, for kitti]calib: {
                R0_rect: ...
                Tr_velo_to_cam: ...
                P2: ...
            }
            annos: {
                frame: [num_gt] array
                track_id: [num_gt, 3] array
                location: [num_gt, 3] array
                dimensions: [num_gt, 3] array
                rotation_y: [num_gt] angle array
                name: [num_gt] ground truth name array
                [optional]difficulty: kitti difficulty
                [optional]group_ids: used for multi-part object
            }
        }
    }
    """
    root_path = pathlib.Path(path)
    if not isinstance(scene_ids, list):
        scene_ids = list(range(scene_ids))

    def scene_func(scene_idx):
        '''
            Annotations of the scene

            return: dict
                calib: dict
                annos: pd.framework
                frames_info: list
        '''
        info = {}

        image_nums = get_image_index(scene_idx, path, training)
        if scene_idx == 1 and training:
            # training/velodyne/0001 missing files
            image_ids = list(range(177))
            image_ids.extend(list(range(181, image_nums)))
        else:
            image_ids = list(range(image_nums))

        info["image_ids"] = image_ids

        calib_info = {}
        if calib:
            calib_path = get_calib_path(
                scene_idx, path, training, relative_path=False)
            with open(calib_path, "r") as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(" ")[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(" ")[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(" ")[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(" ")[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            R0_rect = np.array([
                float(info) for info in lines[4].split(" ")[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.0
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(" ")[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(" ")[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            calib_info["P0"] = P0
            calib_info["P1"] = P1
            calib_info["P2"] = P2
            calib_info["P3"] = P3
            calib_info["R0_rect"] = rect_4x4
            calib_info["Tr_velo_to_cam"] = Tr_velo_to_cam
            calib_info["Tr_imu_to_velo"] = Tr_imu_to_velo
            info["calib"] = calib_info

        if label_info:
            label_path = get_label_path(scene_idx, path, training,
                                        relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            df = pd.read_csv(
                label_path,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df.insert(loc=0, column="scene", value=scene_idx)
            info["annos"] = df

        frames_info = []
        for idx in tqdm(image_ids):
            # info of single frame in the scene
            # TODO: need to prepare the prev and next frame
            single_info = {}
            annotations = None

            pc_info = {"num_features": 4}
            if velodyne:
                pc_info["velodyne_path"] = get_velodyne_path(
                    scene_idx, idx, path, training, relative_path)
            image_info = {}
            image_info["image_idx"] = idx
            image_info["scene_idx"] = scene_idx
            image_info["image_path"] = get_image_path(scene_idx, idx, path,
                                                      training, relative_path)
            if with_imageshape:
                img_path = image_info["image_path"]
                if relative_path:
                    img_path = str(root_path / img_path)
                image_info["image_shape"] = np.array(
                    io.imread(img_path).shape[:2], dtype=np.int32)
            if label_info:
                label = info["annos"][info["annos"]["frame"] == idx]
                annotations = get_label_anno(np.array(label))

            if annotations is not None:
                single_info["annos"] = annotations
                add_difficulty_to_annos(single_info)

            track_info = {}
            if track:
                inds = image_ids.index(idx)
                track_info["prev"] = 1 if image_ids[inds - 1] == idx - 1 else 0
                track_info["next"] = 1 if image_ids[
                    (inds + 1) % len(image_ids)] == idx + 1 else 0

            single_info["image"] = image_info
            single_info["point_cloud"] = pc_info
            single_info["calib"] = calib_info
            single_info["track"] = track_info
            frames_info.append(single_info)

        return frames_info

    image_infos = []
    for i in scene_ids:
        image_infos.extend(scene_func(i))

    return image_infos


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_label_anno(label):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'track_id': []
    })
    annotations['track_id'] = label[:, 2].astype(float)
    num_objects = len(label[:, 3] != 'DontCare')
    annotations['name'] = label[:, 3]
    num_gt = len(annotations['name'])
    annotations['truncated'] = label[:, 4].astype(float)
    annotations['occluded'] = label[:, 5].astype(float)
    annotations['alpha'] = label[:, 6].astype(float)
    annotations['bbox'] = label[:, 7:11].astype(float)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = label[:, [13, 11, 12]].astype(float)
    annotations['location'] = label[:, 14:17].astype(float)
    annotations['rotation_y'] = label[:, 17].astype(float)
    if label.shape[0] != 0 and label.shape[1] == 19:  # have score
        annotations['score'] = label[:, 18].astype(float)
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff

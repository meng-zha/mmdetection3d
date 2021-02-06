import mmcv
import numpy as np
import pandas as pd
import pdb

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
    annotations['name'] = np.array(['Car']*label.shape[0])
    annotations['alpha'] = label[:, -1].astype(float)
    annotations['bbox'] = label[:, 2:6].astype(float)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = label[:, [9, 7, 8]].astype(float)
    annotations['location'] = label[:, 10:13].astype(float)
    annotations['rotation_y'] = label[:, 13].astype(float)
    annotations['score'] = label[:, 6].astype(float)
    return annotations


data_infos = mmcv.load(
    '/Extra/zhangmeng/kitti_multiobject_tracking/kitti_track_infos_trainval.pkl')
gt_annos = [info['annos'] for info in data_infos]
dfs = []
for i in range(0,21):
    scene_idx = i
    label_path = f'/home/zhangmeng/AB3DMOT/data/KITTI/pointrcnn_Car_val/{scene_idx:0>4d}.txt'
    df = pd.read_csv(
        label_path,
        sep=',',
        names=[
            "frame", "type", "truncated", "bbox_left", "bbox_top",
            "bbox_right", "bbox_bottom", "score", "height", "width", "length",
            "x", "y", "z", "alpha"
        ])
    dfs.append(df)

det_annos = []
for info in data_infos:
    scene_idx = info['image']['scene_idx']
    image_idx = info['image']['image_idx']
    df = dfs[scene_idx]
    label = df[df["frame"] == image_idx]
    annotations = get_label_anno(np.array(label))
    det_annos.append(annotations)

from mmdet3d.core.evaluation import kitti_eval
ap_result_str, ap_dict = kitti_eval(gt_annos, det_annos,["Car"])
print(ap_result_str)

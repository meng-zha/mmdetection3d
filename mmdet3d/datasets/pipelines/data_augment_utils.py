import numba
import numpy as np
import warnings
from numba.errors import NumbaPerformanceWarning

from mmdet3d.core.bbox import box_np_ops

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    """Rotate 2D boxes.

    Args:
        corners (np.ndarray): Corners of boxes.
        angle (float): Rotation angle.
        rot_mat_T (np.ndarray): Transposed rotation matrix.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
        clockwise (bool): Whether the corners are in clockwise order.
            Default: True.
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                    min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                    max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                                       D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                                       D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


@numba.jit
def noise_per_box(boxes, track, valid_mask, loc_noises, rot_noises):
    """Add noise to every box (only on the horizontal plane).

    Args:
        boxes (list of np.ndarray): Input boxes with shape (N, 7) with length as time series, the last 2 contain offsets
        track (list of np.ndarray)： tracking id of each frame
        valid_mask (np.ndarray): Mask to indicate which boxes are valid
            with shape (N).
        loc_noises (np.ndarray): Location noises with shape (N, M, 3).
        rot_noises (np.ndarray): Rotation noises with shape (N, M).

    Returns:
        np.ndarray: Mask to indicate whether the noise is
            added successfully (pass the collision test).
    """
    series = len(boxes)
    num_boxes = boxes[0].shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = []
    for t in range(series):
        temp_box = np.ascontiguousarray(boxes[t][:, :5])
        box_corners.append(box_np_ops.box2d_to_corner_jit(temp_box))
    current_corners = np.zeros((4, 2), dtype=boxes[0].dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes[0].dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    valid_corners = []

    for i in range(num_boxes):
        if valid_mask[i]:
            box_flag = True
            for j in range(num_tests):
                flag = True
                for t in range(series):
                    # generate noises
                    if t == 0:
                        current_corners[:] = box_corners[t][i]
                        current_corners -= boxes[t][i, :2]
                        _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                        current_corners += boxes[t][i, :2] + loc_noises[i, j, :2]
                        # detect collision
                        coll_mat = box_collision_test(
                            current_corners.reshape(1, 4, 2), box_corners[t])
                        coll_mat[0, i] = False
                    # translate into next frame if needed
                    elif t > 0:
                        id_t = track[0][i]  # tracking id in 1st frame
                        ind = np.where(track[t] == id_t, True, False)
                        ind_pre = np.where(track[t - 1] == id_t, True, False)
                        if not ind.any() or not ind_pre.any():
                            box_flag = False  # the specific box in 1st frame disappears
                            flag = False
                            valid_corners.clear()
                            break
                        angle = np.arctan2(boxes[t - 1][ind_pre, 6], boxes[t - 1][ind_pre, 5])
                        angle += rot_noises[i, j]
                        radius = np.sqrt(np.square(boxes[t - 1][ind_pre, 6]) + np.square(boxes[t - 1][ind_pre, 5]))
                        delta_x = radius * np.cos(angle)
                        delta_y = radius * np.sin(angle)
                        current_corners += np.array([delta_x, delta_y]).squeeze()
                        # detect collision
                        coll_mat = box_collision_test(
                            current_corners.reshape(1, 4, 2), box_corners[t])
                        coll_mat[0, ind] = False
                    if coll_mat.any():
                        # collision detected
                        flag = False
                        valid_corners.clear()
                        break
                    else:
                        valid_corners.append(current_corners)
                if flag:
                    success_mask[i] = j
                    for t in range(series):
                        id_t = track[0][i]  # tracking id in 1st frame
                        ind = np.where(track[t] == id_t, True, False)
                        box_corners[t][ind] = valid_corners[t]
                    break
                if not box_flag:
                    break
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises,
                      global_rot_noises):
    """Add noise to every box (only on the horizontal plane). Version 2 used
    when enable global rotations.

    Args:
        boxes (np.ndarray): Input boxes with shape (N, 5).
        valid_mask (np.ndarray): Mask to indicate which boxes are valid
            with shape (N).
        loc_noises (np.ndarray): Location noises with shape (N, M, 3).
        rot_noises (np.ndarray): Rotation noises with shape (N, M).

    Returns:
        np.ndarray: Mask to indicate whether the noise is
            added successfully (pass the collision test).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2, ), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0]**2 + boxes[i, 1]**2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += (dst_grot - current_grot)

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = current_box[
                    0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += (dst_pos - boxes[i, :2])
                    rot_noises[i, j] += (dst_grot - current_grot)
                    break
    return success_mask


def _select_transform(transform, indices):
    """Select transform.

    Args:
        transform (np.ndarray): Transforms to select from.
        indices (np.ndarray): Mask to indicate which transform to select.

    Returns:
        np.ndarray: Selected transforms.
    """
    result = np.zeros((transform.shape[0], *transform.shape[2:]),
                      dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    """Get the 3D rotation matrix.

    Args:
        rot_mat_T (np.ndarray): Transposed rotation matrix.
        angle (float): Rotation angle.
        axis (int): Rotation axis.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, valid_mask):
    """Apply transforms to points and box centers.

    Args:
        points (np.ndarray): Input points.
        centers (np.ndarray): Input box centers.
        point_masks (np.ndarray): Mask to indicate which points need
            to be transformed.
        loc_transform (np.ndarray): Location transform to be applied.
        rot_transform (np.ndarray): Rotation transform to be applied.
        valid_mask (np.ndarray): Mask to indicate which boxes are valid.
    """
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform
    return points


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    """Transform 3D boxes.

    Args:
        boxes (np.ndarray): 3D boxes to be transformed.
        loc_transform (np.ndarray): Location transform to be applied.
        rot_transform (np.ndarray): Rotation transform to be applied.
        valid_mask (np.ndarray | None): Mask to indicate which boxes are valid.
    """
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]
    return boxes


def generate_new_loc(gt_boxes, rot_transforms, ind_pre, t, i):
    """Transform 3D boxes.

    Args:
        gt_boxes (list of np.ndarray): 3D boxes to be transformed.
        rot_transforms (np.ndarray): Rotation transform to be applied.
        ind_pre (int): index of box in the previous frame
        t (int): time slice
        i (int): box index

    Return:
        new location translation vector
    """
    angle = np.arctan2(gt_boxes[t - 1][ind_pre, 8], gt_boxes[t - 1][ind_pre, 7])
    angle += rot_transforms[i]
    radius = np.sqrt(np.square(gt_boxes[t - 1][ind_pre, 8]) + np.square(gt_boxes[t - 1][ind_pre, 7]))
    delta_x = radius * np.cos(angle)
    delta_y = radius * np.sin(angle)

    return np.array([delta_x, delta_y]).squeeze()


def noise_per_object_v3_(gt_boxes,
                         points=None,
                         track=None,
                         valid_mask=None,
                         rotation_perturb=np.pi / 4,
                         center_noise_std=1.0,
                         global_random_rot_range=np.pi / 4,
                         num_try=100):
    """Random rotate or remove each groundtruth independently. use kitti viewer
    to test this function points_transform_

    Args:
        gt_boxes (list of np.ndarray): Ground truth boxes with shape (N, 7).
        points (list of np.ndarray | None): Input point cloud with shape (M, 4).
            Default: None.
        track (list of np.ndarray | None): tracking ID for each box.
            Default: None.
        valid_mask (np.ndarray | None): Mask to indicate which boxes are valid.
            Default: None.
        rotation_perturb (float): Rotation perturbation. Default: pi / 4.
        center_noise_std (float): Center noise standard deviation.
            Default: 1.0.
        global_random_rot_range (float): Global random rotation range.
            Default: pi/4.
        num_try (int): Number of try. Default: 100.
    """
    series = len(gt_boxes)
    num_boxes = gt_boxes[0].shape[0]  # boxes in 1st frame
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [
            -global_random_rot_range, global_random_rot_range
        ]
    enable_grot = np.abs(global_random_rot_range[0] -
                         global_random_rot_range[1]) >= 1e-3

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [
            center_noise_std, center_noise_std, center_noise_std
        ]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes, ), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes[0].dtype)

    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[0][:, 0], gt_boxes[0][:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try])

    # Cautious: we dont need to use global rot in cfgs so we dont modify the func v2
    if not enable_grot:
        selected_noise = noise_per_box([gt_boxes[t][:, [0, 1, 3, 4, 6, 7, 8]] for t in range(series)], track,
                                       valid_mask, loc_noises, rot_noises)
    else:
        selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises, rot_noises,
                                           global_rot_noises)

    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)

    # print('!!track', track)

    origin = (0.5, 0.5, 0)
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[0][:, :3],
        gt_boxes[0][:, 3:6],
        gt_boxes[0][:, 6],
        origin=origin,
        axis=2)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    # process 1st frame points
    if points is not None:
        point_masks = box_np_ops.points_in_convex_polygon_3d_jit(points[0][:, :3], surfaces)
        points[0] = points_transform_(points[0], gt_boxes[0][:, :3], point_masks, loc_transforms,
                                      rot_transforms, valid_mask)
    gt_boxes[0] = box3d_transform_(gt_boxes[0], loc_transforms, rot_transforms, valid_mask)

    if series > 1:
        # process subsequent frames
        for t in range(1, series):
            # print('!!!!t = ', t)
            for i in range(gt_boxes[0].shape[0]):
                if valid_mask[i]:
                    id_t = track[0][i]  # tracking id in 1st frame
                    ind = np.where(track[t] == id_t, True, False)
                    # print('? track_t: ', track[t])
                    # print('?', id_t, ind)
                    if not ind.any():
                        # the specific box in 1st frame disappears
                        # print('? disappears!')
                        continue
                    # print('?box:', i, 'ind', np.argwhere(ind == True))
                    gt_box_corners = box_np_ops.center_to_corner_box3d(
                        gt_boxes[t][ind, :3],
                        gt_boxes[t][ind, 3:6],
                        gt_boxes[t][ind, 6],
                        origin=origin,
                        axis=2)
                    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
                    point_masks = box_np_ops.points_in_convex_polygon_3d_jit(points[t][:, :3], surfaces)
                    # first apply traditional point and boxes transform to series t then translate for each gt_points
                    points[t] = points_transform_(points[t], gt_boxes[t][ind, :3], point_masks, [loc_transforms[i]],
                                                  [rot_transforms[i]], valid_mask)
                    point_masks = np.squeeze(point_masks)
                    # print('?mask count:', np.sum(point_masks == 1, axis=0))
                    gt_boxes[t][ind, :3] += loc_transforms[i]
                    gt_boxes[t][ind, 6] += rot_transforms[i]
                    # the points and boxes in subsequent frames should be translated to the 1st then modified
                    for s in range(t, 0, -1):
                        ind_s = np.where(track[s - 1] == id_t, True, False)
                        points[t][point_masks, :2] -= gt_boxes[s - 1][ind_s, 7:]
                        gt_boxes[t][ind, :2] -= gt_boxes[s - 1][ind_s, 7:]
                    # then translate to the new location
                    for s in range(0, t):
                        ind_s = np.where(track[s] == id_t, True, False)
                        trans = generate_new_loc(gt_boxes, rot_transforms, ind_s, s + 1, i)
                        points[t][point_masks, :2] += trans
                        gt_boxes[t][ind, :2] += trans
                    # delete points in new boxes?
                    # TODO: the problem is the new location may contain background points
                    # then handle points?

    return gt_boxes, points

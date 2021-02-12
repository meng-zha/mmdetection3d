from collections import OrderedDict
import torch

import torch.distributed as dist
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .single_stage import SingleStage3DDetector


@DETECTORS.register_module()
class R3DNet(SingleStage3DDetector):
    """VoteNet model.

    https://arxiv.org/pdf/1904.09664.pdf
    """

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(R3DNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.min_id = 0

    def _parse_losses(self, losses_list):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars_sum = OrderedDict()
        loss_sum = []

        for i, losses in enumerate(losses_list):
            log_vars = OrderedDict()
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean()
                                              for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            loss = sum(_value for _key, _value in log_vars.items()
                       if 'loss' in _key)
            loss_sum.append(loss)

            log_vars['loss'] = loss
            for loss_name, loss_value in log_vars.items():
                # reduce loss when distributed training
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
                if log_vars_sum.get(loss_name + f'_{i}', None):
                    log_vars_sum[loss_name + f'_{i}'] += log_vars[loss_name]
                else:
                    log_vars_sum[loss_name + f'_{i}'] = log_vars[loss_name]
        log_vars_sum['loss'] = sum(loss_sum).item()

        return sum(loss_sum), log_vars_sum

    def train_step(self, data, optimizer):
        hidden_dict = None
        losses_list = []
        for data_t in data:
            losses, hidden_dict = self(**data_t, hidden_dict=hidden_dict)
            losses_list.append(losses)
        loss, log_vars = self._parse_losses(losses_list)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data[0]['img_metas']))
        return outputs

    def val_step(self, data, optimizer):
        losses, hidden_dict = self(**data)
        loss, log_vars = self._parse_losses([losses])

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data[0]['img_metas']),
            hidden_dict=hidden_dict)
        return outputs

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_offset,
                      hidden_dict=None,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)
        batch, num_points = points_cat.shape[:2]

        x = self.extract_feat(points_cat)
        if hidden_dict is not None:
            hidden_xyz = hidden_dict['xyz']
            pad_hidden_xyz = torch.cat([
                hidden_xyz,
                hidden_xyz.new_ones(hidden_xyz.shape[0], hidden_xyz.shape[1],
                                    1)
            ],
                                       dim=2)
            for i in range(batch):
                # pose = torch.tensor(
                #     img_metas[i]['pose']).to(device=hidden_xyz.device).float()
                # local_xyz = pad_hidden_xyz[i] @ torch.inverse(pose.T)
                local_xyz = pad_hidden_xyz[i]
                hidden_dict['xyz'][i] = local_xyz[..., :3]
                # prepare for the loss calculation
                points[i] = torch.cat([points[i], local_xyz], axis=0)
            hidden_dict['indices'] = torch.arange(
                num_points, num_points + hidden_dict['xyz'].shape[1]).to(
                    device=hidden_xyz.device).unsqueeze(0).repeat(batch, 1)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod, hidden_dict)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, gt_offset,
                       pts_semantic_mask, pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        hidden_offset = bbox_preds['offset']
        hidden_dict = {'xyz': None, 'features': None}
        hidden_xyz = bbox_preds['hidden_points']
        hidden_xyz[..., :2] += hidden_offset

        # filter background seed points by topk
        probablity = torch.clamp(torch.sigmoid(bbox_preds['obj_scores'])/self.train_cfg['keep_thr'],0,1)
        ind = torch.bernoulli(1-probablity).to(bool)
        hidden_xyz[ind.transpose(1,2).expand(-1,-1,hidden_xyz.shape[2])] = 0
        hidden_dict['xyz'] = hidden_xyz
        hidden_dict['features'] = bbox_preds['hidden_features']
        hidden_dict['features'][ind.expand(-1,bbox_preds['hidden_features'].shape[2],-1)] = 0
        # ind = torch.topk(bbox_preds['obj_scores'],
        #                  int(self.train_cfg['keep_thr'] * hidden_xyz.shape[1]),
        #                  2)[1]
        # hidden_dict['xyz'] = torch.gather(
        #     hidden_xyz, 1, ind.transpose(1,2).expand(-1, -1, hidden_xyz.shape[2]))
        # hidden_dict['features'] = torch.gather(
        #     bbox_preds['hidden_features'], 2,
        #     ind.expand(-1, bbox_preds['hidden_features'].shape[2], -1))
        return losses, hidden_dict

    def simple_test(self,
                    points,
                    img_metas,
                    imgs=None,
                    hidden_dict=None,
                    rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)
        batch, num_points = points_cat.shape[:2]

        x = self.extract_feat(points_cat)
        if not self.test_cfg['with_hidden']:
            hidden_dict = None
        used_hidden = None
        if hidden_dict is not None:
            hidden_xyz = hidden_dict['xyz']
            pad_hidden_xyz = torch.cat([
                hidden_xyz,
                hidden_xyz.new_ones(hidden_xyz.shape[0], hidden_xyz.shape[1],
                                    1)
            ],
                                       dim=2)
            for i in range(batch):
                pose = torch.tensor(
                    img_metas[i]['pose']).to(device=hidden_xyz.device).float()
                local_xyz = pad_hidden_xyz[i] @ torch.inverse(pose.T)
                hidden_dict['xyz'][i] = local_xyz[..., :3]
                # prepare for the loss calculation
                points[i] = torch.cat([points[i], local_xyz], axis=0)
            hidden_dict['indices'] = torch.arange(
                num_points, num_points + hidden_dict['xyz'].shape[1]).to(
                    device=hidden_xyz.device).unsqueeze(0).repeat(batch, 1)
            used_hidden = hidden_dict['xyz'].detach().clone()
            old_hidden_id = hidden_dict['id']

        points_cat = torch.stack(points)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod, hidden_dict)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels, _ in bbox_list
        ]

        hidden_dict = {'xyz': None, 'features': None}
        if used_hidden is not None:
            hidden_dict['show_xyz'] = used_hidden
        else:
            hidden_dict['show_xyz'] = bbox_preds['hidden_points'].detach(
            ).clone()
        hidden_xyz = bbox_preds['hidden_points']
        # the batchsize of test must be 1
        hidden_offset, assignment = self.bbox_head.assign_seeds(
            bbox_results, hidden_xyz, bbox_list)
        pad_hidden_xyz = torch.cat([
            hidden_xyz,
            hidden_xyz.new_ones(hidden_xyz.shape[0], hidden_xyz.shape[1], 1)
        ],
                                   dim=2)
        for i in range(batch):
            pose = torch.tensor(
                img_metas[i]['pose']).to(device=hidden_xyz.device).float()
            hidden_xyz[i] = (pad_hidden_xyz[i] @ pose.T)[..., :3]
        hidden_xyz[..., :2] += torch.stack(hidden_offset, dim=0)

        # filter background seed points by topk
        ind = torch.stack(assignment,dim=0)==-1
        hidden_xyz[ind.unsqueeze(2).expand(-1,-1,hidden_xyz.shape[2])] = 0
        hidden_dict['xyz'] = hidden_xyz
        hidden_dict['features'] = bbox_preds['hidden_features']
        hidden_dict['features'][ind.unsqueeze(1).expand(-1,bbox_preds['hidden_features'].shape[2],-1)] = 0
        # ind = torch.topk(bbox_preds['obj_scores'],
        #                  int(self.test_cfg['keep_thr'] * hidden_xyz.shape[1]),
        #                  2)[1]
        # hidden_dict['xyz'] = torch.gather(
        #     hidden_xyz, 1, ind.transpose(1,2).expand(-1, -1, hidden_xyz.shape[2]))
        # hidden_dict['features'] = torch.gather(
        #     bbox_preds['hidden_features'], 2,
        #     ind.expand(-1, bbox_preds['hidden_features'].shape[2], -1))

        # track id
        if used_hidden is None:
            # new id assignment
            num_boxes = len(bbox_results[0]['boxes_3d'])
            box_id = -1*torch.ones(num_boxes+1,device=hidden_xyz.device)
            box_id[:-1] = torch.arange(self.min_id,self.min_id+num_boxes,device=hidden_xyz.device,dtype=torch.int64)
            self.min_id += num_boxes
            bbox_results[0]['id'] = box_id[:-1].cpu()
            hidden_dict['id'] = -1*torch.ones(hidden_xyz.shape[1]+1,device=hidden_xyz.device,dtype=torch.int64)
            hidden_dict['id'][:-1] = box_id[assignment]
        else:
            hidden_indices = bbox_preds['hidden_indices']
            hidden_indices -= 16384
            hidden_indices[hidden_indices<0] = -1
            hidden_id = old_hidden_id[hidden_indices].to(torch.int64)
            num_boxes = len(bbox_results[0]['boxes_3d'])
            box_id = -1*torch.ones(num_boxes+1,device=hidden_xyz.device,dtype=torch.int64)
            used_ids = set()
            for box in range(num_boxes):
                ids = torch.where(assignment[0]==box)[0]
                if ids.shape[0]!=0:
                    inner_ids = hidden_id[:,ids]
                    inner_ids = inner_ids[inner_ids!=-1]
                    if inner_ids.shape[0] != 0:
                        while torch.mode(inner_ids)[0].item() in used_ids:
                            inner_ids = inner_ids[inner_ids!=inner_ids.mode()[0].item()]
                        box_id[box] = torch.mode(inner_ids)[0].item()
                    else:
                        box_id[box] = self.min_id
                        self.min_id += 1
                else:
                    box_id[box] = self.min_id
                    self.min_id += 1
            bbox_results[0]['id'] = box_id[:-1].cpu()
            hidden_dict['id'] = -1*torch.ones(hidden_xyz.shape[1]+1,device=hidden_xyz.device,dtype=torch.int64)
            hidden_dict['id'][:-1] = box_id[assignment]

        return bbox_results, hidden_dict

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import os.path as osp
import numpy as np
import torch

from ..utils import get_root_logger
from .base import BaseDataset, get_group
from .builder import DATASETS

import copy

@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 transform_before_merge=False,
                 **kwargs):
        modality = 'Pose'
        self.split = split

        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, memcached=memcached, mc_cfg=mc_cfg, **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None and isinstance(self.valid_ratio, float) and self.valid_ratio > 0:
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                assert 'box_score' in item, 'if valid_ratio is a positive number, item should have field `box_score`'
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds

        if not transform_before_merge:
            for item in self.video_infos:
                item.pop('valid', None)
                item.pop('box_score', None)
                if self.memcached:
                    item['key'] = item['frame_dir']

        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        return data


@DATASETS.register_module()
class PoseDatasetMV(PoseDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 pair_mode=False,
                 is_split_by_group=True,
                 transform_before_merge=False,
                 **kwargs):
        self.pair_mode = pair_mode
        self.is_split_by_group = is_split_by_group
        self.transform_before_merge = transform_before_merge
        super().__init__(
            ann_file, pipeline, split, valid_ratio, box_thr, class_prob, memcached, mc_cfg, transform_before_merge, **kwargs)


    def create_new_group_annot(self, annot, group):
        annotmv = annot
        annotmv['frame_dir'] = group
        return annotmv

    def merge_annot_into_other(self, annot, other):
        if other['keypoint'].shape[1] != annot['keypoint'].shape[1] and False:
            print("watchout, temporal dimension mismatch : ", other['keypoint'].shape[1], " vs ", annot['keypoint_score'].shape[1])

        concat = copy.copy(other)

        T = min(concat['keypoint'].shape[1], annot['keypoint'].shape[1])
        concat['total_frames'] = T
        concat['keypoint'] = np.concatenate((concat['keypoint'][:, :T, :, :], annot['keypoint'][:, :T, :, :]))
        concat['keypoint_score'] = np.concatenate((concat['keypoint_score'][:, :T, :], annot['keypoint_score'][:, :T, :]))
        
        return concat

    def datamv_from_split(self, data, split):
        datamv = {}

        for annot in data:
            group = get_group(annot['frame_dir'])

            # do not take data not in this split
            if group not in split:
                continue
            
            if group not in datamv:
                # copie first annotation in the group
                datamv[group] = []

            if annot['keypoint'].shape[0] != 1 and annot['keypoint'].shape[0] != 2:
                print("different : ", annot['keypoint'].shape)

            datamv[group].append(annot)

        return datamv

    def data_from_split(self, data, split):
        datamv = self.datamv_from_split(data, split)
        
        idxToRem = []
        for idx, datas in enumerate(datamv):
            if len(datas) != 3:
                print('watchout: num of view is different from 3 : ', len(datamv))
                idxToRem.append(idx)
        
        for idx in sorted(idxToRem, reverse=True):
            del datamv[idx]

        return datamv

    def data_pair_from_split(self, data, split):

        datamv = self.datamv_from_split(data, split)

        idxPairs = {0: 1, 1: 2, 2: 0}
        pairs = []
        for group, datas in datamv.items():
            if len(datas) != 3:
                print('watchout: num of view is different from 3 : ', len(datas))
                continue

            for i in range(3):
                annot = datas[i]
                annot_linked = datas[idxPairs[i]]
                pairs.append((annot, annot_linked))

        idxToRem = []
        for idx, pair in enumerate(pairs):
            if len(pair) != 2:
                print('watchout: num of view is different from 2 : ', len(pair))
                idxToRem.append(idx)
        
        for idx in sorted(idxToRem, reverse=True):
            del pairs[idx]

        return pairs
    
    def merge_datamv(self, datamv):
        annotsmv = []
        for group, datas in datamv.items():
            annotmv = self.create_new_group_annot(datas[0], group)
            for idx in range(1, len(datas)):
                annot = datas[idx]
                annotmv = self.merge_annot_into_other(annot, annotmv)
                annotsmv.append(annotmv)

        for amv in annotsmv:
            if amv['keypoint'].shape[0] != 3 and amv['keypoint'].shape[0] != 6:
                print('watchout: num of view is different from 3 : ', amv['keypoint'].shape[0])

        return annotsmv

    def merge_pairs(self, pairs, removeIfWrong=False):
        # merge pairs
        pairs_merged = []
        for pair in pairs:
            annot = pair[0]
            annot_linked = pair[1]
            annot = self.merge_annot_into_other(annot_linked, annot)
            pairs_merged.append(annot)

        # check valid keypoints shape 
        idxToRem = []
        for idx, pair in enumerate(pairs_merged):
            isdatacorrect = pair['keypoint'].shape[0] == 2 or pair['keypoint'].shape[0] == 4

            if removeIfWrong:
                if not isdatacorrect:
                    idxToRem.append(idx)
            else:
                assert isdatacorrect, "wrong keypoint shape : " + str(pair['keypoint'].shape[0])


        if removeIfWrong:
            for idx in sorted(idxToRem, reverse=True):
                del pairs_merged[idx]

        return pairs_merged

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            if not self.is_split_by_group:
                split = [get_group(s) for s in split]

            data = self.data_pair_from_split(data, split) if self.pair_mode else self.data_from_split(data, split)

            if not self.transform_before_merge:
                data = self.merge_pairs(data, True) if self.pair_mode else self.merge_datamv(data)

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        return data
    
    def merge_results_trans(self, results_trans):
        imgs = [res["imgs"] for res in results_trans]
        labels = [res["label"] for res in results_trans]

        assert labels.count(labels[0]) == len(labels), 'not all labels from item are the same : ' + labels

        imgs = torch.cat(imgs)
        return {"imgs": imgs, "label": labels[0]}
    
    def prepare_train_frames(self, idx):
        if self.transform_before_merge:
            results_groups = self.video_infos[idx]
            results_trans = []
            for res in results_groups:
                results = copy.deepcopy(res)
                res_trans = self.prepare_train_frames_for_results(results)
                results_trans.append(res_trans)
            return self.merge_results_trans(results_trans)

        else:
            results = copy.deepcopy(self.video_infos[idx])
            return self.prepare_train_frames_for_results(results)
        

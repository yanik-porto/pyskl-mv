# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv import digit_version as dv
from mmcv import load
from mmcv.cnn import fuse_conv_bn
from mmcv.engine import multi_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.utils import cache_checkpoint, mc_off, mc_on, test_port
from pyskl.datasets.base import get_cam

from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()

    cfg = Config.fromfile(args.config)

    out = osp.join(cfg.work_dir, 'result.pkl') if args.out is None else args.out


    mmcv.mkdir_or_exist(osp.dirname(out))
    _, suffix = osp.splitext(out)
    assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    print("type : ", type(model))

    args.checkpoint = cache_checkpoint(args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    label_map = [x.strip() for x in open("tools/data/label_map/nturgbd_120.txt").readlines()]
    
    data_infos = [ann['frame_dir'] for ann in dataset.video_infos]
    cams = [get_cam(info) for info in data_infos]

    maxiter = 10000

    onlylabels = [10, 11]

    feats = []
    labels = []
    for idx, input_batch in tqdm(enumerate(data_loader)):
        if idx >= maxiter:
            break

        labelid = input_batch['label'].detach().numpy()[0]
        if labelid not in onlylabels:
            continue

        # data = None
        onepersfeat = None
        if "keypoint" in input_batch:
            keypoint = input_batch["keypoint"]
            bs, nc = keypoint.shape[:2]
            keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])
            data = keypoint

            x = model.extract_feat(data)
            x = x.detach().numpy()
            onepersfeat = x[0, 0, :, :, :].reshape(-1)

        elif "imgs" in input_batch:
            imgs = input_batch["imgs"]
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            data = imgs

            x = model.extract_feat(data)
            x = x.detach().numpy()
            onepersfeat = x[0, :, :, :, :].reshape(-1)


        feats.append(onepersfeat)
        labels.append(label_map[labelid])


    print("compute tsne for ", len(feats), " sequences")

    tsne = TSNE(n_components=2, random_state=42)

    X_tsne = tsne.fit_transform(feats)

    print("divergence: ", tsne.kl_divergence_)

    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=labels)
    fig.update_layout(
        title="t-SNE visualization of Custom Classification dataset",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )
    fig.show()

if __name__ == '__main__':
    main()

import copy
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from .builder import DATASETS
from .coco import CocoDataset
from torch import distributed as dist
from .builder import DATASETS
from .pipelines import Compose
import pandas as pd
from .organ_lvis import OrganLVIS
from mmdet.models.roi_heads.class_name import organ_novel_label_ids, ORGAN_CLASSES

@DATASETS.register_module()
class OrganDataset(CocoDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root,
                 img_prefix,
                 test_mode=False,
                 proposal_file=None):
        # super(OrganDataset, self).__init__(ann_file, pipeline)
        self.CLASSES = ORGAN_CLASSES
        self.cat_ids = [_ for _ in range(len(self.CLASSES))]
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.img_ids = []
        self.data_infos = self.load_annotations(ann_file)
        self.pipeline = Compose(pipeline)
        if not test_mode:
            self._set_group_flag()
        self.coco = OrganLVIS(ann_file, self.img_ids, self.cat_ids, self.CLASSES)

    def load_annotations(self, ann_file):
        df = pd.read_csv(ann_file, index_col=False)
        df = df[df['is_unknown'] == 0]
        if not self.test_mode:
            novel_class_name = np.array(self.CLASSES)[organ_novel_label_ids]
            df['is_novel'] = df['label'].apply(lambda x: x in novel_class_name)
            out_image_id = set(df[df['is_novel'] == True]['image_id'])
            # df = df[df['is_novel'] == False]
            df['has_out'] = df['image_id'].apply(lambda x: x in out_image_id)
            df = df[df['has_out'] == False]

        # # L10 for cls
        # df = df.drop_duplicates(subset=['ori_x1', 'ori_x2', 'ori_y1', 'ori_y2'])
        ######
        data_infos = []
        for image_id, df_cur in df.groupby('image_id'):
            info = dict()
            info['img_prefix'] = self.img_prefix
            info['ann_info'] = dict()
            info['ann_info']['bboxes'] = df_cur[['x1', 'y1', 'x2', 'y2']].values.tolist()
            info['ann_info']['labels'] = np.array([self.CLASSES.index(lb) for lb in df_cur['label'].values.tolist()])
            info['img_info'] = dict()
            info['img_info']['filename'] = df_cur.iloc[0]['filename_gfp']
            info['img_info']['height'] = df_cur.iloc[0]['size']
            info['img_info']['width'] = df_cur.iloc[0]['size']
            info['img_info']['day'] = df_cur.iloc[0]['day']
            info['img_info']['image_id'] = image_id
            info['bbox_fields'] = []
            self.img_ids.append(image_id)
            data_infos.append(info)
        self.rank = dist.get_rank()
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            results = copy.deepcopy(self.data_infos[idx])
            data = self.pipeline(results)
            return data
        else:
            while True:
                try:
                    results = copy.deepcopy(self.data_infos[idx])
                    # used for grl and triplet
                    # results['ann_info']['labels'] = results['ann_info']['labels'] * 1000 + results['img_info']['day']
                    # results['proposals'] = None
                    data = self.pipeline(results)
                except:
                    idx = self._rand_another()
                    print_log("sample error: {}".format(idx), logger=None)
                    pass
                return data

    def _rand_another(self):
        return np.random.choice(self.__len__())

    def get_cat_ids(self, idx):
        return self.data_infos[idx]['ann_info']['labels'].astype(np.int64).tolist()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        # for i in range(len(self)):
        #     img_info = self.data_infos[i]
        #     if img_info['ann_info']['width'] / img_info['ann_info']['height'] > 1:
        #         self.flag[i] = 1

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',
                    # noqa: E501
                    UserWarning)
            from lvis import LVISResults, LVISEval
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'
                # noqa: E501
            )
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)

        eval_results = OrderedDict()
        # get original api
        lvis_gt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                lvis_dt = LVISResults(lvis_gt, result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type)
            # lvis_eval.params.area_rng
            lvis_eval.params.imgIds = self.img_ids
            if metric == 'proposal':
                lvis_eval.params.useCats = 0
                lvis_eval.params.maxDets = list(proposal_nums)
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                for k, v in lvis_eval.get_results().items():
                    if k.startswith('AR'):
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[k] = val
            else:
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                lvis_results = lvis_eval.get_results()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = lvis_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.load_cats([catId])[0]
                        precision = precisions[:, :, idx, 0]
                        # precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                for k, v in lvis_results.items():
                    if k.startswith('AP'):
                        key = '{}_{}'.format(metric, k)
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[key] = val
                ap_summary = ' '.join([
                    '{}:{:.3f}'.format(k, float(v))
                    for k, v in lvis_results.items() if k.startswith('AP')
                ])
                eval_results['{}_mAP_copypaste'.format(metric)] = ap_summary
            lvis_eval.print_results()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


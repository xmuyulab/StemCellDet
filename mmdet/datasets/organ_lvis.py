from lvis.lvis import LVIS
import json
import os
import logging
from collections import defaultdict
from urllib.request import urlretrieve
import pandas as pd
import pycocotools.mask as mask_utils
from mmdet.models.roi_heads.class_name import organ_novel_label_ids

class OrganLVIS(LVIS):
    def __init__(self, ann_file, img_ids, cat_ids, CLASSES):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")

        self.dataset = dict()
        self.dataset['images'] = [{"idx": ind, "id": ids} for ind, ids in enumerate(img_ids)]
        self.dataset['categories'] = [{"idx": ind,
                                       "id": ids,
                                       "frequency": "r" if ids in organ_novel_label_ids else "f",
                                       "name": CLASSES[ids]} for ind, ids in enumerate(cat_ids)]

        df = pd.read_csv(ann_file, index_col=False)
        self.dataset['annotations'] = []
        for ind in range(len(df)):
            bbox = df.iloc[ind][['x1', 'y1', 'x2', 'y2']].values.tolist()
            info = dict({"id": ind,
                         "image_id": df.iloc[ind]['image_id'],
                         "filename": df.iloc[ind]['filename_gfp'],
                         "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                         "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                         "category_id": CLASSES.index(df.iloc[ind]['label'])})
            self.dataset['annotations'].append(info)

        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        self._create_index()

        for i in range(len(self.dataset['images'])):
            neg_category_ids = []
            for ids in cat_ids:
                if self.dataset['images'][i]['id'] not in self.cat_img_map[ids]:
                    neg_category_ids.append(ids)
            self.dataset['images'][i]['neg_category_ids'] = neg_category_ids
            self.dataset['images'][i]['not_exhaustive_category_ids'] = []

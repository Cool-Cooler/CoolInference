#coding: utf-8

from flask import Flask, redirect, url_for, request
from werkzeug.utils import secure_filename

import torch, torchvision

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances

import os
import sys

app = Flask(__name__)

def init_setup():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE='cpu'
    return cfg

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def detect_object(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATH = os.path.join(PATH_TO_TEST_IMAGES_DIR, filename)
    PRED_IMAGE_PATH = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'pred_' + filename)
    
    im = cv2.imread(TEST_IMAGE_PATH)
    cfg = app.config['detectron2_cfg']

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # filterout bana and orage
    data_set = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # print(data_set.thing_classes)
    pred_inst = outputs["instances"].to("cpu")

    show_inst = []
    pred_res = []
    for tc in app.config['THING_CLASSES']:
        if tc not in data_set.thing_classes:
            print("Thing Class:"+ tc +", Not found in the training set")
            continue
        t_idx = data_set.thing_classes.index(tc)
        filt_inst = pred_inst[pred_inst.pred_classes == t_idx]
        cat_cnt = len(filt_inst)
        if cat_cnt > 0:
            show_inst.append(filt_inst)
            pred_res.append({"t_class": tc, "t_count":cat_cnt})

    if len(show_inst) > 0:
        pred_inst = Instances.cat(show_inst)

    # Comment this out later
    # v = Visualizer(im[:, :, ::-1],data_set , scale=0.3)
    # out = v.draw_instance_predictions(pred_inst)
    # cv2.imwrite(PRED_IMAGE_PATH, out.get_image()[:, :, ::-1])
    
    response = app.response_class(
        response=json.dumps({'result': pred_res}),
        status=200,
        mimetype='application/json'
    )

    return response


@app.route("/infer", methods=['POST'])
def infer():
    file = request.files['fimg']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return detect_object(filename=filename)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = '/app/imgstore/'
    app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
    app.config['detectron2_cfg'] = init_setup()
    app.config['THING_CLASSES'] = ['banana', 'orange', 'carrot', 'apple', 'bottle']
    app.run(debug=False,host='0.0.0.0')

from detectron2.utils.logger import setup_logger

import streamlit as st

setup_logger()

import collections

import torch, torchvision

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import masks

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper

import os
import cv2
from matplotlib import pyplot as plt

from IPython.display import display
from PIL import Image


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import copy
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# Зарегистрировать как коко формат
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils


def predictor_define(cfg, CONFIDENCE=0.7):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE  # set a custom testing threshold 0.7
    predictor = DefaultPredictor(cfg)


def inference(image, THR=0.5, path_to_model="/content/output/model_final.pth"):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"  # 'cpu' or 'cuda'

    out_dct = dict()

    cat_lst = ["0.2", "0.3", "0.5", "0.8"]

    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THR
    cfg.MODEL.WEIGHTS = path_to_model
    MetadataCatalog.get("dataset").thing_classes = cat_lst

    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    box_lst = [i.tolist() for i in outputs["instances"].pred_boxes]
    cl_lst = outputs["instances"].pred_classes.tolist()
    score_lst = outputs["instances"].scores.tolist()

    out_dct["box_lst"] = box_lst
    out_dct["cl_lst"] = [cat_lst[i] for i in cl_lst]
    out_dct["scores"] = score_lst

    return out_dct


def inference_2(image, path_to_model, dataset_name, YAML_FILE, cat_lst=["0.2", "0.3", "0.5", "0.8"], thr=0.25,
                aa_dct='None', device='cuda', loader='default', aug=[            T.Resize((512, 512))        ]):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device

    out_dct = dict()

    cfg.merge_from_file(
        model_zoo.get_config_file(YAML_FILE)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cat_lst)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr
    cfg.INPUT.MIN_SIZE_TEST = 900
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.MODEL.WEIGHTS = path_to_model

    if aa_dct != 'None':
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = aa_dct['anchor_sizes']
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = aa_dct['aspect_ratio']

    # MetadataCatalog.get(dataset_name).thing_classes = cat_lst
    predictor = DefaultPredictor(cfg)

    # if loader == 'default':
    #   val_loader = build_detection_test_loader(cfg, dataset_name)
    # else:
    #   val_loader = build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False, augmentations = aug))

    # return_inference = inference_on_dataset(predictor.model, val_loader, evaluator)

    outputs = predictor(image)

    box_lst = [i.tolist() for i in outputs["instances"].pred_boxes]
    cl_lst = outputs["instances"].pred_classes.tolist()
    score_lst = outputs["instances"].scores.tolist()

    out_dct["box_lst"] = box_lst
    out_dct["cl_lst"] = [cat_lst[i] for i in cl_lst]
    out_dct["scores"] = score_lst

    return out_dct


def imagename(str):
    return str.split('/')[-1]


def read_image(im_path, YAML_FILE, path_to_model, THR=0.5, dim=(500, 500),
                                cat_lst=['pomeranc', 'poteklina']):

    out_dct = {}

    im_name = imagename(im_path)
    # img = cv2.imread(os.path.join(root_path, d["file_name"]))
    img = cv2.imread(im_path)
    out_dct[im_name] = inference_2(image=img, path_to_model=path_to_model,
                                   dataset_name=None, YAML_FILE=YAML_FILE,
                                   cat_lst=cat_lst, thr=THR)
    print(im_path)
    # print(os.path.join(im_path, d["file_name"]))
    labels_list = []
    boxes = []
    name_dict = {}

    for i in range(len(out_dct[im_name]['cl_lst'])):
        box = out_dct[im_name]['box_lst'][i]
        box = [int(i) for i in box]
        label = out_dct[im_name]['cl_lst'][i]
        scores = out_dct[im_name]['scores'][i]
        # print(label, scores)
        # print(box)
        labels_list.append(label)
        boxes.append(box[0])

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        cv2.putText(img, label, (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    # cv2.putText(img, str(int(scores*100)), (box[0], box[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # print(d["file_name"])
    # print(labels_list)

    for num, name in zip(boxes, labels_list):
        name_dict[num] = name

    # print(name_dict)

    od = collections.OrderedDict(sorted(name_dict.items()))
    digits_out_sorted = []
    for k, v in od.items():
        digits_out_sorted.append(v)

    # print(od)

    print(digits_out_sorted)

    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    im_output_path = ROOT_FOLDER + 'output_images/' + im_path.split('/')[-1]
    print(im_output_path)
    cv2.imwrite(im_output_path, img)

def process_one_image(img, im_name, YAML_FILE, path_to_model, THR=0.5, dim=(500, 500),
                                cat_lst=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']):

    out_dct = {}

    # im_name = imagename(im_path)
    # img = cv2.imread(os.path.join(root_path, d["file_name"]))
    # img = cv2.imread(im_path)
    out_dct[im_name] = inference_2(image=img, path_to_model=path_to_model,
                                   dataset_name=None, YAML_FILE=YAML_FILE,
                                   cat_lst=cat_lst, thr=THR)
    # print(im_path)
    # print(os.path.join(im_path, d["file_name"]))
    labels_list = []
    boxes = []
    name_dict = {}

    for i in range(len(out_dct[im_name]['cl_lst'])):
        box = out_dct[im_name]['box_lst'][i]
        box = [int(i) for i in box]
        label = out_dct[im_name]['cl_lst'][i]
        scores = out_dct[im_name]['scores'][i]
        # print(label, scores)
        # print(box)
        labels_list.append(label)
        boxes.append(box[0])

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        cv2.putText(img, label, (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    # cv2.putText(img, str(int(scores*100)), (box[0], box[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # print(d["file_name"])
    # print(labels_list)

    for num, name in zip(boxes, labels_list):
        name_dict[num] = name

    # print(name_dict)

    od = collections.OrderedDict(sorted(name_dict.items()))
    digits_out_sorted = []
    for k, v in od.items():
        digits_out_sorted.append(v)

    # print(od)

    print(digits_out_sorted)

    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # im_output_path = ROOT_FOLDER + 'output_images/' + im_path.split('/')[-1]
    # print(im_output_path)
    # cv2.imwrite(im_output_path, img)

    return img, digits_out_sorted

ROOT_FOLDER = '/home/roman/PycharmProjects/streamlit/digits/'

train_dct1 = {
    '10cl':
        {
            'cat_lst': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            'train': {
                '13': {
                    'json': ROOT_FOLDER + '/json/digits-13.json',
                    'data': ROOT_FOLDER + '/data/train/',

                },
                '14': {
                    'json': ROOT_FOLDER + '/json/digits-14.json',
                    'data': ROOT_FOLDER + '/data/train/',

                },
                '17': {
                    'json': ROOT_FOLDER + '/annotations/digits_04-17.json',
                    'data': ROOT_FOLDER + '/images/train/',

                },
            },
            'val': {
                '16': {
                    'json': ROOT_FOLDER + '/annotations/digits_02-16.json',
                    'data': ROOT_FOLDER + '/images/val/',

                },
            }
        }
}

if __name__ == '__main__':

    st.title('DIGITS')

    model_name = 'model_final'  # '30im_10cl_2000it_faster_rcnn_R_50_FPN_3x_None'
    model_name = 'model_0003999'
    # model_name = 'model_final_8000'
    # model_name = 'model_final_4000_nms_01'
    model_path = ROOT_FOLDER + 'output/' + model_name + '.pth'

    YAML_FILE = 'Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml'
    # YAML_FILE = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

    DATASET_NAME = model_name

    val_dataset_name = DATASET_NAME + "_val_" + str(random.randint(1, 1000))

    num_classes = '10cl'  # 'poteklina' # '1cl'
    dataset = '16'  #

    # MetadataCatalog.get(val_dataset_name).thing_classes = train_dct1['10cl']['cat_lst'] #["person", "dog"]
    register_coco_instances(val_dataset_name, {}, train_dct1[num_classes]['val'][dataset]['json'],
                            train_dct1[num_classes]['val'][dataset]['data'])
    dataset_dicts = DatasetCatalog.get(val_dataset_name)

    filenames = []
    uploaded_files = st.file_uploader("Choose a images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

    num_ok = 0
    num_nok = 0
    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")
        # Convert PIL image to array
        image = np.array(image)

        img, predicted_digits = process_one_image(image, uploaded_file.name, YAML_FILE=YAML_FILE, path_to_model=model_path, THR = 0.7)

        print(uploaded_file.name)
        print(predicted_digits)
        st.image(image)

        wrong_indexes = []
        result_in_string = ''

        for el in predicted_digits:
            result_in_string += el

        if len(predicted_digits) >= 6:
            for num, el in enumerate(uploaded_file.name[:6]):

                if el != predicted_digits[num]:
                    wrong_indexes.append(num)
                    print(el, num)
        else:
            wrong_indexes = ['0']

        status = 'NOK'
        if len(wrong_indexes) == 0:
            status = 'OK'

        if status == 'OK':
            num_ok +=1
        else:
            num_nok +=1

        file_details = {"filename": uploaded_file.name, "detection_result": result_in_string,
         "status": status}

        st.write(file_details)

        st.write('Точність визначення = ', num_ok /(num_ok + num_nok) * 100, ' %')
        st.write('NOK = ', num_nok, 'OK = ', num_ok)
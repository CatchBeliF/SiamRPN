import multiprocessing as mp
import os
from glob import glob
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from net.config import Config
from lib.util import get_instance_img
from multiprocessing import Pool
import pickle


def process_single(sequence_dir, output_dir):
    all_images = glob(os.path.join(sequence_dir, '*.JPEG'))
    all_images = sorted(all_images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    img_name = sequence_dir.split('/')[-1]
    img_path = os.path.join(output_dir, img_name)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    trajs = {}

    for single_img in all_images:
        img = cv2.imread(single_img)
        img_h, img_w, _ = img.shape
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        # 修改为对应gt的目录
        gt_dir = single_img.replace('Data', 'Annotations')
        gt_dir = gt_dir.replace('JPEG', 'xml')
        tree = ET.parse(gt_dir)
        root = tree.getroot()
        # bbox = []
        filename = root.find('filename').text
        for obj in root.iter('object'):
            trkid = int(obj.find('trackid').text)
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            # bbox是整数
            if trkid in trajs:
                trajs[trkid].append(filename)
            else:
                trajs[trkid] = [filename]

            # 这里算gt的宽高需要加1，因为长度和坐标不一样
            bbox = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0] + 1,
                             bbox[3] - bbox[1] + 1])
            # bbox是浮点数
            instance_img, _, w, h = get_instance_img(img, bbox, Config.exemplar_size, Config.crop_size, img_mean)

            instance_img_path = os.path.join(img_path,
                                             filename + ".{:02d}.gt_w_{:.2f}.gt_h_{:.2f}.img_w_{:.0f}.img_h_.{:.0f}.jpg".format(
                                                 trkid, w, h, img_w, img_h))

            # cv2.imwrite写入图片需要两个变量（文件名，图片）
            cv2.imwrite(instance_img_path, instance_img)
    return img_name, trajs


def process_all(video_dir, output_dir, num_threads=mp.cpu_count()):
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))
    meta_data = []
    len = 0
    total = all_videos.__len__()
    with Pool(processes=num_threads) as pool:
        print("starting")
        for sequence_dir in all_videos:
            meta_data.append(process_single(sequence_dir, output_dir))
            len += 1
            print("finish process ", sequence_dir, "\n")
            print("left ", total - len, "waiting \n")

    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    video_dir = "/home/cbf/datasets/ILSVRC2015_VID/ILSVRC2015/Data/VID"
    output_dir = "/home/cbf/datasets/ILSVRC2015_VID/ILSVRC2015_VID_curation"
    process_all(video_dir, output_dir, num_threads=mp.cpu_count())

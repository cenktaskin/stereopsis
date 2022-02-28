from pathlib import Path
from cv2 import cv2
import numpy as np
import yaml


data_path = Path.cwd().joinpath('../data/').resolve()


def parse_stereo_img(st_img, cam):
    return st_img[:, st_img.shape[1] * cam // 2:st_img.shape[1] * (cam + 1) // 2]


def get_img_from_dataset(img_path, cam_index):
    if cam_index == 2:
        # Getting the first layer of 3 channel img
        ir_img = cv2.imread(img_path.with_stem("ir" + img_path.stem[2:]).as_posix(), flags=cv2.IMREAD_UNCHANGED)
        # Returns flipped image since the camera setup is inverted
        return cv2.flip(ir_img, -1)
    elif cam_index > 2:
        return upsample_ir_img(get_img_from_dataset(img_path, 2), resize=(cam_index == 4))
    else:
        stereo_img = cv2.imread(img_path.as_posix())
        return parse_stereo_img(stereo_img, cam_index)


def upsample_ir_img(src, dst_size=(720, 1280), resize=False):
    target_size = np.array(dst_size)
    curr_size = np.array(src.shape)
    factor = min(target_size // curr_size)
    if not resize:
        factor = 1
    diff = target_size - curr_size * factor
    pad = diff // 2
    if resize:
        src = cv2.resize(src, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(src, pad[0], pad[0], pad[1], pad[1], cv2.BORDER_CONSTANT, value=0)


def load_camera_info(cam_index, parameter_type):
    """parameter type has to be either intrinsic or extrinsic"""
    with open(data_path.joinpath("processed", f"cam{cam_index}-{parameter_type}.yaml"), 'r') as f:
        print(f"Loaded {parameter_type} for camera{cam_index} from {f.name}")
        return yaml.load(f, Loader=yaml.Loader)


def save_camera_info(obj, cam_index, info_type):
    with open(data_path.joinpath("processed", f"cam{cam_index}-{info_type}.yaml"), 'w') as f:
        yaml.dump(obj, f)
        print(f"Dumped {info_type} for camera{cam_index} to {f.name}")


# this would be obsolete soon, maybe change it with a random dataset frame
def get_random_calibration_frame():
    from random import sample
    calibration_imgs_dir = data_path.joinpath('raw', 'calibration-images-20210609')
    calibration_images = sorted(calibration_imgs_dir.glob('st*.jpeg'))
    return sample(calibration_images, 1)[0]
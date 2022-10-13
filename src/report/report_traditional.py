import torch
import cv2
from src.preprocessing.data_io import CalibrationDataHandler, show_images
from src.dataset import data_path, StereopsisDataset

epochs = 200
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = data_path.joinpath("processed/report-material/test/")

calib_dh = CalibrationDataHandler("20220610")
rectification = calib_dh.load_camera_info(0, "rectification-wrt-cam1")

ds = StereopsisDataset(img_dir=img_path.joinpath(str(0)), val_split_ratio=1.0)
for ts in ds.timestamp_list:
    img0 = cv2.imread(img_path.joinpath(str(0), f"sl_{ts}.tiff").as_posix())
    img1 = cv2.imread(img_path.joinpath(str(0), f"sr_{ts}.tiff").as_posix())

    block_size = 11
    min_disp = 0
    max_disp = 48
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=max_disp,
                                   blockSize=block_size,
                                   P1=8 * 3 * block_size ** 2,
                                   P2=32 * 3 * block_size ** 2,
                                   uniquenessRatio=10,
                                   speckleWindowSize=125,
                                   speckleRange=2,
                                   mode=cv2.STEREO_SGBM_MODE_HH4
                                   )

    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    import numpy as np

    disparity = stereo.compute(img0_gray, img1_gray)
    disparity = disparity.astype(float)
    disparity[disparity == -16] = np.nan
    # depth = cv2.reprojectImageTo3D(disparity, rectification["Q"], handleMissingValues=True)[..., 2] * -1 / 100
    # depth[depth < 0] = 0
    disparity = cv2.normalize(disparity * -1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(img_path.parent.joinpath("preds_by_sgbm", f"pred_{ts}.tiff").as_posix(), disparity)

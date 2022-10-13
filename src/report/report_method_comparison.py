from pathlib import Path
from src.preprocessing.data_io import data_path
import cv2
import matplotlib.pyplot as plt
import numpy as np
report_dir = data_path.joinpath("processed", "report-material")

ts_list = [f.stem[3:] for f in report_dir.joinpath("test","0").glob("dp*.tiff")]
for ts in ts_list:
    label = cv2.imread(report_dir.joinpath("test", "0", f"dp_{ts}.tiff").as_posix(),
                           flags=cv2.IMREAD_UNCHANGED)
    label2 = cv2.imread(report_dir.joinpath("test", "3", f"dp_{ts}.tiff").as_posix(),
                       flags=cv2.IMREAD_UNCHANGED)
    disp_pred = cv2.imread(report_dir.joinpath("preds_by_disp", "0", f"pred_{ts}.tiff").as_posix(),
                           flags=cv2.IMREAD_UNCHANGED)
    disp_pred2 = cv2.imread(report_dir.joinpath("preds_by_disp", "3", f"pred_{ts}.tiff").as_posix(),
                           flags=cv2.IMREAD_UNCHANGED)
    psm_pred = 255 -cv2.imread(report_dir.joinpath("preds_by_psm", "0", f"pred_{ts}.png").as_posix(),
                          flags=cv2.IMREAD_UNCHANGED)
    sgbm_pred = cv2.imread(report_dir.joinpath("preds_by_sgbm", f"pred_{ts}.tiff").as_posix(),
                           flags=cv2.IMREAD_UNCHANGED)/25

    sgbm_pred[np.isnan(sgbm_pred)] = 0

    imgs = [label, label2, disp_pred, disp_pred2, sgbm_pred, psm_pred]
    row_count = 3
    col_count = 2
    fig, axs = plt.subplots(row_count, col_count)
    mng = plt.get_current_fig_manager()
    for i, img in enumerate(imgs):
        img = img.squeeze()
        plt.subplot(row_count, col_count, i + 1)
        if img.ndim > 2:  # bgr -> rgb
            img = img[:, :, ::-1]
        plt.imshow(img, interpolation=None)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    mng.resize(600, 550)
    fig.show()
    while not plt.waitforbuttonpress():
        pass
    plt.close(fig)

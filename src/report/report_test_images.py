import cv2
import matplotlib.pyplot as plt

from src.preprocessing.data_io import data_path, RawDataHandler, show_images
from report_model_comparison import runs

imgs_dir = data_path.joinpath(f"processed/report-material/test")
ts_list = [f.stem[3:] for f in list(imgs_dir.joinpath("0").glob("sl*.tiff"))]

for ts in ts_list:
    img = cv2.imread(imgs_dir.joinpath(str(0), f"sr_{ts}.tiff").as_posix())
    raw_label = cv2.imread(data_path.joinpath("raw", "rawdata-202207011356-cleaned", f"dp_{ts}.tiff").as_posix(),
                           flags=cv2.IMREAD_UNCHANGED)
    seen_label = [cv2.imread(imgs_dir.joinpath(str(i), f"dp_{ts}.tiff").as_posix(), flags=cv2.IMREAD_UNCHANGED) for i in
                  range(5)]
    preds = [
        cv2.imread(imgs_dir.parent.joinpath("preds_by_disp", str(i), f"pred_{ts}.tiff").as_posix(), flags=cv2.IMREAD_UNCHANGED)
        for i in range(5)]

    res = [raw_label, img]
    for p, l in zip(preds, seen_label):
        res.append(l)
        res.append(p)

    imgs = res
    row_count = 6
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
    mng.resize(600, 950)
    fig.show()
    while not plt.waitforbuttonpress():
        pass
    plt.close(fig)

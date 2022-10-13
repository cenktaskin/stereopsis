from pathlib import Path
from src.preprocessing.data_io import data_path
import cv2
import matplotlib.pyplot as plt
from report_model_comparison import runs
from report_middleburry import read_pfm
from src.preprocessing.data_preprocessor import ImageResizer

mb_path = data_path.joinpath("raw/middlebury2014")

f = 3997.684
baseline = 193.001

image_l = cv2.cvtColor(cv2.imread(mb_path.joinpath("piano/im0.png").as_posix()), cv2.COLOR_BGR2RGB)
image_r = cv2.cvtColor(cv2.imread(mb_path.joinpath("piano/im1.png").as_posix()), cv2.COLOR_BGR2RGB)
label, _ = read_pfm(mb_path.joinpath("piano/disp1.pfm").as_posix())
label = (label + 131.111) ** -1 * f * baseline / 1000

resizer = ImageResizer(target_size=(384, 768))
image_l = resizer(image_l)
image_r = resizer(image_r)
label = resizer(label)

report_dir = data_path.joinpath("processed", "report-material")

raw = cv2.imread(data_path.joinpath("raw", "middlebury2014", "piano", f"im1.png").as_posix())
imgs = [image_l[..., ::-1], image_r[..., ::-1], label]
for r in runs:
    pred = cv2.imread(report_dir.joinpath("preds_on_mbury", f"pred_{r}.tiff").as_posix(),
                      flags=cv2.IMREAD_UNCHANGED)
    imgs.append(pred)

row_count = 4
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
mng.resize(600, 700)
fig.show()
while not plt.waitforbuttonpress():
    pass
plt.close(fig)


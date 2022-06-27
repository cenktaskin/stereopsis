from data_io import data_path, MultipleDirDataHandler
import cv2

data_id = "processed/dataset-20220610-registered"
data_dir = data_path.joinpath(data_id)
a = MultipleDirDataHandler(data_id, ("sl", "sr", "dp"))

for ts, raw_st, raw_depth in a.iterate_over_imgs():
    cv2.imwrite(data_dir.joinpath(f"dp_{ts}.tiff").as_posix(), raw_depth / 1000)

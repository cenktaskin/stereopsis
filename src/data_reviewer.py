from data_io import data_path
from cv2 import cv2
import yaml

dataset = '20220301'
dataset_path = data_path.joinpath('raw', f"data-{dataset}")

valid_frames = []
for img_path in dataset_path.glob('st*'):
    img = cv2.imread(img_path.as_posix())
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.resizeWindow("img", 2048, 720)
    while (key_press := cv2.waitKey() & 0xFF) not in [32, 8]:
        print(f"Wrong key is pressed, probably {chr(key_press)}")
        print(f"Try again...")
    if key_press == 32:
        valid_frames.append(img_path.stem[3:])
    cv2.destroyAllWindows()

with open(data_path.joinpath("processed", dataset, f"valid-frames.yaml"), 'w') as f:
    yaml.dump(valid_frames, f)
    print(f"Dumped valid frames for dataset {dataset} to {f.name}")


with open(data_path.joinpath("processed", dataset, f"valid-frames.yaml"), 'r') as f:
    print(f"Loaded valid frames for dataset {dataset} from {f.name}")
    a = yaml.load(f, Loader=yaml.Loader)

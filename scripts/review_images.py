from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd

images_path = Path('/home/cenkt/projektarbeit/img_out/')

index = 0
# 0 shit 1 only stereo 2 only ir 3 both good
review_dict = {}
for img_file in images_path.glob("st_*.jpeg"):
    pair = img_file.with_stem("ir"+img_file.stem[2:])
    img_st = cv.resize(cv.imread(str(img_file)), (0,0), None, .5, .5)
    img_ir = cv.rotate(cv.imread(str(pair)),cv.ROTATE_180)
    diff = np.array(img_st.shape)-np.array(img_ir.shape)
    img_ir_padded = cv.copyMakeBorder( img_ir, 0,0, diff[1]//2, diff[1]//2, cv.BORDER_CONSTANT)
    numpy_vertical = np.vstack((img_st, img_ir_padded))
    cv.imshow(str(img_file.stem),numpy_vertical)
    keystroke = cv.waitKey(0) & 0xFF
    if keystroke == ord('p'):
        print(img_file.stem,'passed')
        continue
    elif keystroke == ord('s'):
        break
    review_dict[img_file.stem] = chr(keystroke)
    print(img_file.stem,'marked as',chr(keystroke))
    index+=1
    print(index,'images done')
    cv.destroyAllWindows()

df = pd.DataFrame.from_dict(review_dict,orient='index',columns=['label'])
csv_path = Path.cwd().joinpath('review_of_calib_images1.csv')
while csv_path.exists():
    csv_path = csv_path.with_stem(csv_path.stem[:-1]+str(int(csv_path.stem[-1])+1))
df.to_csv(csv_path)
print(df)
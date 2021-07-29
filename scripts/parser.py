from pathlib import Path
import yaml

txt_path = Path('/home/cenkt/projektarbeit/calibrationdata/left.yaml')

with open(txt_path, 'r') as stream:
    left = yaml.safe_load(stream)

print(left['camera_matrix']['data'])

with open(txt_path.with_stem('right'), 'r') as stream:
    left = yaml.safe_load(stream)

print(left['camera_matrix']['data'])
import glob
import os
import shutil
import pandas as pd

data = pd.read_csv(r'..\data\oasis_longitudinal.csv')
scans_path = r'..\data\images\epi_part_2'

demented_path = r"..\data\images\demented"
nondemented_path = r"..\data\images\nondemented"

if not os.path.exists(demented_path):
    os.makedirs(demented_path)

if not os.path.exists(nondemented_path):
    os.makedirs(nondemented_path)

for scan_path in glob.glob(os.path.join(scans_path, 'OAS2_*')):
    i = 0
    mri_id = scan_path.split('\\')[-1]
    group = data.loc[data['MRI ID'] == mri_id]['Group'].values[0] == 'Demented'
    q = os.path.join(scan_path, 'RAW', '*.epi')
    for image in glob.glob(os.path.join(scan_path, '*.png')):
        i += 1
        dst_dir = demented_path if group else nondemented_path
        shutil.copy(image, dst_dir)

        dst_img = os.path.join(dst_dir, image.split('\\')[-1])
        new_dst_img = os.path.join(dst_dir, mri_id + "_" + str(i) + ".png")
        os.rename(dst_img, new_dst_img)

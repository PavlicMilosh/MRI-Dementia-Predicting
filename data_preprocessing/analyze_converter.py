# -*- coding: utf-8 -*-

import os
import re

import nibabel as nib
import numpy as np
import shutil
from nilearn import plotting
from nilearn import image
import time
import gc


def print_directory(src_path="."):
    """
    Prints desired directory, searches for Analyze files (HDR/IMG, NII)
    """
    hdr_count = 0
    img_count = 0
    nii_count = 0
    png_count = 0
    for root, dirs, files in os.walk(src_path):

        path = root.split(os.sep)
        print("\n" + (len(path) - 1) * "     ", os.path.basename(root))

        for file in files:

            extension = ""
            if file.lower().endswith(".hdr"):
                hdr_count += 1
                extension = "<=HDR"
            elif file.lower().endswith(".img"):
                extension = " <=IMG"
                img_count += 1
            elif file.lower().endswith(".nii"):
                extension = " <=NII"
                nii_count += 1
            elif file.lower().endswith(".png"):
                extension = " <=PNG"
                png_count += 1

            print(len(path) * "     ", file + extension)

    print("\n\nFound: \nHDR:\t %d \nIMG:\t %d \nPNG:\t %d\h\n" % (hdr_count, img_count, png_count))


def convert2nii(src_path=".", ext="nii"):
    """
    Converts pair of .img and .hdr analyze format to .nii format

    src_path:       source/root to look for .img files
    ext:            supported output extensions: .nii / .nii.gz
    """

    for root, dirs, files in os.walk(src_path):
        # path = root.split(os.sep)
        for file in files:
            if file.lower().endswith(".img"):
                file_base = os.path.splitext(os.path.basename(file))
                src = os.path.join(os.path.abspath(root), file)
                dst = os.path.join(os.path.abspath(root), file_base[0] + "." + ext)
                img = nib.load(src)
                nib.save(img, dst)


def delete_nii_files(src_path=".", ext="nii"):
    """
    Removes all .nii files from selected dir and it's subdirs

    src_path:       source/root to look for .img files
    ext:            supported output extensions: .nii / .nii.gz
    """

    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith("." + ext):
                dst = os.path.join(os.path.abspath(root), os.path.basename(file))
                os.remove(dst)


ext         = "png"
x_cuts      = []
y_cuts      = []
z_cuts      = []
fwhm        = 0
dpi         = 128
annotate    = False
black_bg    = True
mode        = "anat"
start       = None


def nii2image(src_path=".", p_ext="png", p_sagittal_cuts=30, p_coronal_cuts=30, p_axial_cuts=30, p_fwhm=0, p_dpi=128,
              p_annotate=False, p_black_bg=True, p_mode="anat"):
    """
    Converts .nii files to desired extension

    src_path:       source/root to look for .nii files
    ext:            supported output extensions: png, jpeg, svg ?
    sagittal_cuts:  no of cuts in x-direction
    coronal_cuts:   no of cuts in y-direction
    axial_cuts:     no of cuts in z-direction
    fwhm:           desired level for image smoothing
    dpi:            desired resolution for output images
    annotate:       annoting images (markers for left and right side of brain)
    black_bg:       black or white? BRECKU SENPAI
    mode:           1 for anatomical image, others for epi
    """

    global ext, x_cuts, y_cuts, z_cuts, fwhm, dpi, annotate, black_bg, mode, start

    ext = p_ext
    x_cuts = np.linspace(-60, 50, p_sagittal_cuts, endpoint=False)
    y_cuts = np.linspace(-80, 30, p_coronal_cuts, endpoint=False)
    z_cuts = np.linspace(-50, 50, p_axial_cuts, endpoint=False)
    fwhm = p_fwhm
    dpi = p_dpi
    annotate = p_annotate
    black_bg = p_black_bg
    mode = p_mode
    start = time.time()

    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith(".nii"):
                src = os.path.join(os.path.abspath(root), os.path.basename(file))
                img = image.smooth_img(src, fwhm=fwhm)
                _process(img, root, file)
                del img
                del src
                gc.collect()
                # p = Process(target=_process, args=(img, root, file,))
                # processes.append(p)
                # p.start()


def _process(img, root, file):
    p_start = time.time()

    for x in x_cuts:
        _save(img, root, file, x, "x")
    for y in y_cuts:
        _save(img, root, file, y, "y")
    for z in z_cuts:
        _save(img, root, file, z, "z")

    p_finish = time.time()
    src = os.path.join(os.path.abspath(root), os.path.basename(file))
    print("[took: p=%d s g=%d s]\t\tFinished:\t %s" % (p_finish - p_start, p_finish - start, src))


def _save(img, root, file, coord, display_mode):
    display = None

    if mode.lower() == "anat":
        display = plotting.plot_anat(img, cut_coords=[coord], display_mode=display_mode, annotate=annotate,
                                     black_bg=black_bg)
    elif mode.lower() == "epi":
        display = plotting.plot_epi(img, cut_coords=[coord], display_mode=display_mode, annotate=annotate,
                                    black_bg=black_bg)

    (dst_dir, dst_full_path) = _create_names(root, file, coord, display_mode)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    display.savefig(dst_full_path, dpi=dpi)
    display.close()

    del display


def _create_names(root, file, coord, display_mode):
    file_base = os.path.splitext(os.path.basename(file))[0]
    dst_new_dir = file_base + "." + mode
    dst_file_name = mode + "_" + display_mode + "_" + str(int(coord)) + "_" + file_base + "." + ext

    dst_dir = os.path.abspath(root) + os.sep + dst_new_dir
    dst_full_path = dst_dir + os.sep + dst_file_name

    return (dst_dir, dst_full_path)


def move(src=".", dst=".", mode="anat"):

    pattern = re.compile("^OAS2_[0-9]{4}_MR[0-9]*$")

    for root, dirs, files in os.walk(src):

        path = root.split(os.sep)

        id_dir = "some_dir"

        for directory in path:

            if re.match(pattern, directory):
                id_dir = directory

            if directory.find(mode) != -1:

                dst_dir = dst + os.sep + id_dir + os.sep

                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                copytree(src=root, dst=dst_dir)

                print("Copied:\nFrom:\t" + root + "\nTo:\t\t" + dst_dir + "\n\n")
                break


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


if __name__ == '__main__':
    # nii2image(src_path="C:\\Users\\Ognjen\\Downloads\\OAS2_RAW_PART2\\OAS2_RAW_PART2", p_fwhm=2)
    move(src="C:\\Users\\Ognjen\\Downloads\\OAS2_RAW_PART2\\OAS2_RAW_PART2",
         dst="D:\\Projects\\SOFT\\Dementia\\data\\epi_part_2",
         mode="anat")
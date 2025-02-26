import glob
import os
import csv

import numpy as np
import pydicom

# import matplotlib.pyplot as plt
# import SimpleITK as sitk


######################################################

age_path = "/home/omnisky/home/skull_age.npy"
age_ary = np.load(age_path)
sex_ary = np.load(label_path)
index = 875
sex_ary = np.delete(sex_ary, index)  # new_ary = np.delete(ary, index) 重新赋值才有
age_ary = np.delete(age_ary, index)
print(len(age_ary))
print(len(sex_ary))
print("#####################")
print(age_ary[870:880])  # 一致
print(min(age_ary))
print(max(age_ary))




# print(len(sex_ary), len(age_ary))
# print(sex_ary[875])     # delete
# # print(len(age_ary))
# print(age_ary[875])     # delete

#######################################################


# img_path = "/home/omnisky/home/skullct"
# img_list = sorted(np.array(os.listdir(img_path), dtype=np.int32))
# f = open("imgpath.txt", "w")
# csv_writer = csv.writer(f)
# for item in img_list:
#     img3d_path = os.path.join(img_path, f"{item}")
#     csv_writer.writerow([img3d_path])
#     # print(img3d_path)
# print("sucess")


with open("imgpath.txt", "r") as f:
    allimg = f.readlines()
#
# print(len(allimg))
image3d_path = []
for item in allimg:
    item = item.split("\n")[0]
    # print(item)
    image3d_path.append(item)
    # print(item, "\n")

# print(len(image3d_path))

# print(image3d_path[1])
# print(len(glob.glob(os.path.join(image3d_path[1], "ST0/SE0/IM*"))))  # 356
# for i in range(350):
#     print(os.path.join(image3d_path[1], "ST0/SE0/IM") + f"{i}")
#     dcm = pydicom.read_file(os.path.join(image3d_path[1], "ST0/SE0/IM") + f"{i}").pixel_array
#     plt.imshow(dcm)
#     plt.show()

# dcm = pydicom.dcmread(os.path.join("/home/omnisky/home/skullct/1", "ST0/SE0/IM") + "1")
# intercept = dcm.RescaleIntercept
# slope = dcm.RescaleSlope


def get_pixel(image):
    per = []
    per_dict = {}
    for i in range(len(glob.glob(os.path.join(image, "ST0/SE0/IM*")))):
        dcm = pydicom.dcmread(os.path.join(image, "ST0/SE0/IM") + f"{i}")
        slope = dcm.RescaleSlope
        intercept = dcm.RescaleIntercept

        # get HU value: dcm * slope + intercept
        per_dict[int(dcm.InstanceNumber)] = dcm.pixel_array * slope + intercept    # ct: -1000 - 3700

    for i in range(1, len(glob.glob(os.path.join(image, "ST0/SE0/IM*")))+1):
        pixel_ary = per_dict[i][50:470, 50:470]  # 单层420：420
        pixel_ary = np.copy(pixel_ary)
        pixel_ary_contig = np.ascontiguousarray(pixel_ary)

        # normlization and 0-mean
        max_value = np.max(pixel_ary_contig)
        min_value = np.min(pixel_ary_contig)
        pixel_ary_contig = (pixel_ary_contig - min_value) / (max_value - min_value)

        mean_value = np.mean(pixel_ary_contig)
        # print(mean_value)
        pixel_ary_contig -= mean_value

        pixel_ary_contig = np.resize(pixel_ary_contig, (224, 224))
        pixel_ary_contig = np.array(pixel_ary_contig, dtype=np.float32)


        per.append(pixel_ary_contig)

    person_final = np.stack(per, axis=0)[5:229, :, :]
    # part_up = person[:185, :, :]
    # part_down = person[260:299, :, :]
    # person_final = np.concatenate((part_up, part_down), axis=0)  # 按深度叠加
    return person_final  # (224, 224, 224)





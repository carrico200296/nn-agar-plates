import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from skimage.io import imread
from skimage.transform import resize


def get_image_files(path):
    images = []
    ext = {'.jpg', '.png'}
    for root, dirs, files in os.walk(path):
        print('loading image files ' + root)
        for file in files:
            if os.path.splitext(file)[1] in ext:
                images.append(os.path.join(root, file))
    return sorted(images)



if __name__ == "__main__":
    #path = "C:/Users/DMD_SVM/Documents/DMD_projects/NN_Vision_Cloud/AgarPlates_project/DFR-dev/dataset"
    dataset_path = os.getcwd() + "/dataset/bottle"

    # training set images
    trainset_path = dataset_path + "/train"
    trainset_images = get_image_files(path=trainset_path)
    # testing set images
    testset_path = dataset_path + "/test"
    testset_images = get_image_files(path=testset_path)

    print(testset_images)

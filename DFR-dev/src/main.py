import argparse
from anoseg_dfr import AnoSegDFR
import os
import torch
import numpy as np

def config():
    parser = argparse.ArgumentParser(description="Settings of DFR")

    # DEFAULT VALUES

    # feature extractor
    #cnn_layers = ('relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')
    cnn_layers = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
                    'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
                    'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')

    # positional args
    parser.add_argument('--mode', type=str, choices=["train", "test", "inference"],
                        default="train", help="train, test or unic inference")

    # general
    parser.add_argument('--data_name', type=str, default="data_name", help="dataset name (type of object)")
    parser.add_argument('--model_name', type=str, default="", help="specifed model name used to save the model in the save_path")
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help="model and results saving path")
    parser.add_argument('--img_size', type=int, nargs="+", default=(256, 256), help="image size (hxw)")
    parser.add_argument('--device', type=str, default="cuda:0", help="device for training and testing")

    # parameters for the regional feature generator
    parser.add_argument('--backbone', type=str, default="vgg19", help="backbone net")
    parser.add_argument('--cnn_layers', type=str, nargs="+", default=cnn_layers, help="cnn feature layers to use")
    parser.add_argument('--upsample', type=str, default="bilinear", help="operation for resizing cnn map")
    parser.add_argument('--is_agg', type=bool, default=True, help="if to aggregate the features")
    parser.add_argument('--kernel_size', type=int, nargs="+", default=(4, 4), help="aggregation kernel (hxw)")
    parser.add_argument('--stride', type=int, nargs="+", default=(4, 4), help="stride of the kernel (hxw)")
    parser.add_argument('--dilation', type=int, default=1, help="dilation of the kernel") 
    parser.add_argument('--featmap_size', type=int, nargs="+", default=(256, 256), help="feat map size (hxw)")

    # Parameters for training, testing and inference models.
    parser.add_argument('--train_data_path', type=str, default=os.getcwd(), help="training data path")
    parser.add_argument('--test_data_path', type=str, default=os.getcwd(), help="testing data path")
    parser.add_argument('--test_defect', type=str, default="test_inference_images", help="type of defect to be detected during the testing images")
    parser.add_argument('--inference_image', type=str, default=os.getcwd(), help="input image - inference image")

    # CAE (Convolutional Auto-Encoder)
    parser.add_argument('--latent_dim', type=int, default=None, help="latent dimension of CAE")
    parser.add_argument('--is_bn', type=bool, default=True, help="if using bn layer in CAE")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=700, help="epochs for training")    # default 700, for wine 150

    # segmentation evaluation
    parser.add_argument('--thred', type=float, default=0.5, help="threshold for segmentation")
    parser.add_argument('--except_fpr', type=float, default=0.005, help="fpr to estimate segmentation threshold")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    #torch.cuda.empty_cache()
    cfg = config()
    #IMPORTANT: the model has to be loaded with the same bath_size used during the training (example: --batch_size 2)

    # dataset
    # training and testing path - default values
    cfg.train_data_path = os.getcwd() + "/dataset/" + cfg.data_name + "/train"
    cfg.test_data_path = os.getcwd() + "/dataset/" + cfg.data_name + "/test"

    # path where the models will be stored.
    cfg.save_path = os.getcwd() + "/DFR_trained_models"
    inference_image_path = os.getcwd() + cfg.inference_image[1:]
    
    # train or evaluation
    dfr = AnoSegDFR(cfg)
    if cfg.mode == "train":
        dfr.train()
    elif cfg.mode == "test":
        dfr.evaluation_test_images()
    else:
        dfr.evaluation_one_test_image(inference_image_path)
# Hyperparameter and training settings for all the datasets

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', nargs=1, type=str, default=['mnist'])
parser.add_argument('-mode', nargs=1, type=str, default=['test'])
parser.add_argument('-model_file', nargs=1, type=str, default=['saved_models/MNIST/MNIST.pth'])
parser.add_argument('-expl', nargs=1, type=bool, default=[False])
args = parser.parse_args()
data_name = args.data[0]
mode = args.mode[0]
model_file = args.model_file[0]
expl = args.expl[0]


data_path = 'Data/'

coefs = {
        'crs_ent': 1,
        'recon': 1,
        'kl': 1,
        'ortho': 1,
    }


if (data_name == "mnist"):
    img_size = 28
    latent = 256
    num_prototypes = 50
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10



elif (data_name == "fmnist"):
    img_size = 28
    latent = 256
    num_prototypes = 100
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10



if (data_name == "cifar10"):
    img_size = 32
    latent = 512
    num_prototypes = 100
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 36



if (data_name == "svhn"):
    img_size = 32
    latent = 512
    num_prototypes = 50
    num_classes = 10
    batch_size = 64
    lr = 1e-3
    num_train_epochs = 36



if (data_name == "quickdraw"):
    img_size = 28
    latent = 512
    num_prototypes = 100
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10
    data_path = data_path + 'quickdraw/'



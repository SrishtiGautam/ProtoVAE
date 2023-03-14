import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from helpers import makedir
import model
import train_and_test as tnt
import save
import matplotlib.pyplot as plt
import numpy as np
import dataloader_qd as dl
from settings import *
from matplotlib.pyplot import show

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

model_dir = './saved_models/' + data_name + '/'
makedir(model_dir)
prototype_dir = model_dir + 'prototypes/'
makedir(prototype_dir)


# all datasets
if (data_name == "cifar10"):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.CIFAR10(root=data_path, train=True,
                                download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_path, train=False,
                               download=True, transform=transform_test)

elif (data_name == "mnist"):
    mean = (0.5)
    std = (0.5)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.MNIST(root=data_path, train=True,
                              download=True, transform=transform)
    testset = datasets.MNIST(root=data_path, train=False,
                             download=True, transform=transform)

elif (data_name == "fmnist"):
    mean = (0.5)
    std = (0.5)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.FashionMNIST(root=data_path, train=True,
                              download=True, transform=transform)
    testset = datasets.FashionMNIST(root=data_path, train=False,
                             download=True, transform=transform)

elif (data_name == "svhn"):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = datasets.SVHN(root=data_path, split="train",
                             download=True, transform=transform)
    testset = datasets.SVHN(root=data_path, split="test",
                            download=True, transform=transform)


elif (data_name == "quickdraw"):
    trainset = dl.QuickDraw(ncat=num_classes, mode='train', root_dir=data_path)
    testset = dl.QuickDraw(ncat=num_classes, mode='test', root_dir=data_path)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

test_loader_expl = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=num_workers)

print('data : ',data_name)
print('training set size: {0}'.format(len(train_loader.dataset)))
print('test set size: {0}'.format(len(test_loader.dataset)))

jet = False
if(data_name == "mnist" or data_name=="fmnist" or data_name=="quickdraw"):
    jet = True

# construct the model
protovae = model.ProtoVAE().to(device)

## Training
if(mode=="train"):
    print('start training')
    optimizer_specs = \
            [{'params': protovae.features.parameters(), 'lr': lr},
             {'params': protovae.prototype_vectors, 'lr':lr},
             {'params': protovae.decoder_layers.parameters(), 'lr':lr},
             {'params': protovae.last_layer.parameters(), 'lr':lr}
             ]
    optimizer = torch.optim.Adam(optimizer_specs)

    for epoch in range(num_train_epochs):
        print('epoch: \t{0}'.format(epoch))
        train_acc, train_ce, train_recon, train_kl, train_ortho = tnt.train(model=protovae, dataloader=train_loader,
                                                               optimizer=optimizer)

        test_acc, test_ce, test_recon, test_kl, test_ortho = tnt.test(model=protovae, dataloader=test_loader)


    print("saving..")
    save.save_model_w_condition(model=protovae, model_dir=model_dir, model_name=str(epoch), accu=test_acc,
                                target_accu=0)

    ## Save and plot learned prototypes
    protovae.eval()
    prototype_images = protovae.get_prototype_images()
    prototype_images = (prototype_images + 1) / 2.0
    num_prototypes = len(prototype_images)
    num_p_per_class = protovae.num_prototypes_per_class

    plt.figure("Prototypes")
    for j in range(num_prototypes):
        p_img_j = prototype_images[j, :, :, :].detach().cpu().numpy()
        if(jet!=True):
            p_img_j = np.transpose(p_img_j, (1, 2, 0))
        else:
            p_img_j = np.squeeze(p_img_j)

        if(jet!=True):
            plt.imsave(os.path.join(prototype_dir, 'prototype' + str(j) + '.png'), p_img_j, vmin=0.0, vmax=1.0)
        else:
            plt.imsave(os.path.join(prototype_dir, 'prototype' + str(j) + '.png'), p_img_j, cmap="jet",vmin=0.0, vmax=1.0)

        plt.subplot(num_classes, num_p_per_class, j + 1)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.imshow(p_img_j)

    plt.show()
    print("Prototypes stored in: ", prototype_dir)

else:
    ## Testing
    protovae.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)
    protovae.eval()
    test_acc, test_ce, test_recon, test_kl, test_ortho = tnt.test(model=protovae, dataloader=test_loader)


    ## Save and plot learned prototypes
    prototype_images = protovae.get_prototype_images()
    prototype_images = (prototype_images + 1) / 2.0
    num_prototypes = len(prototype_images)
    num_p_per_class = protovae.num_prototypes_per_class

    plt.figure("Prototypes")
    for j in range(num_prototypes):
        p_img_j = prototype_images[j, :, :, :].detach().cpu().numpy()
        if (jet != True):
            p_img_j = np.transpose(p_img_j, (1, 2, 0))
        else:
            p_img_j = np.squeeze(p_img_j)

        plt.subplot(num_classes, num_p_per_class, j + 1)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.imshow(p_img_j)
    show()

    if(expl):
        ## Generate LRP based location explanation maps
        print("\n")
        print("Generating explanations")
        from prp import *

        prp_path = model_dir + 'prp/'
        prp_train_path = prp_path + 'train/'
        prp_test_path = prp_path + 'test/'
        orig_test_path = prp_path + 'test-orig/'


        wrapper = model_canonized()
        # construct the model for generating LRP based explanations
        model_wrapped = model.ProtoVAE().to(device)
        wrapper.copyfrommodel(model_wrapped.features, protovae.features, lrp_params=lrp_params_def1,
                              lrp_layer2method=lrp_layer2method)

        generate_explanations(test_loader_expl, model_wrapped.features,protovae.prototype_vectors, protovae.num_prototypes, prp_path, orig_test_path,protovae.epsilon)





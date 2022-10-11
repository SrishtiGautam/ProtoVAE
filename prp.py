from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from settings import *
from lrp_general6 import *
from helpers import makedir
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0



class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



class  Modulenotfounderror(Exception):
  pass

class model_canonized():

    def __init__(self):
        super(model_canonized, self).__init__()
    # runs in your current module to find the object layer3.1.conv2, and replaces it by the obkect stored in value (see         success=iteratset(self,components,value) as initializer, can be modified to run in another class when replacing that self)
    def setbyname(self, model, name, value):

        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                # print('found!!', components[0])
                # exit()
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split('.')
        success = iteratset(model, components, value)
        return success

    def copyfrommodel(self, model, net, lrp_params, lrp_layer2method):
        # assert( isinstance(net,ResNet))

        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation dependent

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():

            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                # m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                # store conv2d layers
                foundsth = True
                last_src_module_name = src_module_name
                last_src_module = src_module
            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # conv-bn chain
                foundsth = True

                if (True == lrp_params['use_zbeta']) and (last_src_module_name == '0'):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(m, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta)


                if False == self.setbyname(model,last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + last_src_module_name + " in target net to copy")

                updated_layers_names.append(last_src_module_name)

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)

                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)


        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in model.named_modules():

            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + target_module_name + " in target net , impossible!")
                updated_layers_names.append(target_module_name)



### Save heatmaps overlayed on original images
def imshow_im(hm,imgtensor,q=100,folder="folder", folder_orig="orig", name="name"):

    def invert_normalize(ten, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
      # print(ten.shape)
      s=torch.tensor(np.asarray(std,dtype=np.float32)).unsqueeze(1).unsqueeze(2)
      m=torch.tensor(np.asarray(mean,dtype=np.float32)).unsqueeze(1).unsqueeze(2)

      res=ten*s+m
      return res

    def showimgfromtensor(inpdata):

      ts=invert_normalize(inpdata)
      a=ts.data.squeeze(0).numpy()
      saveimg=(a*255.0).astype(np.uint8)

    hm = hm.squeeze().detach().numpy()
    clim = np.percentile(np.abs(hm), q)
    hm = hm / clim


    makedir(folder+"/")
    plt.imsave(folder + name, hm, cmap="seismic", vmin=-1, vmax=+1)

    ### OVERLAY FINAL
    heatmap = np.array(Image.open(folder+name).convert('RGB'))
    heatmap = np.float32(heatmap) / 255
    ts = invert_normalize(imgtensor.squeeze())
    a = ts.data.numpy().transpose((1, 2, 0))
    makedir(folder_orig + "/")
    plt.imsave(folder_orig + name,
               a,
               vmin=0,
               vmax=+1.0)
    overlayed_original_img_j = 0.2 * a + 0.6 * heatmap
    plt.imsave(folder+name,
               overlayed_original_img_j,
               vmin=-1,
               vmax=+1.0)


## Generating protoypical explanations for each prototypes for 100 test images.
def generate_explanations(test_loader,model,prototypes,n_prototypes,write_path, write_path_orig,epsilon):
    model.eval()

    def x_prp(test_loader,write_path, write_path_orig,epsilon):
        im = 0
        for data in test_loader:
        # for data in itertools.islice(test_loader, stop=100):
            # get the inputs
            inputs = data[0]
            labels = data[1]
            # d = torch.cdist(zx_mean, model.module.prototype_vectors, p=2)


            inputs = inputs.to(device)
            inputs.requires_grad = True

            with torch.enable_grad():
                zx_mean = model(inputs)
                zx_mean = zx_mean[:, :latent]
                p_vector = prototypes[pno,:]
                d = (zx_mean-p_vector)**2
                R_zx = 1/(d+epsilon)
                R_zx.backward(torch.ones_like(R_zx))
                rel = inputs.grad.data

                # print(write_path+'/prototype'+str(pno)+'/'+str(labels.item())+"-"+str(im)+"-PRP.png")
                imshow_im(rel.to('cpu'), imgtensor=inputs.to('cpu'), folder=write_path+'/prototype'+str(pno)+'/', folder_orig = write_path_orig,name=str(labels.item())+"-"+str(im)+"-PRP.png")
                im += 1
                if(im==100):
                    return


    for pno in range(n_prototypes):
        print("Protoype: ", pno)
        print("Saving LRP maps for 100 test images in ", write_path+'/prototype'+str(pno)+'/...')
        x_prp(test_loader,write_path, write_path_orig,epsilon)


### LRP parameters
lrp_params_def1={
    'conv2d_ignorebias': True,
    'eltwise_eps': 1e-6,
    'linear_eps': 1e-6,
    'pooling_eps': 1e-6,
    'use_zbeta': True ,
  }

lrp_layer2method={
'nn.ReLU':          relu_wrapper_fct,
'nn.Sigmoid':          sigmoid_wrapper_fct,
'nn.BatchNorm2d':   relu_wrapper_fct,
'nn.Conv2d':        conv2d_beta0_wrapper_fct,
'nn.Linear':        linearlayer_eps_wrapper_fct,
'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
'nn.MaxPool2d': maxpool2d_wrapper_fct,
'nn.AvgPool2d': avgpool2d_wrapper_fct
}







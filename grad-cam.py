import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import pickle

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
#         output = output.view(output.size(0), -1)
#         output = self.model.fc3(output) # changed from model.classifier 
        return target_activations, output


# def preprocess_image(img):
#     means = [0.485, 0.456, 0.406]
#     stds = [0.229, 0.224, 0.225]

#     preprocessed_img = img.copy()[:, :, ::-1]
#     for i in range(3):
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
#     preprocessed_img = \
#         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
#     preprocessed_img = torch.from_numpy(preprocessed_img)
#     preprocessed_img.unsqueeze_(0)
#     input = preprocessed_img.requires_grad_(True)
#     #______________________________
    
# #      img = self.X[idx].reshape(1, self.img_height, self.img_width)/255.
        
# #         # Crop images into squares for cnn
# #         if self.img_resize:
# #             # Define box for resize
# #             lower = int(self.img_height/2 - self.img_resize/2)
# #             upper = int(self.img_height/2 + self.img_resize/2)
# #             left = int(self.img_width/2 - self.img_resize/2)
# #             right = int(self.img_width/2 + self.img_resize/2)
            
# #             # crop the image
# #             img = img[:, lower:upper, left:right]
        
#     return input


def show_cam_on_image(img, mask, pred_index):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(f"{pred_index}_cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

#         self.model.features.zero_grad()
#         self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (136, 136)) # changed from (224, 224)
        cam = cam - np.min(cam) 
        cam = cam / np.max(cam)
        return cam, index


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--model', type=str)
    parser.add_argument('--label', type=str)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

# Define a flatten class to be picked up by the 
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
       
        # Define the Parameters for the neural network
        # see if neutron can help visualize my network
        # Look at fast ai tutorials for CNN
        
        # Convolution layer 1
        self.conv1_input_channels = 1
        self.conv1_kernel_size = 9
        self.conv1_stride = 1
        self.conv1_output_channels = 16
        self.conv1_output_dim = output_size(img_resize, 
                                            self.conv1_kernel_size, 
                                            self.conv1_stride)
        
        # Pooling layer 1
        self.pool1_kernel_size = 11
        self.pool1_stride = 2
        self.pool1_output_dim = output_size(self.conv1_output_dim, 
                                            self.pool1_kernel_size, 
                                            self.pool1_stride)
       
        #conv 2
        self.conv2_input_channels = self.conv1_output_channels
        self.conv2_kernel_size = 8
        self.conv2_stride = 1
        self.conv2_output_channels = 32
        self.conv2_output_dim = output_size(self.pool1_output_dim,
                                            self.conv2_kernel_size, 
                                            self.conv2_stride)
        
        # Pooling layer 2
        self.pool2_kernel_size = 8
        self.pool2_stride = 2
        self.pool2_output_dim = output_size(self.conv2_output_dim,
                                           self.pool2_kernel_size,
                                           self.pool2_stride)
        
        # Fully connected 1 (input is batch_size x height x width after pooling)
        self.fc1_input_features = self.conv2_output_channels * self.pool2_output_dim**2
        self.fc1_output_features = 256
       
        # Fully connected 2
        self.fc2_input_features = self.fc1_output_features
        self.fc2_output_features = 200
           
        # Fully Connected 3 (output is number of features)
        self.fc3_input_features = self.fc2_output_features
        self.fc3_output_features = 168
        
        # Create the layers
        self.conv1 = nn.Conv2d(self.conv1_input_channels, 
                               self.conv1_output_channels, 
                               self.conv1_kernel_size, 
                               stride=self.conv1_stride)
        
        self.max_pool1 = nn.MaxPool2d(self.pool1_kernel_size, self.pool1_stride)
        
        self.conv2 = nn.Conv2d(self.conv2_input_channels, 
                               self.conv2_output_channels, 
                               self.conv2_kernel_size,
                               stride=self.conv2_stride)
        
        self.max_pool2 = nn.MaxPool2d(self.pool2_kernel_size, self.pool2_stride)
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(self.fc1_input_features, self.fc1_output_features)
        
        self.fc2 = nn.Linear(self.fc2_input_features, self.fc2_output_features)
        
        self.fc3 = nn.Linear(self.fc3_input_features, self.fc3_output_features)
        
        self.features = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

      

    def forward(self, x):
        # run the tensor through the layers
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # number of flat features to determine the size of the first fully connected layer
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    # Define a features attribute

def copy_bw_to_rgb(input_image):
    # Converts torch image to RGB from BW and copies to cpu
    
    # Create a cpu copy of the input called img for plotting
    from copy import deepcopy
    img = deepcopy(input_image).cpu().numpy()
    
    # Reshape
    img = np.squeeze(img)
    
    # copy to RGB
    img = np.stack((img,)*3, axis=-1)
    
    return img
    
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    
    # Import model
    model = torch.load(args.model)
    
    
    # Import label
    with open(args.label, 'rb') as f:
        label = np.asscalar(pickle.load(f).cpu().numpy())
    
    
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model=model, \
                       target_layer_names=['conv1', 'conv2'], use_cuda=args.use_cuda)

#     img = cv2.imread(args.image_path, 1)
#     img = np.float32(cv2.resize(img, (224, 224))) / 255
#     input = preprocess_image(img)
## Use my own input image

    with open(args.image_path, 'rb') as f:
        input = torch.Tensor(pickle.load(f)).to(torch.device('cuda:0'))
    # prepare input
    input = input.reshape((1,1,136,136))
    
    # post process input to image
    img = copy_bw_to_rgb(input)
    
    # Turn on gradient tracking for input
    input = input.requires_grad_(True)
   
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = label
    mask, pred_index = grad_cam(input, target_index)

    show_cam_on_image(img, mask, pred_index)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{pred_index}_gb.jpg', gb)
    cv2.imwrite(f'{pred_index}_cam_gb.jpg', cam_gb)
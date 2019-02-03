import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import json


#from train.py import load_checkpoint, Network

parser = argparse.ArgumentParser(description = 'predict flower name from an image')
parser.add_argument('path', help = 'path to image')
parser.add_argument('checkpoint', help = 'checkpoint for the model')
parser.add_argument('--top_k', type = int, help = 'the number of classes with top probabilities you wish to see', default = 1)
parser.add_argument('--category_names', help = 'category names for the classes')
parser.add_argument('--gpu', action = 'store_true', help = 'GPU')
args = parser.parse_args()

image_path = args.path
checkpoint = args.checkpoint
# optional inputs
top_k = args.top_k
category_names = args.category_names  
gpu = args.gpu



import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, drop_p = 0.3):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size[0])])
        layer_sizes = zip(hidden_size[:-1], hidden_size[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_size[-1], output_size)
        self.dropout = nn.Dropout(p = drop_p)
        
    def forward(self, x):
        
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))
            
        x = F.log_softmax(self.output(x), dim = 1)
        
        return x

    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model_name = checkpoint['model_name']
    if model_name == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        model = models.densenet121(pretrained = True)
        
    classifier = Network(checkpoint['input_size'], 
                        checkpoint['output_size'],
                        checkpoint['hidden_layers'])
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

###################################### preprocessing the image #################################################
# fonksiyon icindeki image aslinda image_path
def preprocess_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    
    im = Image.open(image)
    # getting sizes of image
    width, height = im.size
    
    # setting short edge to 256 pixels
    aspect_ratio = width/height
    new_width, new_height = int(max(aspect_ratio, 1) * 256), int(max(1/aspect_ratio, 1) * 256)
    size = new_width, new_height
    new_im = im.resize(size)
    
    # cropping out center portion of 224x224
    width, height = new_im.size
    
    left = (width - 224)/2
    right = (width + 224)/2
    upper = (height - 224)/2
    lower = (height + 224)/2
    
    im_cropped = new_im.crop((left, upper, right, lower))
    
    np_image = np.array(im_cropped)
    np_image = np_image/255
    
    # normalization numpy image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(np_image.shape[0]):
        for j in range(np_image.shape[1]):
            np_image[i, j] = (np_image[i, j] - mean) / std
    
    # transposing color channels
    image = np_image.transpose(2, 0, 1)
    
    return image

#################################### visualize ##########################################

def visualize(image, probs, classes, names = None):
    fig, ax = plt.subplots(2, 1, figsize = (5, 8)) 

    image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax[0].imshow(image)
    ax[0].axis('off')
    ax[1].barh(np.arange(0, len(classes)), probs)

    ax[1].set_yticks(np.arange(0, len(classes)))
    if names:
        ax[1].set_yticklabels(names)
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

##################################### predicting image ##################################################    
def predict(image_path, model_checkpoint, topk = 1, gpu = False, category_names = None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image_numpy = preprocess_image(image_path)
    image = torch.from_numpy(image_numpy)
    image.unsqueeze_(0)
    image = image.float()
    model = load_checkpoint(model_checkpoint)
    
    if gpu:
        model, image = model.to('cuda'), image.to('cuda')
        
    with torch.no_grad():
        model.eval()
    
        log_ps = model(image)
        ps = torch.exp(log_ps)
        probs, indices = ps.topk(k = topk, dim = 1) 
    probs = probs.cpu()
    indices = indices.cpu()
    probs = probs.numpy()[0]
    indices = indices.numpy()[0]
    
    index_to_class = dict()
    
    for key in model.class_to_idx.keys():
        index_to_class[model.class_to_idx[key]] = key
            
    classes = list()
    for index in indices:
        classes.append(index_to_class[index])
        
        
    #print('The most likely image class(es): {}'.format(classes),
          #'and its probabilities: {}'.format(probs))
    print('\nThe most likely image class(es) and its probabilities: \n{}'.format(list(zip(classes, probs))))
    
    if category_names:
        with open(category_names, 'r') as file:
            cat_names = json.load(file)
        names = list()
        for i in classes:
            names.append(cat_names[i])
        print('\nFlower names:\n')
        for name in names:
            print(name)
        print('\n')
            
        #visualize(image_numpy, probs, classes, names)
    #else:
        #visualize(image_numpy, probs, classes, names = None)
           
        
    return probs, classes
    # TODO: Implement the code to predict the class from an image file
    
 
    
if __name__ == '__main__':
       
            
    probs, classes = predict(image_path, checkpoint, top_k, gpu, category_names)

          
      
          
    
            
        
    
##
import argparse
from collections import OrderedDict
import torch
from torch import nn
from torchvision import models, transforms, datasets



parser = argparse.ArgumentParser(description = 'Getting some inputs to train the model')
parser.add_argument('data_directory', help = 'directory of data to train the model')
parser.add_argument('--save_dir', help = 'directory to save the checkpoint', default = 'checkpoint.pth')
parser.add_argument('--arch', help = 'architecture of the model', \
                    choices = ['densenet121', 'vgg13'], default = 'densenet121')
parser.add_argument('--learning_rate', type = float, help = 'learning rate', default = 0.003)
parser.add_argument('--hidden_units', nargs = '+', type = int, \
                    help = 'number of perceptrons in each layer', default = [800, 300])
parser.add_argument('--epochs', type = int, help = 'number of epochs in training', default = 3)
group = parser.add_mutually_exclusive_group()
#group.add_argument('--cpu', action = 'store_true', help = 'CPU')
group.add_argument('--gpu', action = 'store_true', help = 'GPU')

args = parser.parse_args()

data_dir = args.data_directory
# optional arguments from argparse
model_name = args.arch
hidden_number = args.hidden_units
learning_rate = args.learning_rate
epochs = args.epochs
gpu = args.gpu
save_dir = args.save_dir

#data directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



######################## transform and load data ###########################################
train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     #transforms.RandomRotation(40),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
trainset = datasets.ImageFolder(train_dir, transform = train_transforms)

validset = datasets.ImageFolder(valid_dir, transform = valid_transforms)

testset = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True) 
validloader = torch.utils.data.DataLoader(validset, batch_size =32, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle = True)



######################################################################################################
####################################### CLASSIFIER ###################################################
######################################################################################################
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

#########################################################################################################
########################################## MODEL ########################################################
#########################################################################################################
    
def create_model(model_arch = 'densenet121', hidden_units = [800, 300]):
    
    if model_arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        input_size = 25088
    else:
        model = models.densenet121(pretrained = True)
        input_size = 1024
        
    output_size = 102
    
    if hidden_units == None:
        hidden_units = [800, 300]
    
    for param in model.parameters():
        param.requires_grad = False
        
    p = 0.3   
    classifier = Network(input_size, output_size, hidden_units, p)
    model.classifier = classifier
    model.class_to_idx = trainset.class_to_idx
    
    print("Model is created...")
    
    return model

####################################### TRAINING MODEL ################################################
def do_validation(model, validloader, criterion, gpu):
    
    validation_loss = 0
    total = 0
    if gpu:
        model = model.to('cuda')
    with torch.no_grad():
        model.eval()
        for images, labels in validloader:
            
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
                
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            validation_loss += loss.item()
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(k = 1, dim = 1)
            equality = top_class == labels.view(*top_class.shape)
            total += torch.sum(equality).item()
        accuracy = (total / (len(validloader) * 32)) * 100
        
        #print('Validation Accuracy: {}'.format(accuracy * 100),
             #'Validation Loss: {}'.format(validation_loss/len(validloader)))
        
    return accuracy, validation_loss/len(validloader)


def train_network(model, trainloader, validloader, lrate = 0.003, epochs = 3, gpu = False):
    
    model = model
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr = lrate)
    
    print_every = 30
    steps = 0
    if gpu:
        model = model.to('cuda')
        
    
        
    for e in range(epochs):
        running_loss = 0
        
        for images, labels in trainloader:
            steps +=1
            
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                acc, valid_loss = do_validation(model, validloader, criterion, gpu)
                print('Epoch {}/{}...'.format(e+1, epochs),
                      'Training Loss: {:.5f}'.format(running_loss / print_every),
                     'Validation Accuracy: {:.2f}%'.format(acc),
                     'Validation Loss: {:.5f}'.format(valid_loss))
                running_loss = 0
        
        #else:
            #do_validation(model, validloader, criterion, gpu)
            
            
    return model
            
######################## testing the model ##############################################
def test_model(model, testloader, gpu = False):
    
    model = model
    if gpu:
        model = model.to('cuda')
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            output = model(images)
            top_p, top_class = output.topk(1, dim = 1)
        
            equality = (top_class == labels.view(*top_class.shape))
            #print(torch.sum(equality).item())
            correct += torch.sum(equality).item()
            total += images.shape[0]
        
        accuracy = correct/total
        
        print('\n\nAccuracy on the test set is {:.2f}%'.format(100 * accuracy))

    return accuracy




##################################### Saving the Checkpoint #####################################
#################################################################################################


def save_checkpoint(model, save_dir = 'checkpoint.pth'):
    
    checkpoint = {'model_name': model_name,
                 'input_size': model.classifier.hidden_layers[0].in_features,
                 'hidden_layers':[layer.out_features for layer in model.classifier.hidden_layers],
                 'output_size': model.classifier.output.out_features,
                 'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx }

    
    torch.save(checkpoint, save_dir)
    
    return save_dir

######################################## load checkpoint ###########################################
###########################################################################################################

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




####################################### MAIN #########################################################

if __name__ == '__main__':
    
    model = create_model(model_name, hidden_number)
    model = train_network(model, trainloader, validloader, learning_rate, epochs, gpu)
    accuracy = test_model(model, testloader, gpu)
    filepath_checkpoint = save_checkpoint(model, save_dir)
    print('\nSaving Directory: {}'.format(filepath_checkpoint))
    
    
    
    
                  
                
                
            
          
          
          
          
          
          
   
    
    
                                      
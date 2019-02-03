# Image Classifier with PyTorch


#### Table of Contents

1. [Installation](#Installation)
2. [Project Motivation](#ProjectMotivation)
3. [File and Data Descriptions](#FileDescription)
4. [Results](#Results)
5. [Licensing, Authors and Acknowledgements](#Licensing)

###  Installation
<a name="installation"></a>

The libraries used in the project are:

- NumPy
- pandas
- Matplotlib
- PyTorch

### Project Motivation
<a name="ProjectMotivation"></a>

The project is an image clasifier that recognizes 102 different flower species.
Classifier is fed with an image of a flower and returns the most likely n 
(inputted by user) flower species along with the corresponding possibilities.

The model uses Densenet121 or VGG13 depending on user's choice that are 
pretrained on imagenet dataset. The top classifier part is replaced with a 
classifier that takes features from pretrained model and gives 102 output 
results. 



### Data and File Description
<a name="FileDescription"></a>

Model is trained on dataset of 102 flower categories. The data can be found at
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

The files included in the folder are:

Image Classifier Project.ipynb - Jupyter Notebook training and saving the model

train.py - trains the model and saves the model as a checkpoint and outputs
training loss, validation loss, validation accuracy as the model is trained

predict.py - uses the saved model to predict the top n categories along with 
				probabilities


To train a new network on a dataset, input train file and path to the data
Example : 
		
			python train.py data_directory

Optional parameters include:

 - Directory to save a checkpoint 
	Ex : python train.py data_dir --save_dir save_directory

- Model Architecture ('densenet121' or 'vgg13')
	Ex : python train.py data_dir --arch 'densenet121'

- Hyperparameters
	Ex : python train.py data_dir --learning_rate 0.01 --hidden_units 512 256 
	 -- epochs 20 (all in one line)

- Usage of GPU for training
	Ex : python train.py data_dir --gpu


To use the trained model to predict flower categories, input predict file, path to image and checkpoint at which model is saved

Ex : 
		python predict.py path/to/image checkpoint.pth

Optional parameters include:

- The number of most likely classes
	Ex : python predict.py path/to/image checkpoint.pth --top_k 3

- Using a map of predicted classes to real names
	Ex : python predict.py path/to/image checkpoint.pth cat_to_name.json

- Usage of GPU for prediction
	Ex : python predict.py path/to/image checkpoint.pth --gpu



### Results
<a name="Results"></a>

The trained model predicts flower images with 76% accuracy. 


### Licensing, Authors and Acknowledgements
<a name="Licensing"></a>

MIT License


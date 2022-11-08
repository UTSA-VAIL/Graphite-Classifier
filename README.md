# Graphite_Classifier
This is the repository for a UNet-based deep-learning semantic segmentation classifier intended for use with the Graphite 23 nuclear materials dataset.

## Environment
A docker file is included to set up the environment.

## Preparing dataset
Download the Graphite 23 dataset. \
Then run the graphite_preprocessing.py script, passing the Path to the dataset as an arument. \
Once the script is finished, there will be a new directory in the dataset called Tiles, that contains the dataset images and labels split into smaller more manageble tiles (default 384x384). These are the images and labels that the model will train and evaluate on.

## Training
Run dist_run.sh bash script. This will perform a distributed data parallel run on multiple GPUs. Default file is set to run on 2 GPU's, you can edit the bash script to change this number. 

**--data_dir**    : Path to the directory containing your image and label tiles\
**--exp_dir**     : Path to your experiment directory. A new directory inside of this directory will be automatically created (or overwritten if it already exists), whose name will be a combination of the various settings listed below. This is where the log of your run, the model weights, and test results will be saved to. This is to help keep experiments with different settings separate from one another. \
**--epochs**     : Number of epochs to train on\
**--batch_size**  : The size of a single training batch.\
**--model**       : The backbone for your UNet model. Supported model backbones listed below\
**--num_classes** : The number of classes in your dataset\
**--image-size**  : The single dimension size of your input images. Images and their labels must be squares.\
**--seed**        : integer value to seed the seperation of the dataset into train/test/validation split. Using the same seed value ensures the split is the same across runs.\
**--distributed** : Sets the run to use multi-GPU training \
**--save-model**  : Saves the model weights to the experiment directory. Default True.\
**--mode**        : 'supervised or 'semi'. Currently only supervised learning is supported.\
**--enable_cw**   : Enables using class weights as part of the categorical cross entropy loss calculation. Disabled by default.


## Testing
Run test.sh bash script. 

**--eval**        : Sets the run in Test mode, changing the behavior of some of the above arguments. Changes are listed below.\
**--exp_dir**     : Path to the directory containing the trained model weights your test will use.\
**--num_classes** : Needs to match the number of classes the model was trained on\
**--distributed** : If the model was trained on multi-GPU, the test must also be multi-GPU and vice versa\
**--seed**        : Used to determine what images to use for the test. Ideally this should match seed you used used for the model training, so that the model is not testing on images it trained on.\
**--image-size**  : Same as above. Must match the same image size used in training.

When running a test, the test images are run through the trained model and their respective labels are used as the ground truth to determine the Intersection Over Union score for each image. Mean Intersection Over Union is computed across all test images.\
For each test image, the original image, groun truth label and predicted image is saved as a single file inside your experiment directory for human readability and result comparison.

Note: a test run is automatically done after a training run.

## Supported UNet backbones
For the --model argument the code supports the following inputs:\
'vgg11',\
'vgg19',\
'vgg19-bn',\
'resnet18',\
'resnet50',\
'resnet101',\
'resnet152',\
'deeplabv3',\
'effnet-b0', \
'effnet-b3',\
'dpn68',\
'dpn98',\
'dpn92',\
'dpn107',\
'dpn131',\
'senet154',\
'densenet121'

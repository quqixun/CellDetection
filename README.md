# Cell Detection
A course project that detect cells in image by a simple full convolution neural network. The project is driven by TensorFlow.

# Dependencies

+ python          3.5.2
+ numpy           1.11.3
+ scipy           0.18.1
+ pillow          4.1.0
+ tensorflow      1.0.0
+ matplotlib      2.0.0
+ tensorlayer     1.4.1

This demo is tested only in Ubuntu 16.04.

# Data Organization

50 full-scale images are composed of cells whose positions have been marked, from which training batch is extracted from 30 images, validation batch is extract from 10 images, and the rest 10 images are used to test. Image set is not included in this repositery, you could eamil to quqixun@gmail.com to request dataset.

### Training and Validating Data

Six groups patches are extracted from training and validating images on the basis of the different locations of patches' centers. The dimension of each patch is 35 by 35 by 3.
The groups are shown as follows with one sample patch, in each group, the patch center locates at:
+ **Group 1 - the interaction region of cells**: 
+ **Group 2 - non-goal cell**:
+ **Group 3 - nearby region of cell's edge**:
+ **Group 4 - the gap between cells**:
+ **Group 5 - background**:
+ **Group 6 - the center of cell**:



# Code Organization

+ **read_data.py**: Create TFRecords for training and validating batch to train the model. Training and validating batch is randomly selected according to the batch size.
+ **train_model.py**: In this solution, a simple end-to-end convolution nural network is implemented, being trained and updated by input training set. The model is saved into the file "model.npz".
+ **test_model.py**: Carry out a pixel-wised classification on the input test image, reserving pixels that have highest posibbility to be a cell center.

# Usage



# Result



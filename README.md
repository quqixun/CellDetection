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

50 full-scale images are composed of cells whose positions have been marked, from which training batch is extracted from 30 images, validation batch is extracted from 10 images, and the rest 10 images are used to test. Image set is not included in this repositery, you could eamil to quqixun@gmail.com to request dataset.

### Training and Validating Data

Six groups patches are extracted from training and validating images on the basis of the different locations of patches' centers. The dimension of each patch is 35 by 35 by 3.
The groups are shown as follows with one sample patch, in each group, the patch center locates at:
+ **Group 1 - the interaction region of cells**: ![alt text](https://github.com/quqixun/CellDetection/blob/master/ImageSet/Train/1/img1.png)
+ **Group 2 - non-goal cell**: ![alt text](https://github.com/quqixun/CellDetection/blob/master/ImageSet/Train/2/img1.png)
+ **Group 3 - nearby region of cell's edge**: ![alt text](https://github.com/quqixun/CellDetection/blob/master/ImageSet/Train/3/img1.png)
+ **Group 4 - the gap between cells**: ![alt text](https://github.com/quqixun/CellDetection/blob/master/ImageSet/Train/4/img1.png)
+ **Group 5 - background**: ![alt text](https://github.com/quqixun/CellDetection/blob/master/ImageSet/Train/5/img1.png)
+ **Group 6 - the center of cell**: ![alt text](https://github.com/quqixun/CellDetection/blob/master/ImageSet/Train/6/img1.png)

### Testing Data

A sample of testing image is shown below.

![alt text](https://github.com/quqixun/CellDetection/blob/master/ImageSet/Test/img_41.png)

# Code Organization

+ **read_data.py**: Create TFRecords for training and validating batch to train the model. Training and validating batch is randomly selected according to the batch size.
+ **train_model.py**: In this solution, a simple end-to-end convolution nural network is implemented, being trained and updated by input training set. The model is saved into the file "model.npz".
+ **test_model.py**: Carry out a pixel-wised classification on the input test image, reserving pixels that have highest posibbility to be a cell center.

# Usage

In terminal,

+ **Step 1**: run **python read_data.py** to create TFRecords
+ **Step 2**: run **python train_model.py** to train and save model
+ **Step 3**: run **python test_image.py** to test full-scale images

# Result

### A good case:

<img src="https://github.com/quqixun/CellDetection/blob/master/Results/test1.png" width="500">

### A bad case:

<img src="https://github.com/quqixun/CellDetection/blob/master/Results/test7.png" width="500">

Here is a bad case, in which several cells have not been detected. Increasing the number of training patches is able to solve this problem. The model is trainded by 29,818 patches generates the bad result as shown above. If the number of data is augmented by rotating and modifing HSV color space, the model is likely to perform better. The better result image is shown as below, which is detected by the model that is trained with 321,985 training patches.

<img src="https://github.com/quqixun/CellDetection/blob/master/Results/bad_2_good.png" width="500">

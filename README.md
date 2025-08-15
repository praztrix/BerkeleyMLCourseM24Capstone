### Evaluation of Image Classification Models

**Author: Prasanna Iyengar**

### Executive summary

The ability to accurately understand image content holds immense value across diverse applications. Despite the availability of numerous open-source image classification architectures, their practical performance—particularly concerning memory footprint, CPU utilization, inference latency, and classification accuracy—varies significantly. This project addresses this question by evaluating these key metrics for a selection of prominent Convolutional Neural Networks (CNNs) utilizing transfer learning with a custom dataset. The analysis will provide insights into the trade-offs between different model architectures, enabling informed decision-making for deploying  image classification solutions.


#### Evolution of Image Classification

Image classification is the task of assigning a specific label or a category to images. Before CNNs, image classification largely relied on hand-crafted features and traditional machine learning algorithms. Researchers would meticulously design algorithms to extract specific visual characteristics from images, like edges, corners, or color histograms. These extracted features were then fed into classifiers such as Support Vector Machines (SVMs) or decision trees.  This approach was highly dependent on human expertise in feature engineering, making it rigid and often struggling with variations in lighting, pose, and background. It was also incredibly time-consuming and often failed to capture the complex, hierarchical patterns present in real-world images. 

<!--![Convolutional Neural Network](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/images/CNNAI.png "Convolutional Neural Network") -->
<center>
  <center>
     <img src="https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/images/CNNAI.png" width="700" height="600" class = "center" alt="CNNAI"/>
  </center>
</center>
The breakthrough came with Convolutional Neural Networks (CNNs), inspired by the human visual cortex. Instead of relying on manual feature extraction, CNNs learn to automatically identify and extract hierarchical features directly from raw image data. This "deep learning" approach allowed models to discover intricate patterns, from simple edges in early layers to complex object parts and entire objects in deeper layers. This ability to learn powerful and robust representations, coupled with increased computational power and larger datasets, led to a dramatic leap in image classification accuracy, far surpassing traditional methods and becoming the dominant approach in the field. 

While numerous open-source image classification CNN architectures exist, their performance varies considerably in terms of memory footprint, latency, and accuracy. This project aims to evaluate these critical metrics for  the following  model architectures using **transfer learning**.

 - MobileNetV2
 - ResNet50V2
 - EfficientNetB0
 - DenseNet121

<!--![Transfer Learning](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/images/transfer_learning_cnn_1.png "Transfer Learning")-->

<center>
  <center>
     <img src="https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/images/transfer_learning_cnn_1.png" width="600" height="600" class = "center" alt="Transfer Learning"/>
  </center>

**Transfer learning** is an efficient technique used for benchmarking advanced deep learning models. It allows us to leverage pre-existing knowledge of these models and  makes the evaluation process faster and more realistic. Pre-trained models, particularly those trained on vast datasets like ImageNet have already learned highly generalizable features. For instance, the early layers of a CNN trained on ImageNet can effectively recognize basic shapes, edges, and textures in any image. By leveraging these robust features, your new model often achieves higher accuracy and better overall performance, especially when your specific dataset is small.


### Data Sources

#### CIFAR-10: Training and Testing Data
The CIFAR-10 dataset is a widely recognized benchmark in computer vision for image classification. It consists of 60,000 32x32 pixel color images categorized into 10 distinct classes. Each class contains 6,000 images, equally split into 50,000 images for the training set and 10,000 images for the test set.

The 10 classes are [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]

For this project, CIFAR-10 serves as the primary dataset for training the base model and the transfer-learned architectures (ResNet50V2, DenseNet121, MobileNetV2, EfficientNetB0) and for their in-training validation.

#### CINIC-10: Independent Performance Evaluation Data

[CINIC-10](https://www.kaggle.com/datasets/mengcius/cinic10) an extensive dataset designed as a "bridge" between CIFAR-10 and the much larger ImageNet dataset. It comprises a total of 270,000 images, which is 4.5 times larger than CIFAR-10. These images are also 32x32 pixels color images and belong to the same 10 classes as CIFAR-10. The dataset is constructed by combining all images from CIFAR-10 with a selection of downsampled images from the ImageNet database.

CINIC-10 is equally split into three subsets: training, validation, and test, each containing 90,000 images. Within each subset, there are 9,000 images per class.

For this project, **500 images of CINIC-10 test set of every class**  will be used to evaluate the final performance of the models trained on CIFAR-10. This allows us to assess how well the models generalize to a broader and potentially more diverse set of unseen images, which is critical for real-world application where data distribution might subtly shift. This setup helps to expose potential overfitting to the CIFAR-10 training data and provides a more robust measure of the models' true generalization capabilities.
 
### Benchmarking

The benchmarking of models typically iinvolves the following steps:

- Data Loading /Preparation - Create training anf test data. Training data is used to teach the model and test data is used to evaluate the model.
- Data Preprocessing - Typically involves image re-sizing and pixel scaling
- Data Augmentation - Augment data during the training phase to improve the diversity of the dataset
- Neural Network Creation - Design the Neural Network Architecture with transfer learning
- Train the Network 
- Hyper Parameter Tuning
- Model Evaluation
 
#### Key Metrics for Evaluation

1. Accuracy
2. Error/Loss
3. Inference Time Per Sample (s)
4. In-Memory and disk Model Size (MB)





### Results - Model Performance Comparison

![Model Performance](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/images/CIFAR10MultiModelPerformancePlots.png "Model Performance")

- Refer to [CIFAR-10MultiModelTransferLearningEvaluation.ipynb](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/CIFAR-10MultiModelTransferLearningEvaluation.ipynb) for performance analysis of models. 

#### Overall Analysis

- The accuracy of the base model is low and therefore is not analyzed for other metrics.
- The validation accuracy of ResNet50, EfficientNetB0, and DenseNet121 models is comparable.
- ResNet50 has the highest training accuracy.
- The CINIC10 accuracy of ResNet50, EfficientNetB0, and DenseNet121 models is comparable.
- The ResNet50 model has the lowest training loss.
- The MobileNetV2 model has the lowest inference time per sample. This is not surprising as it was developed for mobile devices with resource constraints.
- The ResNet50 model consumes about three times more memory than the next highest memory-consuming model. This is due to the depth of the ResNet50 architecture.  

#### Model Selction
- Based on accuracy and loss, ResNet50  and EfficientNetB0 models are comparable.
- Inference time per image is slightly better for ResNet50, whereas the EfficientNetB0 model is better at memory and disk consumption.
- CINIC10 performance on EfficientNetB0 was slightly better than ResNet50, particularly in terms of validation loss.
-  The EfficientNetB0 model will also work better on mobile devices due to its low memory footprint.
- **EfficientNetB0** is the final choice as it can also be used on mobile devices.


### Next steps

- Run more trials for Keras search.
- Train the best-fit model for more epochs (30+). Ten epochs were chosen due to compute considerations. Early Stopping could be used to make efficient use of comoute resources.
- Train with a larger subset of data from the CINIC10 dataset and use CIFAR10 for performance validation. This is a function of compute and memory availability.
- Fix the code to use validation_split during training. I couldn't get tensorflow.data.Dataset used for storing the training data to work with validation_split.

### Outline of project

Repository: 

[BerkeleyMLCourseM24Capstone](https://github.com/praztrix/BerkeleyMLCourseM24Capstone) 

Files:

- [capstone_utils.py](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/capstone_utils.py) - Utility functions
- [custom_cinic10_data.zip](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/custom_cinic10_data.zip) - Contains a subset of CINIC10 files for the performanve evaluation of models.
- [EDAForCIFAR-10MultiModelTransferLearningEvaluation.ipynb](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/EDAForCIFAR-10MultiModelTransferLearningEvaluation.ipynb) - Notebook for EDA
- [CIFAR-10MultiModelTransferLearningEvaluation.ipynb](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/CIFAR-10MultiModelTransferLearningEvaluation.ipynb) - Notebook for model evaluation and performance analysis.
- [Notebook Run Instructions](https://github.com/praztrix/BerkeleyMLCourseM24Capstone/blob/main/run_instructions.md) - Run instructions for Notebooks.

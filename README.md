
## Cat-vs-Dog-Classification

A Convolutional Neural Network (CNN) model for classifying images of cats and dogs. The model is built using TensorFlow Keras.
### 1. Data Preprocessing

- **Normalization**: Scale pixel values to the range [0, 1] by dividing by 255.
- **Resizing**: Resize all images to 150x150 pixels.
- **Data Augmentation**: Apply random transformations such as rotation, width/height shift, shear, zoom, and horizontal flip to increase dataset variability and improve generalization.

### 2. Model Architecture 
Convolutional Neural Network (CNN): CNNs are well-suited for image classification tasks due to their ability to capture spatial hierarchies in images.
Its typical Layers are as follows:
- **Input Layer**: Accepts the preprocessed images.
- **Convolutional Layers**: Extract features from the images using filters/kernels.
- **Activation Functions**: Apply non-linear functions like ReLU (Rectified Linear Unit) to introduce non-linearity.
- **Pooling Layers**: Reduce the dimensionality of the feature maps, retaining important information while reducing computational load.
- **Fully Connected Layers**: Perform high-level reasoning on the extracted features.
- **Output Layer**: Typically a softmax or sigmoid layer for binary classification (cat or dog).

### 3. Training
- **Loss Function**:Used binary cross-entropy loss for binary classification.
- **Optimizer**:Used Adam to minimize the loss function.
- **Training Process**: Fed the model batches of images and labels, iteratively adjusting the weights to minimize the loss.

### 4. Evaluation
After training, the model is evaluated on a separate validation dataset. Key performance metrics include accuracy, precision, recall, and F1-score.

### 5. Testing
After training, the model is evaluated on a separate test dataset to assess its generalization performance.

### 6. Deployment
Saved the trained model in a format suitable for deployment and used it to classify new, unseen images of cats and dogs in a production environment.


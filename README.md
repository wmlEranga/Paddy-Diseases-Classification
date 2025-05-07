# Paddy Disease Classification

A deep learning model for automatically classifying diseases in paddy (rice) plants using leaf images. This project helps farmers identify plant diseases early and take appropriate actions to prevent crop loss.

## Overview

Rice (Paddy) is a staple food for over half of the world's population, particularly in Asia. Diseases in paddy crops can lead to significant yield loss and economic impact. Early and accurate identification of diseases allows for timely intervention and appropriate treatment.

This project uses Convolutional Neural Networks (CNNs) to classify paddy leaf images into 10 different classes (9 diseases and 1 healthy class).

## Dataset

The model is trained on the [Paddy Doctor dataset](https://www.kaggle.com/competitions/paddy-disease-classification/data) which contains images of paddy leaves with various diseases. The dataset includes the following classes:

- bacterial_leaf_blight
- bacterial_leaf_streak
- bacterial_panicle_blight
- blast
- brown_spot
- dead_heart
- downy_mildew
- hispa
- normal
- tungro

## Model Architecture

The model uses a CNN architecture with:

- Three convolutional blocks with batch normalization and max pooling
- Global average pooling
- Dense layers with dropout for regularization
- Softmax output layer for multi-class classification

The model also includes:

- Image resizing and rescaling for preprocessing
- Data augmentation techniques (flipping and rotation) to improve generalization

## Requirements

```
tensorflow==2.8.0
numpy==1.23.5
matplotlib
h5py==3.7.0
protobuf==3.19.4
```

## Usage

1. **Setup Environment**:

   ```
   pip install tensorflow==2.8.0 numpy==1.23.5 matplotlib h5py==3.7.0 protobuf==3.19.4
   ```

2. **Train the Model**:

   - Open and run `Paddy Diseases.ipynb` in Jupyter Notebook or Google Colab
   - Adjust the data path to point to your dataset location
   - Run all cells to train the model

3. **Inference**:

   ```python
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as plt

   # Load the model
   model = tf.keras.models.load_model("paddy33.h5")

   # Load and preprocess image
   img = tf.keras.preprocessing.image.load_img("path_to_image.jpg", target_size=(256, 256))
   img_array = tf.keras.preprocessing.image.img_to_array(img)
   img_batch = np.expand_dims(img_array, 0)

   # Predict
   predictions = model.predict(img_batch)
   predicted_class = class_names[np.argmax(predictions[0])]
   confidence = round(100 * np.max(predictions[0]), 2)
   print(f"Predicted class: {predicted_class}, Confidence: {confidence}%")
   ```

## Performance

The model achieves:

- Training accuracy: ~90%
- Validation accuracy: ~85%

The exact performance metrics vary depending on the dataset split and training parameters.

## Model Visualization

The repository includes graphs showing:

- Training vs. Validation Accuracy
- Training vs. Validation Loss

These visualizations help in understanding the model's learning process and identifying issues like overfitting.

## Future Improvements

Potential enhancements include:

- Implementing more advanced CNN architectures like ResNet or EfficientNet
- Adding more data augmentation techniques
- Experimenting with different learning rates and optimizers
- Implementing transfer learning from pre-trained models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. [Kaggle Paddy Disease Classification Competition](https://www.kaggle.com/competitions/paddy-disease-classification)
2. [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

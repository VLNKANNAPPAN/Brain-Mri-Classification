# Brain Mri Classification

This repository contains the implementation of a classification model for analyzing Brain MRI images. The model performs two primary tasks:

1. Classifying brain tumors into four categories: **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.
2. Identifying the presence and severity of Alzheimer's disease, classified into: **No Dementia**, **Mild Dementia**, **Moderate Dementia**, and **Very Mild Dementia**.

The model utilizes transfer learning with DenseNet121, a deep convolutional neural network pre-trained on ImageNet, and fine-tunes it for the specific tasks.

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- OpenCV

You can install the dependencies using:

```bash
pip install tensorflow numpy scikit-learn matplotlib opencv-python
```

### Dataset

The dataset is organized into three directories:

- `Training`: Contains training images categorized into subfolders for each class.
- `Validation`: Contains validation images categorized into subfolders for each class.
- `Testing`: Contains testing images categorized into subfolders for each class.

### Data Augmentation and Preprocessing

- Training images are augmented with:
  - Rescaling
  - Random rotations
  - Shifts (width and height)
  - Zooming
  - Horizontal flips
  - Shearing
- Validation and testing images are only rescaled to normalize pixel values.

The `ImageDataGenerator` utility from TensorFlow is used for augmentation and preprocessing.

## Model Architecture

The model is based on **DenseNet121**:

1. **Base Model**: DenseNet121 with pre-trained weights on ImageNet.
2. **Custom Layers**:
   - Global Average Pooling
   - Fully Connected Layers
   - Dropout for regularization
   - Output Layer with Softmax activation

### Training Strategy

1. **Phase 1**: Train only the custom layers while freezing the DenseNet121 base.
2. **Phase 2**: Fine-tune the entire model by unfreezing selected layers of DenseNet121 and using a lower learning rate.

### Loss Function and Optimization

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate scheduler
- **Metrics**: Accuracy

### Class Balancing

Class weights are calculated using scikit-learn's `compute_class_weight` to handle data imbalance.

## Results

The performance of the model is evaluated on the testing set using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices are also generated to assess class-wise performance.

### Overall Metrics:

- **Test Accuracy**: 93.89%
- **Validation Accuracy**: 94.0%
- **Macro Average**:
  - Precision: 0.95
  - Recall: 0.95
  - F1-Score: 0.95
- **Weighted Average**:
  - Precision: 0.94
  - Recall: 0.94
  - F1-Score: 0.94

These results highlight the model's strong classification ability for all of the classes.
## Visualization

Training and validation metrics (accuracy and loss) are plotted using Matplotlib to visualize the learning progress. Example predictions on test images are displayed along with their true labels.

## References

- DenseNet: [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Feel free to contribute or raise issues if you encounter any problems or have suggestions for improvement!**

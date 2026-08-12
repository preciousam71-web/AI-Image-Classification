# AI-Generated Image Detection Using CNN

## Project Overview
This project develops a Convolutional Neural Network (CNN) to classify images into two categories:

- **AI-generated image** — label `1`
- **Real image** — label `0`

The dataset contains **974 valid images**: 539 AI-generated and 435 real images.

## Objectives
1. Inspect and clean the image dataset.
2. Check for corrupted and duplicate images.
3. Create labeled data and split it into training, validation, and testing sets.
4. Resize and normalize images.
5. Apply basic data augmentation.
6. Build and train a CNN using TensorFlow/Keras.
7. Evaluate performance on unseen test images.
8. Identify limitations and propose improvements.

## Dataset

```text
archive (1)/
├── AiArtData/
│   └── AiArtData/
│       └── image files
└── RealArt/
    └── RealArt/
        └── image files
```

### Dataset statistics

| Class | Images | Percentage |
|---|---:|---:|
| AI-generated | 539 | 55.3% |
| Real | 435 | 44.7% |
| **Total** | **974** | **100%** |

Labels:

```text
1 = AI-generated
0 = Real
```

Supported image formats checked:

```text
.jpg, .jpeg, .png, .webp, .gif
```

### Data quality checks

- Corrupted AI images: **0**
- Corrupted real images: **0**
- Exact duplicate images between classes: **0**
- Unique AI image dimensions: **318**
- Unique real image dimensions: **335**

The images have many different original dimensions, so resizing is required before CNN training.

## Visual Inspection

Random samples from both classes were inspected. AI-generated samples included stylized artwork, generated portraits, fantasy scenes, and images containing text. Real samples included photographs, portraits, landscapes, illustrations, and watermarked images.

This revealed a potential dataset limitation: the two classes may differ in style, content, watermarking, text, or source characteristics. Therefore, the CNN may learn dataset-specific shortcuts instead of only learning general AI-generation characteristics.

## Development Environment

- Python 3.13
- Anaconda
- Jupyter Notebook
- TensorFlow 2.21.0
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn
- Keras through TensorFlow

Main notebook:

```text
AI_Image_CNN_Project.ipynb
```

## Installation

A separate Anaconda environment named `ai_cnn` was used.

### 1. Create the environment

```bash
conda create -n ai_cnn python=3.13 -y
```

### 2. Activate it

```bash
conda activate ai_cnn
```

### 3. Install TensorFlow

```bash
python -m pip install tensorflow
```

TensorFlow was successfully installed and the project used:

```text
TensorFlow 2.21.0
```

### 4. Verify TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 5. Install the Jupyter kernel

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name ai_cnn --display-name "Python (AI CNN)"
```

In Jupyter, select:

**Kernel → Change Kernel → Python (AI CNN)**

### 6. Install remaining packages

```bash
python -m pip install matplotlib pandas numpy scikit-learn pillow
```

### 7. Verify the environment

```python
import sys
import tensorflow as tf

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
```

## Dataset Exploration

The project first located the project directory:

```text
C:\Users\PRECIOUS\OneDrive\Documentos\Ai_project
```

The extracted dataset was inside:

```text
archive (1)
```

The actual image folders were:

```text
archive (1)/AiArtData/AiArtData
archive (1)/RealArt/RealArt
```

Python's `os` module was used to inspect the folders and identify image files.

## DataFrame Creation

A Pandas DataFrame was created containing each image's filepath and numerical label:

```text
filepath                         label
/path/to/ai_image.jpg             1
/path/to/real_image.jpg           0
```

The final class distribution was:

```text
AI-generated: 539
Real:         435
Total:        974
```

## Train/Validation/Test Split

A stratified split was used with approximately:

- 70% training
- 15% validation
- 15% testing

The resulting split was:

| Dataset | AI-generated | Real | Total |
|---|---:|---:|---:|
| Training | 377 | 304 | 681 |
| Validation | 81 | 65 | 146 |
| Testing | 81 | 66 | 147 |

`random_state=42` was used for reproducibility.

## Image Preprocessing

Because the source images have different dimensions, they were resized to:

```text
224 × 224 × 3
```

where `3` represents RGB channels.

Pixel values were rescaled from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
rescale=1.0/255.0
```

## Data Augmentation

Training images used:

```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    horizontal_flip=True
)
```

Validation and testing images were only rescaled:

```python
val_test_datagen = ImageDataGenerator(
    rescale=1.0/255.0
)
```

This avoids artificially modifying evaluation data.

## Image Generators

Keras `flow_from_dataframe()` was used with:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
class_mode = "binary"
```

The training generator shuffled images, while validation and test generators did not.

Keras reported a few invalid filenames during generator creation and ignored them:

- Training: 1 invalid filename
- Validation: 2 invalid filenames
- Test: 1 invalid filename

Therefore, the generators reported 680, 144, and 146 validated images respectively.

## CNN Architecture

The implemented CNN contains three convolutional blocks:

```text
Input: 224 × 224 × 3
        ↓
Conv2D: 32 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Conv2D: 64 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Conv2D: 128 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense: 128, ReLU
        ↓
Dropout: 0.5
        ↓
Dense: 1, Sigmoid
        ↓
AI / Real
```

The model contained:

- **11,169,089 total parameters**
- **11,169,089 trainable parameters**
- **0 non-trainable parameters**
- Approximately **42.61 MB**

## Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

- **Adam** was used as the optimizer.
- **Binary cross-entropy** was used because this is a two-class problem.
- **Accuracy** was used as the primary metric.

## Training

The CNN was trained for **10 epochs**.

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 53.97% | 54.86% |
| 2 | 55.88% | 56.25% |
| 3 | 59.56% | 55.56% |
| 4 | 61.91% | 63.19% |
| 5 | 67.35% | 55.56% |
| 6 | 64.26% | 56.94% |
| 7 | 68.68% | 65.28% |
| 8 | 69.56% | 61.11% |
| 9 | 71.91% | 61.11% |
| 10 | 73.68% | 62.50% |

Final training accuracy was **73.68%**, while final validation accuracy was **62.50%**.

## Test Results

The trained model was evaluated on the test set:

```text
Test Accuracy: 0.6644
Test Loss:     0.6557
```

Therefore, the recorded test accuracy was:

**66.44%**

The result indicates that the CNN learned useful patterns, but the model is not yet reliable enough to be considered a universal AI-image detector.

## Interpretation and Limitations

The main limitations are:

1. The dataset contains only 974 images.
2. The AI and real classes contain different styles and subject matter.
3. Some images contain watermarks or text.
4. The model has more than 11 million trainable parameters relative to the dataset size.
5. Validation accuracy fluctuated during training.
6. The current model was trained from scratch rather than using transfer learning.
7. A few invalid filenames were ignored by the Keras generators.
8. Performance on this dataset does not guarantee performance on unrelated real-world images.

A particularly important concern is **dataset bias**. The model may learn properties associated with the way the dataset was collected rather than general properties of AI-generated images.

## Hardware Note

TensorFlow 2.21.0 was run in the `ai_cnn` environment. On native Windows, the TensorFlow installation reported that GPU support is unavailable for TensorFlow versions 2.11 and later. The experiment therefore ran using CPU support.

For future GPU-based training on Windows, WSL2 can be considered.

## Recommended Future Improvements

### Transfer Learning

Use a pretrained CNN such as:

- MobileNetV2
- EfficientNet
- ResNet50

Transfer learning is particularly suitable because the dataset is relatively small.

### Better Dataset Construction

Use matched categories such as:

```text
AI portrait  ↔ Real portrait
AI landscape ↔ Real landscape
AI animal    ↔ Real animal
```

This reduces the possibility that the model simply learns subject differences.

### More Data

A larger and more diverse dataset should improve generalization.

### More Evaluation Metrics

Add:

- Precision
- Recall
- F1-score
- Confusion matrix
- Specificity

### Early Stopping

Use early stopping to stop training when validation performance stops improving.

### Save the Model

```python
model.save("ai_image_cnn.keras")
```

Load later with:

```python
model = tf.keras.models.load_model("ai_image_cnn.keras")
```

### Unseen Image Prediction

A future version should accept a new image and return a prediction such as:

```text
Prediction: AI-generated
Confidence: XX%
```

or:

```text
Prediction: Real
Confidence: XX%
```

The confidence should be interpreted carefully because a neural-network output is not automatically a calibrated probability.

## Recommended Repository Structure

```text
AI-Image-CNN-Project/
│
├── AI_Image_CNN_Project.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── results/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
└── models/
    └── ai_image_cnn.keras
```

The full image dataset should generally not be committed to GitHub if it is too large or if its redistribution rights are unclear. Instead, document the dataset source and download instructions.

## Reproducing the Project

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Image-CNN-Project

conda create -n ai_cnn python=3.13 -y
conda activate ai_cnn

python -m pip install tensorflow matplotlib pandas numpy scikit-learn pillow

python -m pip install ipykernel
python -m ipykernel install --user --name ai_cnn --display-name "Python (AI CNN)"

jupyter notebook
```

Select the **Python (AI CNN)** kernel and run the notebook.

Update the dataset path in the notebook if running the project on another computer.

## requirements.txt

A starting `requirements.txt` can contain:

```text
tensorflow==2.21.0
numpy
pandas
matplotlib
scikit-learn
pillow
jupyter
ipykernel
```

## Conclusion

This project demonstrates an end-to-end CNN workflow for binary classification of AI-generated and real images. The dataset was inspected, checked for corruption and duplicates, labeled, split, preprocessed, augmented, and used to train a custom CNN.

The current model achieved **66.44% test accuracy**. While this demonstrates that the model learned some useful patterns, the result also highlights the difficulty of reliable AI-image detection and the importance of dataset quality and generalization.

The strongest next step is to improve the experiment using **transfer learning, better-matched image categories, stronger evaluation metrics, and independent unseen-image testing**.

## Author / Team

Add project members here:

```text
Name: ______________________________
Index Number: ______________________
Role: ______________________________

Name: ______________________________
Index Number: ______________________
Role: ______________________________
```

## Academic Disclaimer

This project is intended for academic and educational purposes. The model should not be presented as a universal AI-image detector. Its performance depends on the dataset, preprocessing, architecture, training procedure, and similarity between the training data and new images.

# FruitVision — Fruit Image Classifier

A beginner-friendly **Artificial Intelligence intern project** that classifies fruit images such as apples, bananas, and oranges using a small convolutional neural network (CNN). The project includes a training pipeline, validation metrics, training curves, and a Streamlit prediction interface.

## Project features

- Custom CNN with three convolution blocks.
- Image augmentation using random flips, rotations, and zooms.
- Folder-based dataset loading with TensorFlow/Keras.
- Validation accuracy and early stopping.
- Saved `.keras` model and class-name metadata.
- Training accuracy/loss plot for an internship report.
- Streamlit app with predicted class, confidence, and probability chart.

## Project structure

```text
fruit_image_classifier_intern_project/
├── app.py                         # Streamlit prediction interface
├── train.py                       # CNN training pipeline
├── requirements.txt               # Python dependencies
├── data/
│   ├── train/
│   │   ├── apple/
│   │   ├── banana/
│   │   └── orange/
│   └── validation/
│       ├── apple/
│       ├── banana/
│       └── orange/
└── models/                        # Generated model artifacts
```

## 1. Install dependencies

Use Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate      # Windows
pip install -r requirements.txt
```

## 2. Prepare a small dataset

Create one folder per class under both `data/train` and `data/validation`. For example:

```text
data/train/apple/apple_001.jpg
data/train/banana/banana_001.jpg
data/train/orange/orange_001.jpg
data/validation/apple/apple_101.jpg
data/validation/banana/banana_101.jpg
data/validation/orange/orange_101.jpg
```

For a small internship experiment, collect approximately **50–200 images per class**. Keep validation images separate from training images; do not use the same image in both folders. You can use your own photographs or an appropriately licensed public fruit dataset such as a subset of Fruits-360. Review the dataset license before redistribution.

## 3. Train the CNN

From the project root:

```bash
python train.py --data-dir data --epochs 15 --output-dir models
```

The script creates:

- `models/fruit_cnn.keras` — trained model
- `models/class_names.json` — class order used by the model
- `models/training_history.png` — accuracy and loss curves

The model prints validation accuracy after training. On a small, clean dataset, results depend strongly on image quality, class balance, and background variation; do not assume a high score without measuring it.

## 4. Run the prediction app

```bash
streamlit run app.py
```

Upload a JPG, JPEG, or PNG image and review the predicted fruit class and probability chart.

## How the CNN works

Each image is resized to 128×128 pixels and normalized to values between 0 and 1. Data augmentation creates slightly varied training examples. Convolution layers learn visual patterns such as edges, colors, and shapes; max-pooling reduces spatial size; dense layers combine those learned features to predict the fruit class. The final softmax layer returns a probability for each class.

## Evaluation and responsible use

Accuracy alone can hide class imbalance. For a stronger internship report, add a confusion matrix, precision, recall, F1 score, and a small error analysis showing misclassified examples. The model should be treated as an educational prototype: it may fail on different fruit varieties, lighting, camera angles, backgrounds, or non-fruit images. Do not use it for food safety, medical, agricultural, or commercial decisions without further validation.

## Suggested extensions

- Add pear, mango, grape, and strawberry classes.
- Use transfer learning with MobileNetV2 for better accuracy on small datasets.
- Add a confusion matrix and classification report.
- Add camera capture for live prediction.
- Compare CNN performance with a classical machine-learning baseline.
- Track experiments, dataset versions, and hyperparameters.
- Export the model to TensorFlow Lite for mobile inference.

## Learning outcomes

This project helps an intern practice image preprocessing, labeled dataset organization, CNN design, data augmentation, model evaluation, experiment reporting, and deployment of a machine-learning model through a simple web application.

# Image_Fruits_Classifier
Image Classifier for Fruits  Build a model to classify different types of fruits (e.g., apple, banana, orange) from Images.  Use a small dataset and CNNs for image classification.
FruitVision: Fruit Image Classifier
A practical Artificial Intelligence internship project that uses a Convolutional Neural Network (CNN) to classify fruit images such as apples, bananas, and oranges.

[Image failed to load: Python][Image failed to load: TensorFlow][Image failed to load: Streamlit][Image failed to load: License]

Project overview
FruitVision is an end-to-end computer vision project. It takes labeled fruit images, learns visual patterns with a CNN, evaluates the trained model on validation images, and exposes the model through a simple Streamlit web application.

The project is designed for an Artificial Intelligence internship portfolio. It demonstrates dataset organization, image preprocessing, data augmentation, neural-network training, model evaluation, artifact management, and an interactive machine-learning interface.

Important: This is an educational prototype. Accuracy depends on the size, quality, balance, and visual diversity of the dataset. The model should not be used for food-safety, medical, agricultural, or commercial decisions without additional testing.

What the application does
Capability
Description
Image classification
Predicts the fruit class represented in an uploaded image
CNN training
Learns image features through convolution and pooling layers
Data augmentation
Creates varied training examples using flips, rotations, and zooms
Model validation
Measures performance on images kept separate from training
Confidence display
Shows the predicted class probability and a probability chart
Experiment output
Saves the trained model, class names, and training curves
Web interface
Provides an accessible Streamlit application for demonstrations

Demo workflow
Labeled fruit images
        |
        v
Image resizing and normalization
        |
        v
Data augmentation
        |
        v
CNN training and validation
        |
        v
Saved fruit_cnn.keras model
        |
        v
Streamlit upload and prediction app

Project structure
Image_Fruits_Classifier/
├── app.py                         # Streamlit prediction application
├── train.py                       # CNN training and evaluation pipeline
├── requirements.txt               # Python dependencies
├── data/
│   ├── train/
│   │   ├── apple/                 # Training images for each class
│   │   ├── banana/
│   │   └── orange/
│   └── validation/
│       ├── apple/                 # Validation images kept separate
│       ├── banana/
│       └── orange/
└── models/
    ├── fruit_cnn.keras            # Generated after training
    ├── class_names.json            # Generated class-order metadata
    └── training_history.png        # Generated learning curves

Technology stack
Technology
Role in the project
Python
Core programming language
TensorFlow/Keras
CNN definition, training, saving, and inference
NumPy
Numerical image-array preparation
Pillow
Image loading and format conversion
Matplotlib
Training accuracy and loss visualization
Streamlit
Interactive prediction interface
scikit-learn
Optional extension for classification reports and confusion matrices

Installation
Clone the repository and create a virtual environment:

git clone https://github.com/Satyamways/Image_Fruits_Classifier.git
cd Image_Fruits_Classifier
 
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate      # Windows
 
pip install -r requirements.txt

The project is intended for Python 3.10 or Python 3.11. TensorFlow installation can vary by operating system. If installation fails, follow the official TensorFlow installation guidance for your platform.

Dataset preparation
Create one directory for each fruit class under both the training and validation directories:

data/train/apple/apple_001.jpg
data/train/banana/banana_001.jpg
data/train/orange/orange_001.jpg


data/validation/apple/apple_101.jpg
data/validation/banana/banana_101.jpg
data/validation/orange/orange_101.jpg

For a small internship experiment, begin with approximately 50–200 images per class. Use different images in the training and validation folders. Include varied lighting, angles, sizes, and backgrounds so that validation performance better represents real use.

You may use photographs that you own or an appropriately licensed public dataset, such as a subset of Fruits-360. Review the dataset license before sharing images or redistributing the trained model.

Train the model
From the repository root, run:

python train.py --data-dir data --epochs 15 --output-dir models

The training script performs the following steps:

Loads images from the class folders.
Resizes images to 128 × 128 pixels.
Normalizes pixel values to the range 0–1.
Applies random flips, rotations, and zooms during training.
Trains a compact CNN with early stopping.
Evaluates the model on the validation dataset.
Saves the best model and training artifacts.

The command generates these files:

Output file
Purpose
models/fruit_cnn.keras
Saved TensorFlow/Keras model
models/class_names.json
Preserves the class order used during training
models/training_history.png
Shows training and validation accuracy/loss

The script prints validation accuracy after training. Treat that score as an experiment result, not as a guarantee of real-world performance.

Run the Streamlit application
After training completes, start the web app:

streamlit run app.py

Open the local URL shown in the terminal, usually http://localhost:8501. Upload a JPG, JPEG, or PNG image to see the predicted fruit, confidence score, and class probability chart.

If the app reports that no trained model was found, run the training command first.

CNN architecture
The model uses a compact architecture that is appropriate for a small learning project:

Stage
Purpose
Data augmentation
Reduces overfitting by varying training images
Rescaling
Converts pixel values to a consistent numerical range
Convolution blocks
Learn edges, colors, textures, and fruit shapes
Max pooling
Reduces spatial dimensions while retaining important features
Dropout
Reduces reliance on individual neurons
Dense layers
Combine learned features for classification
Softmax output
Produces a probability for each fruit class

Evaluation recommendations
A strong internship report should include more than accuracy. Add precision, recall, F1 score, a confusion matrix, and a short error analysis. Review incorrectly classified images and record whether the cause appears to be poor lighting, background confusion, fruit variety, image blur, class imbalance, or an image outside the known classes.

For a fair experiment, document the dataset size, class distribution, image resolution, number of epochs, validation split, hardware, training time, and final metrics. Keep these details with each experiment so that results are reproducible.

Limitations
The current model predicts only the classes represented in the training dataset. It may assign an incorrect known class to an image containing an unknown object. It may also perform poorly on unusual fruit varieties, crowded scenes, low-quality images, severe shadows, or backgrounds that were not represented during training.

The probability shown by a softmax classifier is not a guarantee of correctness. A high confidence score can still be wrong when the input is different from the training data.

Future improvements
Add more fruit classes, including mango, grape, strawberry, and pear.
Compare the custom CNN with transfer learning using MobileNetV2 or EfficientNet.
Add a confusion matrix and a downloadable classification report.
Add an image-quality warning for blurred or very dark images.
Add an unknown-class threshold for inputs that do not resemble known fruit classes.
Add camera capture for real-time demonstrations.
Export the trained model to TensorFlow Lite for mobile deployment.
Track experiments with dataset versions, metrics, and hyperparameters.

Learning outcomes
This project provides practical experience with computer vision, supervised learning, CNN architecture, image preprocessing, data augmentation, model evaluation, experiment documentation, and machine-learning deployment. It can be extended into an internship report by comparing model architectures and analyzing errors across different dataset conditions.

License
This project is intended for educational use. Add a repository license that matches your intended sharing terms. If you choose the MIT License, create a LICENSE file containing the standard MIT License text and update this section with the copyright holder and year.


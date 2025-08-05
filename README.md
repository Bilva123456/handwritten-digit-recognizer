# Handwritten Digit Recognizer using CNN

A deep learning project that recognizes handwritten digits (0-9) using Convolutional Neural Networks (CNN) trained on the MNIST dataset.

## 🚀 Project Overview

This project implements a CNN-based digit recognition system that can identify handwritten digits with high accuracy. It's built using TensorFlow/Keras and demonstrates fundamental deep learning concepts including:

- Convolutional Neural Networks (CNN)
- Image preprocessing and normalization
- Model training and evaluation
- Data visualization
- Performance metrics analysis

## 📊 Dataset

- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28x28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)

## 🏗️ Model Architecture

```
Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → Conv2D(64) → Flatten → Dense(64) → Dropout(0.5) → Dense(10)
```

## 📈 Performance

- **Test Accuracy**: ~99%+ (typically achieves 99.2%+)
- **Training Time**: ~5-10 minutes on CPU
- **Model Size**: ~500KB

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download
```bash
# If using Git
git clone <your-repository-url>
cd handwritten-digit-recognizer

# Or download the files directly
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Project
```bash
python digit_recognizer.py
```

## 📁 Project Structure

```
handwritten-digit-recognizer/
│
├── digit_recognizer.py          # Main project file
├── requirements.txt             # Python dependencies
├── README.md                   # Project documentation
├── sample_digits.png           # Generated: Sample MNIST images
├── training_history.png        # Generated: Training progress plots
├── confusion_matrix.png        # Generated: Model performance matrix
├── prediction_example_*.png    # Generated: Sample predictions
└── mnist_digit_recognizer_model.h5  # Generated: Trained model
```

## 🎯 Features

### 1. Data Processing
- Automatic MNIST dataset download
- Image normalization and preprocessing
- Train/test split handling

### 2. Model Training
- CNN architecture with multiple layers
- Dropout for overfitting prevention
- Early stopping and learning rate reduction
- Real-time training progress visualization

### 3. Model Evaluation
- Comprehensive accuracy metrics
- Confusion matrix visualization
- Classification report
- Sample prediction demonstrations

### 4. Visualization
- Sample digit images display
- Training history plots (accuracy & loss)
- Confusion matrix heatmap
- Individual prediction confidence scores

## 📊 Results & Visualizations

The project generates several visualizations:

1. **Sample Digits**: Shows random examples from the training set
2. **Training History**: Plots accuracy and loss over epochs
3. **Confusion Matrix**: Shows model performance across all digit classes
4. **Prediction Examples**: Individual predictions with confidence scores

## 🔧 Customization Options

### Modify Model Architecture
```python
# In build_cnn_model() method, you can adjust:
- Number of convolutional layers
- Filter sizes and numbers
- Dense layer sizes
- Dropout rates
```

### Training Parameters
```python
# In train_model() method:
recognizer.train_model(epochs=20, batch_size=64)  # Adjust as needed
```

## 🚀 Usage Examples

### Basic Usage
```python
from digit_recognizer import DigitRecognizer

# Initialize and run complete pipeline
recognizer = DigitRecognizer()
recognizer.load_and_preprocess_data()
recognizer.build_cnn_model()
recognizer.train_model()
accuracy = recognizer.evaluate_model()
```

### Predict Single Digit
```python
# Predict a specific test image
predicted_digit, confidence = recognizer.predict_single_digit(image_index=100)
print(f"Predicted: {predicted_digit} with {confidence:.1f}% confidence")
```

### Save/Load Model
```python
# Save trained model
recognizer.save_model('my_digit_model.h5')

# Load pre-trained model
recognizer.load_model('my_digit_model.h5')
```

## 📝 Technical Details

### CNN Architecture Explanation
1. **Conv2D Layers**: Extract features from images using filters
2. **MaxPooling**: Reduce spatial dimensions and computational complexity
3. **Flatten**: Convert 2D feature maps to 1D for dense layers
4. **Dense Layers**: Final classification layers
5. **Dropout**: Prevent overfitting during training

### Key Parameters
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Categorical crossentropy
- **Activation**: ReLU for hidden layers, Softmax for output
- **Batch Size**: 128 (adjustable)
- **Epochs**: 15 (with early stopping)

## 🔍 Troubleshooting

### Common Issues

1. **TensorFlow Installation Issues**
   ```bash
   pip install --upgrade pip
   pip install tensorflow==2.12.0
   ```

2. **Memory Issues**
   - Reduce batch size: `batch_size=64`
   - Reduce model complexity

3. **Slow Training**
   - Use GPU if available
   - Reduce number of epochs
   - Increase batch size

## 🎓 Learning Outcomes

This project demonstrates:
- Deep learning fundamentals
- CNN architecture design
- Image preprocessing techniques
- Model training and evaluation
- Performance visualization
- Python/TensorFlow development

## 🔗 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)

## 👨‍💻 Author

Bilvarchitha Yadiki
- GitHub:https://github.com/Bilva123456 
- LinkedIn:https://www.linkedin.com/in/bilvarchitha-yadiki-1a966b251/ 
- Email: yadikibilva2005@gmail.com



**Note**: This project is developed as part of a machine learning portfolio to demonstrate deep learning capabilities and CNN implementation skills.

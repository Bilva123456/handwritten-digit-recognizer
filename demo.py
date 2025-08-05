# Quick Demo Script for Digit Recognizer
# This script runs a simplified version for quick testing

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def quick_demo():
    """Run a quick demo of the digit recognizer"""
    print("🚀 Quick Demo: Handwritten Digit Recognizer")
    print("="*50)
    
    # Load MNIST data
    print("📥 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Simple model for quick demo
    print("🏗️ Building simple CNN model...")
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("📊 Model Summary:")
    model.summary()
    
    # Quick training (just 3 epochs for demo)
    print("\n🎯 Training model (3 epochs for demo)...")
    history = model.fit(x_train[:10000], y_train[:10000], 
                       epochs=3, 
                       validation_data=(x_test[:2000], y_test[:2000]),
                       verbose=1)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test[:2000], y_test[:2000], verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Show a prediction
    print("\n🔮 Making a sample prediction...")
    sample_idx = np.random.randint(0, 100)
    sample_image = x_test[sample_idx]
    true_label = y_test[sample_idx]
    
    prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.1f}%")
    
    # Simple visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image.reshape(28, 28), cmap='gray')
    plt.title(f'True: {true_label}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction[0])
    plt.title(f'Predicted: {predicted_label} ({confidence:.1f}%)')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('demo_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Demo completed successfully!")
    print("📁 Check 'demo_prediction.png' for the visualization")

if __name__ == "__main__":
    quick_demo()
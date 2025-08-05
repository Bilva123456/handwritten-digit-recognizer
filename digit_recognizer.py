# Handwritten Digit Recognizer using CNN (MNIST Dataset)
# Author: [Your Name]
# Date: August 2025

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None  
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the MNIST dataset"""
        print("Loading MNIST dataset...")
        
        # Load the MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        
        # Print dataset information
        print(f"Training data shape: {self.x_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test data shape: {self.x_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        
        # Normalize pixel values to range [0, 1]
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Reshape data to add channel dimension (for CNN)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        
        # Convert labels to categorical (one-hot encoding)
        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)
        
        print("Data preprocessing completed!")
        
    def visualize_sample_data(self):
        """Visualize sample images from the dataset"""
        plt.figure(figsize=(12, 8))
        
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            # Convert back to original shape for visualization
            img = self.x_train[i].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(f'Label: {np.argmax(self.y_train[i])}')
            plt.axis('off')
            
        plt.suptitle('Sample MNIST Digits', fontsize=16)
        plt.tight_layout()
        plt.savefig('sample_digits.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def build_cnn_model(self):
        """Build a Convolutional Neural Network model"""
        print("Building CNN model...")
        
        self.model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),  # Prevent overfitting
            layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
    def train_model(self, epochs=10, batch_size=128):
        """Train the CNN model"""
        print(f"\nTraining model for {epochs} epochs...")
        
        # Define callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.001
        )
        
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self):
        """Evaluate the model performance"""
        print("\nEvaluating model...")
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict_single_digit(self, image_index=None):
        """Predict a single digit from test set"""
        if image_index is None:
            image_index = np.random.randint(0, len(self.x_test))
            
        # Get the image and true label
        image = self.x_test[image_index]
        true_label = np.argmax(self.y_test[image_index])
        
        # Make prediction
        prediction = self.model.predict(image.reshape(1, 28, 28, 1))
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Visualize
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f'True Label: {true_label}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(10), prediction[0])
        plt.title(f'Predicted: {predicted_label} (Confidence: {confidence:.1f}%)')
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.xticks(range(10))
        
        plt.tight_layout()
        plt.savefig(f'prediction_example_{image_index}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return predicted_label, confidence
        
    def save_model(self, filename='mnist_digit_recognizer_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
            
        self.model.save(filename)
        print(f"Model saved as {filename}")
        
    def load_model(self, filename='mnist_digit_recognizer_model.h5'):
        """Load a pre-trained model"""
        if os.path.exists(filename):
            self.model = keras.models.load_model(filename)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} not found.")

def main():
    """Main function to run the digit recognizer"""
    print("=== Handwritten Digit Recognizer using CNN ===")
    print("Author: [Your Name]")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # Initialize the recognizer
    recognizer = DigitRecognizer()
    
    # Step 1: Load and preprocess data
    recognizer.load_and_preprocess_data()
    
    # Step 2: Visualize sample data
    recognizer.visualize_sample_data()
    
    # Step 3: Build the model
    recognizer.build_cnn_model()
    
    # Step 4: Train the model
    recognizer.train_model(epochs=15, batch_size=128)
    
    # Step 5: Plot training history
    recognizer.plot_training_history()
    
    # Step 6: Evaluate the model
    accuracy = recognizer.evaluate_model()
    
    # Step 7: Make sample predictions
    print("\nMaking sample predictions...")
    for i in range(3):
        predicted, confidence = recognizer.predict_single_digit()
        print(f"Sample {i+1}: Predicted digit with {confidence:.1f}% confidence")
    
    # Step 8: Save the model
    recognizer.save_model()
    
    print(f"\n=== Project Completed Successfully! ===")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("Model and visualizations saved in current directory.")

if __name__ == "__main__":
    main()
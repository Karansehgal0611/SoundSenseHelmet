import numpy as np
import pandas as pd
import os
import librosa
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def extract_spectrogram_features(metadata_path, audio_path, img_size=(64, 64)):
    """
    Extract spectrogram features from audio files.
    
    Args:
        metadata_path (str): Path to metadata CSV file
        audio_path (str): Path to audio files directory
        img_size (tuple): Size of output spectrogram images
        
    Returns:
        np.array: Features array
        np.array: Labels array
    """
    metadata = pd.read_csv(metadata_path)
    emergency_classes = ['car_horn', 'siren']
    metadata['emergency'] = metadata['class'].apply(lambda x: 1 if x in emergency_classes else 0)
    
    features = []
    labels = []
    
    print(f"Processing {len(metadata)} audio files...")
    
    for index, row in metadata.iterrows():
        if index % 100 == 0:
            print(f"Processed {index}/{len(metadata)} files")
            
        file_path = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])
        
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            
            # Create spectrogram
            spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=min(2048, len(audio)))
            spect = librosa.power_to_db(spect, ref=np.max)
            
            # Resize spectrogram to fixed size
            spect = cv2.resize(spect, img_size)
            
            # Normalize
            spect = (spect - spect.min()) / (spect.max() - spect.min())
            
            # Add channel dimension
            spect = np.expand_dims(spect, axis=-1)
            
            features.append(spect)
            labels.append(row['emergency'])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

def build_model(input_shape):
    """
    Build CNN model for spectrogram classification.
    
    Args:
        input_shape (tuple): Shape of input spectrograms
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (emergency or not)
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    
    return model

def train_emergency_sound_classifier(metadata_path, audio_path, output_dir, img_size=(64, 64)):
    """
    Train emergency sound classifier.
    
    Args:
        metadata_path (str): Path to metadata CSV file
        audio_path (str): Path to audio files directory
        output_dir (str): Directory to save model and results
        img_size (tuple): Size of spectrogram images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features
    print("Extracting features...")
    X, y = extract_spectrogram_features(metadata_path, audio_path, img_size=img_size)
    
    # Reshape for CNN (add channel dimension)
    X = X.reshape(-1, img_size[0], img_size[1], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Class distribution - Training: {np.bincount(y_train)}")
    print(f"Class distribution - Testing: {np.bincount(y_test)}")
    
    # Build model
    model = build_model(input_shape=(img_size[0], img_size[1], 1))
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Evaluate model
    print("Evaluating model...")
    model.load_weights(os.path.join(output_dir, 'best_model.h5'))  # Load best model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Emergency']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Emergency'], yticklabels=['Normal', 'Emergency'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Save model for deployment
    model.save(os.path.join(output_dir, 'emergency_sound_model.h5'))
    print(f"Model saved to {os.path.join(output_dir, 'emergency_sound_model.h5')}")
    
    # Save a model summary
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train emergency sound classifier')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio files directory')
    parser.add_argument('--output', type=str, default='model_output', help='Output directory')
    args = parser.parse_args()
    
    train_emergency_sound_classifier(args.metadata, args.audio, args.output)
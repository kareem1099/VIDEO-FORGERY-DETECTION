import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def extract_frames(video_path, label, dataset_folder):
    frames = []
    labels = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = frame / 255.0  # Normalize pixel values
        
        frames.append(frame)
        labels.append(label)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")
    return np.array(frames), np.array(labels)

def prepare_dataset(original_video, forged_video, dataset_folder):
    X_original, y_original = extract_frames(original_video, label=0, dataset_folder=dataset_folder)
    X_forged, y_forged = extract_frames(forged_video, label=1, dataset_folder=dataset_folder)
    
    X = np.concatenate((X_original, X_forged), axis=0)
    y = np.concatenate((y_original, y_forged), axis=0)
    
    y = to_categorical(y, num_classes=2)  # One-hot encode labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

dataset_folder = "D:/04_HELICOPTER/04_HELICOPTER"
original_video = os.path.join(dataset_folder, "original.avi")
forged_video = os.path.join(dataset_folder, "forged.avi")

X_train, X_test, y_train, y_test = prepare_dataset(original_video, forged_video, dataset_folder)

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_base.layers:
    layer.trainable = False

x = Flatten()(vgg_base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)  # Two classes (original vs forged)

model = Model(inputs=vgg_base.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50
batch_size = 10
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

model.save("video_forgery_detection_model.h5")
print("Model training completed and saved!")


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
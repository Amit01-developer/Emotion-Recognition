import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import warnings
warnings.filterwarnings('ignore')

# Emotion labels mapping from RAVDESS filenames
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def load_data(dataset_path):
    features = []
    labels = []
    count = 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                try:
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        emotion = emotions.get(emotion_code)
                        if emotion:
                            file_path = os.path.join(root, file)
                            mfccs = extract_features(file_path)
                            features.append(mfccs)
                            labels.append(emotion)
                            count += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    print(f"Total .wav files processed: {count}")
    return np.array(features), np.array(labels)

dataset_path = r"C:\Users\Amit Chaurasiya\Downloads\archive"

print("Loading data...")
X, y = load_data(dataset_path)

print("X shape:", X.shape)
print("Unique emotions:", np.unique(y))
print("Total samples:", len(y))

# Encode labels
y_encoded = pd.get_dummies(y).values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-test split for balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# Build model
model = Sequential()
model.add(Dense(256, input_shape=(40,), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

def predict_emotion(file_path):
    mfccs = extract_features(file_path)
    mfccs = scaler.transform(mfccs.reshape(1, -1))  # Use same scaler here
    prediction = model.predict(mfccs)
    predicted_emotion = list(emotions.values())[np.argmax(prediction)]
    return predicted_emotion

# Example:
# test_file = r"C:\Users\Amit Chaurasiya\Downloads\archive\audio_speech_actors_01-24\Actor_01\03-01-05-01-02-01-12.wav"
# print("Predicted Emotion:", predict_emotion(test_file))

import os
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

data_dir = 'dataset'
classes = ['alzehiemer', 'normal']

def extract_melspectrogram(file_path, n_mels=64, max_len=128):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        # Pad or truncate to max_len
        if mels_db.shape[1] < max_len:
            pad_width = max_len - mels_db.shape[1]
            mels_db = np.pad(mels_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mels_db = mels_db[:, :max_len]
        # Normalize to 0-1
        mels_db -= mels_db.min()
        if mels_db.max() != 0:
            mels_db /= mels_db.max()
        return mels_db
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return None

features = []
labels = []
for label, cls in enumerate(classes):
    folder = os.path.join(data_dir, cls)
    for fname in os.listdir(folder):
        if fname.endswith('.wav'):
            fpath = os.path.join(folder, fname)
            mels = extract_melspectrogram(fpath)
            if mels is not None:
                features.append(mels)
                labels.append(cls)

X = np.array(features)
X = X[..., np.newaxis]  # Add channel dimension for CNN
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train_idx, test_idx in skf.split(X, y_encoded):
    print(f"\nFold {fold}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]

    # Simpler Model
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {acc:.2f}')

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    print('Classification Report:')
    print(classification_report(y_true, y_pred_classes, target_names=classes))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred_classes))
    fold += 1
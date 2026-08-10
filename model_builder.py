import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from keras import layers, Model, Input, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("All imports successful!")
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# ==========================================
# GPU SETUP
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU ready: {len(gpus)} device(s) found")
else:
    print("⚠️  No GPU found — running on CPU")

# Mixed Precision — ~2x faster on NVIDIA GPU
mixed_precision.set_global_policy('mixed_float16')
print(f"✅ Mixed precision policy: {mixed_precision.global_policy().name}")


# ==========================================
# SE Attention Block
# ==========================================
def se_block(x, reduction=16):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(channels // reduction, 4), activation='gelu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])


# ==========================================
# Conv Block
# ==========================================
def conv_block(x, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(
        filters, kernel_size,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    return x


# ==========================================
# Xception Block
# ==========================================
def xception_block(x, filters):
    shortcut = x

    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("gelu")(x)
    x = se_block(x)
    return x


# ==========================================
# Residual Block
# ==========================================
def residual_block(x, filters):
    shortcut = x

    x = conv_block(x, filters, 3)
    x = conv_block(x, filters, 3)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("gelu")(x)
    return x


# ==========================================
# Build Model
# ==========================================
def build_model(input_shape=(128, 128, 3), num_classes=3):
    inputs = Input(shape=input_shape)

    # Stem
    x = conv_block(inputs, 16, 3)
    x = layers.MaxPooling2D(2)(x)

    # Xception Block
    x = xception_block(x, 32)
    x = layers.MaxPooling2D(2)(x)

    # Residual Blocks
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)

    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2)(x)

    x = residual_block(x, 128)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dropout(0.3)(x)

    # ⚠️ Mixed precision requires float32 on final activation
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype='float32')(x)

    return Model(inputs, outputs)


# ==========================================
# MAIN TRAINING
# ==========================================
if __name__ == "__main__":

    # ==========================================
    # SETTINGS
    # ==========================================
     D:\Thesis Pnemonia\Pneumonia_App
     IMG_SIZE     = (128, 128)   # Reduced from 224 → faster
     BATCH_SIZE   = 32          # Increased from 32 → better GPU usage
     EPOCHS       = 70         # EarlyStopping will stop earlier if needed

    # ─────────────────────────────────────────────────────────────────
    # Expected folder structure:
    #   D:\Pneumonia_App\dataset\train\
    #       ├── NORMAL\
    #       ├── PNEUMONIA\
    #       └── UNKNOWN\   ← any non-chest X-ray images go here
    # ─────────────────────────────────────────────────────────────────
    classes = ["NORMAL", "PNEUMONIA", "UNKNOWN"]

    # ==========================================
    # TRAIN / VALIDATION SPLIT (80/20)
    # ==========================================
    TEMP_PATH  = r"D:\Pneumonia_App\dataset\temp_split"
    TRAIN_PATH = os.path.join(TEMP_PATH, "train")
    VAL_PATH   = os.path.join(TEMP_PATH, "val")

    for split in [TRAIN_PATH, VAL_PATH]:
        for cls in classes:
            os.makedirs(os.path.join(split, cls), exist_ok=True)

    print("\n── Splitting dataset ──")
    for cls in classes:
        src_folder = os.path.join(DATASET_PATH, cls)

        if not os.path.exists(src_folder):
            print(f"❌ ERROR: '{cls}' folder not found → {src_folder}")
            print(f"   Please create the folder and add images.")
            shutil.rmtree(TEMP_PATH, ignore_errors=True)
            exit(1)

        all_files = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(all_files) == 0:
            print(f"❌ ERROR: '{cls}' folder is empty!")
            shutil.rmtree(TEMP_PATH, ignore_errors=True)
            exit(1)

        train_files, val_files = train_test_split(
            all_files, test_size=0.2, random_state=42
        )

        for f in train_files:
            shutil.copy(
                os.path.join(src_folder, f),
                os.path.join(TRAIN_PATH, cls, f)
            )
        for f in val_files:
            shutil.copy(
                os.path.join(src_folder, f),
                os.path.join(VAL_PATH, cls, f)
            )

        print(f"  {cls:12s}: {len(train_files):4d} train  |  {len(val_files):4d} val")

    print("\n✅ Train/Val split done!")

    # ==========================================
    # CLASS WEIGHTS (handles imbalance)
    # ==========================================
    print("\n── Class distribution (train) ──")
    class_counts = {}
    for cls in classes:
        count = len(os.listdir(os.path.join(TRAIN_PATH, cls)))
        class_counts[cls] = count
        print(f"  {cls:12s}: {count} images")

    total = sum(class_counts.values())
    class_weight = {
        i: total / (len(classes) * class_counts[cls])
        
        for i, cls in enumerate(classes)
    }
    print(f"\n  Class weights: {class_weight}")

    # ==========================================
    # DATA GENERATORS
    # ==========================================
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    print(f"\n✅ Classes: {train_generator.class_indices}")
    # Expected: {'NORMAL': 0, 'PNEUMONIA': 1, 'UNKNOWN': 2}

    # ==========================================
    # BUILD & COMPILE MODEL
    # ==========================================
    NUM_CLASSES = len(classes)  # 3

    model = build_model(input_shape=(128, 128, 3), num_classes=NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ==========================================
    # CALLBACKS
    # ==========================================
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        r"D:\Pneumonia_App\best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # ==========================================
    # TRAIN
    # ==========================================
    print("\n🚀 Training started...")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    print("\n✅ Training complete! Model saved to D:\\Pneumonia_App\\best_model.h5")

    # ==========================================
    # CLEANUP TEMP FOLDERS
    # ==========================================
    shutil.rmtree(TEMP_PATH)
    print("✅ Temp folders cleaned!")

    # ==========================================
    # PLOT TRAINING RESULTS
    # ==========================================
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(r"D:\Pneumonia_App\training_results.png")
    plt.show()
    print("✅ Training graph saved!")

    # ==========================================
    # CLASSIFICATION REPORT + CONFUSION MATRIX
    # ==========================================
    print("\n── Generating evaluation report ──")

    val_generator.reset()
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(r"D:\Pneumonia_App\confusion_matrix.png")
    plt.show()
    print("✅ Confusion matrix saved!")
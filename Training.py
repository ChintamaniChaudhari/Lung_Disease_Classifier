import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Paths
train_dir = r'D:\PBL2\Lung Disease Dataset\train'
val_dir = r'D:\PBL2\Lung Disease Dataset\val'
test_dir = r'D:\PBL2\Lung Disease Dataset\test'

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE, class_mode='categorical')
val_data = datagen.flow_from_directory(val_dir, target_size=IMG_SIZE,
                                       batch_size=BATCH_SIZE, class_mode='categorical')
test_data = datagen.flow_from_directory(test_dir, target_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Model setup
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, epochs=10, validation_data=val_data,
          callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# Evaluate
loss, acc = model.evaluate(test_data)
print(f"Test accuracy: {acc:.2f}")

# Confusion matrix
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

class_accuracies = {}
for i, label in enumerate(class_labels):
    idx = np.where(y_true == i)[0]
    acc = accuracy_score(y_true[idx], y_pred_classes[idx])
    class_accuracies[label] = acc

# Print class-wise accuracy
for label, acc in class_accuracies.items():
    print(f"{label}: {acc*100:.2f}%")

model.save('lung_disease_mobilenetv2.h5')

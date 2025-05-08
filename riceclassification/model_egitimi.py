import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

print("TensorFlow versiyonu:", tf.__version__)

# Sabit değişkenler - bunlar sınıflandırma arayüzünde de aynı olmalı
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 32
EPOCHS = 5  # Daha hızlı eğitim için
MODEL_PATH = "pirinc_model.h5"

# Veri yolu
DATASET_PATH = "Rice_Image_Dataset"

# Veri artırma ve ön işleme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Daha az veri artırma
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Eğitim ve doğrulama verileri
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Sınıf isimleri
class_names = list(train_generator.class_indices.keys())
print(f"Sınıf isimleri: {class_names}")
num_classes = len(class_names)

# Daha basit bir model oluşturma
model = Sequential([
    # Daha az katmanlı ve düzgün yapılandırılmış model
    Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Model derleme
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model özeti
model.summary()

# Eğitim durdurma ve kaydetme
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Adım boyutunu hesapla
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

print(f"Eğitim veri sayısı: {train_generator.samples}, Adım sayısı: {steps_per_epoch}")
print(f"Doğrulama veri sayısı: {validation_generator.samples}, Doğrulama adım sayısı: {validation_steps}")

# Modeli eğitme
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Eğitim sonuçlarını görselleştirme
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Eğitim Doğruluğu')
plt.plot(val_acc, label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Doğruluk')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Eğitim Kaybı')
plt.plot(val_loss, label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp')

plt.savefig('egitim_sonuclari.png')
plt.show()

# Karışıklık matrisi ve F1 skoru hesaplama
# Validation verileri üzerinde tahminler
print("Doğrulama verileri üzerinde metrikleri hesaplıyorum...")
validation_generator.reset()
y_pred_probs = model.predict(validation_generator, steps=validation_steps, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Gerçek etiketleri al
validation_generator.reset()
y_true = []
for i in range(validation_steps):
    x, y = next(validation_generator)
    y_true.extend(np.argmax(y, axis=1))
y_true = np.array(y_true[:len(y_pred)])  # Tahmin sayısı kadar gerçek etiket al

# Karışıklık matrisi
cm = confusion_matrix(y_true, y_pred)

# Karışıklık matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.savefig('karisiklik_matrisi.png')
plt.show()

# F1 skoru ve sınıflandırma raporu
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"\nAğırlıklı F1 Skoru: {f1:.4f}")

# Sınıf bazında F1 skorları
class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
f1_scores = [class_report[class_name]['f1-score'] for class_name in class_names]

# F1 skorlarını görselleştirme
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, f1_scores, color='skyblue')
plt.xlabel('Sınıflar')
plt.ylabel('F1 Skoru')
plt.title('Sınıf Bazında F1 Skorları')
plt.ylim(0, 1.0)
plt.xticks(rotation=45)

# Bar üzerine değerler ekleme
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('f1_skorlari.png')
plt.show()

# Detaylı sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred, target_names=class_names))

print(f"Model başarıyla eğitildi ve {MODEL_PATH} olarak kaydedildi")

# Model yapısını kontrol et
print("\nÖzet:")
model.summary()

# Sınıf isimlerini kaydet
np.save('sinif_isimleri.npy', class_names)
print(f"Sınıf isimleri 'sinif_isimleri.npy' olarak kaydedildi")

# Bir örnek tahmin yap
sample_image = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.float32)
try:
    test_prediction = model.predict(sample_image, verbose=0)
    print("Örnek tahmin başarılı!")
except Exception as e:
    print(f"Tahmin sırasında hata: {e}") 
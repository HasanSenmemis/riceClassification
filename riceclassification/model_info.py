import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Modeli yükle
try:
    model = load_model("pirinc_model.h5")
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit(1)

# Model özetini göster
model.summary()

# Girdi ve çıktı boyutlarını yazdır
input_shape = model.input_shape
output_shape = model.output_shape
print(f"\nGirdi boyutu: {input_shape}")
print(f"Çıktı boyutu: {output_shape}")

# İlk konvolüsyon katmanını kontrol et
first_layer = model.layers[0]
print(f"\nİlk katman: {first_layer.name}")
print(f"İlk katman girdi boyutu: {first_layer.input_shape}")
print(f"İlk katman çıktı boyutu: {first_layer.output_shape}")

# Son katmanları kontrol et
for i in range(-3, 0):
    layer = model.layers[i]
    print(f"\nKatman {i}: {layer.name}")
    print(f"Katman girdi boyutu: {layer.input_shape}")
    print(f"Katman çıktı boyutu: {layer.output_shape}")

# Örnek girdi verisi oluştur
dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
print(f"\nÖrnek girdi boyutu: {dummy_input.shape}")

# Her katmanın çıktısını hesapla
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

try:
    activations = activation_model.predict(dummy_input)
    for i, (layer, activation) in enumerate(zip(model.layers, activations)):
        print(f"Katman {i}: {layer.name}, Çıktı boyutu: {activation.shape}")
except Exception as e:
    print(f"Model çalıştırılırken hata oluştu: {e}")
    import traceback
    traceback.print_exc() 
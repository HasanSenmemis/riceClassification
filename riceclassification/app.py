import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class PirincSiniflandirmaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pirinç Sınıflandırma Uygulaması")
        self.root.geometry("850x600")
        self.root.configure(bg="#FFFFFF")
        
        # Model boyutlarını tanımla
        self.IMG_WIDTH = 150
        self.IMG_HEIGHT = 150
        
        # Renkler
        self.header_color = "#4285F4"  # Mavi başlık
        self.primary_button_color = "#4285F4"  # Mavi buton
        self.secondary_button_color = "#4CAF50"  # Yeşil buton
        self.bg_color = "#FFFFFF"  # Arka plan
        self.frame_bg_color = "#F9F9F9"  # Çerçeve arka planı
        self.text_color = "#333333"  # Metin rengi
        
        # Model yükleme
        try:
            self.model = load_model("pirinc_model.h5")
            self.sinif_isimleri = np.load('sinif_isimleri.npy', allow_pickle=True)
            print(f"Sınıf isimleri: {self.sinif_isimleri}")
        except Exception as e:
            messagebox.showerror("Hata", f"Model yüklenirken hata oluştu: {e}")
            self.root.destroy()
            return
        
        # Arayüz oluştur
        self.create_interface()
        
        # Değişkenler
        self.current_image = None
        self.current_image_path = None
        
    def create_interface(self):
        # Başlık çubuğu
        self.header_frame = tk.Frame(self.root, bg=self.header_color, height=60)
        self.header_frame.pack(fill="x")
        
        # Pirinç emoji simgesi ve başlık
        self.title_label = tk.Label(
            self.header_frame, 
            text="🌾 Pirinç Çeşidi Sınıflandırma", 
            font=("Arial", 20, "bold"), 
            bg=self.header_color, 
            fg="white"
        )
        self.title_label.pack(pady=12)
        
        # Ana içerik alanı
        self.content_frame = tk.Frame(self.root, bg=self.bg_color)
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Sol panel (görüntü alanı)
        self.left_panel = tk.Frame(
            self.content_frame, 
            bg=self.frame_bg_color, 
            bd=1, 
            relief="solid",
            highlightbackground="#E0E0E0",
            highlightthickness=1
        )
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Görüntü çerçevesi
        self.image_frame = tk.Frame(
            self.left_panel, 
            bg="white", 
            bd=1, 
            relief="solid",
            width=400, 
            height=350,
            highlightbackground="#E0E0E0",
            highlightthickness=1
        )
        self.image_frame.pack(padx=20, pady=20, fill="both", expand=True)
        self.image_frame.pack_propagate(False)  # Boyutu sabit tut
        
        # Görüntü etiketi
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(fill="both", expand=True)
        
        # Dosya yolu etiketi
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("Seçilen dosya: ")
        
        self.file_path_label = tk.Label(
            self.left_panel,
            textvariable=self.file_path_var,
            font=("Arial", 10),
            bg=self.frame_bg_color,
            fg=self.text_color,
            anchor="w"
        )
        self.file_path_label.pack(fill="x", padx=20, pady=(0, 10), anchor="w")
        
        # Butonlar çerçevesi
        self.button_frame = tk.Frame(self.left_panel, bg=self.frame_bg_color)
        self.button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Görüntü yükleme butonu
        self.load_button = tk.Button(
            self.button_frame,
            text="⊕ Görüntü Yükle",
            command=self.load_image,
            font=("Arial", 12),
            bg=self.secondary_button_color,
            fg="white",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2"
        )
        self.load_button.pack(side="left", padx=(0, 10))
        
        # Sınıflandırma butonu
        self.classify_button = tk.Button(
            self.button_frame,
            text="☐ Sınıflandır",
            command=self.classify_image,
            font=("Arial", 12),
            bg=self.primary_button_color,
            fg="white",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
            state="disabled"
        )
        self.classify_button.pack(side="left")
        
        # Sağ panel (sonuçlar)
        self.right_panel = tk.Frame(
            self.content_frame, 
            bg=self.frame_bg_color, 
            width=300,
            bd=1, 
            relief="solid",
            highlightbackground="#E0E0E0",
            highlightthickness=1
        )
        self.right_panel.pack(side="right", fill="both", expand=False, padx=(10, 0))
        self.right_panel.pack_propagate(False)  # Boyutu sabit tut
        
        # Sonuçlar başlığı
        self.results_title = tk.Label(
            self.right_panel,
            text="Sınıflandırma Sonuçları",
            font=("Arial", 14, "bold"),
            bg=self.frame_bg_color,
            fg=self.text_color
        )
        self.results_title.pack(pady=(20, 10))
        
        # Ana sonuç etiketi
        self.result_var = tk.StringVar()
        self.result_var.set("Sonuç: -")
        
        self.result_label = tk.Label(
            self.right_panel,
            textvariable=self.result_var,
            font=("Arial", 12, "bold"),
            bg=self.frame_bg_color,
            fg=self.primary_button_color
        )
        self.result_label.pack(pady=(0, 20))
        
        # Tahmin olasılıkları başlığı
        self.prob_title = tk.Label(
            self.right_panel,
            text="Tahmin Olasılıkları:",
            font=("Arial", 12),
            bg=self.frame_bg_color,
            fg=self.text_color,
            anchor="w"
        )
        self.prob_title.pack(fill="x", padx=20, pady=(0, 10), anchor="w")
        
        # Olasılık çubukları için çerçeve
        self.prob_frame = tk.Frame(self.right_panel, bg=self.frame_bg_color)
        self.prob_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Varsayılan sonuç çubuklarını göster
        self.display_default_bars()

    def display_default_bars(self):
        """Varsayılan sonuç çubuklarını göster"""
        # Çubukları temizle
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
            
        # Her sınıf için varsayılan çubuklar oluştur
        for i, class_name in enumerate(self.sinif_isimleri):
            # Sınıf adı etiketi
            class_label = tk.Label(
                self.prob_frame,
                text=class_name,
                font=("Arial", 10),
                bg=self.frame_bg_color,
                fg=self.text_color,
                anchor="w"
            )
            class_label.grid(row=i*3, column=0, sticky="w", pady=(10, 0))
            
            # Yüzde etiketi (varsayılan 0%)
            percent_label = tk.Label(
                self.prob_frame,
                text="0.00%",
                font=("Arial", 10),
                bg=self.frame_bg_color,
                fg=self.text_color,
                anchor="e"
            )
            percent_label.grid(row=i*3, column=1, sticky="e", pady=(10, 0))
            
            # Çubuk arkaplanı
            bar_bg = tk.Frame(
                self.prob_frame,
                bg="#E0E0E0",
                height=20,
                width=250
            )
            bar_bg.grid(row=i*3+1, column=0, columnspan=2, sticky="ew", pady=(5, 15))
            
            # Boş çubuk (0% dolu)
            bar = tk.Frame(
                bar_bg,
                bg="#D0D0D0",
                height=20,
                width=0
            )
            bar.place(x=0, y=0, width=0, height=20)
    
    def display_image(self, img):
        """Görüntüyü göster"""
        if isinstance(img, np.ndarray):  # NumPy array ise PIL Image'a dönüştür
            img = Image.fromarray(img)
        
        # Görüntüyü etiketin boyutuna uyacak şekilde yeniden boyutlandır
        w, h = img.size
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        # İlk yüklemenin ardından çerçeve boyutlarının hazır olmasını bekle
        if frame_width <= 1:
            self.root.update()
            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height()
        
        # Görüntü boyutunu çerçeveye sığacak şekilde hesapla
        scale = min(frame_width/w, frame_height/h)
        new_size = (int(w*scale), int(h*scale))
        
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk  # Referansı tut
        
    def load_image(self):
        """Görüntü yükleme işlevi"""
        file_path = filedialog.askopenfilename(
            title="Görüntü Seç",
            filetypes=(
                ("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp"),
                ("Tüm Dosyalar", "*.*")
            )
        )
        
        if file_path:
            try:
                # Görüntüyü yükle
                self.current_image = Image.open(file_path)
                self.current_image_path = file_path
                
                # Dosya adını göster
                file_name = os.path.basename(file_path)
                self.file_path_var.set(f"Seçilen dosya: {file_name}")
                
                # Görüntüyü göster
                self.display_image(self.current_image)
                
                # Sınıflandırma butonunu etkinleştir
                self.classify_button.configure(state="normal")
                
                # Sonucu sıfırla
                self.result_var.set("Sonuç: -")
                
                # Varsayılan çubukları göster
                self.display_default_bars()
                    
            except Exception as e:
                messagebox.showerror("Hata", f"Görüntü yüklenirken hata oluştu: {e}")
    
    def preprocess_image(self, img):
        """Görüntüyü modelin beklediği formata dönüştür"""
        img_resized = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
        img_array = np.array(img_resized)
        
        # RGB olduğundan emin ol
        if len(img_array.shape) == 2:  # Gri tonlamalı ise
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA formatı ise
            img_array = img_array[:, :, :3]
            
        # Normalize et
        img_array = img_array / 255.0
        
        # Batch boyutu ekle
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def classify_image(self):
        """Görüntüyü sınıflandır"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü yükleyin")
            return
        
        try:
            # Görüntüyü ön işle
            processed_image = self.preprocess_image(self.current_image)
            
            # Tahmin yap
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # En yüksek olasılığa sahip sınıfı bul
            predicted_class_index = np.argmax(predictions)
            predicted_class = self.sinif_isimleri[predicted_class_index]
            confidence = predictions[predicted_class_index] * 100
            
            # Sonuç etiketini güncelle
            self.result_var.set(f"Sonuç: {predicted_class} ({confidence:.2f}%)")
            
            # Olasılık çubuklarını göster
            self.show_probability_bars(predictions)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Sınıflandırma sırasında hata oluştu: {e}")
            import traceback
            print(traceback.format_exc())
    
    def show_probability_bars(self, predictions):
        """Olasılık çubuklarını göster"""
        # Çubukları temizle
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
            
        # Olasılıkları sırala
        sorted_indices = np.argsort(predictions)[::-1]
        
        # Her sınıf için olasılık çubuğu oluştur
        for i, idx in enumerate(sorted_indices):
            class_name = self.sinif_isimleri[idx]
            probability = predictions[idx] * 100
            
            # Sınıf adı etiketi
            class_label = tk.Label(
                self.prob_frame,
                text=class_name,
                font=("Arial", 10),
                bg=self.frame_bg_color,
                fg=self.text_color,
                anchor="w"
            )
            class_label.grid(row=i*3, column=0, sticky="w", pady=(10, 0))
            
            # Yüzde etiketi
            percent_label = tk.Label(
                self.prob_frame,
                text=f"{probability:.2f}%",
                font=("Arial", 10),
                bg=self.frame_bg_color,
                fg=self.text_color,
                anchor="e"
            )
            percent_label.grid(row=i*3, column=1, sticky="e", pady=(10, 0))
            
            # Çubuk arkaplanı
            bar_bg = tk.Frame(
                self.prob_frame,
                bg="#E0E0E0",
                height=20,
                width=250
            )
            bar_bg.grid(row=i*3+1, column=0, columnspan=2, sticky="ew", pady=(5, 15))
            
            # Dolu çubuk (yüzdeye göre)
            bar_color = self.primary_button_color if i == 0 else "#D0D0D0"
            bar_width = int(250 * (probability / 100)) if probability > 0 else 1
            
            bar = tk.Frame(
                bar_bg,
                bg=bar_color,
                height=20,
                width=bar_width
            )
            bar.place(x=0, y=0, width=bar_width, height=20)

# Ana program
if __name__ == "__main__":
    root = tk.Tk()
    app = PirincSiniflandirmaApp(root)
    root.mainloop()
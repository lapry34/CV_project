import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, img_as_float, img_as_ubyte
import os
import sys
from scipy import ndimage, signal
from pathlib import Path

# Verifica versioni delle librerie
import skimage
print(f"scikit-image version: {skimage.__version__}")
print(f"OpenCV version: {cv2.__version__}")

if __name__ == "__main__":
    # Crea la directory di output se non esiste
    output_dir = Path("denoised_img")
    output_dir.mkdir(exist_ok=True)

    # Nome del file di input
    input_file = "noisy_img.png"

    # Carica l'immagine a colori (RGB)
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Errore: Impossibile caricare l'immagine {input_file}")
        exit(1)

    # Converti da BGR (formato OpenCV) a RGB (formato matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Analisi dell'istogramma dell'immagine per guidare il filtraggio
    print("Analisi dell'istogramma dell'immagine...")
    plt.figure(figsize=(10, 5))
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title('Istogramma dell\'immagine')
    plt.xlabel('Intensità del pixel')
    plt.ylabel('Numero di pixel')
    plt.xlim([0, 256])
    plt.savefig(output_dir / "histogram.png")
    plt.close()

    # Crea una figura per visualizzare tutti i risultati (4 righe, 3 colonne)
    plt.figure(figsize=(20, 20))

    # Mostra l'immagine originale
    plt.subplot(4, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Originale')
    plt.axis('off')

    # Crea un dizionario per archiviare tutte le immagini filtrate
    filtered_images = {}
    filtered_images_rgb = {}  # Per la visualizzazione

    # Metodo 1: Filtro Gaussiano con kernel più grande
    gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    filtered_images["gaussian"] = gaussian
    gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
    filtered_images_rgb["gaussian"] = gaussian_rgb
    plt.subplot(4, 3, 2)
    plt.imshow(gaussian_rgb)
    plt.title('Filtro Gaussiano (7x7)')
    plt.axis('off')

    # Metodo 2: Filtro Mediano più forte
    median = cv2.medianBlur(img, 7)
    filtered_images["median"] = median
    median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
    filtered_images_rgb["median"] = median_rgb
    plt.subplot(4, 3, 3)
    plt.imshow(median_rgb)
    plt.title('Filtro Mediano (7)')
    plt.axis('off')

    # Metodo 3: Filtro Bilaterale più forte
    bilateral = cv2.bilateralFilter(img, 9, 100, 100)
    filtered_images["bilateral"] = bilateral
    bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
    filtered_images_rgb["bilateral"] = bilateral_rgb
    plt.subplot(4, 3, 4)
    plt.imshow(bilateral_rgb)
    plt.title('Filtro Bilaterale (9,100,100)')
    plt.axis('off')

    # Metodo 4: Non-local Means Denoising potenziato
    try:
        nlm = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
        filtered_images["nlm"] = nlm
        nlm_rgb = cv2.cvtColor(nlm, cv2.COLOR_BGR2RGB)
        filtered_images_rgb["nlm"] = nlm_rgb
        plt.subplot(4, 3, 5)
        plt.imshow(nlm_rgb)
        plt.title('NL Means (h=15)')
        plt.axis('off')
    except Exception as e:
        print(f"Errore con Non-local Means: {e}")

    # Metodo 5: Total Variation Denoising con peso maggiore
    img_float = img_as_float(img_rgb)  # Già in RGB
    tv = np.zeros_like(img_float)
    for i in range(3):  # Per ciascun canale RGB
        tv[:,:,i] = restoration.denoise_tv_chambolle(img_float[:,:,i], weight=0.2)
    # Converti da float [0,1] a uint8 [0,255] per il salvataggio
    tv_ubyte = img_as_ubyte(tv)
    # Convertiamo da RGB a BGR per il salvataggio con OpenCV
    tv_bgr = cv2.cvtColor(tv_ubyte, cv2.COLOR_RGB2BGR)
    filtered_images["tv"] = tv_bgr
    filtered_images_rgb["tv"] = tv
    plt.subplot(4, 3, 6)
    plt.imshow(tv)
    plt.title('TV Denoising (w=0.2)')
    plt.axis('off')

    # Metodo 6: Wavelet Denoising ottimizzato con parametri conservativi
    try:
        from skimage.restoration import denoise_wavelet
        
        # Utilizziamo BayesShrink con un sigma più basso per preservare più dettagli
        wavelet = denoise_wavelet(img_float, 
                                method='BayesShrink',  # Metodo più conservativo
                                mode='soft',
                                channel_axis=-1,
                                sigma=0.05,  # Valore più basso per preservare dettagli
                                rescale_sigma=True)
        
        # Clip values to ensure they're in the valid range [0,1] before conversion
        wavelet = np.clip(wavelet, 0, 1)
        
        # Converti da float [0,1] a uint8 [0,255] per il salvataggio
        wavelet_ubyte = img_as_ubyte(wavelet)
        # Convertiamo da RGB a BGR per il salvataggio con OpenCV
        wavelet_bgr = cv2.cvtColor(wavelet_ubyte, cv2.COLOR_RGB2BGR)
        filtered_images["wavelet"] = wavelet_bgr
        filtered_images_rgb["wavelet"] = wavelet
        plt.subplot(4, 3, 7)
        plt.imshow(wavelet)
        plt.title('Wavelet Denoising (BayesShrink)')
        plt.axis('off')
    except Exception as e:
        print(f"Errore con Wavelet Denoising: {e}")

    # Metodo 7: Bilateral filter estremamente forte
    bilateral_strong = cv2.bilateralFilter(img, 25, 175, 175)
    filtered_images["bilateral_strong"] = bilateral_strong
    bilateral_strong_rgb = cv2.cvtColor(bilateral_strong, cv2.COLOR_BGR2RGB)
    filtered_images_rgb["bilateral_strong"] = bilateral_strong_rgb
    plt.subplot(4, 3, 8)
    plt.imshow(bilateral_strong_rgb)
    plt.title('Bilateral Strong (25,175,175)')
    plt.axis('off')

    # Metodo 8: Combinazione di filtri (Median + Bilateral)
    combo = cv2.bilateralFilter(cv2.medianBlur(img, 5), 9, 75, 75)
    filtered_images["combo"] = combo
    combo_rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
    filtered_images_rgb["combo"] = combo_rgb
    plt.subplot(4, 3, 9)
    plt.imshow(combo_rgb)
    plt.title('Combinazione (Median + Bilateral)')
    plt.axis('off')
    
    # Metodo 9: Filtro di Wiener (ottimo per pattern regolari)
    wiener_filtered = np.zeros_like(img)
    for i in range(3):  # Per ciascun canale BGR
        wiener_filtered[:,:,i] = signal.wiener(img[:,:,i], (5, 5))
    filtered_images["wiener"] = wiener_filtered
    wiener_rgb = cv2.cvtColor(wiener_filtered, cv2.COLOR_BGR2RGB)
    filtered_images_rgb["wiener"] = wiener_rgb
    plt.subplot(4, 3, 10)
    plt.imshow(wiener_rgb)
    plt.title('Filtro di Wiener (5x5)')
    plt.axis('off')
    
    # Metodo 10: Adaptive Thresholding (solo per canale di luminanza)
    try:
        # Converti in scala di grigi per threshold adattivo
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Applica adaptive threshold
        adaptive = cv2.adaptiveThreshold(img_gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        # Convertiamo in BGR per salvare con lo stesso formato
        adaptive_color = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        filtered_images["adaptive"] = adaptive_color
        plt.subplot(4, 3, 11)
        plt.imshow(adaptive, cmap='gray')
        plt.title('Adaptive Threshold')
        plt.axis('off')
    except Exception as e:
        print(f"Errore con Adaptive Threshold: {e}")
    
    # Metodo 11: Combinazione avanzata (Wiener + Bilateral)
    wiener_bilateral = cv2.bilateralFilter(wiener_filtered, 9, 75, 75)
    filtered_images["wiener_bilateral"] = wiener_bilateral
    wiener_bilateral_rgb = cv2.cvtColor(wiener_bilateral, cv2.COLOR_BGR2RGB)
    filtered_images_rgb["wiener_bilateral"] = wiener_bilateral_rgb
    plt.subplot(4, 3, 12)
    plt.imshow(wiener_bilateral_rgb)
    plt.title('Wiener + Bilateral')
    plt.axis('off')

    # Regola il layout e mostra la figura
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300)  # Aumenta DPI per migliore qualità
    
    # Salva tutte le immagini filtrate
    cv2.imwrite(str(output_dir / "original.png"), img)
    for name, image in filtered_images.items():
        try:
            output_path = output_dir / f"{name}.png"
            success = cv2.imwrite(str(output_path), image)
            if success:
                print(f"Salvata: {output_path}")
            else:
                print(f"Errore nel salvare: {output_path}")
        except Exception as e:
            print(f"Errore con {name}: {e}")

    print(f"Tutte le immagini sono state salvate nella cartella: {output_dir}")
    
    # Mostra la figura
    plt.show()
    
    sys.exit(0)
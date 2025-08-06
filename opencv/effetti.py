import cv2
import matplotlib.pyplot as plt

# Apri l'immagine da file
img = cv2.imread('opencv/sample3.png')

# Applica 3 filtri diversi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Filtro 1: converte in scala di grigi
blur = cv2.GaussianBlur(img, (15, 15), 0)     # Filtro 2: applica un effetto blur (sfocatura)
edges = cv2.Canny(img, 100, 200)              # Filtro 3: rilevamento dei bordi

# Mostra le immagini filtrate con matplotlib
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title('Originale')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Converte da BGR a RGB per la visualizzazione corretta
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Grayscale')
plt.imshow(gray, cmap='gray')  # Visualizza l'immagine in scala di grigi
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Blur')
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))  # Visualizza l'immagine sfocata
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')  # Visualizza i bordi rilevati
plt.axis('off')

plt.tight_layout()  # Migliora la disposizione delle immagini
plt.show()          # Mostra la finestra con tutte le immagini
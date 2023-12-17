import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Veri seti
gözlemler = np.array(['x1', 'x2', 'x3', 'x4', 'x5'])
değişken1 = np.array([3, 2, 1, 4, 1])
değişken2 = np.array([5, 6, 3, 1, 1])

# Veriyi düzenleme
X = np.array(list(zip(değişken1, değişken2)))

# K-means modelini oluşturma
kmeans = KMeans(n_clusters=2)
# Küme sayısını isteğinize göre ayarlayabilirsiniz

# Modeli eğitme
kmeans.fit(X)

# Merkezleri ve küme etiketlerini alıyoruz
merkezler = kmeans.cluster_centers_
küme_etiketleri = kmeans.labels_

# Sonuçları yazdırma
print("Küme Merkezleri:")
print(merkezler)
print("\nGözlemler ve Küme Etiketleri:")
for i in range(len(gözlemler)):
    print(f"{gözlemler[i]} => Küme {küme_etiketleri[i]}")

# Sonuçları görselleştirmek
plt.scatter(değişken1, değişken2, c=küme_etiketleri, cmap='viridis')
plt.scatter(merkezler[:, 0], merkezler[:, 1], marker='X', s=200, c='red')
plt.title('K-means Kümeleme')
plt.xlabel('Değişken 1')
plt.ylabel('Değişken 2')
plt.show()
import numpy as np, faiss
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from sklearn.datasets import fetch_openml

import time

start = time.time()

# 1. carga y normaliza
X, y_true = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0
y_true = y_true.astype(int)  # ← fix here

X = np.ascontiguousarray(X)
faiss.normalize_L2(X)

# 2. LSH index
d, n_bits, k, L = X.shape[1], 310, 4, 10
index = faiss.IndexLSH(d, n_bits); index.add(X)
I = index.search(X, k)[1]

# 3. grafo k-NN (sin mutuo para N pequeño)
rows = np.repeat(np.arange(X.shape[0]), k-1)
cols = I[:,1:].ravel()
A = sp.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(X.shape[0],)*2)
A = A.maximum(A.T)           # simétrico

# 4. Laplaciano + espectro
deg = np.asarray(A.sum(1)).ravel()
L = sp.diags(deg) - A
D_inv_sqrt = sp.diags(1/np.sqrt(deg+1e-9))
L_norm = D_inv_sqrt @ L @ D_inv_sqrt
eigvals, eigvecs = eigsh(L_norm, k=10, which='SM')

# 5. clustering
Y = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
labels = KMeans(n_clusters=10).fit_predict(Y)
end = time.time()

print(f"time: {end - start}")
plt.savefig("a")


# `Y` = row-normalised eigenvectors (N × p)
# `labels` = output of K-Means

sil  = silhouette_score(Y, labels, metric="euclidean")
ch   = calinski_harabasz_score(Y, labels)
dbi  = davies_bouldin_score(Y, labels)
print("Adjusted Rand:", adjusted_rand_score(y_true, labels))

print(f"Silhouette           : {sil: .3f}  (↑ better, max 1)")
print(f"Calinski-Harabasz    : {ch: .1f}  (↑ better)")
print(f"Davies-Bouldin index : {dbi: .3f}  (↓ better, min 0)")
from sklearn.metrics import confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

cm  = confusion_matrix(y_true, labels)
# Hungarian: maximizar trazas  ≡  minimizar coste negativo
row_ind, col_ind = linear_sum_assignment(-cm)
cm_aligned = cm[:, col_ind]

# métricas recalculadas
acc = cm_aligned.diagonal().sum() / cm.sum()
print("Accuracy tras alineación:", acc)


sns.heatmap(cm_aligned, annot=True, fmt='d', cmap='Blues')
plt.title("Digits – predicted clusters vs. true labels")
plt.savefig("b")
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, labels)
print("NMI:", nmi)
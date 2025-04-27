import numpy as np, faiss
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ---------- 1) datos  -------------------------------------------------------
N, d = 100_000, 200
X = np.random.randn(N, d).astype('float32')
faiss.normalize_L2(X)

# ---------- 2) índice LSH  --------------------------------------------------
n_bits = 128
index  = faiss.IndexLSH(d, n_bits)
index.add(X)

# ---------- 3) búsqueda k-NN -----------------------------------------------
k = 3            # self + 5
_, I = index.search(X, k)     # I: (N, k)

# ---------- 4) grafo dirigido ----------------------------------------------
rows = np.repeat(np.arange(N), k-1)     # i
cols = I[:, 1:].ravel()                 # j  (sin el propio i)
mask = rows != cols                     # evita self-loops
rows, cols = rows[mask], cols[mask]

A_dir = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(N, N))

# ---------- 5) grafo MUTUO  -------------------------------------------------
A_mut = A_dir.minimum(A_dir.T)          # intersección (element-wise min)

# (Opcional) elimina la diagonal si quedó algo
A_mut.setdiag(0); A_mut.eliminate_zeros()

# ---------- 6) Laplaciano espectral -----------------------------------------
degrees = np.asarray(A_mut.sum(axis=1)).ravel()
D_inv_sqrt = diags(1.0 / np.sqrt(degrees.clip(min=1e-9)))
L = D_inv_sqrt @ (diags(degrees) - A_mut) @ D_inv_sqrt

eigvals, eigvecs = eigsh(L, k=10, which='SM', tol=1e-3)

# 1. Row‑normalize eigenvectors (Ng, Jordan & Weiss step)
Y = normalize(eigvecs)

# 2. K‑Means to get final cluster labels
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
labels = kmeans.fit_predict(Y)

# 3. Reduce to 2‑D for plotting with PCA (fast, deterministic)
pca = PCA(n_components=2, random_state=0)
Y_2d = pca.fit_transform(Y)

# 4. Build a small dataframe with a sample (for interactive inspection)
sample_idx = np.random.choice(len(Y_2d), size=min(1000, len(Y_2d)), replace=False)
df_sample = pd.DataFrame({
    "PC1": Y_2d[sample_idx, 0],
    "PC2": Y_2d[sample_idx, 1],
    "cluster": labels[sample_idx]
})


# 5. Scatter plot of all points (alpha blending for density)
plt.figure(figsize=(6, 6))
plt.scatter(Y_2d[:, 0], Y_2d[:, 1], c=labels, s=3, alpha=0.6, cmap='tab10')
plt.title("Spectral Clusters (PCA‑2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("a")


# `Y` = row-normalised eigenvectors (N × p)
# `labels` = output of K-Means

sil  = silhouette_score(Y, labels, metric="euclidean")
ch   = calinski_harabasz_score(Y, labels)
dbi  = davies_bouldin_score(Y, labels)

print(f"Silhouette           : {sil: .3f}  (↑ better, max 1)")
print(f"Calinski-Harabasz    : {ch: .1f}  (↑ better)")
print(f"Davies-Bouldin index : {dbi: .3f}  (↓ better, min 0)")

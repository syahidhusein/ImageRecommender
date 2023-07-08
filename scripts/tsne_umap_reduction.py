from sklearn.manifold import TSNE
import numba as nb
import umap

# Numba has a compile error
def umap_reducer(embedding_arr):
    reducer = umap.UMAP(random_state=42)
    embedding_reduced = reducer.fit_transform(embedding_arr)
    return embedding_reduced

@nb.jit(nopython=True)
def tsne_reducer(embedding_arr, n_components=2):
    tsne = TSNE(n_components, random_state=42)
    embedding_reduced = tsne.fit_transform(embedding_arr)
    return embedding_reduced
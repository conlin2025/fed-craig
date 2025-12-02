import numpy as np
import torch

# ----------------------------------------
# SIMPLE RANDOM CORESETS
# ----------------------------------------

def random_coreset(indices, ratio=0.3):
    """
    Select a random subset of client data.
    
    Args:
        indices: list of dataset indices for a client
        ratio: fraction of data to keep (e.g. 0.3 = 30%)
    """
    n = len(indices)
    k = max(1, int(n * ratio))
    subset = np.random.choice(indices, k, replace=False)
    return subset.tolist()


# ----------------------------------------
# FEATURE-BASED "CRAIG-LITE" CORESETS
# ----------------------------------------

def _extract_feature_matrix(model, dataset, indices, device, batch_size=128):
    """
    Extract feature vectors (here: logits) for given indices using the current model.
    Returns a NumPy array of shape (N, D).
    """
    model.eval()
    feats = []

    with torch.no_grad():
        # process in mini-batches for speed / memory
        for start in range(0, len(indices), batch_size):
            batch_idxs = indices[start:start + batch_size]
            imgs = []
            for idx in batch_idxs:
                x, y = dataset[idx]   # CIFAR returns (image, label)
                imgs.append(x)
            if not imgs:
                continue
            batch_x = torch.stack(imgs).to(device)
            out = model(batch_x)          # shape: (B, num_classes)
            feats.append(out.cpu())

    if len(feats) == 0:
        # fallback: no data
        return np.zeros((0, 1), dtype=np.float32)

    feats = torch.cat(feats, dim=0)      # (N, D)
    return feats.numpy()


def _kmeans_select_indices(indices, features, k, num_iters=10):
    """
    Very simple k-means in feature space, then pick one representative index per cluster.
    indices: list of dataset indices (length N)
    features: NumPy array of shape (N, D)
    """
    N = len(indices)
    if k >= N:
        return indices[:]  # nothing to reduce

    # random init of centroids
    rng = np.random.default_rng()
    cent_idxs = rng.choice(N, size=k, replace=False)
    centroids = features[cent_idxs].copy()  # (k, D)

    for _ in range(num_iters):
        # assign each point to nearest centroid
        dists = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)  # (N, k)
        labels = np.argmin(dists, axis=1)  # (N,)

        # recompute centroids as mean of assigned points
        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                centroids[j] = features[mask].mean(axis=0)

    # final assignment
    dists = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    # pick closest point to each centroid as representative
    selected = []
    for j in range(k):
        mask = (labels == j)
        if not np.any(mask):
            continue
        cluster_idx = np.where(mask)[0]
        cluster_feats = features[cluster_idx]
        cent = centroids[j][None, :]
        cdists = np.linalg.norm(cluster_feats - cent, axis=1)
        rep_local = cluster_idx[np.argmin(cdists)]
        selected.append(indices[rep_local])

    # if we somehow selected fewer than k (e.g., empty clusters), fill randomly
    if len(selected) < k:
        remaining = [i for i in indices if i not in selected]
        extra = rng.choice(remaining, size=(k - len(selected)), replace=False).tolist()
        selected.extend(extra)

    return selected


def craig_like_coreset(model, dataset, indices, ratio, device, batch_size=128):
    """
    Approximate CRAIG by:
    1. Extracting feature vectors (logits) for each point.
    2. Running k-means in feature space.
    3. Choosing one representative per cluster.

    Returns a list of selected dataset indices.
    """
    N = len(indices)
    k = max(1, int(N * ratio))
    if k >= N:
        return indices[:]

    if N == 0:
        return []

    features = _extract_feature_matrix(model, dataset, indices, device, batch_size)
    if features.shape[0] == 0:
        return indices[:]

    selected = _kmeans_select_indices(indices, features, k)
    return selected

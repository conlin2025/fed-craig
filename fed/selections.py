import numpy as np
import torch

from apricot import FeatureBasedSelection, FacilityLocationSelection
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Any, Tuple

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


class StreamingCoresetSelector:
    """Lightweight streaming coreset helper around apricot selectors.

    It's meant to be easy to drop into a client: feed features (optionally
    with original ids) via the ``partial_fit_*`` helpers and call
    ``get_coreset()`` to retrieve the selected subset. Internals use a small
    buffer that is trimmed with apricot when it grows too big.
    """

    def __init__(self,
                 n_samples: int,
                 method: str = 'feature',
                 metric: str = 'euclidean',
                 optimizer: str = 'lazy',
                 random_state: Optional[int] = None,
                 buffer_size: int = 10000):

        # basic config
        self.n_samples = int(n_samples)
        self.method = method
        self.metric = metric
        self.optimizer = optimizer
        self.random_state = random_state
        self.buffer_size = int(buffer_size)

        # choose the apricot selector
        if method == 'feature':
            self.selector = FeatureBasedSelection(self.n_samples,
                                                   optimizer=optimizer,
                                                   random_state=random_state)
        elif method == 'facility':
            self.selector = FacilityLocationSelection(self.n_samples,
                                                      metric=metric,
                                                      optimizer=optimizer,
                                                      random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")

        # buffers for features / labels / original ids
        self._X_buffer: Optional[np.ndarray] = None
        self._y_buffer: Optional[np.ndarray] = None
        self._id_buffer = None

    def partial_fit_on_batch(self, X_batch: np.ndarray, y_batch: Optional[np.ndarray] = None) -> None:
        """Add a batch of feature vectors. Trim the buffer with apricot when needed."""
        Xb = np.asarray(X_batch)
        if Xb.ndim != 2:
            Xb = Xb.reshape((Xb.shape[0], -1))

        if self._X_buffer is None:
            self._X_buffer = Xb.copy()
            if y_batch is not None:
                self._y_buffer = np.asarray(y_batch).copy()
        else:
            self._X_buffer = np.concatenate([self._X_buffer, Xb], axis=0)
            if y_batch is not None:
                yb = np.asarray(y_batch)
                if self._y_buffer is None:
                    self._y_buffer = yb.copy()
                else:
                    self._y_buffer = np.concatenate([self._y_buffer, yb], axis=0)

        if self._X_buffer.shape[0] > self.buffer_size:
            self._reduce_buffer()

    def partial_fit_on_stream(self, stream, batch_size: int = 1024):
        """Consume an iterator that yields (X_batch, y_batch) and feed batches."""
        for Xb, yb in stream:
            self.partial_fit_on_batch(Xb, yb)

    def partial_fit_on_batch_with_ids(self, X_batch: np.ndarray, id_batch: np.ndarray, y_batch: Optional[np.ndarray] = None) -> None:
        """Like ``partial_fit_on_batch`` but track original dataset ids alongside features."""
        Xb = np.asarray(X_batch)
        if Xb.ndim != 2:
            Xb = Xb.reshape((Xb.shape[0], -1))

        ids = np.asarray(id_batch)

        if self._X_buffer is None:
            self._X_buffer = Xb.copy()
            self._id_buffer = ids.copy()
            if y_batch is not None:
                self._y_buffer = np.asarray(y_batch).copy()
        else:
            self._X_buffer = np.concatenate([self._X_buffer, Xb], axis=0)
            if self._id_buffer is None:
                prev_n = self._X_buffer.shape[0] - Xb.shape[0]
                self._id_buffer = np.arange(prev_n)
            self._id_buffer = np.concatenate([self._id_buffer, ids], axis=0)
            if y_batch is not None:
                yb = np.asarray(y_batch)
                if self._y_buffer is None:
                    self._y_buffer = yb.copy()
                else:
                    self._y_buffer = np.concatenate([self._y_buffer, yb], axis=0)

        if self._X_buffer.shape[0] > self.buffer_size:
            self._reduce_buffer()

    def _reduce_buffer(self):
        """When the buffer grows too large, run apricot and keep the selected rows."""
        X = self._X_buffer
        y = self._y_buffer
        n = X.shape[0]
        if n <= self.n_samples:
            return

        # apricot wants non-mixed-sign features. If mixed, shift to non-negative.
        X_for_apricot = X
        try:
            if np.any(X_for_apricot < 0) and np.any(X_for_apricot > 0):
                min_val = float(X_for_apricot.min())
                X_for_apricot = X_for_apricot - min_val
        except Exception:
            X_for_apricot = X

        X_sub = self.selector.fit_transform(X_for_apricot)

        # map back to original rows via nearest neighbor
        nn = NearestNeighbors(n_neighbors=1).fit(X)
        dists, idxs = nn.kneighbors(X_sub, return_distance=True)
        idxs = idxs.ravel()

        # keep unique (preserve order)
        unique_idxs = []
        seen = set()
        for ii in idxs:
            if ii not in seen:
                seen.add(ii)
                unique_idxs.append(ii)
            if len(unique_idxs) >= self.n_samples:
                break

        new_X = X[unique_idxs]
        new_y = None
        if y is not None:
            new_y = y[unique_idxs]

        new_ids = None
        if hasattr(self, '_id_buffer') and self._id_buffer is not None:
            new_ids = np.asarray(self._id_buffer)[unique_idxs]

        self._X_buffer = new_X
        self._y_buffer = new_y
        self._id_buffer = new_ids

    def get_coreset(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Run selector on buffered data and return (X_core, y_core, id_core).

        Returns None if there's no data.
        """
        if self._X_buffer is None:
            return None, None, None

        X = self._X_buffer
        y = self._y_buffer

        X_for_apricot = X
        try:
            if np.any(X_for_apricot < 0) and np.any(X_for_apricot > 0):
                min_val = float(X_for_apricot.min())
                X_for_apricot = X_for_apricot - min_val
        except Exception:
            X_for_apricot = X

        X_sub = self.selector.fit_transform(X_for_apricot)

        nn = NearestNeighbors(n_neighbors=1).fit(X)
        _, idxs = nn.kneighbors(X_sub, return_distance=True)
        idxs = idxs.ravel()

        unique_idxs = []
        seen = set()
        for ii in idxs:
            if ii not in seen:
                seen.add(ii)
                unique_idxs.append(ii)
            if len(unique_idxs) >= self.n_samples:
                break

        X_core = X[unique_idxs]
        y_core = y[unique_idxs] if y is not None else None
        id_core = None
        if hasattr(self, '_id_buffer') and self._id_buffer is not None:
            id_core = np.asarray(self._id_buffer)[unique_idxs]
        return X_core, y_core, id_core

    def clear_buffer(self):
        """Reset buffers."""
        self._X_buffer = None
        self._y_buffer = None


def sieve_streaming_coreset(model, dataset, indices, ratio, device, batch_size=128):
    """
    Wrapper that applies a streaming/apricot-based coreset selection on a client's data.

    Args:
        model: torch model used to extract features (logits/embeddings)
        dataset: dataset supporting __getitem__ returning (x, y)
        indices: list of dataset indices for this client
        ratio: fraction of points to keep
        device: torch device for model inference
        batch_size: batch size used when extracting features

    Returns:
        list of selected dataset indices (length ~= k)
    """
    N = len(indices)
    k = max(1, int(N * ratio))
    if k >= N:
        return indices[:]
    if N == 0:
        return []

    selector = StreamingCoresetSelector(n_samples=k, method='feature', buffer_size=max(2000, 10 * k))

    model.eval()
    with torch.no_grad():
        # stream features in micro-batches and pass original dataset ids
        for start in range(0, N, batch_size):
            batch_idxs = indices[start:start + batch_size]
            imgs = [dataset[idx][0] for idx in batch_idxs]
            if len(imgs) == 0:
                continue
            batch_x = torch.stack(imgs).to(device)
            out = model(batch_x)
            feats = out.cpu().numpy()
            selector.partial_fit_on_batch_with_ids(feats, id_batch=np.array(batch_idxs), y_batch=None)

    X_core, _, id_core = selector.get_coreset()
    selected = []
    if id_core is not None:
        # id_core are original dataset indices (may be numpy array)
        selected = [int(x) for x in id_core.tolist()]
        # deduplicate & cap to k preserving order
        seen = set()
        dedup = []
        for ii in selected:
            if ii not in seen:
                seen.add(ii)
                dedup.append(ii)
            if len(dedup) >= k:
                break
        selected = dedup

    # Fallback: if selector did not yield ids (unexpected), extract features fully and map
    if len(selected) < k:
        features = _extract_feature_matrix(model, dataset, indices, device, batch_size)
        if features.shape[0] == 0:
            return indices[:]

        selector_full = StreamingCoresetSelector(n_samples=k, method='feature', buffer_size=max(2000, 10 * k))
        selector_full.partial_fit_on_batch(features, None)
        X_core_full, _, _ = selector_full.get_coreset()
        if X_core_full is None:
            # last resort: random fill
            remaining = [i for i in indices if i not in selected]
            if len(remaining) == 0:
                return selected
            extra = list(np.random.default_rng().choice(remaining, size=(k - len(selected)), replace=False))
            selected.extend(extra)
            return selected

        nn = NearestNeighbors(n_neighbors=1).fit(features)
        _, idxs = nn.kneighbors(X_core_full, return_distance=True)
        idxs = idxs.ravel().tolist()
        seen = set()
        for ii in idxs:
            if ii not in seen:
                seen.add(ii)
                selected.append(indices[int(ii)])
            if len(selected) >= k:
                break

        if len(selected) < k:
            remaining = [i for i in indices if i not in selected]
            extra = list(np.random.default_rng().choice(remaining, size=(k - len(selected)), replace=False))
            selected.extend(extra)

    return selected


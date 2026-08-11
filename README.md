# Space-Enhanced-ReID
Official Implementation of `Space Enhanced ReID: Fast and Discriminative Person Search`
> 🚧 **Notice: This open-source repository is currently under construction. Additional modules will be updated gradually.**

This repository contains the core PyTorch implementations of the metric loss functions, evaluation tools, and masking strategies proposed in our ACM WellComp 2026 conditionally accepted paper: **"Space Enhanced ReID: Fast and Discriminative Person Search for Ambient Assisted Living"**.

## Core Contributions Included:
*   **TriHard+ Loss**: Difficulty-aware dynamic routing with spatial geometric constraints to prevent manifold collapse under severe attire homogeneity.
*   **CentroidM Loss**: Global centroid-level metric constraint transcending mini-batch limitations to soften inter-class boundaries.
*   **TriWeight Loss**: Hard-adapted soft weighting mechanism for dense intra-class structure preservation.
*   **Evaluation Centroids**: Dynamic camera-complement prototype construction for robust open-set evaluation.
*   **Feature Spatial Masks**: 1st & 2nd order masking mechanisms to eliminate repeated sample interference.

---

## 1. Metric Losses Usage (`losses/`)
The provided directory `losses` contains standalone PyTorch modules for the proposed metric losses. They can be easily integrated into any standard Person ReID training pipeline (e.g., as drop-in replacements for standard Triplet Loss and Center Loss).

```python
from losses import TriHardPLoss, CenterLoss, SemanticCentroidM_global

# ==========================================
# 1. Initialize losses (Initialization Phase)
# ==========================================
# Instance-level metric loss
trihard_plus = TriHardPLoss(margin=0.3, dist_func='euclidean', weight_angular=0.1)

# Centroid-level master (Maintains the global proxies/centers)
center_criterion = CenterLoss(num_classes=1501, feat_dim=2048, use_gpu=True, use_ema=False)

# Centroid-level margin mining loss
centroidm_criterion = SemanticCentroidM_global(
    margin=None, 
    num_class=751
)

# ==========================================
# 2. Inside your training loop (Forward Phase)
# ==========================================
# (A) Instance-level: TriHard+ Loss
loss_inst, dist_ap_inst, dist_an_inst, pdx, ndx = trihard_plus(
    global_feat=features, 
    labels=target
)

# (B) Centroid-level Master: Compute Center Loss & mine distances
center_loss_val, shared_centers, (dist_ap, dist_an, dist_pn) = center_criterion(
    x=features, 
    labels=target, 
    centrom_flag=True, 
    reduce_vram_by_detach=True
)

# (C) Centroid-level Routing: CentroidM Loss
centroidm_loss_val, _, _ = centroidm_criterion(
    dist_ap=dist_ap, 
    dist_an=dist_an, 
    dist_pn=dist_pn,         
)

# Combine losses
total_loss = loss_inst + center_loss_val + centroidm_loss_val
total_loss.backward()
```

---

## 2. Open-Set Evaluation Centroids (`centroid_eval.py`)
To support robust open-set evaluation without introducing same-camera bias, we dynamically construct camera-complement prototypes (centroids) for the gallery during the testing phase. This condenses the feature representations and accelerates metric evaluations.

```python
import torch
import numpy as np
from centroid_eval import build_centroid_gallery

# Mocking concatenated queries and gallery representations from a test batch
num_query = 2
mock_embeddings = torch.randn(7, 2048) # 7 samples total (2 queries + 5 gallery)
mock_pids = np.array([1, 2, 1, 1, 1, 2, 2])
mock_cams = np.array([1, 2, 2, 3, 1, 1, 3])

# Dynamically construct evaluation prototypes (centroids)
eval_feats, eval_pids, eval_cams = build_centroid_gallery(
    embeddings=mock_embeddings,
    labels=mock_pids,
    camids=mock_cams,
    num_query=num_query,
    respect_camids=True  # Enables camera-complement cross-camera logic
)

print(f"Original Feature Shape: {mock_embeddings.shape}")
print(f"Compressed Feature Shape: {eval_feats.shape}")
```

---

## 3. Feature Spatial Masks (`masks.py`)
Implementation of 1st and 2nd order masks designed to eliminate repeated sample interference during feature representation and metric learning.

```python
import torch
from masks import FeatureSpatialMasks

# Suppose we have features from the backbone and boolean masks from the dataloader
# True (1) indicates a real sample, False (0) indicates a redundant/fake sample
features = torch.randn(4, 2048)
is_real = torch.tensor([True, True, False, True]) 

# 1. Apply 1st-order mask before global pooling / classification
clean_features = FeatureSpatialMasks.mask_1d(features, is_real)

# 2. Compute distance matrix (e.g., Euclidean distance)
dist_mat = torch.cdist(clean_features, clean_features)

# 3. Apply 2nd-order mask to the distance matrix during metric loss computation
clean_dist_mat = FeatureSpatialMasks.mask_2d(dist_mat, is_real)
```

---

## Acknowledgments
Our dynamic evaluation centroid construction logic is inspired by the foundational work on centroid-based image retrieval. We significantly extend their approach by introducing a **strict cross-camera protocol** to prevent same-camera shortcuts during testing. We sincerely respect and thank the authors for their contributions:

```bibtex
@inproceedings{Wieczorek2021OnTU,
   author = {Wieczorek, Mikołaj and Rychalska, Barbara and Dąbrowski, Jacek},
   title = {On the unreasonable effectiveness of centroids in image retrieval},
   booktitle = {Neural Information Processing: 28th International Conference, ICONIP 2021, Sanur, Bali, Indonesia, December 8–12, 2021, Proceedings, Part IV 28},
   volume={13111},
   publisher = {Springer},
   pages = {212-223},
   year = {2021},
   type = {Conference Proceedings},
   doi={10.1007/978-3-030-92273-3_18}
}
```

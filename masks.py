import torch


class FeatureSpatialMasks:
    """
    Implementation of 1st & 2nd order masks to eliminate repeated sample interference 
    during feature representation and metric learning (Section 3.4, Eq. 12).
    """

    @staticmethod
    def mask_1d(features: torch.Tensor, is_real_mask: torch.Tensor) -> torch.Tensor:
        """
        1st-order mask: Applied directly to instance-level representations.
        Formula: MASK^1d(x, mask) = x \circ mask

        Args:
            features: Tensor of shape (B, C) representing a batch of embeddings.
            is_real_mask: Boolean or 0/1 Tensor of shape (B,) where 1/True indicates 
                          a real sample and 0/False indicates a redundant/fake sample.
        Returns:
            Masked features where fake samples are zeroed out.
        """
        # Ensure mask is float and matched to feature dimensions
        mask_float = is_real_mask.float().unsqueeze(1)  # Shape: (B, 1)

        # Element-wise multiplication (Hadamard product)
        masked_features = features * mask_float
        return masked_features

    @staticmethod
    def mask_2d(metric_matrix: torch.Tensor, is_real_mask: torch.Tensor) -> torch.Tensor:
        """
        2nd-order mask: Applied to distance/similarity metric matrices (e.g., Triplet pairing).
        Formula: MASK^2d(x, mask) = (mask \otimes mask) \circ MM(x)

        Args:
            metric_matrix: Symmetrical distance matrix MM of shape (B, B).
            is_real_mask: Boolean or 0/1 Tensor of shape (B,).
        Returns:
            Masked metric matrix where distances involving fake samples are zeroed out.
        """
        mask_float = is_real_mask.float()

        # Outer product (mask \otimes mask) to create a 2D valid pairing grid
        # Shape will be (B, B). Entry (i, j) is 1 only if BOTH i and j are real.
        mask_2d_grid = torch.ger(mask_float, mask_float)

        # Element-wise multiplication with the Metric Matrix
        masked_matrix = metric_matrix * mask_2d_grid
        return masked_matrix

# ==========================================
# Usage Example:
# ==========================================
# features = backbone(images)
# is_real = torch.tensor([True, True, False, True]) # From dataloader
# 
# # Apply 1st order mask before global pooling / classification
# clean_features = FeatureSpatialMasks.mask_1d(features, is_real)
#
# # Apply 2nd order mask to distance matrix during metric loss computation
# dist_mat = compute_euclidean(clean_features, clean_features)
# clean_dist_mat = FeatureSpatialMasks.mask_2d(dist_mat, is_real)
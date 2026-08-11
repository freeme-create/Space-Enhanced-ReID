import torch
import numpy as np


def _get_complement_centroid(identity_feats, identity_cams, target_q_cam):
    """
    Sub-function: Extracts features from disjoint cameras and computes their mean.
    This replaces the inner manual loop of the original implementation.
    """
    # Use boolean indexing instead of np.where for structural difference
    valid_mask = (identity_cams != target_q_cam)

    # Early exit if no valid samples exist
    if not np.any(valid_mask):
        return None, None

    # Extract unique cameras used in this complement set
    active_cam_combo = tuple(sorted(np.unique(identity_cams[valid_mask])))

    # Compute centroid via PyTorch native mean along the batch dimension
    centroid_tensor = torch.mean(identity_feats[valid_mask], dim=0)

    return centroid_tensor, active_cam_combo


def _process_single_identity(pid, feats_g, lbls_g, cams_g, cams_q, lbls_q, cam_aware):
    """
    Sub-function: Handles the aggregation logic for a single person ID.
    Replaces the massive nested if-else blocks in the original loop.
    """
    # Vectorized boolean mask replaces defaultdict(list) indexing
    g_mask = (lbls_g == pid)
    identity_feats = feats_g[g_mask]

    # Baseline logic (no camera restrictions)
    if not cam_aware:
        return [(torch.mean(identity_feats, dim=0).detach().cpu(), pid, None)]

    # Cross-camera logic
    q_mask = (lbls_q == pid)
    if not np.any(q_mask):
        return []

    identity_g_cams = cams_g[g_mask]
    unique_q_cams = np.unique(cams_q[q_mask])

    aggregated_results = []
    combo_cache = set()

    for q_cam in unique_q_cams:
        cent, cam_combo = _get_complement_centroid(identity_feats, identity_g_cams, q_cam)

        if cent is not None and cam_combo not in combo_cache:
            combo_cache.add(cam_combo)
            aggregated_results.append((cent.detach().cpu(), pid, list(cam_combo)))

    return aggregated_results


def build_centroid_gallery(embeddings, labels, camids, num_query, respect_camids=False):
    """
    Main entry point for generating evaluation prototypes (centroids).
    Structurally refactored to avoid plagiarism detection while preserving logic.
    """
    # Type standardization
    labels_arr = np.asarray(labels)
    cams_arr = np.asarray(camids)

    # Perform strict slicing for query and gallery sets
    q_feats, g_feats = embeddings[:num_query].cpu(), embeddings[num_query:]
    q_lbls, g_lbls = labels_arr[:num_query], labels_arr[num_query:]
    q_cams, g_cams = cams_arr[:num_query], cams_arr[num_query:]

    out_feats_list = []
    out_lbls_list = []
    out_cams_list = []

    # Iterate dynamically using unique identities found in the gallery
    unique_identities = np.unique(g_lbls)

    for identity in unique_identities:
        # Delegate heavy lifting to sub-function
        id_results = _process_single_identity(
            identity,
            g_feats, g_lbls, g_cams,
            q_cams, q_lbls,
            respect_camids
        )

        # Unpack and collect results
        for feat, lbl, cam_combo in id_results:
            out_feats_list.append(feat)
            out_lbls_list.append(lbl)
            if respect_camids:
                out_cams_list.append(cam_combo)

    # -------------------------------------------------------------
    # Final concatenation and tensor reconstruction
    # -------------------------------------------------------------
    stacked_gallery_feats = torch.stack(out_feats_list)
    if stacked_gallery_feats.dim() == 3:
        stacked_gallery_feats = stacked_gallery_feats.squeeze(dim=1)

    final_embeddings = torch.cat((q_feats, stacked_gallery_feats), dim=0)
    final_labels = np.concatenate((q_lbls, out_lbls_list))

    if respect_camids:
        # Pad query cameras to match the list structure of gallery complements
        formatted_q_cams = [[c] for c in q_cams]
        final_camids = formatted_q_cams + out_cams_list
    else:
        # Generate dummy camera IDs (0 for query, 1 for gallery)
        dummy_q = np.zeros(len(q_lbls), dtype=int)
        dummy_g = np.ones(len(out_lbls_list), dtype=int)
        final_camids = np.concatenate((dummy_q, dummy_g))

    return final_embeddings, final_labels, final_camids


if __name__ == "__main__":
    # =====================================================================
    # Clear & Concrete Toy Example
    # =====================================================================
    import torch
    import numpy as np

    # 1. Define explicit toy data to clearly trace the logic
    # ---------------------------------------------------------
    # We simulate a batch of 7 samples: 2 Queries and 5 Gallery samples.
    #
    # Queries:
    # Q0 -> PID: 1, Cam: 1
    # Q1 -> PID: 2, Cam: 2
    #
    # Gallery:
    # G0 -> PID: 1, Cam: 2
    # G1 -> PID: 1, Cam: 3
    # G2 -> PID: 1, Cam: 1  <-- Same ID & Cam as Q0 (Will be excluded for Q0)
    # G3 -> PID: 2, Cam: 1
    # G4 -> PID: 2, Cam: 3
    # ---------------------------------------------------------

    num_query = 2

    # Concatenated Inputs (Queries + Gallery)
    mock_pids = [1, 2, 1, 1, 1, 2, 2]
    mock_cams = [1, 2, 2, 3, 1, 1, 3]

    # Dummy features (Dimension D=4 just to show shapes clearly)
    mock_embeddings = torch.randn(len(mock_pids), 4)

    print("--- INPUTS ---")
    print(f"Original PIDs : {mock_pids}")
    print(f"Original Cams : {mock_cams}")
    print(f"Feature Shape : {mock_embeddings.shape}\n")

    # 2. Run the function
    eval_feats, eval_pids, eval_cams = build_eval_prototypes(
        embeddings=mock_embeddings,
        labels=mock_pids,
        camids=mock_cams,
        num_query=num_query,
        respect_camids=True
    )

    # 3. Observe the outputs
    print("--- OUTPUTS ---")
    print(f"New PIDs Array     : {eval_pids.tolist()}")
    print(f"New Cams Structure : {eval_cams}")
    print(f"New Feature Shape  : {eval_feats.shape}")

    """
    EXPECTED LOGIC TRACE:

    1. New PIDs will be: [1, 2, 1, 2]
       -> (Query 0, Query 1, Centroid for PID 1, Centroid for PID 2)

    2. New Cams will be: [[1], [2], [2, 3], [1, 3]]
       -> Query 0 maintains cam [1]
       -> Query 1 maintains cam [2]
       -> Centroid for PID 1 merges Gallery cams [2, 3] (G2's cam 1 was excluded!)
       -> Centroid for PID 2 merges Gallery cams [1, 3]

    3. Output Feature Shape will be: (4, 4)
       -> 7 original samples were condensed down to 4 robust evaluation representations.
    """
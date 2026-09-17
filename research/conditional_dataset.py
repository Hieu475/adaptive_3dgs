"""Phase 12-J: Conditional Counterfactual Dataset Generation.

Generates the training dataset for the interaction-aware utility model (Phase 14).
Each record contains:
    (scene, t, S, i, s_i, h_S, ΔQ, ΔC, U)

where h_S is a permutation-invariant representation of the selected set S.
"""
import os
import json
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import csv

from .oracle_utility import OracleUtilityExperiment
from .phase6_oracle import ConditionalOracleExperiment, ConditionalOracleConfig
from .phase6_context import (
    build_full_context,
    _get_pixel_mask,
    _get_knn_indices,
    ContextConfig,
    PHASE6_FEATURE_NAMES,
)
from .utility_features import CANONICAL_FEATURE_NAMES, extract_feature_vector
from .phase12_protocol import PHASE12_OUTPUT_DIR, SEEDS

H_S_DIM = 51  # Total dimension of h_S

@dataclass 
class ConditionalDatasetConfig:
    """Configuration for Phase 12-J dataset generation."""
    n_candidates_per_frame: int = 30
    context_sizes: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    context_types: List[str] = field(default_factory=lambda: [
        'empty', 'spatial_knn', 'overlap_top', 'random', 'high_utility', 'mixed'
    ])
    n_opt_steps: int = 5
    min_influence_pixels: int = 25
    contribution_threshold: float = 0.01
    epsilon: float = 1e-6
    seed: int = 42
    output_dir: str = ''


class ConditionalDatasetGenerator:
    """Phase 12-J: Generates conditional counterfactual dataset."""
    
    def __init__(self, pipeline, config: Optional[ConditionalDatasetConfig] = None):
        self.pipeline = pipeline
        self.config = config or ConditionalDatasetConfig()
        
        # Initialize oracles
        self.cond_config = ConditionalOracleConfig(
            context_sizes=self.config.context_sizes,
            context_types=self.config.context_types,
            n_opt_steps=self.config.n_opt_steps,
            contribution_threshold=self.config.contribution_threshold,
        )
        self.cond_oracle = ConditionalOracleExperiment(self.pipeline, self.cond_config)
        self.oracle = getattr(self.cond_oracle, 'oracle_engine', getattr(self.cond_oracle, 'oracle', None))
        
        self.output_dir = Path(self.config.output_dir) if self.config.output_dir else Path(PHASE12_OUTPUT_DIR) / "phase12j_dataset"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_h_S(
        self,
        candidate_idx: int,
        context_indices: List[int],
        all_features: np.ndarray,
        positions: torch.Tensor,
        contrib_indices: torch.Tensor,
        contrib_weights: torch.Tensor,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Compute permutation-invariant context representation h_S.
        
        Components:
        1. Aggregate statistics of S members' features (44 dims):
           mean, std, max, min of 11 canonical features
        2. Interaction descriptors (5 dims):
           mean_iou, max_iou, mean_depth_conflict, alpha_competition, attribution_redundancy
        3. Set metadata (2 dims):
           |S|, budget_fraction
           
        Returns:
            h_S as np.ndarray of shape (51,)
        """
        if not context_indices:
            return np.zeros(H_S_DIM, dtype=np.float32)
            
        # 1. Aggregate statistics (44 dims)
        S_features = all_features[context_indices]  # (|S|, 11)
        mean_feat = np.mean(S_features, axis=0)
        std_feat = np.std(S_features, axis=0)
        max_feat = np.max(S_features, axis=0)
        min_feat = np.min(S_features, axis=0)
        agg_stats = np.concatenate([mean_feat, std_feat, max_feat, min_feat])  # 44 dims
        
        # 2. Interaction descriptors (5 dims)
        cand_mask = _get_pixel_mask(candidate_idx, contrib_indices, contrib_weights, self.config.contribution_threshold)
        cand_pixels = cand_mask.sum().item()
        
        ious = []
        depth_conflicts = []
        S_union_mask = torch.zeros_like(cand_mask, dtype=torch.bool)
        
        cand_pos = positions[candidate_idx]
        
        for j in context_indices:
            j_mask = _get_pixel_mask(j, contrib_indices, contrib_weights, self.config.contribution_threshold)
            intersection = (cand_mask & j_mask).sum().item()
            union = (cand_mask | j_mask).sum().item()
            iou = intersection / max(union, 1)
            ious.append(iou)
            
            S_union_mask |= j_mask
            
            j_pos = positions[j]
            depth_conflict = torch.abs(cand_pos[2] - j_pos[2]).item()
            depth_conflicts.append(depth_conflict)
            
        mean_iou = np.mean(ious) if ious else 0.0
        max_iou = np.max(ious) if ious else 0.0
        mean_depth_conflict = np.mean(depth_conflicts) if depth_conflicts else 0.0
        alpha_competition = mean_iou  # As specified
        
        redundant_pixels = (cand_mask & S_union_mask).sum().item()
        attribution_redundancy = redundant_pixels / max(cand_pixels, 1)
        
        inter_desc = np.array([
            mean_iou,
            max_iou,
            mean_depth_conflict,
            alpha_competition,
            attribution_redundancy
        ], dtype=np.float32)
        
        # 3. Set metadata (2 dims)
        set_size = len(context_indices)
        budget_fraction = set_size / 200.0  # arbitrary normalization, assuming max budget 200
        meta = np.array([set_size, budget_fraction], dtype=np.float32)
        
        h_S = np.concatenate([agg_stats, inter_desc, meta])
        return h_S
    
    def generate_frame_dataset(
        self,
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor,
        all_features: np.ndarray,
        scene_name: str,
        frame_idx: int,
        split: str = 'train',
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generate conditional dataset for a single frame.
        
        For each candidate i, for each context condition (type, size):
        1. Sample context set S
        2. Compute h_S (permutation-invariant representation)
        3. Measure ΔQ(i|S), ΔC(i|S) via oracle
        4. Compute U(i|S) = ΔQ / (ΔC + ε)
        5. Record (scene, t, S, i, s_i, h_S, ΔQ, ΔC, U)
        """
        # Fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 1. Base render with attribution
        render_dict = self.oracle._render_with_attribution()
        contrib_indices = render_dict["contrib_indices"]
        contrib_weights = render_dict["contrib_weights"]
        h, w = rgb_gt.shape[1], rgb_gt.shape[2]
        
        # Find candidates
        inf_mask = self.oracle._get_influence_mask(contrib_indices, contrib_weights)
        visible_indices = torch.nonzero(inf_mask).squeeze(1).cpu().numpy()
        
        if len(visible_indices) < self.config.n_candidates_per_frame:
            candidates = visible_indices
        else:
            candidates = np.random.choice(visible_indices, self.config.n_candidates_per_frame, replace=False)
            
        # We need positions for depth conflict
        positions = self.pipeline.model.get_xyz.detach()
        
        records = []
        
        for cand_idx in candidates:
            # Get local features s_i
            s_i = all_features[cand_idx].tolist()
            
            # Baseline quality
            cand_mask = _get_pixel_mask(cand_idx, contrib_indices, contrib_weights, self.config.contribution_threshold)
            if cand_mask.sum().item() < self.config.min_influence_pixels:
                continue
                
            # Sample contexts
            context_schedule = self.cond_oracle._build_context_schedule(
                int(cand_idx),
                visible_indices.tolist(),
                contrib_indices,
                contrib_weights,
                positions,
                h, w
            )
            
            # First, evaluate empty context to get delta_q_single
            empty_res = self.cond_oracle.measure_conditional_utility(
                int(cand_idx), [], rgb_gt, depth_gt
            )
            delta_q_single = empty_res["delta_q"]
            utility_single = delta_q_single / (empty_res["delta_c_ms"] + self.config.epsilon)
            
            for ctx in context_schedule:
                ctx_type = ctx["type"]
                S = ctx["indices"]
                ctx_size = len(S)
                
                # compute h_S
                h_S = self.compute_h_S(
                    int(cand_idx), S, all_features, positions, contrib_indices, contrib_weights, h, w
                )
                
                # Measure delta Q(i|S)
                res = self.cond_oracle.measure_conditional_utility(
                    int(cand_idx), S, rgb_gt, depth_gt
                )
                
                delta_q = res["delta_q"]
                delta_c_ms = res["delta_c_ms"]
                utility = delta_q / (delta_c_ms + self.config.epsilon)
                
                # Compute descriptors again for logging
                if S:
                    ious = []
                    depth_confs = []
                    S_union = torch.zeros_like(cand_mask)
                    cand_pos = positions[cand_idx]
                    for j in S:
                        j_mask = _get_pixel_mask(j, contrib_indices, contrib_weights, self.config.contribution_threshold)
                        intersection = (cand_mask & j_mask).sum().item()
                        union = (cand_mask | j_mask).sum().item()
                        ious.append(intersection / max(union, 1))
                        depth_confs.append(torch.abs(cand_pos[2] - positions[j][2]).item())
                        S_union |= j_mask
                    
                    mean_iou = float(np.mean(ious))
                    max_iou = float(np.max(ious))
                    mean_dc = float(np.mean(depth_confs))
                    attr_red = float((cand_mask & S_union).sum().item() / max(cand_mask.sum().item(), 1))
                else:
                    mean_iou = max_iou = mean_dc = attr_red = 0.0
                
                record = {
                    'scene': scene_name,
                    'frame': frame_idx,
                    'split': split,
                    'seed': seed,
                    'candidate_idx': int(cand_idx),
                    'context_indices': S,
                    'context_type': ctx_type,
                    'context_size': ctx_size,
                    's_i': s_i,
                    'h_S': h_S.tolist(),
                    'delta_q': float(delta_q),
                    'delta_c_ms': float(delta_c_ms),
                    'utility': float(utility),
                    'delta_q_single': float(delta_q_single),
                    'utility_single': float(utility_single),
                    'interaction_descriptors': {
                        'mean_iou': mean_iou,
                        'max_iou': max_iou,
                        'mean_depth_conflict': mean_dc,
                        'attribution_redundancy': attr_red,
                    },
                    'q_baseline_psnr': float(res.get("q_baseline_psnr", 0.0)),
                    'q_after_psnr': float(res.get("q_after_psnr", 0.0)),
                }
                records.append(record)
                
        return records
    
    def save_dataset(
        self,
        records: List[Dict],
        output_path: str,
        scene_name: str,
        split: str,
        seed: int,
    ) -> None:
        """Save dataset to JSON and CSV."""
        if not records:
            return
            
        out_path = Path(output_path)
        out_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        base_name = f"phase12j_{scene_name}_{split}_s{seed}_{timestamp}"
        
        json_path = out_path / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2)
            
        csv_path = out_path / f"{base_name}.csv"
        # Flatten structure for CSV
        csv_keys = [
            'scene', 'frame', 'split', 'seed', 'candidate_idx', 'context_type', 'context_size',
            'delta_q', 'delta_c_ms', 'utility', 'delta_q_single', 'utility_single',
            'q_baseline_psnr', 'q_after_psnr'
        ]
        # Adding interaction descriptors
        desc_keys = list(records[0]['interaction_descriptors'].keys())
        csv_keys.extend([f"desc_{k}" for k in desc_keys])
        
        # We don't save s_i and h_S to CSV to keep it lightweight, or we can if we want
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys)
            writer.writeheader()
            for r in records:
                row = {k: r[k] for k in csv_keys if k in r and not k.startswith("desc_")}
                for k in desc_keys:
                    row[f"desc_{k}"] = r['interaction_descriptors'][k]
                writer.writerow(row)

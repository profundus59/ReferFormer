#!/usr/bin/env python3
"""
Evaluate GroundingDINO bounding box predictions against ground truth masks.

This script:
1. Loads GT masks from dataset_visor/Annotations_Sparse
2. Computes GT bounding boxes using the same method as ActionVOSDataset
3. Loads GroundingDINO predictions from prediction folders
4. Computes IoU metrics and outputs evaluation results
"""

"""
HOW TO RUN:

# Evaluate training set
python evaluate_groundingdino_bbox.py --split train --output_json train_results.json

# Evaluate validation set  
python evaluate_groundingdino_bbox.py --split val --output_json val_results.json
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import re


def bounding_box_from_mask(mask_array):
    """
    Compute bounding box from binary mask (same as ActionVOSDataset.bounding_box).
    
    Args:
        mask_array: numpy array (H, W) with binary values
    
    Returns:
        box: [x1, y1, x2, y2] in xyxy format, or [0, 0, 0, 0] if empty
    """
    if not (mask_array > 0).any():
        return np.array([0, 0, 0, 0], dtype=np.float32)
    
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # return as [x1, y1, x2, y2] (xyxy format)
    return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in xyxy format.
    
    Args:
        box1, box2: arrays of shape [4] with [x1, y1, x2, y2]
    
    Returns:
        iou: float
    """
    # Compute intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Compute union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def box_iou_batch(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes.
    
    Args:
        boxes1: numpy array [N, 4] in xyxy format
        boxes2: numpy array [M, 4] in xyxy format
    
    Returns:
        iou_matrix: numpy array [N, M]
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    # Expand dims for broadcasting
    boxes1 = boxes1[:, np.newaxis, :]  # [N, 1, 4]
    boxes2 = boxes2[np.newaxis, :, :]  # [1, M, 4]
    
    # Compute intersection
    x1_inter = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1_inter = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2_inter = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2_inter = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Compute union
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = boxes1_area + boxes2_area - inter_area
    
    iou = inter_area / (union_area + 1e-8)
    return iou.squeeze()


def load_gt_boxes_for_frame(mask_path, obj_id):
    """
    Load GT mask and compute bounding box for a specific object.
    
    Args:
        mask_path: path to mask PNG file
        obj_id: object ID to extract from the mask
    
    Returns:
        box: [x1, y1, x2, y2] in xyxy format
    """
    if not os.path.exists(mask_path):
        return None
    
    mask = Image.open(mask_path).convert('P')
    mask = np.array(mask)
    
    # Extract binary mask for this object
    binary_mask = (mask == obj_id).astype(np.float32)
    
    # Compute bounding box
    box = bounding_box_from_mask(binary_mask)
    
    return box


def load_prediction_boxes(pred_path):
    """
    Load predicted bounding boxes from GroundingDINO output.
    
    Args:
        pred_path: path to prediction .pt file
    
    Returns:
        boxes: numpy array [N, 4] in xyxy format (pixel coordinates)
        scores: numpy array [N] with confidence scores (parsed from labels)
        labels: list of label strings
    """
    if not os.path.exists(pred_path):
        return None, None, None
    
    pred_data = torch.load(pred_path, map_location='cpu')
    
    # Handle GroundingDINO format
    if isinstance(pred_data, dict):
        boxes = pred_data.get('boxes', pred_data.get('bboxes', None))
        labels = pred_data.get('labels', [])
        
        # Parse scores from labels (format: "class_name(0.XX)")
        scores = []
        if labels:
            for label in labels:
                if '(' in label and ')' in label:
                    score_str = label.split('(')[1].split(')')[0]
                    scores.append(float(score_str))
                else:
                    scores.append(1.0)
            scores = np.array(scores)
        else:
            scores = pred_data.get('scores', pred_data.get('confidences', None))
    elif isinstance(pred_data, (list, tuple)):
        boxes = pred_data[0] if len(pred_data) > 0 else None
        scores = pred_data[1] if len(pred_data) > 1 else None
        labels = pred_data[2] if len(pred_data) > 2 else []
    else:
        boxes = pred_data
        scores = None
        labels = []
    
    if boxes is not None:
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        boxes = np.array(boxes)
    
    if scores is not None and not isinstance(scores, np.ndarray):
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        scores = np.array(scores)
    
    return boxes, scores, labels


def extract_class_name(text):
    """
    Extract object class name from expression text or prediction label.
    
    Args:
        text: expression like "knife used in the action..." or label like "knife(0.53)"
    
    Returns:
        normalized class name (lowercase, simplified)
    """
    if not text:
        return "unknown"
    
    # Remove confidence score if present: "knife(0.53)" -> "knife"
    if '(' in text and ')' in text:
        text = text.split('(')[0].strip()
    
    # Take first word/phrase before "used in" or other keywords
    text = text.lower().strip()
    
    # Common patterns in expressions
    if 'used in the action' in text:
        text = text.split('used in the action')[0].strip()
    if 'involved in' in text:
        text = text.split('involved in')[0].strip()
    
    # Normalize common variations
    text = text.replace('board:cutting', 'cutting board')
    text = text.replace('_', ' ')
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text


def class_names_match(gt_class, pred_class):
    """
    Check if ground truth and predicted class names match.
    Handles variations like "hand" matching "left hand" or "right hand".
    
    Args:
        gt_class: ground truth class name (extracted from expression)
        pred_class: predicted class name (from model output)
    
    Returns:
        bool: True if classes match (with flexibility for variations)
    """
    gt_class = gt_class.lower().strip()
    pred_class = pred_class.lower().strip()
    
    # Exact match
    if gt_class == pred_class:
        return True
    
    # One contains the other (handles "hand" vs "left hand", "right hand")
    if gt_class in pred_class or pred_class in gt_class:
        return True
    
    # Check if they share significant keywords
    gt_words = set(gt_class.split())
    pred_words = set(pred_class.split())
    
    # Remove common articles/prepositions
    stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at'}
    gt_words -= stop_words
    pred_words -= stop_words
    
    # If they share at least one significant word (and both have words left)
    if gt_words and pred_words and len(gt_words & pred_words) > 0:
        return True
    
    # Common synonyms/variations
    synonyms = [
        {'tap', 'faucet'},
        {'cupboard', 'cabinet'},
        {'chopping board', 'cutting board'},
        {'refrigerator', 'fridge'},
        {'container', 'box'},
    ]
    
    for syn_set in synonyms:
        if gt_class in syn_set and pred_class in syn_set:
            return True
    
    return False


def evaluate_video(video_name, image_set, dataset_root, pred_root, meta_info, verbose=False):
    """
    Evaluate predictions for a single video.
    
    Args:
        video_name: name of the video
        image_set: 'train' or 'val'
        dataset_root: root path of dataset_visor
        pred_root: root path of prediction folder
        meta_info: dict with video metadata (from expressions JSON)
        verbose: whether to print per-frame results
    
    Returns:
        results: dict with evaluation metrics
    """
    results = {
        'ious': [],
        'matched': 0,
        'total_gt': 0,
        'total_pred': 0,
        'missing_predictions': 0,
        'missing_gt': 0,
        'class_correct': 0,
        'class_total': 0,
        'class_correct_high_iou': 0,  # class correct AND IoU >= 0.5
        'class_correct_low_iou': 0,   # class correct BUT IoU < 0.5
        'class_wrong_high_iou': 0,    # class wrong BUT IoU >= 0.5
    }
    
    frames = sorted(meta_info['frames'])
    expressions = meta_info['expressions']
    
    for exp_id, exp_dict in expressions.items():
        obj_id = int(exp_dict['obj_id'])
        positive = exp_dict.get('positive', True)
        exp_text = exp_dict.get('exp', '')
        
        # Skip negative samples
        if not positive:
            continue
        
        # Extract GT class name from expression
        gt_class = extract_class_name(exp_text)
        
        for frame_name in frames:
            # GT mask path
            mask_path = os.path.join(
                dataset_root, 'Annotations_Sparse', image_set, 
                video_name, f"{frame_name}.png"
            )
            
            # Prediction path
            pred_path = os.path.join(
                pred_root, video_name, f"{frame_name}_bbx.pt"
            )
            
            # Load GT box
            gt_box = load_gt_boxes_for_frame(mask_path, obj_id)
            if gt_box is None:
                results['missing_gt'] += 1
                continue
            
            # Skip empty GT boxes
            if np.all(gt_box == 0):
                continue
            
            results['total_gt'] += 1
            
            # Load predicted boxes
            pred_boxes, pred_scores, pred_labels = load_prediction_boxes(pred_path)
            
            if pred_boxes is None or len(pred_boxes) == 0:
                results['missing_predictions'] += 1
                if verbose:
                    print(f"  {video_name}/{frame_name}: No predictions")
                continue
            
            results['total_pred'] += len(pred_boxes)
            
            # Compute IoU with all predicted boxes and take the best match
            ious = [compute_iou(gt_box, pred_box) for pred_box in pred_boxes]
            best_iou = max(ious)
            best_idx = np.argmax(ious)
            
            results['ious'].append(best_iou)
            if best_iou >= 0.5:
                results['matched'] += 1
            
            # Check class prediction accuracy
            pred_label = pred_labels[best_idx] if pred_labels and best_idx < len(pred_labels) else ""
            pred_class = extract_class_name(pred_label)
            class_match = class_names_match(gt_class, pred_class)
            
            results['class_total'] += 1
            if class_match:
                results['class_correct'] += 1
                if best_iou >= 0.5:
                    results['class_correct_high_iou'] += 1
                else:
                    results['class_correct_low_iou'] += 1
            else:
                if best_iou >= 0.5:
                    results['class_wrong_high_iou'] += 1
            
            if verbose:
                class_status = "✓" if class_match else "✗"
                print(f"  {video_name}/{frame_name} obj_{obj_id}: "
                      f"IoU={best_iou:.3f}, Class={class_status}, "
                      f"GT_class='{gt_class}', Pred_class='{pred_class}' ({pred_label})")
    
    return results


def evaluate_dataset(image_set='train', 
                     dataset_root='/home/jihun/workspace/repositories/vos_task/actionvos/dataset_visor',
                     pred_root=None,
                     expression_file='train_meta_expressions_promptaction.json',
                     verbose=False):
    """
    Evaluate all videos in a dataset split.
    
    Args:
        image_set: 'train' or 'val'
        dataset_root: root path of dataset_visor
        pred_root: root path of prediction folder
        expression_file: JSON file with expressions
        verbose: whether to print detailed per-video results
    
    Returns:
        metrics: dict with overall metrics
    """
    if pred_root is None:
        pred_root = f'/nas_data2/jihun/actionvos_seim/groundingdino/dataset_visor_{image_set}_bbox'
    
    # Load metadata
    meta_path = os.path.join(dataset_root, 'ImageSets', expression_file)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    videos = metadata['videos']
    
    print(f"\n{'='*60}")
    print(f"Evaluating {image_set.upper()} set")
    print(f"Dataset root: {dataset_root}")
    print(f"Prediction root: {pred_root}")
    print(f"Total videos: {len(videos)}")
    print(f"{'='*60}\n")
    
    # Aggregate results
    all_ious = []
    total_matched = 0
    total_gt = 0
    total_pred = 0
    missing_predictions = 0
    missing_gt = 0
    class_correct = 0
    class_total = 0
    class_correct_high_iou = 0
    class_correct_low_iou = 0
    class_wrong_high_iou = 0
    
    # Evaluate each video
    for video_name, video_meta in tqdm(videos.items(), desc=f"Evaluating {image_set}"):
        video_results = evaluate_video(
            video_name, image_set, dataset_root, pred_root, 
            video_meta, verbose=verbose
        )
        
        all_ious.extend(video_results['ious'])
        total_matched += video_results['matched']
        total_gt += video_results['total_gt']
        total_pred += video_results['total_pred']
        missing_predictions += video_results['missing_predictions']
        missing_gt += video_results['missing_gt']
        class_correct += video_results.get('class_correct', 0)
        class_total += video_results.get('class_total', 0)
        class_correct_high_iou += video_results.get('class_correct_high_iou', 0)
        class_correct_low_iou += video_results.get('class_correct_low_iou', 0)
        class_wrong_high_iou += video_results.get('class_wrong_high_iou', 0)
    
    # Compute metrics
    all_ious = np.array(all_ious)
    
    metrics = {
        'mean_iou': float(np.mean(all_ious)) if len(all_ious) > 0 else 0.0,
        'median_iou': float(np.median(all_ious)) if len(all_ious) > 0 else 0.0,
        'iou@0.5': float(np.mean(all_ious >= 0.5)) if len(all_ious) > 0 else 0.0,
        'iou@0.75': float(np.mean(all_ious >= 0.75)) if len(all_ious) > 0 else 0.0,
        'total_gt_boxes': total_gt,
        'total_pred_boxes': total_pred,
        'matched_boxes': total_matched,
        'missing_predictions': missing_predictions,
        'missing_gt': missing_gt,
        'num_evaluated': len(all_ious),
        'class_accuracy': float(class_correct / class_total) if class_total > 0 else 0.0,
        'class_correct': class_correct,
        'class_total': class_total,
        'class_correct_high_iou': class_correct_high_iou,
        'class_correct_low_iou': class_correct_low_iou,
        'class_wrong_high_iou': class_wrong_high_iou,
    }
    
    return metrics, all_ious


def print_metrics(metrics, image_set):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Results for {image_set.upper()} set")
    print(f"{'='*60}")
    print(f"Mean IoU:              {metrics['mean_iou']:.4f}")
    print(f"Median IoU:            {metrics['median_iou']:.4f}")
    print(f"IoU >= 0.5:            {metrics['iou@0.5']:.2%}")
    print(f"IoU >= 0.75:           {metrics['iou@0.75']:.2%}")
    print(f"")
    print(f"Classification Accuracy: {metrics['class_accuracy']:.2%} ({metrics['class_correct']}/{metrics['class_total']})")
    print(f"  Class correct + High IoU (>=0.5): {metrics['class_correct_high_iou']}")
    print(f"  Class correct + Low IoU (<0.5):   {metrics['class_correct_low_iou']}")
    print(f"  Class wrong + High IoU (>=0.5):   {metrics['class_wrong_high_iou']}")
    print(f"")
    print(f"Total GT boxes:        {metrics['total_gt_boxes']}")
    print(f"Total pred boxes:      {metrics['total_pred_boxes']}")
    print(f"Matched (IoU>=0.5):    {metrics['matched_boxes']}")
    print(f"Missing predictions:   {metrics['missing_predictions']}")
    print(f"Missing GT:            {metrics['missing_gt']}")
    print(f"Evaluated frames:      {metrics['num_evaluated']}")
    print(f"{'='*60}\n")


def inspect_sample_mask(dataset_root='/home/jihun/workspace/repositories/vos_task/actionvos/dataset_visor',
                        image_set='train'):
    """
    Inspect a sample GT mask and show its computed bounding box.
    """
    print(f"\n{'='*60}")
    print("Inspecting Sample GT Mask")
    print(f"{'='*60}\n")
    
    # Find a sample mask
    ann_dir = os.path.join(dataset_root, 'Annotations_Sparse', image_set)
    videos = os.listdir(ann_dir)
    
    if len(videos) == 0:
        print(f"No videos found in {ann_dir}")
        return
    
    sample_video = videos[0]
    video_dir = os.path.join(ann_dir, sample_video)
    masks = [f for f in os.listdir(video_dir) if f.endswith('.png')]
    
    if len(masks) == 0:
        print(f"No masks found in {video_dir}")
        return
    
    sample_mask_name = masks[0]
    mask_path = os.path.join(video_dir, sample_mask_name)
    
    print(f"Video: {sample_video}")
    print(f"Frame: {sample_mask_name}")
    print(f"Path: {mask_path}\n")
    
    # Load mask
    mask = Image.open(mask_path).convert('P')
    mask_array = np.array(mask)
    
    print(f"Mask shape: {mask_array.shape}")
    print(f"Unique object IDs in mask: {np.unique(mask_array)}\n")
    
    # Compute bounding box for each object
    unique_ids = np.unique(mask_array)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude background (0)
    
    for obj_id in unique_ids:
        binary_mask = (mask_array == obj_id).astype(np.float32)
        bbox = bounding_box_from_mask(binary_mask)
        
        # Count pixels
        pixel_count = int(np.sum(binary_mask))
        
        print(f"Object ID {obj_id}:")
        print(f"  Bounding box (xyxy): [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"  Box size: {bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f} pixels")
        print(f"  Mask pixels: {pixel_count}")
        print()
    
    print(f"{'='*60}\n")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate GroundingDINO bounding box predictions')
    parser.add_argument('--dataset_root', type=str, 
                       default='/home/jihun/workspace/repositories/vos_task/actionvos/dataset_visor',
                       help='Root directory of dataset_visor')
    parser.add_argument('--pred_root_train', type=str,
                       default='/nas_data2/jihun/actionvos_seim/groundingdino/dataset_visor_train_bbox',
                       help='Root directory of training predictions')
    parser.add_argument('--pred_root_val', type=str,
                       default='/nas_data2/jihun/actionvos_seim/groundingdino/dataset_visor_val_bbox',
                       help='Root directory of validation predictions')
    parser.add_argument('--train_expression_file', type=str,
                       default='train_meta_expressions_promptaction.json',
                       help='Expression file for training set')
    parser.add_argument('--val_expression_file', type=str,
                       default='val_meta_expressions_promptaction.json',
                       help='Expression file for validation set')
    parser.add_argument('--split', type=str, default='both', choices=['train', 'val', 'both'],
                       help='Which split to evaluate')
    parser.add_argument('--inspect_sample', action='store_true',
                       help='Inspect a sample GT mask before evaluation')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed per-frame results')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Inspect sample mask if requested
    if args.inspect_sample:
        inspect_sample_mask(args.dataset_root, 'train')
    
    results = {}
    
    # Evaluate training set
    if args.split in ['train', 'both']:
        train_metrics, train_ious = evaluate_dataset(
            image_set='train',
            dataset_root=args.dataset_root,
            pred_root=args.pred_root_train,
            expression_file=args.train_expression_file,
            verbose=args.verbose
        )
        print_metrics(train_metrics, 'train')
        results['train'] = train_metrics
    
    # Evaluate validation set
    if args.split in ['val', 'both']:
        val_metrics, val_ious = evaluate_dataset(
            image_set='val',
            dataset_root=args.dataset_root,
            pred_root=args.pred_root_val,
            expression_file=args.val_expression_file,
            verbose=args.verbose
        )
        print_metrics(val_metrics, 'val')
        results['val'] = val_metrics
    
    # Save results to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()

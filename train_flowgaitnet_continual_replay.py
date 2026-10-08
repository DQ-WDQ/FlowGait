import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from data.rds_experiments import RDSExperimentDataset
from tqdm import tqdm
from torch.optim import lr_scheduler
import copy
from model_img.flowgait_net import CenterLoss
from torch.utils.data import ConcatDataset, Subset
import pandas as pd
import random  # Used to sample exemplars.
from collections import defaultdict  # Groups sample indices by identity label.
from training.engine import resolve_device

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from training.half_common import FineTunedFlowgaitNet, OnlineTripletLoss, PseudoLabeledDataset

# ==========================================================================================
# Select a bounded exemplar set for each identity.
# This keeps replay data bounded as new pseudo-labeled samples arrive.
def update_exemplar_set(current_exemplars, new_data, cfg, num_classes):
    """
    Merge the current exemplars with newly accepted pseudo-labeled samples,
    then cap the number of stored samples per class.
    
    Args:
        current_exemplars (Dataset): The current exemplar set.
        new_data (Dataset): Newly accepted pseudo-labeled samples.
        cfg (argparse.Namespace): Configuration containing exemplars_per_class.
        num_classes (int): Total number of classes in the dataset.

    Returns:
        Dataset: The updated exemplar set as a Subset.
    """
    print("\n--- Updating exemplar set ---")
    
    # 1. Combine the current exemplars and new samples.
    combined_dataset = ConcatDataset([current_exemplars, new_data])
    print(f"Combined temporary dataset size: {len(combined_dataset)}")

    # 2. Group sample indices by class. This runs once per learning step.
    class_indices = defaultdict(list)
    for i in tqdm(range(len(combined_dataset)), desc="Grouping samples by class"):
        # `__getitem__` returns (image, (identity_label, action_label)).
        # The identity label is stored at labels[0].
        _, labels = combined_dataset[i]
        class_id = labels[0].item()
        class_indices[class_id].append(i)

    # 3. Randomly sample up to k exemplars per class.
    k_per_class = cfg.exemplars_per_class
    all_selected_indices = []
    
    for class_id in sorted(class_indices.keys()):
        indices = class_indices[class_id]
        # Keep all samples when a class has fewer than k examples.
        num_to_sample = min(k_per_class, len(indices))
        selected = random.sample(indices, num_to_sample)
        all_selected_indices.extend(selected)

    print(f"Total classes seen so far: {len(class_indices)}")
    print(f"Target exemplars per class: {k_per_class}")
    print(f"New exemplar set size: {len(all_selected_indices)}")

    # 4. Build the new exemplar set from the selected indices.
    new_exemplar_dataset = Subset(combined_dataset, all_selected_indices)
    
    return new_exemplar_dataset
# ==========================================================================================







def train_model(model, criterion, criterion_triplet, criterion_center, optimizer, scheduler, 
                labeled_loader, unlabeled_loader, val_loader, cfg, device, 
                epochs_per_step, model_save_path):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    id_loss_weight = cfg.id_loss_weight
    action_loss_weight = cfg.action_loss_weight
    pseudo_label_weight = cfg.pseudo_label_weight
    confidence_threshold = cfg.confidence_threshold

    for epoch in range(epochs_per_step):
        print(f"--- [Step Epoch {epoch+1}/{epochs_per_step}] Semi-Supervised Training ---")
        
        # Generate pseudo-labels for the current learning step.
        print("Generating pseudo-labels for the current step...")
        model.eval()
        
        all_preds, all_probs, all_routes, all_original_indices = [], [], [], []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(unlabeled_loader, desc="Predicting on Unlabeled Data", leave=False)):
                inputs = inputs.to(device)
                routes = labels[2] 
                
                id_outputs, _ = model(inputs)
                probs = F.softmax(id_outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(max_probs.cpu().numpy())
                all_routes.extend(routes.cpu().numpy())
                
                batch_indices = unlabeled_loader.dataset.indices[i * unlabeled_loader.batch_size : i * unlabeled_loader.batch_size + len(inputs)]
                all_original_indices.extend(batch_indices)

        df = pd.DataFrame({
            'original_idx': all_original_indices,
            'route': all_routes,
            'pred': all_preds,
            'prob': all_probs,
            'pseudo_label': -1
        })

        high_conf_mask = df['prob'] >= confidence_threshold
        df.loc[high_conf_mask, 'pseudo_label'] = df.loc[high_conf_mask, 'pred']
        print(f"Initially generated {high_conf_mask.sum()} pseudo-labels based on confidence threshold.")
        
        propagated_labels_count = 0
        for route_id, group_df in tqdm(df.groupby('route'), desc="Applying Route Logic", leave=False):
            total_samples = len(group_df)
            if total_samples == 0: continue
            labeled_in_group = group_df[group_df['pseudo_label'] != -1]
            num_labeled = len(labeled_in_group)
            if num_labeled / total_samples >= cfg.proportion_threshold:
                if num_labeled > 0 and labeled_in_group['pseudo_label'].nunique() == 1:
                    common_label = labeled_in_group['pseudo_label'].iloc[0]
                    unlabeled_indices_in_group = group_df[group_df['pseudo_label'] == -1].index
                    if not unlabeled_indices_in_group.empty:
                        df.loc[unlabeled_indices_in_group, 'pseudo_label'] = common_label
                        propagated_labels_count += len(unlabeled_indices_in_group)

        if propagated_labels_count > 0:
            print(f"Propagated labels to {propagated_labels_count} additional samples using route logic.")

        final_pseudo_labeled_df = df[df['pseudo_label'] != -1]
        all_pseudo_labels = torch.tensor(final_pseudo_labeled_df['pseudo_label'].values, dtype=torch.long)
        all_unlabeled_data_indices = final_pseudo_labeled_df['original_idx'].tolist()

        print(f"Total pseudo-labels for this epoch: {len(all_pseudo_labels)}")
        
        if len(all_pseudo_labels) > 0:
            selected_unlabeled_subset = data.Subset(unlabeled_loader.dataset.dataset, all_unlabeled_data_indices)
            pseudo_dataset = PseudoLabeledDataset(selected_unlabeled_subset, all_pseudo_labels)
            # The labeled dataset is the current exemplar set.
            combined_dataset = ConcatDataset([labeled_loader.dataset, pseudo_dataset])
            print(f"Combined dataset size for this epoch: {len(combined_dataset)}")
        else:
            combined_dataset = labeled_loader.dataset
            print("No pseudo-labels generated, using labeled data only for this epoch.")
        
        train_loader = data.DataLoader(combined_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        # Train and evaluate the current model.
        model.train()
        running_loss, running_loss_id, running_loss_pseudo, running_loss_trip = 0.0, 0.0, 0.0, 0.0
        id_correct, total = 0, 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            optimizer.zero_grad()
            inputs, id_labels, action_labels = inputs.to(device), labels[0].to(device), labels[1].to(device)
            id_outputs, features = model(inputs)
            
            real_mask = action_labels != -1
            pseudo_mask = action_labels == -1
            loss = 0.0
            
            if torch.any(real_mask):
                loss_id_real = criterion(id_outputs[real_mask], id_labels[real_mask])
                loss_triplet = criterion_triplet(features[real_mask], id_labels[real_mask])
                loss += id_loss_weight * loss_id_real + action_loss_weight * loss_triplet
                running_loss_id += loss_id_real.item() * inputs[real_mask].size(0)
                running_loss_trip += loss_triplet.item() * inputs[real_mask].size(0)
            
            if torch.any(pseudo_mask):
                loss_id_pseudo = criterion(id_outputs[pseudo_mask], id_labels[pseudo_mask])
                loss += pseudo_label_weight * loss_id_pseudo
                running_loss_pseudo += loss_id_pseudo.item() * inputs[pseudo_mask].size(0)
            
            loss.backward()
            optimizer.step()

            id_correct += (torch.max(id_outputs, 1)[1] == id_labels).sum().item()
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / total if total > 0 else 0
        id_acc = id_correct / total if total > 0 else 0
        print(f"Train Loss: {epoch_loss:.4f}, Accuracy_id: {id_acc:.4f}")

        model.eval()
        print("Validation with route-based logic:")
        all_val_preds, all_val_probs, all_val_routes, all_val_labels = [], [], [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="1/2: Predicting on Validation Data", leave=False):
                inputs = inputs.to(device)
                true_labels = labels[0].cpu().numpy()
                routes = labels[2].cpu().numpy()
                
                id_outputs, _ = model(inputs)
                probs = F.softmax(id_outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(max_probs.cpu().numpy())
                all_val_routes.extend(routes)
                all_val_labels.extend(true_labels)

        val_df = pd.DataFrame({
            'true_label': all_val_labels,
            'route': all_val_routes,
            'pred': all_val_preds,
            'prob': all_val_probs,
        })
        val_df['final_pred'] = val_df['pred']
        
        propagated_predictions_count = 0
        for route_id, group_df in tqdm(val_df.groupby('route'), desc="2/2: Applying Route Logic to Val Data", leave=False):
            total_samples = len(group_df)
            if total_samples == 0: continue
            high_conf_group = group_df[group_df['prob'] >= confidence_threshold]
            num_high_conf = len(high_conf_group)
            if (num_high_conf / total_samples) >= cfg.proportion_threshold:
                if num_high_conf > 0 and high_conf_group['pred'].nunique() == 1:
                    common_label = high_conf_group['pred'].iloc[0]
                    val_df.loc[group_df.index, 'final_pred'] = common_label
                    propagated_predictions_count += (total_samples - num_high_conf)
        
        if propagated_predictions_count > 0:
            print(f"Propagated predictions to {propagated_predictions_count} samples in validation set.")

        initial_correct = (val_df['pred'] == val_df['true_label']).sum()
        final_correct = (val_df['final_pred'] == val_df['true_label']).sum()
        total_val_samples = len(val_df)
        initial_val_acc = initial_correct / total_val_samples
        final_val_acc = final_correct / total_val_samples
        
        print(f"Validation Accuracy (Initial): {initial_val_acc:.4f}")
        print(f"Validation Accuracy (After Route Logic): {final_val_acc:.4f}")
        
        val_id_acc = final_val_acc

        if val_id_acc > best_acc:
            best_acc = val_id_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)
            print(f"New best model for this step saved to {model_save_path} with accuracy: {best_acc:.4f}")
            
    print(f"Finished step. Best validation accuracy: {best_acc:.4f}")


def promote_pseudo_labels(model, device, unlabeled_dataset_step, cfg):
    print("\n--- Promoting high-confidence pseudo-labels to the trusted dataset ---")
    model.eval()
    loader = data.DataLoader(unlabeled_dataset_step, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    all_preds, all_probs = [], []
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Final Prediction for Promotion"):
            inputs = inputs.to(device)
            id_outputs, _ = model(inputs)
            probs = F.softmax(id_outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.cpu())
            all_probs.extend(max_probs.cpu())
            
    all_preds = torch.stack(all_preds)
    all_probs = torch.stack(all_probs)
    
    high_conf_indices_mask = all_probs >= cfg.promotion_threshold
    
    original_indices = torch.tensor(unlabeled_dataset_step.indices)
    promoted_original_indices = original_indices[high_conf_indices_mask].tolist()
    
    promoted_pseudo_labels = all_preds[high_conf_indices_mask]
    
    if len(promoted_original_indices) > 0:
        print(f"Promoting {len(promoted_original_indices)} samples with confidence >= {cfg.promotion_threshold}")
        promoted_subset = Subset(unlabeled_dataset_step.dataset, promoted_original_indices)
        promoted_dataset = PseudoLabeledDataset(promoted_subset, promoted_pseudo_labels)
        return promoted_dataset
    else:
        print("No samples met the promotion threshold in this step.")
        return None


def main(cfg):
    device = resolve_device(cfg.device)
    device_id = device.index or 0
    cfg.save_dir = cfg.save_dir.expanduser().resolve()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    transform_train = transforms.Compose([transforms.ToTensor()])

    # --- 1. Initialize the model and load an optional checkpoint ---
    model = FineTunedFlowgaitNet(image_size=(11,220), patch_size=11, num_classes=cfg.num_classes, 
                             dim=256, depth=5, heads=8, mlp_dim=256, device_id=device_id,
                             channels=1, dropout=0.1, emb_dropout=0.1)
    
    if cfg.pretrained_checkpoint is not None:
        pretrained_path = cfg.pretrained_checkpoint.expanduser().resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "mlp_head" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Pretrained backbone weights loaded.")
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    criterion_triplet = OnlineTripletLoss(margin=0.3).to(device)
    criterion_center = CenterLoss(num_classes=cfg.num_classes, feat_dim=cfg.feat_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 2. Prepare datasets ---
    initial_labeled_dataset = RDSExperimentDataset(
        root_dir=cfg.train_dir,
        actions=['1', '2'],
        transform=transform_train,
        preload="sequential",
    )
    full_unlabeled_dataset = RDSExperimentDataset(
        root_dir=cfg.unlabeled_dir,
        actions=['3', '4', '5', '6', '7'],
        transform=transform_train,
        action_classes=10,
        preload="parallel",
        include_route=True,
    )
    val_dataset = RDSExperimentDataset(
        root_dir=cfg.valid_dir,
        actions=['8'],
        transform=transform_train,
        action_classes=10,
        preload="parallel",
        include_route=True,
    )
    val_loader = data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, 
                                 drop_last=True, num_workers=cfg.num_workers)

    # --- 3. Incremental learning with exemplar replay ---
    actions_to_learn = ['3','4','5','6','7']
    
    # Initialize the exemplar set with the labeled training data.
    exemplar_dataset = initial_labeled_dataset 

    for i, action in enumerate(actions_to_learn):
        print("\n" + "="*50)
        print(f"GRADUAL LEARNING STEP {i+1}/{len(actions_to_learn)}: Focusing on Action '{action}'")
        print("="*50)
        
        # Prepare data loaders for this step.
        # The labeled loader uses the bounded exemplar set.
        print(f"Current exemplar set size: {len(exemplar_dataset)}")
        labeled_loader_step = data.DataLoader(exemplar_dataset, batch_size=cfg.batch_size, 
                                              shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        # Select unlabeled samples assigned to this action.
        unlabeled_indices_step = [
            idx for idx, path in enumerate(full_unlabeled_dataset.img_paths)
            if path.split(os.sep)[-2] == action
        ]
        
        if not unlabeled_indices_step:
            print(f"Warning: No samples found for action '{action}'. Skipping this step.")
            continue
            
        unlabeled_subset_step = Subset(full_unlabeled_dataset, unlabeled_indices_step)
        print(f"Unlabeled samples for this step: {len(unlabeled_subset_step)}")
        unlabeled_loader_step = data.DataLoader(unlabeled_subset_step, batch_size=cfg.batch_size, 
                                                shuffle=False, num_workers=cfg.num_workers)
        
        # Train on this action step.
        step_model_path = cfg.save_dir / f"best_model_step_{action}.pth"
        train_model(model, criterion, criterion_triplet, criterion_center, optimizer, scheduler,
                    labeled_loader_step, unlabeled_loader_step, val_loader, cfg, device,
                    epochs_per_step=cfg.epochs_per_step, model_save_path=step_model_path)
        
        # Update the model and exemplar set after this step.
        print(f"Loading best model from step '{action}' to promote labels and update exemplars...")
        if step_model_path.exists():
             best_step_weights = torch.load(step_model_path, map_location=device)
             model.load_state_dict(best_step_weights)
        else:
            print(f"Warning: Model for step {action} not found. Skipping exemplar update.")
            continue

        newly_promoted_dataset = promote_pseudo_labels(model, device, unlabeled_subset_step, cfg)
        
        if newly_promoted_dataset:
            # Keep the exemplar set bounded instead of accumulating every sample.
            exemplar_dataset = update_exemplar_set(
                exemplar_dataset, 
                newly_promoted_dataset, 
                cfg, 
                cfg.num_classes
            )

    print("\n" + "="*50)
    print("GRADUAL LEARNING PROCESS COMPLETED!")
    print(f"Final exemplar set size: {len(exemplar_dataset)}")
    print("="*50)
    
    torch.save(model.state_dict(), cfg.save_dir / "final_model_exemplar_replay.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--unlabeled-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/half_continual"))
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--epochs-per-step", type=int, default=20)
    parser.add_argument("--promotion-threshold", type=float, default=0.95)
    parser.add_argument("--exemplars-per-class", type=int, default=300)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--proportion-threshold", type=float, default=0.6)
    parser.add_argument("--pseudo-label-weight", type=float, default=1.0)
    parser.add_argument("--id-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    main(parser.parse_args())

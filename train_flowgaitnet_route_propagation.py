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
from torch.utils.data import ConcatDataset
import pandas as pd
from training.engine import resolve_device

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from training.half_common import FineTunedFlowgaitNet, OnlineTripletLoss, PseudoLabeledDataset



def train_model(model, criterion, criterion_triplet, criterion_center, optimizer, scheduler, 
                labeled_loader, unlabeled_loader, val_loader, cfg, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Read settings from the experiment configuration.
    epochs = cfg.epochs
    warmup_epochs = cfg.warmup_epochs
    id_loss_weight = cfg.id_loss_weight
    action_loss_weight = cfg.action_loss_weight
    pseudo_label_weight = cfg.pseudo_label_weight
    confidence_threshold = cfg.confidence_threshold

    for epoch in range(epochs):

        if epoch < warmup_epochs:
            # --- Phase 1: Supervised warm-up ---
            print(f"--- [Epoch {epoch+1}/{epochs}] Phase 1: Supervised Warm-up ---")
            train_loader = labeled_loader
        else:
            # --- Phase 2: Semi-supervised training ---
            print(f"--- [Epoch {epoch+1}/{epochs}] Phase 2: Semi-Supervised Training ---")
            
            # Generate pseudo-labels with route-level propagation.
            print("Generating pseudo-labels with route-based logic...")
            model.eval()
            
            # Step 1: Collect predictions for all unlabeled samples.
            all_preds = []
            all_probs = []
            all_routes = []
            all_original_indices = []

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(tqdm(unlabeled_loader, desc="Predicting on Unlabeled Data", leave=False)):
                    inputs = inputs.to(device)
                    # The third label contains the route ID.
                    routes = labels[2] 
                    
                    id_outputs, _ = model(inputs)
                    probs = F.softmax(id_outputs, dim=1)
                    max_probs, preds = torch.max(probs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(max_probs.cpu().numpy())
                    all_routes.extend(routes.cpu().numpy())
                    
                    # Preserve each sample's index in the original dataset.
                    batch_indices = list(range(i * unlabeled_loader.batch_size, i * unlabeled_loader.batch_size + len(inputs)))
                    all_original_indices.extend(batch_indices)

            # Step 2: Organize predictions and route IDs for propagation.
            df = pd.DataFrame({
                'original_idx': all_original_indices,
                'route': all_routes,
                'pred': all_preds,
                'prob': all_probs,
                'pseudo_label': -1  # -1 means no pseudo-label has been assigned.
            })

            # Step 3: Apply the initial confidence threshold.
            high_conf_mask = df['prob'] >= confidence_threshold
            df.loc[high_conf_mask, 'pseudo_label'] = df.loc[high_conf_mask, 'pred']
            
            print(f"Initially generated {high_conf_mask.sum()} pseudo-labels based on confidence threshold.")

            # Step 4: Propagate labels when the configured route proportion is met.
            propagated_labels_count = 0
            # Process samples route by route.
            for route_id, group_df in tqdm(df.groupby('route'), desc="Applying Route Logic", leave=False):
                total_samples = len(group_df)
                if total_samples == 0:
                    continue

                # Keep samples that already have pseudo-labels.
                labeled_in_group = group_df[group_df['pseudo_label'] != -1]
                num_labeled = len(labeled_in_group)

                # Condition 1: Enough samples in the route have pseudo-labels.
                if num_labeled / total_samples >= cfg.proportion_threshold:
                    # Condition 2: All assigned pseudo-labels agree.
                    if num_labeled > 0 and labeled_in_group['pseudo_label'].nunique() == 1:
                        common_label = labeled_in_group['pseudo_label'].iloc[0]
                        
                        # Find unassigned samples in this route.
                        unlabeled_indices_in_group = group_df[group_df['pseudo_label'] == -1].index
                        
                        if not unlabeled_indices_in_group.empty:
                            # Assign the shared label to the remaining samples.
                            df.loc[unlabeled_indices_in_group, 'pseudo_label'] = common_label
                            propagated_labels_count += len(unlabeled_indices_in_group)

            if propagated_labels_count > 0:
                print(f"Propagated labels to {propagated_labels_count} additional samples using route logic.")

            # Step 5: Collect samples with final pseudo-labels.
            final_pseudo_labeled_df = df[df['pseudo_label'] != -1]
            all_pseudo_labels = torch.tensor(final_pseudo_labeled_df['pseudo_label'].values, dtype=torch.long)
            all_unlabeled_data_indices = final_pseudo_labeled_df['original_idx'].tolist()

            print(f"Total pseudo-labels after all steps: {len(all_pseudo_labels)}")
            # <<< MODIFICATION END >>>

            if len(all_pseudo_labels) > 0:
                # Build the subset from the original unlabeled dataset.
                selected_unlabeled_subset = data.Subset(unlabeled_loader.dataset, all_unlabeled_data_indices)
                pseudo_dataset = PseudoLabeledDataset(selected_unlabeled_subset, all_pseudo_labels)
                combined_dataset = ConcatDataset([labeled_loader.dataset, pseudo_dataset])
                print(f"Combined dataset size: {len(combined_dataset)}")
            else:
                combined_dataset = labeled_loader.dataset
                print("No pseudo-labels generated, using labeled data only for this epoch.")

            # Train on the combined labeled and pseudo-labeled dataset.
            train_loader = data.DataLoader(combined_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

        # --- Shared training and evaluation loop ---
        
        # Training.
        model.train()
        running_loss = 0.0
        running_loss_id = 0.0
        running_loss_pseudo = 0.0
        running_loss_center = 0.0
        running_loss_trip = 0.0
        id_correct = 0
        total = 0

        # Train with the loader selected for this phase.
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            id_labels = labels[0].to(device)
            action_labels = labels[1].to(device)  # -1 marks a pseudo-labeled sample.
            
            id_outputs, features = model(inputs)
            
            real_mask = action_labels != -1
            pseudo_mask = action_labels == -1
            
            loss = 0.0
            
            if torch.any(real_mask):
                loss_id_real = criterion(id_outputs[real_mask], id_labels[real_mask])
                loss_triplet = criterion_triplet(features[real_mask], id_labels[real_mask])
                # loss_center = criterion_center(features[real_mask], id_labels[real_mask])
                loss_center = 0
                loss += id_loss_weight * loss_id_real + action_loss_weight * loss_triplet + action_loss_weight * 0 * loss_center * 0.01
                running_loss_id += loss_id_real.item() * inputs[real_mask].size(0)
                running_loss_trip += loss_triplet.item() * inputs[real_mask].size(0)
                # running_loss_center += loss_center.item() * inputs[real_mask].size(0)
                running_loss_center = 0
            if torch.any(pseudo_mask):
                loss_id_pseudo = criterion(id_outputs[pseudo_mask], id_labels[pseudo_mask])
                loss += pseudo_label_weight * loss_id_pseudo
                running_loss_pseudo += loss_id_pseudo.item() * inputs[pseudo_mask].size(0)
            
            loss.backward()
            optimizer.step()

            id_correct += (torch.max(id_outputs, 1)[1] == id_labels).sum().item()
            total += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Log metrics, validate the model, and save checkpoints.
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_loss_id = running_loss_id / total if total > 0 else 0
        epoch_loss_pseudo = running_loss_pseudo / total if total > 0 else 0
        epoch_loss_trip = running_loss_trip / total if total > 0 else 0
        epoch_loss_center = running_loss_center / total if total > 0 else 0
        id_acc = id_correct / total if total > 0 else 0
        
        print(f"Train Loss: {epoch_loss:.4f}, Accuracy_id: {id_acc:.4f}")
        print(f"Breakdown -> ID Loss: {epoch_loss_id:.4f}, Pseudo Loss: {epoch_loss_pseudo:.4f}, Triplet: {epoch_loss_trip:.4f}, Center: {epoch_loss_center*action_loss_weight:.4f}")

        model.eval()
        
        # Step 1: Collect predictions and route IDs for validation samples.
        print("Validation with route-based logic:")
        all_val_preds = []
        all_val_probs = []
        all_val_routes = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="1/2: Predicting on Validation Data", leave=False):
                inputs = inputs.to(device)
                true_labels = labels[0].cpu().numpy()
                # The validation dataset must return the route ID as labels[2].
                routes = labels[2].cpu().numpy()
                
                id_outputs, _ = model(inputs)
                probs = F.softmax(id_outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(max_probs.cpu().numpy())
                all_val_routes.extend(routes)
                all_val_labels.extend(true_labels)

        # Step 2: Organize validation results for route-level post-processing.
        val_df = pd.DataFrame({
            'true_label': all_val_labels,
            'route': all_val_routes,
            'pred': all_val_preds,
            'prob': all_val_probs,
        })
        
        # Start with the original predictions as the final predictions.
        val_df['final_pred'] = val_df['pred']

        # Step 3: Apply the same route proportion rule used during training.
        propagated_predictions_count = 0
        for route_id, group_df in tqdm(val_df.groupby('route'), desc="2/2: Applying Route Logic to Val Data", leave=False):
            total_samples = len(group_df)
            if total_samples == 0:
                continue

            # Keep high-confidence predictions in this route.
            high_conf_group = group_df[group_df['prob'] >= confidence_threshold]
            num_high_conf = len(high_conf_group)

            # Condition 1: The high-confidence proportion reaches the threshold.
            if (num_high_conf / total_samples) >= cfg.proportion_threshold:
                # Condition 2: All high-confidence predictions agree.
                if num_high_conf > 0 and high_conf_group['pred'].nunique() == 1:
                    common_label = high_conf_group['pred'].iloc[0]
                    
                    # Propagate the shared prediction to the entire route.
                    # Use the group's original indices to update the correct rows.
                    val_df.loc[group_df.index, 'final_pred'] = common_label
                    propagated_predictions_count += (total_samples - num_high_conf)
        
        if propagated_predictions_count > 0:
            print(f"Propagated predictions to {propagated_predictions_count} samples in validation set.")

        # Step 4: Compare accuracy before and after route propagation.
        initial_correct = (val_df['pred'] == val_df['true_label']).sum()
        final_correct = (val_df['final_pred'] == val_df['true_label']).sum()
        total_val_samples = len(val_df)
        
        initial_val_acc = initial_correct / total_val_samples
        final_val_acc = final_correct / total_val_samples  # Final validation metric.

        print(f"Validation Accuracy (Initial): {initial_val_acc:.4f}")
        print(f"Validation Accuracy (After Route Logic): {final_val_acc:.4f}")
        
        val_id_acc = final_val_acc  # Select the best model using propagated accuracy.
        # End route-aware validation.

        # scheduler.step()

        if val_id_acc > best_acc:
            best_acc = val_id_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(cfg.save_dir, "best_model.pth"))
            print(f"New best model saved with accuracy: {best_acc:.4f}")
            
    print("best:" + str(best_acc))


def main(cfg):
    device = resolve_device(cfg.device)
    device_id = device.index or 0
    cfg.save_dir = cfg.save_dir.expanduser().resolve()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0., std=0.1)),
    ])
    # '3','4','5','6','7','8','9','10','11'
    # Load datasets.
    labeled_dataset = RDSExperimentDataset(
        root_dir=cfg.train_dir,
        actions=['1', '2'],
        transform=transform_train,
        preload="sequential",
    )
    unlabeled_dataset = RDSExperimentDataset(
        root_dir=cfg.unlabeled_dir,
        actions=['3', '4', '5', '6', '7', '8', '9', '10'],
        transform=transform_train,
        action_classes=10,
        preload="parallel",
        include_route=True,
    )
    val_dataset = RDSExperimentDataset(
        root_dir=cfg.valid_dir,
        actions=['11'],
        transform=transform_train,
        action_classes=10,
        preload="parallel",
        include_route=True,
    )
    
    # Create data loaders.
    labeled_loader = data.DataLoader(labeled_dataset, batch_size=cfg.batch_size, 
                                     shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    # Keep shuffling disabled so loader indices match the dataset order.
    unlabeled_loader = data.DataLoader(unlabeled_dataset, batch_size=cfg.batch_size, 
                                       shuffle=False, num_workers=cfg.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, 
                                 drop_last=True, num_workers=cfg.num_workers)

    # Build the model.
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
        print("Pretrained weights loaded.")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    criterion_triplet = OnlineTripletLoss(margin=0.3).to(device)
    criterion_center = CenterLoss(num_classes=cfg.num_classes, feat_dim=cfg.feat_dim).to(device)

    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model, criterion, criterion_triplet, criterion_center, optimizer, scheduler, 
                labeled_loader, unlabeled_loader, val_loader, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--unlabeled-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/half_route"))
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--proportion-threshold", type=float, default=0.6)
    parser.add_argument("--pseudo-label-weight", type=float, default=1.0)
    parser.add_argument("--id-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    main(parser.parse_args())

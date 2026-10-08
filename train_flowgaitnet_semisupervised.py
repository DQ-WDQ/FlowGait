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
from training.engine import resolve_device

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from training.half_common import FineTunedFlowgaitNet, OnlineTripletLoss, PseudoLabeledDataset

# <<< MODIFICATION START: Define a dataset for pseudo-labeled data >>>
# <<< MODIFICATION END >>>


def train_model(model, criterion, criterion_triplet, criterion_center, optimizer, scheduler, 
                labeled_loader, unlabeled_loader, val_loader, cfg, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Read settings from the experiment configuration.
    epochs = cfg.epochs
    warmup_epochs = cfg.warmup_epochs  # Number of supervised warm-up epochs.
    id_loss_weight = cfg.id_loss_weight
    action_loss_weight = cfg.action_loss_weight
    pseudo_label_weight = cfg.pseudo_label_weight
    confidence_threshold = cfg.confidence_threshold

    for epoch in range(epochs):

        # Select the training phase based on the current epoch.
        if epoch < warmup_epochs:
            # --- Phase 1: Supervised warm-up ---
            print(f"--- [Epoch {epoch+1}/{epochs}] Phase 1: Supervised Warm-up ---")
            # Use only labeled samples during warm-up.
            train_loader = labeled_loader
        else:
            # --- Phase 2: Semi-supervised training ---
            print(f"--- [Epoch {epoch+1}/{epochs}] Phase 2: Semi-Supervised Training ---")
            print("Generating pseudo-labels...")
            model.eval()
            
            all_pseudo_labels = []
            all_unlabeled_data_indices = []

            with torch.no_grad():
                for i, (inputs, _) in enumerate(tqdm(unlabeled_loader, desc="Predicting on Unlabeled Data", leave=False)):
                    inputs = inputs.to(device)
                    id_outputs, _ = model(inputs)
                    probs = F.softmax(id_outputs, dim=1)
                    max_probs, preds = torch.max(probs, dim=1)
                    
                    mask = max_probs >= confidence_threshold
                    high_confidence_indices = torch.arange(len(inputs), device=device)[mask]
                    
                    global_indices = [i * unlabeled_loader.batch_size + idx.item() for idx in high_confidence_indices]
                    
                    all_pseudo_labels.extend(preds[mask].cpu())
                    all_unlabeled_data_indices.extend(global_indices)
            
            print(f"Generated {len(all_pseudo_labels)} pseudo-labels.")

            if len(all_pseudo_labels) > 0:
                selected_unlabeled_subset = data.Subset(unlabeled_loader.dataset, all_unlabeled_data_indices)
                pseudo_dataset = PseudoLabeledDataset(selected_unlabeled_subset, all_pseudo_labels)
                combined_dataset = ConcatDataset([labeled_loader.dataset, pseudo_dataset])
                print(f"Combined dataset size: {len(combined_dataset)}")
            else:
                combined_dataset = labeled_loader.dataset
                print("No pseudo-labels generated, using labeled data only for this epoch.")

            # Train on the combined labeled and pseudo-labeled dataset.
            train_loader = data.DataLoader(combined_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        # <<< MODIFICATION END >>>

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
            action_labels = labels[1].to(device)  # This label is never -1 during warm-up.
            
            id_outputs, features = model(inputs)
            
            # The same loss calculation applies in both phases.
            # During warm-up, pseudo_mask is empty, so the pseudo-label loss is zero.
            real_mask = action_labels != -1
            pseudo_mask = action_labels == -1
            
            loss = 0.0
            
            if torch.any(real_mask):
                loss_id_real = criterion(id_outputs[real_mask], id_labels[real_mask])
                loss_triplet = criterion_triplet(features[real_mask], id_labels[real_mask])
                loss_center = criterion_center(features[real_mask], id_labels[real_mask])
                loss += id_loss_weight * loss_id_real + action_loss_weight * loss_triplet + action_loss_weight * 0 * loss_center * 0.01
                running_loss_id += loss_id_real.item() * inputs[real_mask].size(0)
                running_loss_trip += loss_triplet.item() * inputs[real_mask].size(0)
                running_loss_center += loss_center.item() * inputs[real_mask].size(0)

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

        # Validation.
        model.eval()
        val_id_correct = 0
        val_total = 0
        
        print("Validation:")
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs = inputs.to(device)
                id_labels = labels[0].to(device)
                id_outputs, _ = model(inputs)
                val_id_correct += (torch.max(id_outputs, 1)[1] == id_labels).sum().item()
                val_total += inputs.size(0)

        val_id_acc = val_id_correct / val_total
        print(f"Validation Accuracy_id: {val_id_acc:.4f}")

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

    labeled_dataset = RDSExperimentDataset(
        root_dir=cfg.train_dir,
        actions=['1', '2'],
        transform=transform_train,
        gallery_fraction=0.2,
    )
    unlabeled_dataset = RDSExperimentDataset(
        root_dir=cfg.unlabeled_dir,
        actions=['3'],
        transform=transform_train,
        gallery_split=True,
        gallery_fraction=0.2,
    )
    val_dataset = RDSExperimentDataset(
        root_dir=cfg.valid_dir,
        actions=['3'],
        transform=transform_train,
        gallery_split=False,
        gallery_fraction=0.2,
    )
    
    # Create DataLoaders
    labeled_loader = data.DataLoader(labeled_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    unlabeled_loader = data.DataLoader(unlabeled_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) # Shuffle is False for consistent indexing
    val_loader = data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=cfg.num_workers)
    # Model definition
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
        {'params': model.parameters(), 'lr': 1e-4} # Simplified optimizer for example
    ], weight_decay=1e-4)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # <<< MODIFICATION START: Call the modified training function >>>
    train_model(model, criterion, criterion_triplet, criterion_center, optimizer, scheduler, 
                labeled_loader, unlabeled_loader, val_loader, cfg, device)
    # <<< MODIFICATION END >>>

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--unlabeled-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/half"))
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--pseudo-label-weight", type=float, default=1.0)
    parser.add_argument("--id-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    main(parser.parse_args())

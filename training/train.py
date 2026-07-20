import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.dataset import CarDamageDataset, get_transforms
from model.model import CarDamageModel
from utils.logger import get_logger
from utils.metrics import calculate_metrics

logger = get_logger("Train")

def objective(trial, args):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Setup
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    full_dataset = CarDamageDataset(root_dir=dataset_path, transform=get_transforms(is_train=True))
    
    # Quick split for train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Overwrite val dataset transform
    val_dataset.dataset.transform = get_transforms(is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = CarDamageModel(num_classes=6, model_name="resnet50").to(device)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    num_epochs = args.epochs
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        metrics = calculate_metrics(all_labels, all_preds)
        val_acc = metrics["Accuracy"]
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_acc > best_acc:
            best_acc = val_acc
            # Save model
            os.makedirs(os.path.join(dataset_path, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(dataset_path, "models", "best_model.pth"))

    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per trial")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials")
    args = parser.parse_args()
    
    logger.info("Starting Hyperparameter Tuning with Optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

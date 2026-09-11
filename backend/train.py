import os
import json
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import Dict, Any, Tuple

from model import create_model
from dataset import create_dataloaders

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Error in months"""
    return mean_absolute_error(targets, predictions)

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience: int = 15, min_delta: float = 10.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def train_epoch(model: nn.Module, dataloader, optimizer, criterion, device: str) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    for batch_idx, (images, ages) in enumerate(dataloader):
        images, ages = images.to(device), ages.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs['age_pred'].squeeze(), ages)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs['age_pred'].squeeze().cpu().detach().numpy())
        targets.extend(ages.cpu().numpy())
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    mae = calculate_mae(np.array(predictions), np.array(targets))
    
    return avg_loss, mae

def validate_epoch(model: nn.Module, dataloader, criterion, device: str) -> Tuple[float, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, ages in dataloader:
            images, ages = images.to(device), ages.to(device)
            
            outputs = model(images)
            loss = criterion(outputs['age_pred'].squeeze(), ages)
            
            total_loss += loss.item()
            predictions.extend(outputs['age_pred'].squeeze().cpu().numpy())
            targets.extend(ages.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    mae = calculate_mae(np.array(predictions), np.array(targets))
    
    return avg_loss, mae

def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, 
                   filepath: str, config: Dict[str, Any]):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, filepath)

def train_model(config_path: str = 'config.json'):
    """Main training function"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['training']['seed'])
    
    # Create directories
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    logger = setup_logging(config['paths']['log_dir'])
    
    # Create model and data
    model = create_model(config).to(device)
    train_loader, val_loader = create_dataloaders(config)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training setup
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    writer = SummaryWriter(config['paths']['log_dir'])
    best_val_loss = float('inf')
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Logging
        logger.info(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f} months")
        logger.info(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f} months")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('MAE/Train', train_mae, epoch)
        writer.add_scalar('MAE/Validation', val_mae, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config['paths']['model_dir'], 'best_model.pth'),
                config
            )
            logger.info("New best model saved!")
        
        # Save latest model
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            os.path.join(config['paths']['model_dir'], 'latest_model.pth'),
            config
        )
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    logger.info("Training completed!")

if __name__ == "__main__":
    train_model()
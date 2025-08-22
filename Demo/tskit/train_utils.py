import torch
import math
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from .metrics_utils import *
@dataclass
class TrainingConfig:
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_extension: str = ".pth"
    loss_weights: List[float] = None  # weights for combined loss
    lr_scheduler: Optional[Callable] = None

class Trainer:
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        self.best_score = math.inf
        self.early_stop_counter = 0
        self.epoch = 0
        self.metrics = {
            'train_mse': [],
            'val_mse': [],
            'train_r2': [],
            'val_r2': [],
            'train_mape': [],
            'val_mape': [],
            'train_smape': [],
            'val_smape': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
        }
        self.train_lr = config.lr

    def train_epoch(self, train_loader):
        self.train_lr = self.config.lr/(2**self.epoch)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.train_lr, 
            weight_decay=self.config.weight_decay
        )
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}")
        
        for x, y in progress_bar:
            x, y = x.to(self.config.device), y.to(self.config.device)

            # Forward pass
            pred = self.model(x)

            # loss = self.model.cal_loss(pred, y)
            loss = mse_loss(pred, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            # progress_bar.set_postfix(loss=loss.item())
            
            self.metrics['train_mse'].append(loss.detach().cpu().item())
            self.metrics['train_r2'].append(r2_loss(pred, y).detach().cpu().item())
            self.metrics['train_mape'].append(mape_loss(pred, y).detach().cpu().item())
            self.metrics['train_smape'].append(smape_loss(pred, y).detach().cpu().item())
            self.metrics['train_mae'].append(mae_loss(pred, y).detach().cpu().item())
            self.metrics['train_rmse'].append(rmse_loss(pred, y).detach().cpu().item())
            progress_bar.set_postfix(
                train_loss=loss.detach().cpu().item(), 
                train_r2=self.metrics['train_r2'][-1]
                )
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        metrics = {}
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.config.device), y.to(self.config.device)
                pred = self.model(x)
                
                # Calculate loss and metrics
                # loss = self.model.cal_loss(pred, y)
                loss = mse_loss(pred, y)
                total_loss += loss.item() * len(x)
                
                # Calculate metrics
                self.metrics['val_mse'].append(loss.detach().cpu().item())
                self.metrics['val_r2'].append(r2_loss(pred, y).detach().cpu().item())
                self.metrics['val_mape'].append(mape_loss(pred, y).detach().cpu().item())
                self.metrics['val_smape'].append(smape_loss(pred, y).detach().cpu().item())
                self.metrics['val_mae'].append(mae_loss(pred, y).detach().cpu().item())
                self.metrics['val_rmse'].append(rmse_loss(pred, y).detach().cpu().item())
        
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss
    
    def should_stop(self, val_score):
        if val_score < self.best_score:
            self.best_score = val_score
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            return self.early_stop_counter >= self.config.early_stop
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def train(self, train_loader, val_loader, save_path):
        for _ in range(self.config.num_epochs):
            # Train one epoch
            train_loss = self.train_epoch(train_loader)
            # Validate
            val_loss = self.validate(val_loader)
            
            # Calculate combined score
            val_score = self.calculate_score(val_loss)
            
            # Check early stopping
            if self.should_stop(val_score):
                print(f"Early stopping at epoch {self.epoch}")
                break
                
            # Save best model
            if val_score == self.best_score:
                self.model_path = f"{save_path}_{self.epoch}{self.config.model_extension}"
                self.save_model(self.model_path)
            print("| Epoch: {}/{} || Train Mse: {:.4f} | Dev Mse: {:.4f} | Train R²: {:.4f} | Dev R²: {:.4f} | Model saved to {}".format(
                    self.epoch + 1, self.config.num_epochs, train_loss, val_loss, self.metrics['train_r2'][-1], self.metrics['val_r2'][-1], self.model_path))
            print("| Epoch: {}/{} || Train SMape: {:.4f} | Dev SMape: {:.4f}".format(
                    self.epoch + 1, self.config.num_epochs, self.metrics['train_smape'][-1], self.metrics['val_smape'][-1]))
            self.epoch += 1
            
    
    def calculate_score(self, loss):
        if self.config.loss_weights:
            # Example: weighted combination of loss and metrics
            return (self.config.loss_weights[0] * loss + 
                    self.config.loss_weights[1] * (1 - self.metrics['val_r2'][-1]) + 
                    self.config.loss_weights[2] * self.metrics['val_smape'][-1])
        return loss
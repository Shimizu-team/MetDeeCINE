import torch
import numpy as np
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from datetime import datetime
from preprocessing_new import *
from mignn import *
from create_dics import *
from sklearn.preprocessing import StandardScaler

def torch_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Device settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reset_params(model):
    """Initialize model parameters"""
    for param in model.parameters():
        if param.requires_grad:
            if param.dim() > 1:
                std = 1 / np.sqrt(param.shape[0])
                param.data.normal_(0, std)
            else:
                param.data.uniform_(-0.1, 0.1) # For 1-dimensional parameters

def get_optimizer(model, optimizer_name, lr):
    """Get optimizer"""
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

class EarlyStopping:
    """EarlyStopping class"""
    def __init__(self, patience=7, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

def calculate_pcc(y_pred, y_true):
    """Calculate Pearson correlation coefficient (PCC)"""
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    corrs = []
    for pred_row, true_row in zip(y_pred_np, y_true_np):
        mask = ~np.isnan(true_row)
        if np.sum(mask) > 1: # Need at least 2 data points for calculation
            corr, _ = pearsonr(pred_row[mask], true_row[mask])
            corrs.append(corr if not np.isnan(corr) else 0)
    
    return np.mean(corrs) if corrs else 0.0

def calculate_scc(y_pred, y_true):
    """Calculate Spearman correlation coefficient (SCC)"""
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    corrs = []
    for pred_row, true_row in zip(y_pred_np, y_true_np):
        mask = ~np.isnan(true_row)
        if np.sum(mask) > 1:
            corr, _ = spearmanr(pred_row[mask], true_row[mask])
            corrs.append(corr if not np.isnan(corr) else 0)
            
    return np.mean(corrs) if corrs else 0.0

def calculate_r2_per_dimension(y_pred, y_true):
    """Calculate coefficient of determination (R2 score) for each output dimension"""
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    n_outputs = y_pred_np.shape[1]
    r2_scores = np.zeros(n_outputs)
    
    for i in range(n_outputs):
        mask = ~np.isnan(y_true_np[:, i])
        if np.sum(mask) > 1:
            r2_scores[i] = r2_score(y_true_np[mask, i], y_pred_np[mask, i])
        else:
            r2_scores[i] = np.nan # NaN if calculation is not possible
            
    return r2_scores

def train_epoch(model, loader, optimizer):
    """Perform training for one epoch"""
    model.train()
    total_loss = 0

    num_batches = len(loader)
    for i, (X_batch, y_batch) in enumerate(loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        y_pred = model(X_batch)
        # Assume user-defined loss calculation
        loss, _, _, _, _ = model.calculate_loss(y_pred, y_batch, X_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        
        # Save prediction and true labels of the last batch
        if i == num_batches - 1:
            last_y_pred = y_pred.detach()
            last_y_true = y_batch.detach()

    avg_loss = total_loss / len(loader.dataset)
    
    # Calculate metrics on the last batch
    pcc = calculate_pcc(last_y_pred, last_y_true)
    scc = calculate_scc(last_y_pred, last_y_true)
    
    return {"loss": avg_loss, "pcc": pcc, "scc": scc}

def validate_epoch(model, loader):
    """Perform validation for one epoch and return overall loss and prediction/true labels"""
    model.eval()
    total_loss = 0
    all_preds, all_trues = [], []

    num_batches = len(loader)
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            loss, _, _, _, _ = model.calculate_loss(y_pred, y_batch, X_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(y_pred)
            all_trues.append(y_batch)

            # Save prediction and true labels of the last batch
            if i == num_batches - 1:
                last_y_pred = y_pred
                last_y_true = y_batch

    avg_loss = total_loss / len(loader.dataset)
    y_pred_all = torch.cat(all_preds)
    y_true_all = torch.cat(all_trues)
    
    # Calculate metrics for logging using the final batch
    pcc = calculate_pcc(last_y_pred, last_y_true)
    scc = calculate_scc(last_y_pred, last_y_true)
    
    metrics = {"loss": avg_loss, "pcc": pcc, "scc": scc}
    
    return metrics, y_pred_all, y_true_all

def run_training_cv(model, train_loader_list, val_loader_list, conf):
    """
    Perform training and evaluation with cross-validation.

    Args:
        model (torch.nn.Module): Model to train
        train_loader_list (list): List of training DataLoaders for each fold
        val_loader_list (list): List of validation DataLoaders for each fold
        conf (dict): Training configuration (e.g., {'epochs': 100, 'lr': 0.001, 'optimizer': 'Adam', 'patience': 10})

    Returns:
        list: List of dictionaries containing results for each fold
    """
    all_folds_results = []

    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loader_list, val_loader_list)):
        print(f"===== Fold {fold_idx+1} / {len(train_loader_list)} | Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
        
        # Initialize model parameters, optimizer, and EarlyStopping for each fold
        reset_params(model)
        model.to(device)
        optimizer = get_optimizer(model, conf.optim, conf.lr)
        early_stopping = EarlyStopping(patience=conf.patience, verbose=True)
        
        history = {'train_loss': [], 'train_pcc': [], 'train_scc': [],
                   'val_loss': [], 'val_pcc': [], 'val_scc': []}

        best_val_loss = float('inf')
        best_y_pred_fold = None
        best_y_true_fold = None

        for epoch in range(1, conf.Epoch + 1):
            # Training
            train_metrics = train_epoch(model, train_loader, optimizer)
            
            # Validation
            val_metrics, y_pred_val, y_true_val = validate_epoch(model, val_loader)
            
            # Record results for each epoch
            history['train_loss'].append(train_metrics['loss'])
            history['train_pcc'].append(train_metrics['pcc'])
            history['train_scc'].append(train_metrics['scc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_pcc'].append(val_metrics['pcc'])
            history['val_scc'].append(val_metrics['scc'])

            # Save best model and apply EarlyStopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                # Save prediction and true values at the best validation loss
                best_y_pred_fold = y_pred_val
                best_y_true_fold = y_true_val
            
            early_stopping(val_metrics['loss'])
            if early_stopping.early_stop:
                print(f"Early stopping at Epoch {epoch}")
                break
        
        # Evaluation after fold completion
        if best_y_pred_fold is not None:
            r2_scores = calculate_r2_per_dimension(best_y_pred_fold, best_y_true_fold)
            best_pcc = calculate_pcc(best_y_pred_fold, best_y_true_fold)
            best_scc = calculate_scc(best_y_pred_fold, best_y_true_fold)
            
            print(f"\nFold {fold_idx+1} Summary:")
            print(f"  Best Validation Loss: {best_val_loss:.4f}")
            print(f"  Best Validation PCC: {best_pcc:.4f}")
            print(f"  Best Validation SCC: {best_scc:.4f}")
            print(f"  R2 Scores per dimension (mean): {np.nanmean(r2_scores):.4f}\n")
            
            fold_summary = {
                "fold_index": fold_idx + 1,
                "best_val_loss": best_val_loss,
                "best_val_pcc": best_pcc,
                "best_val_scc": best_scc,
                "r2_scores_per_dimension": r2_scores,
                "history": history
            }
            all_folds_results.append(fold_summary)
        else:
             print(f"Fold {fold_idx+1} did not complete any epochs.")

    return all_folds_results

def Train_fulldata(model, Epoch_final, conf, train_loader_all, model_path):
    """Train final model using all data"""
    model = model.to(device)
    reset_params(model)
    
    # Use new get_optimizer and get learning rate from conf
    optimizer = get_optimizer(model, conf.optim, conf.lr)

    print(f"Starting full data training for {Epoch_final} epochs...")
    for epoch in range(Epoch_final):
        # Call new train_epoch function (return value is ignored as it's not needed)
        train_epoch(model, train_loader_all, optimizer)
    torch.save(model.state_dict(), model_path)
    print(f"Full data model saved to {model_path}")

def load_model_for_prediction(model_class, model_path, device, **model_args):
    """Initialize model, load trained parameters and return"""
    model = model_class.to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode
    print(f"Model loaded from {model_path} and set to evaluation mode.")
    return model

# This function receives a ready model and focuses on inference
def Predict_RCmatrix(model, test_input, enz_num, met_num, device):
    rc_pred = np.zeros((len(test_input), met_num))
    rc_output = np.zeros((len(test_input), met_num))

    with torch.no_grad():
        for i, input_vec in enumerate(test_input):
            # Model is already on the appropriate device, so only input tensor needs to(device)
            input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
            enz_value = input_vec[i % enz_num]
            output = model(input_tensor).cpu().numpy()[0]
            rc_pred[i] = output / enz_value
            rc_output[i] = output
            if enz_value is None or enz_value == 0:
                print(i, "Enzyme value is None or zero, skipping division.")
                continue

    rc_matrices = np.split(rc_pred, len(test_input) // enz_num)
    output_matrices = np.split(rc_output, len(test_input) // enz_num)

    return rc_matrices, output_matrices
from config import *
from preprocessing_new import *
from mignn import *
from create_dics import *
import random
import numpy as np
import torch
import optuna
from functools import partial
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

def fix_seeds(seed=123):
    """Function to set random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Fix random seed and set number of threads
torch.set_num_threads(conf.num_threads)
fix_seeds()

def torch_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch_device()

def Reset_params(model):
    """Reset model parameters with normal distribution."""
    for param in model.parameters():
        if param.requires_grad:
            std = 1 / np.sqrt(param.size(0))
            param.data.normal_(0, std)

def Optimizer(trial, model, optim_name, lr_suggest):
    lr = conf.lr
    if optim_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        # For Optuna, model weights are discarded for each trial,
        # so only record the best score update here
        self.val_loss_min = val_loss

def train_epoch(model, loader, optimizer):
    """Perform training for one epoch"""
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss, _, _, _, _ = model.calculate_loss(y_pred, y_batch, X_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate_epoch(model, loader):
    """Perform validation for one epoch and return overall loss and prediction/true labels"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss, _, _, _, _ = model.calculate_loss(y_pred, y_batch, X_batch)
            total_loss += loss.item() * X_batch.size(0)
            
            all_preds.append(y_pred.cpu())
            all_trues.append(y_batch.cpu())

    avg_loss = total_loss / len(loader.dataset)
    # Concatenate data and return
    y_pred_all = torch.cat(all_preds, dim=0)
    y_true_all = torch.cat(all_trues, dim=0)
    return avg_loss, y_pred_all, y_true_all

def PCC(pred, answer):
    """Calculate Pearson's correlation coefficient (PCC)"""
    p_corrs = []
    for row_pred, row_true in zip(pred, answer):
        mask = ~(np.isnan(row_true) | np.isnan(row_pred))
        if np.any(mask) and len(row_pred[mask]) > 1:  # Need at least 2 data points to calculate correlation
            p_corr, _ = pearsonr(row_pred[mask], row_true[mask])
            p_corrs.append(p_corr if not np.isnan(p_corr) else 0)
        else:
            p_corrs.append(0)
    return np.mean(p_corrs)

def SCC(pred, answer):
    """Calculate Spearman's rank correlation coefficient (SCC)"""
    s_corrs = []
    for row_pred, row_true in zip(pred, answer):
        mask = ~(np.isnan(row_true) | np.isnan(row_pred))
        if np.any(mask) and len(row_pred[mask]) > 1:  # Need at least 2 data points to calculate correlation
            s_corr, _ = spearmanr(row_pred[mask], row_true[mask])
            s_corrs.append(s_corr if not np.isnan(s_corr) else 0)
        else:
            s_corrs.append(0)
    return np.mean(s_corrs)

def create_optuna_dic(conf, trial, model_name, batch_norm, reg_type, enzyme_name, metabolite_name, EMmatrix, MMmatrix, EMmatrix_rev, MMmatrix_rev):
    model_tuning_params = {
        "exp": conf.exp,
        "enz_num": len(enzyme_name),
        "met_num": len(metabolite_name),
        'EMmatrix': EMmatrix,
        'MMmatrix': MMmatrix,
        'EMmatrix_rev': EMmatrix_rev,
        'MMmatrix_rev': MMmatrix_rev,
        "lr": conf.lr,
        # Model configuration
        'loss_fn': trial.suggest_categorical('loss_fn', conf.loss_fn_suggest),
        'reg_type': trial.suggest_categorical('reg_type', conf.reg_type_suggest),
        'af': trial.suggest_categorical('af', conf.af_suggest),
        'batch_norm': batch_norm,
        
        # GNN-specific parameters (always GNN for public release)
        "ML_model": "GNN",
        'GNN_numlayer': trial.suggest_categorical('GNN_numlayer', conf.GNN_numlayer_suggest),
        'GNN_em_subpro_alpha': trial.suggest_float('GNN_em_subpro_alpha', *conf.GNN_em_subpro_alpha_suggest, log=True),
        'GNN_mm_strong_alpha': trial.suggest_float('GNN_mm_strong_alpha', *conf.GNN_mm_strong_alpha_suggest, log=True),
        'GNN_mm_subpro_alpha_ratio': trial.suggest_categorical('GNN_mm_subpro_alpha_ratio', conf.GNN_mm_subpro_alpha_ratio_suggest)
    }
    return model_tuning_params

class OptunaModel:
    def __init__(self,
                 train_loader_list, 
                 val_loader_list,
                 enzyme_name,
                 metabolite_name,
                 EMmatrix,
                 MMmatrix,
                 EMmatrix_rev,
                 MMmatrix_rev,
                 output_dir,
                 conf):
        
        self.study_name = conf.study_name
        self.n_trials = conf.n_trials
        self.epochs = conf.Epoch
        self.patience = conf.patience
        self.train_loader_list = train_loader_list
        self.val_loader_list = val_loader_list
        self.enzyme_name = enzyme_name
        self.metabolite_name = metabolite_name
        self.EMmatrix = EMmatrix
        self.MMmatrix = MMmatrix
        self.EMmatrix_rev = EMmatrix_rev
        self.MMmatrix_rev = MMmatrix_rev
        self.metric = conf.metric
        self.conf = conf
        self._sampler = optuna.samplers.TPESampler(seed=0)
        
        if conf.load_trial_path:
            # Load existing Optuna trials
            self._study = optuna.load_study(
                storage=f'sqlite:///{conf.load_trial_path}',
                study_name=None)
        else:
            if self.metric == "MSE":
                self._study = optuna.create_study(sampler=self._sampler,
                                                study_name=self.study_name,
                                                direction="minimize",
                                                storage=f'sqlite:///{output_dir}/optuna_study.db',
                                                load_if_exists=True)            
            else:
                self._study = optuna.create_study(sampler=self._sampler,
                                            study_name=self.study_name,
                                            direction="maximize",
                                            storage=f'sqlite:///{output_dir}/optuna_study.db',
                                            load_if_exists=True)

    def prepare_model(self, trial):
        """Prepare model for each Optuna trial"""
        train_params = create_optuna_dic(
            self.conf,
            trial,
            self.conf.ML_model,
            self.conf.batch_norm,
            self.conf.reg_type,
            self.enzyme_name,
            self.metabolite_name,
            self.EMmatrix,
            self.MMmatrix,
            self.EMmatrix_rev,
            self.MMmatrix_rev
        )
        model = MiGNN(train_params)
        return model

    def objective(self, trial):
        """Optuna objective function"""
        
        # Define hyperparameters
        optimizer_name = self.conf.optim  # Use pre-configured optimizer name here
        # Model instantiation (create new model for each trial)      
        model = self.prepare_model(trial).to(device)

        fold_metric = []  # List to store metrics for each fold

        # Cross-validation loop
        for fold_idx, (train_loader, val_loader) in enumerate(zip(self.train_loader_list, self.val_loader_list)):
            
            # Initialize model parameters
            Reset_params(model)
            optimizer = Optimizer(trial, model, optimizer_name, self.conf.lr_suggest)
            early_stopping = EarlyStopping(patience=self.patience)
            
            best_val_loss = float('inf')
            best_y_pred = None
            best_y_true = None

            # Epoch loop
            for epoch in range(self.epochs):
                # Training
                train_epoch(model, train_loader, optimizer)
                
                # Validation
                val_loss, y_pred_val, y_true_val = validate_epoch(model, val_loader)

                # Save best validation loss and corresponding prediction/true labels
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_y_pred = y_pred_val.numpy()
                    best_y_true = y_true_val.numpy()

                # Early stopping check
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at Epoch {epoch+1}")
                    break
            
            # Fold evaluation
            # Calculate PCC and SCC using saved best predictions
            if self.metric == "PCC":
                pcc = PCC(best_y_pred, best_y_true)
                fold_metric.append(pcc)
            elif self.metric == "SCC":
                scc = SCC(best_y_pred, best_y_true)
                fold_metric.append(scc)
            elif self.metric == "MSE":
                mse = np.mean((best_y_pred - best_y_true) ** 2)
                fold_metric.append(mse)
            
        final_score = np.mean(fold_metric)
        return final_score

    def optimize(self):
        self._study.optimize(self.objective, n_trials=self.n_trials)

    def get_best_params(self):
        return self._study.best_params

    def save_study_results(self, save_dir):
        study_df = self._study.trials_dataframe(multi_index=True)
        study_df.to_csv(save_dir)
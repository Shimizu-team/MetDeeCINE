from config import *
import numpy as np
import torch
import torch.nn as nn
import math
conf = config()
torch.set_num_threads(conf.num_threads)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MiGNN(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        # Core parameters
        self.enz_num = param['enz_num']
        self.met_num = param['met_num']
        self.lr = param['lr']
        self.af = param['af']
        self.lossfn = param['loss_fn']
        self.reg_type = param['reg_type']
        
        # Experimental data settings
        self.exp = True  # Always True for public release

        # Loss function
        if self.lossfn == "MSE":
            self.loss_fn = nn.MSELoss()
        elif self.lossfn == "L1":
            self.loss_fn = nn.L1Loss()
            
        # Xavier initialization
        self.em_start = 1 / math.sqrt(self.enz_num)
        self.mm_start = 1 / math.sqrt(self.met_num)
        
        # GNN-specific parameters
        self.EMmatrix = param['EMmatrix']
        self.EMmatrix_rev = param['EMmatrix_rev']
        self.MMmatrix = param["MMmatrix"]    
        self.MMmatrix_rev = param['MMmatrix_rev']
        self.GNN_em_subpro_alpha = param['GNN_em_subpro_alpha']
        self.GNN_mm_strong_alpha = param['GNN_mm_strong_alpha']
        self.GNN_mm_subpro_alpha_ratio = param['GNN_mm_subpro_alpha_ratio']
        self.num_layer = param['GNN_numlayer']
        
        # E→M layer
        em_linear_weight = torch.empty([self.enz_num, self.met_num]).normal_(-1 * self.em_start, self.em_start)
        self.em_linear_weight = nn.Parameter(em_linear_weight)
        
        # M→M layer
        mm_linear_weight = torch.empty([self.met_num, self.met_num]).normal_(-1 * self.mm_start, self.mm_start)
        self.mm_linear_weight = nn.Parameter(mm_linear_weight)

    def activation(self, input):
        # Activation functions
        if self.af == "tanh":
            af = nn.Tanh()
        elif self.af == "relu":
            af = nn.ReLU()
        elif self.af == "elu":
            af = nn.ELU()
        elif self.af == "swish":
            af = nn.SiLU()
        elif self.af == "mish":
            af = nn.Mish()
        return af(input)

    def forward(self, input):
        # GNN forward pass
        metabolite = torch.mm(input, torch.mul(self.EMmatrix, self.em_linear_weight))
        for _ in range(self.num_layer):
            metabolite_change_before = torch.mm(metabolite, self.mm_linear_weight)
            metabolite = metabolite + self.activation(metabolite_change_before)
        return metabolite

    def missing_mask(self, y_batch):
        mask = torch.ones(y_batch.shape, device=y_batch.device)
        mask[torch.isnan(y_batch)] = 0
        return mask

    def loss_fn_exp(self, y_pred, y_batch, mask):
        # Replace NaN with 0
        y_batch[torch.isnan(y_batch)] = 0
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print("[error] y_pred contains nan or inf")
        if torch.isnan(y_batch).any() or torch.isinf(y_batch).any():
            print("[error] y_batch contains nan or inf")
        
        if self.lossfn == "L1":
            loss = torch.sum(torch.abs(mask * (y_pred - y_batch))) / torch.sum(mask)
        elif self.lossfn == "MSE":
            loss = torch.sum(mask * ((y_pred - y_batch) ** 2)) / torch.sum(mask)
        
        return loss

    def fit_loss(self, pred, answer):
        # Always use experimental data loss (exp=True)
        mask = self.missing_mask(answer)
        fit_loss = self.loss_fn_exp(pred, answer, mask)
        return fit_loss

    def compute_regularization(self):
        # GNN regularization
        if self.reg_type == "l1":
            self.em_subpro_regularization = torch.abs(torch.mul(self.EMmatrix, self.em_linear_weight)).sum()
            self.mm_related_regularization = torch.abs(torch.mul(self.MMmatrix, self.mm_linear_weight)).sum()
            self.mm_strong_regularization = torch.abs(torch.mul(self.MMmatrix_rev, self.mm_linear_weight)).sum()
        elif self.reg_type == "l2":
            self.em_subpro_regularization = torch.mul(torch.mul(self.EMmatrix, self.em_linear_weight),
                                                torch.mul(self.EMmatrix, self.em_linear_weight)).sum()
            self.mm_related_regularization = torch.mul(torch.mul(self.MMmatrix, self.mm_linear_weight),
                                                torch.mul(self.MMmatrix, self.mm_linear_weight)).sum()
            self.mm_strong_regularization = torch.mul(torch.mul(self.MMmatrix_rev, self.mm_linear_weight),
                                                torch.mul(self.MMmatrix_rev, self.mm_linear_weight)).sum()

    def edge_penalty(self):
        edge_loss = \
                self.GNN_em_subpro_alpha * self.em_subpro_regularization +\
                self.GNN_mm_strong_alpha * self.mm_strong_regularization +\
                self.GNN_mm_strong_alpha * self.GNN_mm_subpro_alpha_ratio * self.mm_related_regularization
        return edge_loss

    def calculate_loss(self, y_pred, y_batch, X_batch=None):
        self.compute_regularization()
        fit_loss = self.fit_loss(y_pred, y_batch)
        edge_penalty = self.edge_penalty()
        loss = fit_loss + edge_penalty
        return loss, fit_loss, edge_penalty, 0, 0  # bias_penalty=0, autograd_penalty=0 for compatibility
    
    def compute_loss(self, y_pred, y_batch, X_batch=None):
        self.compute_regularization()
        fit_loss = self.fit_loss(y_pred, y_batch)
        edge_penalty = self.edge_penalty()
        loss = fit_loss + edge_penalty
        return loss
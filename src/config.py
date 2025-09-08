import os
import pandas as pd
from datetime import datetime

class config:
    def __init__(self):
        # Data settings
        self.exp_root="./input"
        self.exp_name="Uematsu_2022" # Folder name under /input
        self.exp_strains="WT,WTOG,OB,OBOG" # Experimental conditions (comma-separated)

        # Preprocessing settings
        self.gen_method="allsamples_reverse_nosame"
        # "allsamples_reverse", "allsamples_reverse_nosame" The latter excludes FC between the same experimental condition
        # RECOMMENDED: "allsamples_reverse_nosame"
        self.CV_method="leave_one_strain_out"
        self.train_size=0.8 # For random_split only
        self.random_state=0 # For random_split only
        # Validation method:
        # "leave_one_strain_out", "random_split"
        # RECOMMENDED: "leave_one_strain_out" if multiple strains are available
        self.RCinput_method="mean" # "mean"  "lowfilter" "onesample"
        self.RCinput_threshold=0
        # How to generate input features for inference
        # "mean": mean of all samples available for each experimental condition
        # "lowfilter": use "mean" but if it is below threshold, set to threshold
        # "onesample": use one of the samples available for each experimental condition
        # RECOMMENDED: "mean"
        self.Xstd= "All"
        # Whether to standardize input features (enzyme FC vectors)
        # None, "Each_fold", "All" 
        # RECOMMENDED: "All"
        self.ystd= None
        # Whether to standardize output values (metabolite log2FC)
        # None, "Each_fold", "All"
        # RECOMMENDED: None
        self.ystd= None # Whether to standardize output values (metabolite log2FC)

        # Machine learning model settings
        self.exp_batch_size=32 # Batch size for training
        self.optim="AdamW" # "Adam", "AdamW" You can add more optimizers in train.py if needed
        self.Epoch=200 # Maximum number of epochs
        self.patience=10 # Early stopping patience
        self.lr = 1e-3 # Learning rate
        self.batch_norm=False # RECOMMENDED: False
        # ---Setting below is optimized for the Uematsu_2022 dataset---
        self.GNN_numlayer = 1 # Number of GNN layers
        self.GNN_em_subpro_alpha = 0.0008343997810348038 # Weight for subgraph information in E→M layer
        self.GNN_mm_strong_alpha = 0.009245794280397476 # Weight for edges between unrelated metabolites in M→M layer
        self.GNN_mm_subpro_alpha_ratio = 0.1 # Weight ratio for edges between related metabolites in M→M layer
        self.loss_fn = "L1" # "MSE", "L1", loss function
        self.reg_type = "l2" # "l1", "l2", regularization type for weights
        self.af = "elu" # Activation functions: "tanh", "relu", "elu", "swish", "mish"
        # --- 

        # Optuna hyperparameter tuning settings
        self.metric = "PCC" # "PCC", "SCC", "MSE" 
        # Metric for hyperparameter tuning (higher is better for PCC and SCC and lower is better for MSE)
        self.study_name = "hyperparameter_tuning" # Name for hyperparameter tuning
        self.load_trial_path = None # Path to load previous Optuna study
        self.n_trials=2 # Number of trials for hyperparameter tuning
        self.lr_suggest = [1e-5,1e-1]
        self.loss_fn_suggest = ["MSE","L1"]
        self.eval_fn_suggest = ["MSE","MAE"]
        self.reg_type_suggest = ["l1","l2"]
        self.af_suggest = ["tanh","relu","elu","swish","mish"]
        self.GNN_numlayer_suggest = [1,2,3,4]
        self.GNN_em_subpro_alpha_suggest = [1e-7,1e-1]
        self.GNN_mm_strong_alpha_suggest = [1e-7,1e-1]
        self.GNN_mm_subpro_alpha_ratio_suggest = [0,0.1,0.01]

        # Other settings
        self.num_threads=1
        self.exp=True # Always true for public release
        self.ML_model="GNN" # Always GNN for public release
        self.transform="log" # Always "log" unless csv file is already log-transformed
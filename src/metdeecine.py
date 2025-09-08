# Import required libraries
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import copy
import math

# Import MetDeeCINE modules
from config import config
from preprocessing_new import Preprocessing
from mignn import MiGNN
from train import *
from create_dics import *
from hyperparameter_tuning import OptunaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetDeeCINE:
    def __init__(self,conf):
        self.config = conf
        preprocessing_params = create_preprocessing_param_dic(self.config, self.config.exp)
        self.preprocessing_class = Preprocessing(preprocessing_params)
        self.preprocessing_class.def_enzmetname()
        self.enzyme_list, self.metabolite_list = self.preprocessing_class.enzyme_name, self.preprocessing_class.metabolite_name
        self.enz_num, self.met_num = len(self.enzyme_list), len(self.metabolite_list)
        print(f"Number of enzymes: {self.enz_num}")
        print(f"Number of metabolites: {self.met_num}")

    def data_preprocessing(self):
        train_loader_list, val_loader_list = self.preprocessing_class.loader_generator()
        train_loader_all = self.preprocessing_class.loader_all()
        inference_input = self.preprocessing_class.rc_input(self.config.RCinput_method)

        EMmatrix, MMmatrix, EMmatrix_rev, MMmatrix_rev = self.preprocessing_class.matrices()
        model_settings = create_model_params_dic(
            self.config,
            self.enz_num,
            self.met_num,
            EMmatrix,
            MMmatrix,
            EMmatrix_rev,
            MMmatrix_rev
        )
        return train_loader_list, val_loader_list, train_loader_all, inference_input, model_settings
    
    def fc_training(self, model_settings, train_loader_list, val_loader_list, train_loader_all, parameter_save_path):
        mignn = MiGNN(model_settings)
        cross_validation_results = run_training_cv(mignn, train_loader_list, val_loader_list, self.config)
        stop_epochs = [len(f['history']['val_loss']) for f in cross_validation_results]
        epoch_fulltrain = math.floor(sum(stop_epochs) / len(stop_epochs)) - self.config.patience
        print(f"Full training epochs: {epoch_fulltrain}")
        Train_fulldata(mignn, epoch_fulltrain, self.config, train_loader_all, parameter_save_path)
        print(f"Model parameters saved to {parameter_save_path}")

    def ccc_inference(self, model_settings, inference_input, output_dir, parameter_load_path):
        mignn = MiGNN(model_settings)
        mignn.load_state_dict(torch.load(parameter_load_path))
        mignn.to(device)
        RCs_pred, Outputs_pred = Predict_RCmatrix(
            mignn,
            inference_input,
            self.enz_num,
            self.met_num,
            device
        )
        mean_RC_pred = np.nanmean(RCs_pred, axis=0)
        RC_pred_df = pd.DataFrame(mean_RC_pred, columns=self.metabolite_list, index=self.enzyme_list)
        RC_pred_df.to_csv(output_dir + "meanCCC.csv")

    def hyperparameter_tuning(self, train_loader_list, val_loader_list, output_dir):
 
        EMmatrix, MMmatrix, EMmatrix_rev, MMmatrix_rev = self.preprocessing_class.matrices()
        
        optuna_model = OptunaModel(
            train_loader_list=train_loader_list,
            val_loader_list=val_loader_list,
            enzyme_name=self.enzyme_list,
            metabolite_name=self.metabolite_list,
            EMmatrix=EMmatrix,
            MMmatrix=MMmatrix,
            EMmatrix_rev=EMmatrix_rev,
            MMmatrix_rev=MMmatrix_rev,
            output_dir=output_dir,
            conf=self.config)
        
        print("Starting hyperparameter tuning with Optuna...")
        optuna_model.optimize()
        
        best_params = optuna_model.get_best_params()
        print("Best parameters found: ", best_params)
        
        optuna_model.save_study_results(save_dir=output_dir + "/optuna_study.csv")
    
        return best_params

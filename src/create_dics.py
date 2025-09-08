from config import *
import itertools

def create_preprocessing_param_dic(conf,exp):
    if exp:
        preprocessing_params = {
            "exp":True,
            'exp_root': conf.exp_root,
            'exp_name': conf.exp_name,
            'exp_strains': conf.exp_strains,
            'exp_transform': conf.transform,
            'exp_gen_method': conf.gen_method,
            'exp_CV_method': conf.CV_method,
            'exp_Xstd': conf.Xstd,
            'exp_ystd': conf.ystd,
            'exp_batch_size': conf.exp_batch_size,
            'exp_train_size': conf.train_size,
            'exp_random_state': conf.random_state,
        }    
    return preprocessing_params

def create_model_params_dic(conf, enz_num, met_num, EMmatrix, MMmatrix, EMmatrix_rev, MMmatrix_rev):
    model_params = {
        "exp": conf.exp,
        "ML_model": conf.ML_model,
        "enz_num": enz_num,
        "met_num": met_num,
        'EMmatrix': EMmatrix,
        'MMmatrix': MMmatrix,
        'EMmatrix_rev': EMmatrix_rev,
        'MMmatrix_rev': MMmatrix_rev,
        # Model configuration
        'lr': conf.lr,
        'loss_fn': conf.loss_fn,
        'reg_type': conf.reg_type,
        'af': conf.af,
        'batch_norm': conf.batch_norm,

        # GNN-specific parameters
        'GNN_numlayer': conf.GNN_numlayer,
        'GNN_em_subpro_alpha': conf.GNN_em_subpro_alpha,
        'GNN_mm_strong_alpha': conf.GNN_mm_strong_alpha,
        'GNN_mm_subpro_alpha_ratio': conf.GNN_mm_subpro_alpha_ratio,
    }
    return model_params
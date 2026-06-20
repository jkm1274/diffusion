from datetime import datetime
import optuna
import argparse
import os
import torch
import random
import numpy as np
import warnings
from src.exp.exp_basic_diffusion import Exp_Basic_Diffusion
from src.exp.exp_diffusionts import Exp_DiffusionTS
from src.exp.exp_classifier_free_conditional_diffusion import Exp_Classifier_Free_Conditional_Diffusion
from src.exp.exp_diffusion_denoised_x import Exp_Diffusion_Denoised_X
from src.exp.exp_diffusion_denoised_x_conditional import Exp_Diffusion_Denoised_X_Conditional
from src.exp.exp_diffusion_direct_x import Exp_Diffusion_Direct_X
from src.exp.exp_diffusion_direct_x_conditional import Exp_Diffusion_Direct_X_Conditional
from src.utils.utils import fix_seed, build_config, dotdict

# Change working directory to the root of the project
working_dir = r'F:\Projects\Ask2.ai\Diffusion'
os.chdir(working_dir)
warnings.filterwarnings('ignore')
fix_seed()

exp_dict = {
    'basic_diffusion': Exp_Basic_Diffusion,
    'diffusion_ts': Exp_DiffusionTS,
    'conditional_diffusion': Exp_Classifier_Free_Conditional_Diffusion,
    'diffusion_denoised_x': Exp_Diffusion_Denoised_X,
    'diffusion_denoised_x_conditional': Exp_Diffusion_Denoised_X_Conditional,
    'diffusion_direct_x': Exp_Diffusion_Direct_X,
    'diffusion_direct_x_conditional': Exp_Diffusion_Direct_X_Conditional
}

benchmark_performance = {
    'M_KL_Div': 0.182869501557954,
    'Auto-Corr_DTW': 0.824090200553693,
    'Covariance_Riemannian': 3.79260704015992,
    'Correlation_Riemannian': 2.45916866578517
}


def objective(trial):
    # Define hyperparameter search space
    search_space = {
        'data_path': './warehouse/processed/benchmark_data_log_ret_10.csv',
        'enc_in': 10,
        'checkpoints': './tuning_results/alt_checkpoints/',
        'test_results': './tuning_results/results/',
        'task_name': 'diffusion_denoised_x',
        'model': 'UniTST_MP',
        'loss': 'KL2_N+Corr+FFT',
        'causal_mask': True,
        'channel_embed': True,
        'ind_proj': True,
        'pcgrad': False,
        'grad_norm': False,
        'RoPE': True,
        'compile': True,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'lr_decay_rounds': trial.suggest_categorical('lr_decay_rounds', [5, 10, 25]),
        'temperature': trial.suggest_loguniform('temperature', 1e-2, 1),
    }

    args = dotdict(build_config(search_space))
    args.description = f"optuna{trial.number}"
    Exp = exp_dict[args.task_name]
    exp = Exp(args)

    exp.train()
    results_df = exp.test(size=1024, method='discrete', temperature=search_space['temperature'], save_data=True,
                          sample_step=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0], no_compile=False)
    results_df.drop(columns=['Step'], inplace=True)
    results_df = results_df.iloc[3:].reset_index(drop=True)
    for metric in benchmark_performance.keys():
        results_df[metric] = results_df[metric] / benchmark_performance[metric]
    results_df['mean_performance'] = results_df[benchmark_performance.keys()].mean(axis=1)
    return np.min(results_df['mean_performance'])


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print("  Value: ", study.best_value)
    print("  Params: ", study.best_params)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study.save(f"./tuning_results/10mfs_optuna_study_{timestamp}.pkl")

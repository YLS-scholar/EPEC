"""
Multi-seed statistical analysis for EPEC.
Input: all_experiment_results.csv with columns:
    seed, data, model, pred_len, train_time, mse, mape
Output: CSV files in ./output/
"""

import os
import pandas as pd
import numpy as np

# ================== Configuration ==================
LAMBDA = 0.25
SEEDS = [3047, 1234, 5678, 9012, 3456]
INPUT_FILE = "all_experiment_results.csv"
OUTPUT_DIR = "output"

DATASETS = {
    'BeijingMetro': {
        'error_col': 'mape',
        'horizons': [1, 7, 14, 28],
        'is_multi_line': True,
        'line_name': None
    },
    'electricity': {
        'error_col': 'mse',
        'horizons': [6, 24, 48, 96],
        'is_multi_line': False,
        'line_name': 'electricity'
    },
    'exchange_rate': {
        'error_col': 'mse',
        'horizons': [6, 24, 48, 96],
        'is_multi_line': False,
        'line_name': 'exchange_rate'
    },
    'weather': {
        'error_col': 'mse',
        'horizons': [6, 24, 72, 144],
        'is_multi_line': False,
        'line_name': 'weather'
    },
    'illness': {
        'error_col': 'mse',
        'horizons': [1, 16, 24, 52],
        'is_multi_line': False,
        'line_name': 'national_illness'
    }
}

ALL_MODELS = ['XGBoost', 'SARIMAX', 'DLinear', 'SegRNN', 'LightTS',
              'TimeMixer', 'iTransformer', 'TimeXer', 'PatchTST', 'SCINet']

# ================== Helper functions ==================
def get_dataset_subset(df, ds_name, ds_info):
    if ds_info['is_multi_line']:
        mask = df['data'].str.startswith('Line')
        ds_df = df[mask].copy()
        ds_df['dataset_name'] = 'BeijingMetro'
    else:
        mask = df['data'] == ds_info['line_name']
        ds_df = df[mask].copy()
        ds_df['dataset_name'] = ds_name
    return ds_df

def unify_beijing_mape(df_sub, ds_name):
    if ds_name == 'BeijingMetro':
        error_col = 'mape'
        decimal_models = [m for m in ALL_MODELS if m not in ['XGBoost', 'SARIMAX']]
        mask = df_sub['model'].isin(decimal_models)
        df_sub.loc[mask, error_col] = df_sub.loc[mask, error_col] * 100
    return df_sub

# ================== Main ==================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    error_stats = []
    traintime_stats = []
    traintime_horizons = []
    epec_stats = []
    per_horizon_epec = []

    for ds_name, ds_info in DATASETS.items():
        print(f"\nProcessing {ds_name}...")
        ds_df = get_dataset_subset(df, ds_name, ds_info)
        ds_df = unify_beijing_mape(ds_df, ds_name)
        if ds_df.empty:
            print(f"  No data for {ds_name}")
            continue

        models_in_data = sorted([m for m in ALL_MODELS if m in ds_df['model'].unique()])
        horizons = ds_info['horizons']
        is_multi = ds_info['is_multi_line']

        # Storage for seeds
        seed_error_vals = {m: {h: [] for h in horizons} for m in models_in_data}
        seed_total_time_values = {m: [] for m in models_in_data}
        seed_train_vals = {m: [] for m in models_in_data}
        seed_train_mean = {m: {h: [] for h in horizons} for m in models_in_data}
        seed_epec_vals = {m: [] for m in models_in_data}
        seed_horizon_epec_vals = {m: {h: [] for h in horizons} for m in models_in_data}
        seed_rank_vals = {m: [] for m in models_in_data}

        for seed in SEEDS:
            seed_df = ds_df[ds_df['seed'] == seed].copy()

            # 1. Error per horizon
            for model in models_in_data:
                for h in horizons:
                    mask = (seed_df['model'] == model) & (seed_df['pred_len'] == h)
                    sub = seed_df[mask]
                    mean_err = sub[ds_info['error_col']].mean() if len(sub) > 0 else np.nan
                    seed_error_vals[model][h].append(mean_err)

            # 2. Training time (average and total)
            for model in models_in_data:
                mask = (seed_df['model'] == model)
                sub = seed_df[mask]
                if len(sub) > 0:
                    avg_time = sub['train_time'].mean()
                    total_time = sub['train_time'].sum()
                else:
                    avg_time = total_time = np.nan
                seed_train_vals[model].append(avg_time)
                seed_total_time_values[model].append(total_time)

            # 3. Training time per horizon
            for model in models_in_data:
                for h in horizons:
                    mask = (seed_df['model'] == model) & (seed_df['pred_len'] == h)
                    sub = seed_df[mask]
                    mean_time = sub['train_time'].mean() if len(sub) > 0 else np.nan
                    seed_train_mean[model][h].append(mean_time)

            # 4. Overall EPEC for this seed
            if is_multi:
                lines = seed_df['data'].unique()
                tasks = [f"{line}_{h}" for line in lines for h in horizons]
            else:
                tasks = [str(h) for h in horizons]
            n_models = len(models_in_data)
            n_tasks = len(tasks)
            error_mat = np.full((n_models, n_tasks), np.nan)
            train_time_vec = np.zeros(n_models)

            for i, m in enumerate(models_in_data):
                m_df = seed_df[seed_df['model'] == m]
                train_time_vec[i] = m_df['train_time'].mean()
                if is_multi:
                    for j, task in enumerate(tasks):
                        line, h = task.split('_')
                        h = int(h)
                        val = m_df[(m_df['data'] == line) & (m_df['pred_len'] == h)][ds_info['error_col']].values
                        if len(val) > 0:
                            error_mat[i, j] = val[0]
                else:
                    for j, h in enumerate(horizons):
                        val = m_df[m_df['pred_len'] == h][ds_info['error_col']].values
                        if len(val) > 0:
                            error_mat[i, j] = val[0]

            min_per_task = np.nanmin(error_mat, axis=0, keepdims=True)
            min_per_task = np.where(min_per_task == 0, 1e-12, min_per_task)
            loss_mat = (error_mat - min_per_task) / min_per_task
            w_mat = np.exp(-LAMBDA * loss_mat)
            avg_weight = np.nanmean(w_mat, axis=1)
            epec = avg_weight / train_time_vec

            for i, m in enumerate(models_in_data):
                seed_epec_vals[m].append(epec[i])
            # Rank for this seed
            epec_with_model = [(models_in_data[i], epec[i]) for i in range(len(models_in_data))]
            sorted_by_epec = sorted(epec_with_model, key=lambda x: x[1], reverse=True)
            rank_this_seed = {m: idx + 1 for idx, (m, _) in enumerate(sorted_by_epec)}
            for m in models_in_data:
                seed_rank_vals[m].append(rank_this_seed[m])

            # 5. Per-horizon EPEC for this seed
            for h in horizons:
                if is_multi:
                    tasks_h = [f"{line}_{h}" for line in lines]
                else:
                    tasks_h = [str(h)]
                n_tasks_h = len(tasks_h)
                error_mat_h = np.full((n_models, n_tasks_h), np.nan)
                train_time_h = np.zeros(n_models)
                for i, m in enumerate(models_in_data):
                    m_df = seed_df[seed_df['model'] == m]
                    train_times_for_h = []
                    if is_multi:
                        for j, task in enumerate(tasks_h):
                            line, _ = task.split('_')
                            val = m_df[(m_df['data'] == line) & (m_df['pred_len'] == h)][ds_info['error_col']].values
                            if len(val) > 0:
                                error_mat_h[i, j] = val[0]
                            val_time = m_df[(m_df['data'] == line) & (m_df['pred_len'] == h)]['train_time'].values
                            if len(val_time) > 0:
                                train_times_for_h.append(val_time[0])
                    else:
                        val_err = m_df[m_df['pred_len'] == h][ds_info['error_col']].values
                        if len(val_err) > 0:
                            error_mat_h[i, 0] = val_err[0]
                        val_time = m_df[m_df['pred_len'] == h]['train_time'].values
                        if len(val_time) > 0:
                            train_times_for_h.append(val_time[0])
                    train_time_h[i] = np.mean(train_times_for_h) if train_times_for_h else np.nan

                min_h = np.nanmin(error_mat_h, axis=0, keepdims=True)
                min_h = np.where(min_h == 0, 1e-12, min_h)
                loss_h = (error_mat_h - min_h) / min_h
                w_h = np.exp(-LAMBDA * loss_h)
                avg_w_h = np.nanmean(w_h, axis=1)
                epec_h = avg_w_h / train_time_h
                for i, m in enumerate(models_in_data):
                    seed_horizon_epec_vals[m][h].append(epec_h[i])

        # Aggregate across seeds
        for m in models_in_data:
            for h in horizons:
                vals = seed_error_vals[m][h]
                valid = [v for v in vals if not np.isnan(v)]
                mean_e = np.mean(valid) if valid else np.nan
                std_e = np.std(valid) if len(valid) > 1 else 0.0
                error_stats.append({
                    'Dataset': ds_name, 'Model': m, 'Horizon': h,
                    'Error_Mean_OverSeeds': mean_e,
                    'Error_Std_OverSeeds': std_e,
                    'NumSeeds': len(valid)
                })

        for model in models_in_data:
            vals = seed_train_vals[model]
            valid = [v for v in vals if not np.isnan(v)]
            mean_t = np.mean(valid) if valid else np.nan
            std_t = np.std(valid) if len(valid) > 1 else 0.0
            total_vals = seed_total_time_values[model]
            valid_total = [v for v in total_vals if not np.isnan(v)]
            total_mean = np.mean(valid_total) if valid_total else np.nan
            total_std = np.std(valid_total) if len(valid_total) > 1 else 0.0
            traintime_stats.append({
                'Dataset': ds_name, 'Model': model,
                'TrainTime_Mean_OverSeeds': mean_t,
                'TrainTime_Std_OverSeeds': std_t,
                'TotalTrainTime_Mean_OverSeeds': total_mean,
                'TotalTrainTime_Std_OverSeeds': total_std,
                'NumSeeds': len(valid)
            })
        for m in models_in_data:
            for h in horizons:
                vals = seed_train_mean[m][h]
                valid = [v for v in vals if not np.isnan(v)]
                mean_time = np.mean(valid) if valid else np.nan
                std_time = np.std(valid) if len(valid) > 1 else 0.0
                traintime_horizons.append({
                    'Dataset': ds_name, 'Model': m, 'Horizon': h,
                    'TrainTime_Mean_OverSeeds_horizons': mean_time,
                    'TrainTime_Std_OverSeeds_horizons': std_time,
                    'NumSeeds': len(valid)
                })

        for m in models_in_data:
            epec_vals = seed_epec_vals[m]
            valid_epec = [v for v in epec_vals if not np.isnan(v)]
            mean_epec = np.mean(valid_epec) if valid_epec else np.nan
            std_epec = np.std(valid_epec) if len(valid_epec) > 1 else 0.0
            rank_vals = seed_rank_vals[m]
            valid_rank = [r for r in rank_vals if not np.isnan(r)]
            mean_rank = np.mean(valid_rank) if valid_rank else np.nan
            std_rank = np.std(valid_rank) if len(valid_rank) > 1 else 0.0
            epec_stats.append({
                'Dataset': ds_name, 'Model': m,
                'EPEC_Mean_OverSeeds': mean_epec,
                'EPEC_Std_OverSeeds': std_epec,
                'Rank_Mean_OverSeeds': mean_rank,
                'Rank_Std_OverSeeds': std_rank,
                'NumSeeds': len(valid_epec)
            })

        for m in models_in_data:
            for h in horizons:
                vals = seed_horizon_epec_vals[m][h]
                valid = [v for v in vals if not np.isnan(v)]
                mean_eph = np.mean(valid) if valid else np.nan
                std_eph = np.std(valid) if len(valid) > 1 else 0.0
                per_horizon_epec.append({
                    'Dataset': ds_name, 'Horizon': h, 'Model': m,
                    'EPEC_Mean': mean_eph, 'EPEC_Std': std_eph,
                    'NumSeeds': len(valid)
                })

    # Save
    pd.DataFrame(error_stats).to_csv(os.path.join(OUTPUT_DIR, 'seed_error_stats.csv'), index=False)
    pd.DataFrame(traintime_stats).to_csv(os.path.join(OUTPUT_DIR, 'seed_traintime_stats.csv'), index=False)
    pd.DataFrame(traintime_horizons).to_csv(os.path.join(OUTPUT_DIR, 'seed_traintime_horizons.csv'), index=False)
    pd.DataFrame(epec_stats).to_csv(os.path.join(OUTPUT_DIR, 'seed_epec_stats.csv'), index=False)
    pd.DataFrame(per_horizon_epec).to_csv(os.path.join(OUTPUT_DIR, 'seed_epec_per_horizon.csv'), index=False)
    print("\nAll outputs saved to ./output/")

if __name__ == "__main__":
    main()

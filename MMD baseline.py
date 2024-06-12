from tqdm import tqdm
import pandas as pd
import torch
import torchdrift.detectors as detectors

SAMPLE = r'20 bin PPO 500 results\baseline_obs.csv'
SAVE_DIR = r'20 bin PPO 500 results' + '/'
SAVE_NAME = 'MMD_baseline_random_daily_samples'
BOOTSTRAP = 10_000
PVAL = 0.05
REPETITIONS = 80

kernel = detectors.mmd.GaussianKernel()
samples_per_day = 24

#load data
df_obs = pd.read_csv(SAMPLE, 
                        index_col=0,
                        dtype='float32',
                        )
df_obs.set_index(df_obs.index.astype(int), inplace=True) #all data is loaded as float32, but the index should be an int

#remove actions if present
if 'a' in df_obs.columns:
    df_obs.drop(columns=['a'], inplace=True)
elif 'actions' in df_obs.columns:
    df_obs.drop(columns=['actions'], inplace=True)

for rep in tqdm(range(REPETITIONS)):
    # Split the DataFrame into two equal parts day by day
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for i in range(0, len(df_obs), samples_per_day):
        daily_samples = df_obs.iloc[i:i+samples_per_day]
        daily_samples = daily_samples.sample(frac=1)  # shuffle the daily samples
        df1 = df1.append(daily_samples.iloc[:samples_per_day//2])
        df2 = df2.append(daily_samples.iloc[samples_per_day//2:])

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    #compute MMD
    result = detectors.kernel_mmd(torch.from_numpy(df1.values).to('cuda'), #clean obs from adv trace
                                  torch.from_numpy(df2.values).to('cuda'), #perturbed obs from adv trace
                                  n_perm=BOOTSTRAP,
                                  kernel=kernel)
    cpu_result = [tensor.item() for tensor in result]
    print(f'mmd:{cpu_result[0]}, p-value:{cpu_result[1]}')
    
    #Save results
    mmd_savename = SAVE_DIR+'MMDs.csv'
    save_name = SAVE_NAME + f'_{rep}'
    try:
        df_mmd = pd.read_csv(mmd_savename,
                            index_col=0)
        df_mmd = df_mmd.append(
                    pd.Series(cpu_result,
                            index=df_mmd.columns,
                            name=save_name,),
                )
        df_mmd.to_csv(mmd_savename)
        print(f'{mmd_savename} updated')
    except:
        df_mmd = pd.DataFrame([cpu_result],
                        columns=['MMD','p_value'],
                        index=[save_name])
        df_mmd.to_csv(mmd_savename)
        print(f'{mmd_savename} created')
"""
ref: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
"""

import os

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, Subset

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

STORAGE = None
N_TRIALS = 400
N_STARTUP_TRIALS = N_TRIALS//10
DEVICE = torch.device("cuda")
JOBS = 1 # 2 is much slower, but I also doubled the epochs...
CLASSES = 20
INPUTS = 31
DIR = os.getcwd()
FOLDS = 10 #5 for first
EPOCHS = 1000//FOLDS #500 for first
DATADIR = '20 bin PPO 500 results/'
NAME = 'SA Surrogate 20 bin PPO 500 Policy 2'

class MyDataset(Dataset):
    def __init__(self, df):
        self.data = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
def load_SA_dataset():
    #path to data directory
    data_dir = os.path.join(os.getcwd(), DATADIR) #run this from citylearn folder
    data_dir = os.path.normpath(data_dir) #resolve '..'
    #load features
    df_obs = pd.read_csv(data_dir + '/rebaseline obs.csv',
                    header=None,
                    dtype='float32')
    df_obs.set_index(df_obs.index.astype(int), inplace=True)
    #load labels
    df_a = pd.read_csv(data_dir + '/rebaseline a.csv',
                header=None,
                dtype='float32')
    df_a.set_index(df_a.index.astype(int), inplace=True)
    df_obs['a'] = df_a[0]
    return df_obs



def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    #n_layers = trial.suggest_int("n_layers", 1, 3)
    n_layers = 2 #studies quickly converged here
    trial.set_user_attr('n_layers', n_layers)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"]) 
    #activation_fn = 'tanh'
    #trial.set_user_attr('activation_fn', activation_fn)
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    layers = []

    in_features = INPUTS
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 128, 512) #64-256 starting
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation_fn())
        p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES)) #only linear output needed for CE loss
    if(trial.suggest_categorical("output_layer",['linear','softmax']) == 'softmax'):
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)

def get_data_subset(dataset,train_ids, valid_ids, batch_size):
    """returns dataloaders with no shuffling as timeseries samples are NOT iid"""
    train_subsampler = Subset(dataset, 
                              train_ids)
    valid_subsampler = Subset(dataset, 
                              valid_ids)
    train_loader = torch.utils.data.DataLoader(train_subsampler,
                                              batch_size=batch_size,)
    valid_loader = torch.utils.data.DataLoader(valid_subsampler, 
                                             batch_size=batch_size,)
    return train_loader, valid_loader


class objective():
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self,trial):

        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.
        #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        optimizer_name = 'RMSprop'
        trial.set_user_attr('optimizer', optimizer_name)
        #lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        lr = 3.3e-4
        trial.set_user_attr('lr', lr)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        #select batch size
        batch_size = 2**trial.suggest_int('bs_exp', 3, 7) #6-10 starting
        trial.set_user_attr("batch size", batch_size) #display true value


        #generate time series folds, preserving the order of samples, so validation data is in the future of training data
        #ref: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
        tscv = TimeSeriesSplit(n_splits=FOLDS)
        acc_folds_sum = 0
        for fold, (train_ids, valid_ids) in enumerate(tscv.split(self.dataset)):
            # load training fold from dataset
            train_loader, valid_loader = get_data_subset(self.dataset,
                                                         train_ids, 
                                                         valid_ids, 
                                                         batch_size)
            # Training of the model.
            for epoch in range(EPOCHS):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):

                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

                # Validation of the model.
                model.eval()
                correct = 0
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(valid_loader):
                        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                        output = model(data)
                        # Get the index of the max logit
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()

                acc_epoch = correct / len(valid_loader.dataset)
                
            acc_folds_sum += acc_epoch
            #intermediate values are the cumulative mean accuracy at the end of each fold
            trial.report(acc_folds_sum/(fold + 1), step=fold) 
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        
        #TODO save best model if accuracy > trial.study.best_value:
        return acc_folds_sum/FOLDS


if __name__ == "__main__":
    df_SA = load_SA_dataset()
    study_dataset = MyDataset(df=df_SA)
    #defaults to median pruner
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS),
                                #reccomended pruner for TPE sampler
                                pruner=HyperbandPruner(max_resource=FOLDS,
                                                        min_resource=2,
                                                        reduction_factor=2,),
                                #produces 3 brackets ref: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner
                                #this corresponds to 3*N_STARTUP_TRIALS pruner startups
                                #optimally we would have 4, but FOLDS  interations is restrictive
                                load_if_exists=True,
                                storage=STORAGE,
                                study_name=NAME,
                                )
    study.set_user_attr('epochs',EPOCHS)
    study.set_user_attr('time series splits',FOLDS)
    study.set_user_attr('Optimizer','TPE')
    study.set_user_attr('Pruner','Hyperband(2,10,2)')
    study.set_metric_names([f'mean accuracy of {FOLDS} splits'])
    
    try:
        study.optimize(objective(dataset=study_dataset), 
                    n_trials=N_TRIALS, 
                    show_progress_bar=True,
                    n_jobs=JOBS)
    except KeyboardInterrupt:
        pass

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
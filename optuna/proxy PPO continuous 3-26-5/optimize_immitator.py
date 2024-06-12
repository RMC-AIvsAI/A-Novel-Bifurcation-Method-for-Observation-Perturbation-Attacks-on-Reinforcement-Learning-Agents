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

STORAGE = 'mysql+mysqlconnector://root:Broda1^6@localhost/optuna'
N_TRIALS = 100
N_STARTUP_TRIALS = N_TRIALS//10
DEVICE = torch.device("cuda")
JOBS = 1 # 2 is much slower, but I also doubled the epochs...
CLASSES = 20
INPUTS = 31
DIR = os.getcwd()
FOLDS = 10 #5 for first
EPOCHS = 1000//FOLDS #500 for first
DATADIR = 'Karla\3-26-5 PPOc Karla results' + '/'
NAME = 'SA Proxy 3-26-5 PPOc Policy'
SEED = 188

class MyDataset(Dataset):
    def __init__(self, df):
        self.data = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].unsqueeze(dim=-1)
    
def load_SA_dataset():
    #path to data directory
    data_dir = os.path.join(os.getcwd(), DATADIR) #run this from citylearn folder
    data_dir = os.path.normpath(data_dir) #resolve '..'
    #load features
    df_data = pd.read_csv(data_dir + '/baseline_obs-a.csv',
                    header=0,
                    index_col=0,
                    dtype='float32')
    df_data.set_index(df_data.index.astype(int), inplace=True)
    return df_data



def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    trial.set_user_attr('n_layers', n_layers)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"]) 
    trial.set_user_attr('activation_fn', activation_fn)
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    layers = []

    in_features = INPUTS
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 64, 512, step=16) 
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation_fn())
        p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.5, step=0.05)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, 1)) 

    return nn.Sequential(*layers)

def get_data_subset(dataset,train_ids, valid_ids, batch_size):
    """returns data with no shuffling as timeseries samples are NOT iid
    validation data is provided as a single batch"""
    train_subsampler = Subset(dataset, 
                              train_ids)
    valid_subsampler = Subset(dataset, 
                              valid_ids)
    train_loader = torch.utils.data.DataLoader(train_subsampler,
                                              batch_size=batch_size,)
    valid_loader = torch.utils.data.DataLoader(valid_subsampler, 
                                             batch_size=len(valid_subsampler),
                                             )
    assert len(valid_loader) == 1, 'validation data is not a single batch'
    return train_loader, valid_loader


class objective():
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self,trial):

        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        #select  TRAINING loss
        loss_fn = trial.suggest_categorical('loss function',['MSE','MAE','SMAE','Huber'])
        loss_fn = {'MSE':nn.MSELoss(),'MAE':nn.L1Loss(),'SMAE':nn.SmoothL1Loss(),'Huber':nn.HuberLoss()}[loss_fn]

        #select batch size
        batch_size = 2**trial.suggest_int('bs_exp', 3, 10)
        trial.set_user_attr("batch size", batch_size) #display true value


        #generate time series folds, preserving the order of samples, so validation data is in the future of training data
        #ref: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
        tscv = TimeSeriesSplit(n_splits=FOLDS)
        rmse_folds_sum = 0
        for fold, (train_ids, valid_ids) in enumerate(tscv.split(self.dataset)):
            # load training fold from dataset
            train_loader, valid_loader = get_data_subset(self.dataset,
                                                         train_ids, 
                                                         valid_ids, 
                                                         batch_size)
            # Training of the model, using selected loss function
            for epoch in range(EPOCHS):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):

                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

                    optimizer.zero_grad()
                    output = model(data) #output is 2d, with each pred a single item list, tagets is 1d
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                # Validation of the model using RMSE, so the result is always comparable
                model.eval()
                #correct = 0
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(valid_loader): #should run only once
                        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                        pred = model(data)
                        rmse = torch.sqrt(F.mse_loss(pred,target))

                        assert batch_idx < 1, 'validated more than 1 batch'

            #intermediate values are the cumulative mean accuracy at the end of each fold
            rmse_folds_sum+=rmse
            trial.report(rmse_folds_sum.item()/(fold + 1), step=fold)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        trial.set_user_attr('Final split accuracy',rmse.item())
        return rmse_folds_sum.item()/FOLDS # avoids TypeError: Object of type Tensor is not JSON serializable


if __name__ == "__main__":

    # Set pytorch num threads to 1 for faster training, suggested by reference example
    #torch.set_num_threads(1) #only for CPU?
    torch.manual_seed(SEED) #CPU 
    torch.cuda.manual_seed_all(SEED)

    #For reproducibility, but may hurt performance. But we need this to be replroducible for continued training
    #ref: https://pytorch.org/docs/stable/notes/randomness.html
        #https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # debug variable required to use deterministic algos (will increase library footprint in GPU memory by approximately 24MiB)
    torch.use_deterministic_algorithms(True) #raises error for stocastic algos
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #uses same algo every time, rather than trying multiple and choosing fastest

    df_SA = load_SA_dataset()
    study_dataset = MyDataset(df=df_SA)
    #defaults to median pruner
    study = optuna.create_study(direction="minimize",
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
    study.set_user_attr('Sampler',study.sampler.__class__.__name__)
    study.set_user_attr('Pruner',study.pruner.__class__.__name__)
    study.set_metric_names([f'mean RMSE of {FOLDS} splits'])
    
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
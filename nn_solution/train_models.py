import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import yaml

from scipy import sparse
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str)

# Sophia optimizer implementation from https://github.com/kyegomez/Sophia.git
class SophiaG(Optimizer):
    """
    SophiaG optimizer class.
    """

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False, dynamic: bool = False):
        """
        Initialize the optimizer.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter at index 1: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable, dynamic=dynamic)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        """
        Set the state of the optimizer.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
            group.setdefault('dynamic', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        """
        Update the hessian.
        """
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def update_exp_avg(self):
        """
        Update the exponential average.
        """
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        """
        Perform a step of the optimizer.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.update_hessian()
        self.update_exp_avg()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            self._sophiag(params_with_grad,
                          grads,
                          exp_avgs,
                          hessian,
                          state_steps,
                          bs=bs,
                          beta1=beta1,
                          beta2=beta2,
                          rho=group['rho'],
                          lr=group['lr'],
                          weight_decay=group['weight_decay'],
                          maximize=group['maximize'],
                          capturable=group['capturable'])

        return loss

    def _sophiag(self, params,
                 grads,
                 exp_avgs,
                 hessian,
                 state_steps,
                 capturable,
                 *,
                 bs: int,
                 beta1: float,
                 beta2: float,
                 rho: float,
                 lr: float,
                 weight_decay: float,
                 maximize: bool):
        """
        SophiaG function.
        """
        if not all(isinstance(t, torch.Tensor) for t in state_steps):
            raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

        self._single_tensor_sophiag(params,
                                    grads,
                                    exp_avgs,
                                    hessian,
                                    state_steps,
                                    bs=bs,
                                    beta1=beta1,
                                    beta2=beta2,
                                    rho=rho,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    maximize=maximize,
                                    capturable=capturable)

    def _single_tensor_sophiag(self, params,
                               grads,
                               exp_avgs,
                               hessian,
                               state_steps,
                               *,
                               bs: int,
                               beta1: float,
                               beta2: float,
                               rho: float,
                               lr: float,
                               weight_decay: float,
                               maximize: bool,
                               capturable: bool):
        """
        SophiaG function for single tensor.
        """
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            hess = hessian[i]
            step_t = state_steps[i]

            if capturable:
                assert param.is_cuda and step_t.is_cuda and bs.is_cuda

            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                hess = torch.view_as_real(hess)
                param = torch.view_as_real(param)

            # update step
            step_t += 1

            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            if capturable:
                step = step_t
                step_size = lr
                step_size_neg = step_size.neg()

                ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
                param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
            else:
                step = step_t.item()
                step_size_neg = - lr

                ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
                param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)


if __name__ == '__main__':

    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    DATA_DIR = config['base_path']  # sys.argv[1]
    T5_DIR = os.path.join(config['base_path'], config['embeds_path'], 't5')  # sys.argv[2]
    ESM2_DIR = os.path.join(config['base_path'], config['embeds_path'], 'esm_small')  # sys.argv[3]
    FEAT_DIR = os.path.join(config['base_path'], config['helpers_path'], 'feats')  # sys.argv[4]
    FOLDS_DIR = os.path.join(config['base_path'], config['helpers_path'], )  # sys.argv[5]
    MODEL_DIR = os.path.join(config['base_path'], config['models_path'], 'nn_serg')  # sys.argv[6]

    mode_loc = 'full'

    n_samples_to_consider = 142246  # 1000 #   50_000#   142246 #    downsampling - might be useful for debug:  small number - fast run
    n_labels_to_consider = 2000  # Up to 31466 but more than 3000-5000 may crash RAM
    n_folds_to_process = 5  # can reduce number of folds to speed-up - set 1,2,3 .. , , here 100 folds does NOT mean 100 folds - it just will be effectively clipped down the same number as in loaded folds file

    list_main_config_model_feature_etc = []  #

    cfg2 = {'model': {'id': 'pMLP', 'n_selfblend': 12, 'epochs': 20,
                      'custom_model_fit_function_name': 'model_fit_pytorch_Sophia_Andrey1'}}  # Pytorch MLP model
    list_main_config_model_feature_etc.append(cfg2)
    list_features_id = ['t5', 'esm2S1280']

    flag_compute_oof_predictions = True  # Compute local OOF predictions.  Necessary to compute local CV scores/other statistics

    # Compute statistics on OOF for each model
    flag_compute_stat_for_each_model = (False) and (flag_compute_oof_predictions)
    flag_compute_cafa_f1_for_each_model = (True) and (flag_compute_stat_for_each_model) and (
        flag_compute_oof_predictions)

    # Compute statistics on blend of OOF each blend
    flag_compute_each_blend_stat = (False) and (flag_compute_oof_predictions)
    flag_compute_cafa_f1_for_each_blend = (True) and (flag_compute_each_blend_stat) and (flag_compute_oof_predictions)

    # Compute stat on final blend
    flag_compute_final_model_stat = (False) and (flag_compute_oof_predictions)

    #########################################  Furthter params   ########################################################

    cutoff_threshold_low = 0.1  # prediction < cutoff_threshold_low will be set to zero (i.e. no need to save to submission file)

    mode_submit = True  # True # Compute prediction for submission part and prepare submission file in required CAFA5 format. Set to False if you only interested in local score

    ###  Save/not/what predictions   #####
    flag_save_final_submit_file = (True) and (mode_submit)  # Prepare and save final submission txt file.
    #  mode_submit - controls two things - 1) computation Y_submit  (and it save as numpy array) 2) final submission file, and

    # Set these flags to False if do not plan to blend current prediction with other models outside the notebook:
    flag_save_numpy_Y_pred_oof_blend = (True) and (
        flag_compute_oof_predictions)  # Save  Y_pred_oof_blend in numpy format "npy" for possible further blend outside current notebook
    flag_save_numpy_Y_submit = (True) and (
        mode_submit)  # Save Y_submit matrix in numpy  format "npy" for possible further blend outside current notebook

    mode_downsample_train_default = '43k'  # None #   'random_subsample_percent_90' #  '43k'  # None #

    RANDOM_SEED = None  # Fix or Not random seed

    #########################################  Information string  ########################################################

    str_id = str(list_features_id) + '_Y' + str(n_labels_to_consider)
    str_id += '_S' + str(n_samples_to_consider)
    str_id += '_CUT' + str(cutoff_threshold_low)
    print(str_id, len(str_id))
    print(str_id[:48])

    logs_file_path = 'logs.txt'


    def get_available_ram():
        virtual_memory = psutil.virtual_memory()
        available_ram = virtual_memory.available
        return available_ram


    def log_available_ram(str_for_logging_optional=None):
        try:
            virtual_memory = psutil.virtual_memory()
            available_ram_bytes = virtual_memory.available
            # Convert bytes to other units if needed (e.g., megabytes, gigabytes)
            available_ram_megabytes = available_ram_bytes / (1024 ** 2)
            available_ram_gigabytes = available_ram_bytes / (1024 ** 3)

            #     print(f"Available RAM: {available_ram_bytes} bytes")
            #     print(f"Available RAM: {available_ram_megabytes:.2f} MB")
            if str_for_logging_optional is not None:
                print(str_for_logging_optional)
            current_datetime = datetime.datetime.now()
            str1 = f"Available RAM: {available_ram_gigabytes:.2f} G  Current datetime: {current_datetime}"
            print(str1)

            with open(logs_file_path, 'a') as file:
                if str_for_logging_optional is not None:
                    file.write(str_for_logging_optional + '\n')
                file.write(str1 + '\n')
            # print("Data appended successfully.")
        except Exception as e:
            print(f"Error while appending data: {e}")

        #     return available_ram


    log_available_ram('On start')


    def seed_all(RANDOM_SEED):
        if RANDOM_SEED is not None:
            try:
                SEED = RANDOM_SEED
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            except Exception as e:
                print(f"Exception: {e}")


    seed_all(RANDOM_SEED)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)


    def get_paths_to_features(features_id):
        if features_id == 'esm2S1280':
            fn_X = os.path.join(ESM2_DIR, 'train_embeds.npy')
            fn_protein_ids = os.path.join(ESM2_DIR, 'train_ids.npy')
            fn_X_submit = os.path.join(ESM2_DIR, 'test_embeds.npy')
            fn_submit_protein_ids = os.path.join(ESM2_DIR, 'test_ids.npy')

        elif features_id == 't5':
            fn_X = os.path.join(T5_DIR, 'train_embeds.npy')
            fn_protein_ids = os.path.join(T5_DIR, 'train_ids.npy')
            fn_X_submit = os.path.join(T5_DIR, 'test_embeds.npy')
            fn_submit_protein_ids = os.path.join(T5_DIR, 'test_ids.npy')

        return fn_X, fn_protein_ids, fn_X_submit, fn_submit_protein_ids


    ################ load  features #########################

    print();
    print('!!! Pay attention proteins ids should be the same as in all the files !!!!!!!!!!!!!!!!!! ');
    print();


    def get_features(list_features_id, verbose=0):
        if verbose >= 100: print(list_features_id);

        X_submit, submit_protein_ids = None, None
        for i0, features_id in enumerate(list_features_id):
            # Pay attention proteins ids should be the same as in all the files !!!!!!!!!!!!!!!!!!

            fn_X, fn_protein_ids, fn_X_submit, fn_submit_protein_ids = get_paths_to_features(features_id)
            fn = fn_X  # '/kaggle/input/4637427/train_embeds_esm2_t36_3B_UR50D.npy'
            if verbose >= 100:  print(fn)
            if i0 == 0:
                X = np.load(fn).astype(np.float32)[:n_samples_to_consider, :]
            else:
                X = np.concatenate([X, np.load(fn).astype(np.float32)[:n_samples_to_consider, :]], axis=1)
            if verbose >= 100: print(X.shape)
            if verbose >= 100: print(X[:2, :3])
            protein_ids = np.load(fn_protein_ids)[:n_samples_to_consider]
            vec_train_protein_ids = protein_ids
            if verbose >= 100: print('protein_ids.shape:', protein_ids.shape)
            if verbose >= 100: print('protein_ids[:15]:', protein_ids[:15])

            ################ load  features for submit #########################
            if mode_submit:
                # fn = '/kaggle/input/4637427/train_embeds_esm2_t36_3B_UR50D.npy'
                # fn = '/kaggle/input/4637427/test_embeds_esm2_t36_3B_UR50D.npy'
                fn = fn_X_submit
                if verbose >= 100: print(fn)
                # X_submit = np.load(fn).astype(np.float32)
                if i0 == 0:
                    X_submit = np.load(fn).astype(np.float32)
                else:
                    X_submit = np.concatenate([X_submit, np.load(fn).astype(np.float32)], axis=1)
                if verbose >= 100: print(X_submit.shape)
                if verbose >= 100: print(X_submit[:2, :3])

                fn = fn_submit_protein_ids
                submit_protein_ids = np.load(fn)
                if verbose >= 100: print(submit_protein_ids.shape, submit_protein_ids[:10])
        return X, vec_train_protein_ids, X_submit, submit_protein_ids


    X, vec_train_protein_ids, X_submit, submit_protein_ids = get_features(list_features_id, verbose=100)

    # Load targets Y

    fn = os.path.join(FEAT_DIR, 'Y_31466_sparse_float32.npz')
    Y = sparse.load_npz(fn)
    print('Y', Y.shape, 'loaded')
    Y = Y[:n_samples_to_consider, :n_labels_to_consider].toarray()
    print('Y', Y.shape, 'truncated')
    n_labels_to_consider = Y.shape[1]  # in case n_labels_to_consider is greater that Y.shape we decrease it

    fn = os.path.join(FEAT_DIR, 'Y_31466_labels.npy')
    Y_labels = np.load(fn, allow_pickle=True)[:n_labels_to_consider]
    labels_to_consider = Y_labels
    print(Y_labels.shape)
    print(Y_labels[:20])

    import gc

    gc.collect()

    print('X mbytes:', X.nbytes / 1024 / 1024)
    print('Y mbytes:', Y.nbytes / 1024 / 1024)
    try:
        print('X_submit mbytes:', X_submit.nbytes / 1024 / 1024)
    except:
        pass
    log_available_ram('After data load')

    dict_set_allowed_train_indexes = {}

    fn = os.path.join(FEAT_DIR, 'train_ids_cut43k.npy')
    allowed_train_ids = np.load(fn)
    print(allowed_train_ids.shape, allowed_train_ids[:10])
    vec_allowed_train_indexes_43k = [ix for ix in range(len(vec_train_protein_ids)) if
                                     vec_train_protein_ids[ix] in (allowed_train_ids)]
    set_allowed_train_indexes_43k = set(vec_allowed_train_indexes_43k)
    dict_set_allowed_train_indexes['43k'] = set_allowed_train_indexes_43k
    print(len(dict_set_allowed_train_indexes['43k']), list(dict_set_allowed_train_indexes['43k'])[:10])


    def get_downsampled_IX_train(IX_train, mode_downsample_train):
        if mode_downsample_train in dict_set_allowed_train_indexes.keys():
            set_allowed_train_indexes = dict_set_allowed_train_indexes[mode_downsample_train]
            IX_train = [t for t in IX_train if t in set_allowed_train_indexes]
        elif 'random_subsample_percent' in str(mode_downsample_train):
            random_subsample_percent = float(str(mode_downsample_train).split('_')[-1])  # 'random_subsample_percent_90'
            IX_train = np.random.permutation(IX_train)[:int(len(IX_train) * random_subsample_percent / 100)]
        return IX_train


    fn = os.path.join(FOLDS_DIR, 'folds_gkf.npy')
    folds = np.load(fn)[:n_samples_to_consider]
    print(folds.shape, len(set(folds)))
    for k in set(folds):
        m = folds == k
        print(k, m.sum())


    class Model(nn.Module):
        def __init__(self, input_features, output_features):
            super().__init__()

            self.activation = nn.PReLU()

            self.bn1 = nn.BatchNorm1d(input_features)
            self.fc1 = nn.Linear(input_features, 800)
            self.ln1 = nn.LayerNorm(800, elementwise_affine=True)

            self.bn2 = nn.BatchNorm1d(800)
            self.fc2 = nn.Linear(800, 600)
            self.ln2 = nn.LayerNorm(600, elementwise_affine=True)

            self.bn3 = nn.BatchNorm1d(600)
            self.fc3 = nn.Linear(600, 400)
            self.ln3 = nn.LayerNorm(400, elementwise_affine=True)

            self.bn4 = nn.BatchNorm1d(1200)
            self.fc4 = nn.Linear(1200, output_features)
            self.ln4 = nn.LayerNorm(output_features, elementwise_affine=True)

            self.sigm = nn.Sigmoid()

        def forward(self, inputs):
            #         print(inputs.shape)

            fc1_out = self.bn1(inputs)
            fc1_out = self.ln1(self.fc1(inputs))
            fc1_out = self.activation(fc1_out)

            x = self.bn2(fc1_out)

            x = self.ln2(self.fc2(x))
            x = self.activation(x)

            x = self.bn3(x)

            x = self.ln3(self.fc3(x))
            x = self.activation(x)

            x = torch.cat([x, fc1_out], axis=-1)

            x = self.bn4(x)

            x = self.ln4(self.fc4(x))
            out = self.sigm(x)
            return out


    def get_model(model_config):
        str_model_id = str(model_config['id'])
        model = Model(X.shape[1], Y.shape[1])
        model.to(device)
        namepostfix = model_config.get('namepostfix', "")
        str_model_id += namepostfix
        return model, str_model_id


    def model_fit_pytorch_Sophia_Andrey1(model, X_train, Y_train, model_config, str_model_id='', verbose=1000):
        '''
        Fit pytorch model with Sophia optimizer, with certain params borrowed from: https://www.kaggle.com/code/andreylalaley/pytorch-cafa-5-prediction?scriptVersionId=138595845&cellId=32
        '''
        criterion = model_config.get('criterion', nn.BCELoss())
        max_epoch = model_config.get('epochs', 39)  # EPOCHS = 39
        BATCH_SIZE = model_config.get('batch_size', 128)
        LEARNING_RATE = model_config.get('LR', 0.001)

        criterion = model_config.get('criterion', nn.BCELoss())

        lr = LEARNING_RATE  # learning rate
        optimizer = SophiaG(model.parameters(), lr, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
        train_dataset = TensorDataset(X_train, Y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        if verbose >= 100:
            print(str_model_id, 'Start model training. X_train.shape, Y_train.shape', X_train.shape, Y_train.shape)
        log_available_ram(f'After X_train initialization. Right before model train {str_model_id}')

        ##################### Model Training ###################################################
        for epoch in range(max_epoch):
            t0_epoch = time.time()
            model.train()  # switch model into train mode. This helps inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and evaluation. For instance, in training mode, BatchNorm updates a moving average on each new batch; whereas, for evaluation mode, these updates are frozen.
            for i_batch, (x_batch, y_batch) in enumerate(train_dataloader):  # Loop ove batches
                # x_batch, y_batch = x_batch.to(device), y_batch.to(device) # do we need it ? may be already on device
                optimizer.zero_grad()  # technical - set gradients to zero, otherwise they will be accumulated

                preds = model(x_batch)  # Compute predictions only for batch samples

                loss = criterion(preds, y_batch)  # Compute loss function for batch predictions

                loss.backward()  # Compute gradients
                optimizer.step()  # Update NN weights using gradients

            # lr_sched is not used in original code
            # if lr_sched is not None:
            #    lr_sched.step() # Step LR scheduler

            if (verbose >= 10) and (i_batch % 100 == 0):
                print(str_model_id,
                      f'Epoch: {epoch}, batch: {i_batch},  train loss on batch: {loss.item():12.5f} , time: {time.time() - t0:.1f} ')


    def model_predict_pytorch(model, XX, model_config, str_model_id='', verbose=1000):
        t0_submit = time.time()
        XX = torch.tensor(XX, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            Y_pred = model(XX).cpu().numpy()

        if verbose >= 100: print(str_model_id,
                                 f'Y_pred.shape: {Y_pred.shape}, type(Y_pred): {type(Y_pred)}, predict on submit time: {time.time() - t0_submit :.1f} ')

        return Y_pred


    if 0:  # Code updated - now most params are specified for each model or in the top section "Key params"
        mode_submit = True
    verbose = 0

    ######################### Output ##################################################
    df_stat = pd.DataFrame()

    if (mode_submit is not None) and (mode_submit != False):
        Y_submit = np.zeros((141865, Y.shape[1]), dtype=np.float16)  # Predictions for submission will be stored here
        # Results from all models and all folds will be blended
        print('Y_submit mbytes:', Y_submit.nbytes / 1024 / 1024)
    cnt_blend_submit = 0;

    if flag_compute_oof_predictions:
        Y_pred_oof_blend = np.zeros((Y.shape), dtype=np.float16)
        print('Y_pred_oof_blend mbytes:', Y_pred_oof_blend.nbytes / 1024 / 1024)
    cnt_blend_oof = -1;

    ########################## Preparations ###########################################
    log_available_ram('Right before modeling')

    if flag_compute_stat_for_each_model:  # Predictions OOF for each particular model - will be rewritten for each modelling
        Y_pred_oof = np.zeros((Y.shape), dtype=np.float16)
        print('Y_pred_oof mbytes:', Y_pred_oof.nbytes / 1024 / 1024)

    i_model = -1  #
    i_config = -1  # conter for configurations
    t0modeling = time.time()
    list_folds_ix = np.sort(list(set(folds)))
    print();
    print('Start training models', datetime.datetime.now());
    print()
    ########################## Main modelling  ###########################################
    for main_config_model_feature_etc in list_main_config_model_feature_etc:
        i_config += 1
        model_config = main_config_model_feature_etc['model']
        if ('Keras' in model_config.keys()) and (
                model_config['Keras']): continue  # Keras models will be processed in the next cell - RAM leak problem
        if 'list_features_id' in main_config_model_feature_etc.keys():
            print()
            X, vec_train_protein_ids, X_submit, submit_protein_ids = get_features(
                main_config_model_feature_etc['list_features_id'], verbose=100)
            gc.collect()
            log_available_ram(f"New features loaded:  {str(main_config_model_feature_etc['list_features_id'])}")
            print()

        mode_downsample_train = model_config.get('mode_downsample_train', mode_downsample_train_default)

        n_selfblend = model_config.get('n_selfblend', 1)
        if verbose >= 100:
            print();
            print('Starting model_config:', model_config, f'time from start: {(time.time() - t0modeling):.1f}')
        for i_selfblend in range(
                n_selfblend):  # train-predict same model several times and blend predictions - especially useful for NN, but do not fix random seed (!)
            i_model += 1  # Models count
            t0one_model_all_folds = time.time()
            for ix_fold in list_folds_ix[:n_folds_to_process]:

                model, str_model_id = get_model(model_config)
                str_model_id_pure_save = str_model_id
                str_model_id = str(i_model) + ' ' + str_model_id
                if n_selfblend > 1:  str_model_id += ' ' + str(i_selfblend)

                ##################### Prepare train data ###################################################
                mask_fold = folds == ix_fold
                IX_train = np.where(mask_fold == 0)[0];
                # IX_train = [ix for ix in IX_train if ix in  set_allowed_train_indexes]
                IX_train = get_downsampled_IX_train(IX_train, mode_downsample_train)
                X_train = X[IX_train, :];
                Y_train = Y[IX_train, :]

                if verbose >= 10:
                    print(
                        f'fold {ix_fold}, model: {str_model_id},  X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}, time: {(time.time() - t0modeling):12.1f} ')
                    print('X_train Mbytes:', X_train.nbytes / 1024 / 1024, 'Y_train Mbytes:',
                          Y_train.nbytes / 1024 / 1024, )
                ##################### Call train model ###################################################
                t0 = time.time()
                model_fit_pytorch_Sophia_Andrey1(model, X_train, Y_train, model_config, str_model_id, verbose=0)
                time_fit = time.time() - t0
                if verbose >= 1000:
                    print(f'time_fit {time_fit:.1f}')
                del X_train, Y_train
                torch.cuda.empty_cache()
                gc.collect()
                log_available_ram(f'After Model fit. ix_fold {ix_fold}, i_selfblend, {i_selfblend}, {str_model_id}')

                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'model_{i_selfblend}_{ix_fold}.pt'))

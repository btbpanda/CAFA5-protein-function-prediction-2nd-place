import argparse
import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import yaml
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str)

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
    OUT_DIR = os.path.join(config['base_path'], config['models_path'], 'nn_serg')  # sys.argv[7]

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
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'model_{i_selfblend}_{ix_fold}.pt')))
                time_fit = time.time() - t0
                if verbose >= 1000:
                    print(f'time_fit {time_fit:.1f}')
                del X_train, Y_train
                torch.cuda.empty_cache()
                gc.collect()
                log_available_ram(f'After Model fit. ix_fold {ix_fold}, i_selfblend, {i_selfblend}, {str_model_id}')

                ##################### Compute predictions for submission and blend with the previous one ###################################################
                if mode_submit:
                    t0 = time.time()
                    Y_submit = (Y_submit * cnt_blend_submit + model_predict_pytorch(model, X_submit, model_config,
                                                                                    str_model_id, verbose=0)) / (
                                       cnt_blend_submit + 1);  # Average predictions from different folds/models
                    cnt_blend_submit += 1
                    time_pred_submit = time.time() - t0
                    torch.cuda.empty_cache()
                    gc.collect()

                if flag_compute_oof_predictions:
                    t0 = time.time()
                    IX_val = np.where(mask_fold > 0)[0];
                    X_val = X[IX_val, :];  # Y_val = Y[IX_val,:]
                    Y_pred_val = model_predict_pytorch(model, X_val, model_config, str_model_id, verbose=0)
                    time_pred_val = time.time() - t0
                    if verbose >= 10000:
                        print('Y_pred_val.shape', Y_pred_val.shape, f'time_pred_val {time_pred_val:.1f}')

                    if ix_fold == 0: cnt_blend_oof += 1
                    Y_pred_oof_blend[IX_val, :] = (Y_pred_oof_blend[IX_val, :] * cnt_blend_oof + Y_pred_val) / (
                            cnt_blend_oof + 1);

                    if flag_compute_stat_for_each_model:
                        Y_pred_oof[IX_val, :] = (Y_pred_val)

                    del X_val, Y_pred_val
                    torch.cuda.empty_cache()
                    gc.collect()

    if flag_save_numpy_Y_pred_oof_blend and flag_compute_oof_predictions:
        t0 = time.time()
        fn = os.path.join(OUT_DIR, 'Y_pred_oof_blend.npy')
        np.save(fn, Y_pred_oof_blend)
        print(f'File {fn} saved. Y_pred_oof_blend.shape: {Y_pred_oof_blend.shape}. Time: {(time.time() - t0):.1f}')
        t0 = time.time()
        fn = os.path.join(OUT_DIR, 'Y_labels.npy')
        np.save(fn, Y_labels)
        print(f'File {fn} saved. Time: {(time.time() - t0):.1f}')

    if flag_save_numpy_Y_submit and mode_submit:
        t0 = time.time()
        fn = os.path.join(OUT_DIR, 'Y_submit.npy')
        np.save(fn, Y_submit)
        print(f'File {fn} saved. Y_submit.shape: {Y_submit.shape}. Time: {(time.time() - t0):.1f}')
        t0 = time.time()
        fn = os.path.join(OUT_DIR, 'Y_labels.npy')
        np.save(fn, Y_labels)
        print(f'File {fn} saved. Time: {(time.time() - t0):.1f}')
        log_available_ram(f'After Save Y_submit')

    if flag_compute_final_model_stat and flag_compute_oof_predictions:
        # time_one_model = np.round( time.time() - t0one_model_all_folds )
        update_modeling_stat(df_stat, Y_pred_oof_blend, Y, flag_compute_cafa_f1=True, str_model_id='FinalKeras Blend',
                             dict_optional_info={}, verbose=0)
        gc.collect()
        log_available_ram('After Final Stat Calculation')

bash create_nn_env.sh;
mkdir models;
mkdir feats;
python prepare.py "/kaggle/input/cafa-5-protein-function-prediction/" "/kaggle/input/t5embeds/" "feats/";
python t5.py "/kaggle/input/cafa-5-protein-function-prediction/" "t5_embs/";
python esm2sm.py "/kaggle/input/cafa-5-protein-function-prediction/" "esm2_embs/";
python train_models.py "/kaggle/input/cafa-5-protein-function-prediction/" "t5_embs/" "esm2_embs/" "feats/" "folds/" "models/";
python inference_models.py "/kaggle/input/cafa-5-protein-function-prediction/" "t5_embs/" "esm2_embs/" "feats/" "folds/" "models/" "out/";

###########################################################
# Change the following values to train a new model.
# type: the name of the new model, only affects the saved file name.
# dataset: the name of the dataset, as was preprocessed using preprocess.sh
# test_data: by default, points to the validation set, since this is the set that
#   will be evaluated after each training iteration. If you wish to test
#   on the final (held-out) test set, change 'val' to 'test'.

dataset_name=$1
partition=$2
snapshot_name=$3
data_dir=<<<CODE2SEQ_PREPROCESSED_DATA_PATH>>>/${dataset_name}
test_data=${data_dir}/${dataset_name}.${partition}.c2s
parameters_file=<<<MODELS_SAVE_PATH>>>/code2seq/${snapshot_name}
output_dir=<<<MODELS_SAVE_PATH>>>/code2seq-evaluation/${snapshot_name}/${partition}

set -e
python3 -u code2seq.py --test ${test_data} --load ${parameters_file} --save_prefix ${output_dir}

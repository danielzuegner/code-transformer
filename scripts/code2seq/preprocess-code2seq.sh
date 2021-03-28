input_data_path=<<<DATA_PATH_STAGE_1>>>/<<<CODE2SEQ_DATASET>>>
output_data_path=<<<CODE2SEQ_PREPROCESSED_DATA_PATH>>>
language=$1

MAX_DATA_CONTEXTS=1000
MAX_CONTEXTS=200

# For the Code Transformer we use a min_frequency limit of 100 for the vocabulary
# code2seq uses an absolute vocabulary size limit. The following numbers represent exactly our min_frequency=100

case "$1" in
  go)
  # Go
    SUBTOKEN_VOCAB_SIZE=5653
    TARGET_VOCAB_SIZE=5653
    ;;

  javascript)
    # JavaScript
    SUBTOKEN_VOCAB_SIZE=3955
    TARGET_VOCAB_SIZE=3955
    ;;

  ruby)
    # Ruby
    SUBTOKEN_VOCAB_SIZE=1793
    TARGET_VOCAB_SIZE=1793
    ;;

  python)
    # Python
    SUBTOKEN_VOCAB_SIZE=9363
    TARGET_VOCAB_SIZE=9363
    ;;

  java-small)
    # java-small
    SUBTOKEN_VOCAB_SIZE=4936
    TARGET_VOCAB_SIZE=4936
    ;;
esac


PYTHON=python

cd "${0%/*}" || exit

# Extract methods with paths
cd ../
python -m scripts.preprocess-code2seq "$language" train $input_data_path $output_data_path
python -m scripts.preprocess-code2seq "$language" valid $input_data_path $output_data_path
python -m scripts.preprocess-code2seq "$language" test $input_data_path $output_data_path

TRAIN_DATA_FILE=$output_data_path/"$language"/train.txt
VAL_DATA_FILE=$output_data_path/"$language"/valid.txt
TEST_DATA_FILE=$output_data_path/"$language"/test.txt

TARGET_HISTOGRAM_FILE=$output_data_path/"$language"/"$language".histo.tgt.c2s
SOURCE_SUBTOKEN_HISTOGRAM=$output_data_path/"$language"/"$language".histo.ori.c2s
NODE_HISTOGRAM_FILE=$output_data_path/"$language"/"$language".histo.node.c2s

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_SUBTOKEN_HISTOGRAM}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}

${PYTHON} scripts/preprocess-code2seq-helper.py --train_data ${TRAIN_DATA_FILE} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --max_data_contexts ${MAX_DATA_CONTEXTS} --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --subtoken_histogram ${SOURCE_SUBTOKEN_HISTOGRAM} \
  --node_histogram ${NODE_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name $output_data_path/"$language"/"$language"

# If all went well, the raw data files can be deleted, because preprocess.py creates new files
# with truncated and padded number of paths for each example.
#rm ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE} ${TARGET_HISTOGRAM_FILE} ${SOURCE_SUBTOKEN_HISTOGRAM} \
#  ${NODE_HISTOGRAM_FILE}

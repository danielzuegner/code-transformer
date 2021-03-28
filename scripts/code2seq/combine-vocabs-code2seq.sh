#!/usr/bin/env bash

output_data_path=<<<CODE2SEQ_PREPROCESSED_DATA_PATH>>>
language=$1

MAX_DATA_CONTEXTS=1000
MAX_CONTEXTS=200
PYTHON=python

cd "${0%/*}" || exit
cd ../..

IFS=',' read -ra languages <<< "$language"
input_paths=""
for lang in "${languages[@]}"; do
  input_paths="$input_paths $output_data_path/$lang/train.txt"
done

TARGET_HISTOGRAM_FILE=$output_data_path/"$language"/"$language".histo.tgt.c2s
SOURCE_SUBTOKEN_HISTOGRAM=$output_data_path/"$language"/"$language".histo.ori.c2s
NODE_HISTOGRAM_FILE=$output_data_path/"$language"/"$language".histo.node.c2s

mkdir -p "$output_data_path/$language"

echo "Creating histograms from the training data"
cat ${input_paths} | cut -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${input_paths} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_SUBTOKEN_HISTOGRAM}
cat ${input_paths} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}


for lang in "${languages[@]}"; do
  case "$lang" in
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

  train_data_file="$output_data_path/$lang/train.txt"
  val_data_file="$output_data_path/$lang/valid.txt"
  test_data_file="$output_data_path/$lang/test.txt"



  ${PYTHON} scripts/preprocess-code2seq-helper.py --train_data "${train_data_file}" --test_data "${test_data_file}" --val_data "${val_data_file}" \
  --max_contexts ${MAX_CONTEXTS} --max_data_contexts ${MAX_DATA_CONTEXTS} --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --subtoken_histogram "${SOURCE_SUBTOKEN_HISTOGRAM}" \
  --node_histogram "${NODE_HISTOGRAM_FILE}" --target_histogram "${TARGET_HISTOGRAM_FILE}" --output_name $output_data_path/"$language"/"$lang"
done

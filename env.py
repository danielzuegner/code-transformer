_DATA_PATH = <<<FOLDER_TO_STORE_DATASETS>>>
_BINARY_PATH = <<<FOLDER_TO_STORE_BINARIES>>>
MODELS_SAVE_PATH = <<<FOLDER_TO_STORE_MODEL_CHECKPOINTS>>>
LOGS_PATH = <<<FOLDER_TO_STORE_TRAIN_LOGS>>>

"""
Per default, we assume the following folder structure:
_DATA_PATH
 ├── raw
 │   ├── csn
 │   │   ├── python
 │   │   │   └── final
 │   │   │       └── ...
 │   │   :
 │   │   └── go
 │   ├── code2seq
 │   │   └── java-small
 │   └── code2seq-methods
 │       └── java-small
 ├── stage1
 └── stage2
     ├── python
     │   ├── train
     │   ├── valid
     │   ├── test
     │   └── vocabularies.p.gzip
     ├── java-small
     :
     └── python,javascript,go,ruby
     
 
_BINARY_PATH
 ├── java-parser-1.0-SNAPSHOT.jar
 ├── JavaMethodExtractor-1.0.0-SNAPSHOT.jar
 └── semantic
 
MODELS_SAVE_PATH
 ├── ct_lm
 ├── ct_code_summarization
 │   ├── CT-1
 │   │   ├── config.json
 │   │   ├── model_10000.p
 │   │   :
 │   │   └── model_450000.p
 │   :   
 │   └── CT-24
 ├── great_code_summarization
 └── xl_net_code_summarization
"""

CSN_RAW_DATA_PATH = f"{_DATA_PATH}/raw/csn"
CODE2SEQ_RAW_DATA_PATH = f"{_DATA_PATH}/raw/code2seq"
CODE2SEQ_EXTRACTED_METHODS_DATA_PATH = f"{_DATA_PATH}/raw/code2seq-methods"

DATA_PATH_STAGE_1 = f"{_DATA_PATH}/stage1"
DATA_PATH_STAGE_2 = f"{_DATA_PATH}/stage2"

JAVA_EXECUTABLE = f"java"
JAVA_PARSER_EXECUTABLE = f"{_BINARY_PATH}/java-parser-1.0-SNAPSHOT.jar"
JAVA_METHOD_EXTRACTOR_EXECUTABLE = f"{_BINARY_PATH}/JavaMethodExtractor-1.0.0-SNAPSHOT.jar"
SEMANTIC_EXECUTABLE = f"{_BINARY_PATH}/semantic"

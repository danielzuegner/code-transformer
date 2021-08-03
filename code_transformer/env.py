"""
Per default, we assume the following folder structure:
CODE_TRANSFORMER_DATA_PATH
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


CODE_TRANSFORMER_BINARY_PATH
 ├── java-parser-1.0-SNAPSHOT.jar
 ├── JavaMethodExtractor-1.0.0-SNAPSHOT.jar
 └── semantic

CODE_TRANSFORMER_MODELS_PATH
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

from environs import Env
from pathlib import Path

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/code_transformer/.env")
if env_file_path.exists():
    env.read_env(env_file_path, recurse=False)

with env.prefixed("CODE_TRANSFORMER_"):

    _DATA_PATH = env("DATA_PATH")
    _BINARY_PATH = env("BINARY_PATH")
    MODELS_SAVE_PATH = env("MODELS_PATH")
    LOGS_PATH = env("LOGS_PATH")

    CSN_RAW_DATA_PATH = env("CSN_RAW_DATA_PATH", f"{_DATA_PATH}/raw/csn")
    POJ_RAW_DATA_PATH = env("POJ_RAW_DATA_PATH", f"{_DATA_PATH}/poj_104/raw")
    CODEFORCES_RAW_DATA_PATH = env("CODEFORCES_RAW_DATA_PATH", f"{_DATA_PATH}/codeforces/raw")
    CODE2SEQ_RAW_DATA_PATH = env("CODE2SEQ_RAW_DATA_PATH", f"{_DATA_PATH}/raw/code2seq")
    CODE2SEQ_EXTRACTED_METHODS_DATA_PATH = env("CODE2SEQ_EXTRACTED_METHODS_DATA_PATH", f"{_DATA_PATH}/raw/code2seq-methods")

    DATA_PATH_STAGE_1 = env("DATA_PATH_STAGE_1", f"{_DATA_PATH}/stage1")
    DATA_PATH_STAGE_2 = env("DATA_PATH_STAGE_2", f"{_DATA_PATH}/stage2")
    POJ_DATA_PATH_STAGE_1 = env("DATA_PATH_STAGE_1", f"{_DATA_PATH}/poj_104/stage1")
    POJ_DATA_PATH_STAGE_2 = env("DATA_PATH_STAGE_2", f"{_DATA_PATH}/poj_104/stage2")
    CODEFORCES_DATA_PATH_STAGE_1 = env("DATA_PATH_STAGE_1", f"{_DATA_PATH}/codeforces/stage1")
    CODEFORCES_DATA_PATH_STAGE_2 = env("DATA_PATH_STAGE_2", f"{_DATA_PATH}/codeforces/stage2")

    JAVA_EXECUTABLE = env("JAVA_EXECUTABLE", "java")
    JAVA_PARSER_EXECUTABLE = env("JAVA_PARSER_EXECUTABLE", f"{_BINARY_PATH}/java-parser-1.0-SNAPSHOT.jar")
    JAVA_METHOD_EXTRACTOR_EXECUTABLE = env("JAVA_METHOD_EXTRACTOR_EXECUTABLE", f"{_BINARY_PATH}/JavaMethodExtractor-1.0.0-SNAPSHOT.jar")
    SEMANTIC_EXECUTABLE = env("SEMANTIC_EXECUTABLE", f"{_BINARY_PATH}/semantic")

# =============================================================================
# Preprocess stage 1
# =============================================================================
UNKNOWN_TOKEN = "<unk>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "[PAD]"  # Padding for sub tokens
CLS_TOKEN = "[CLS]"  # Used for classification tasks
NUM_CLS_TOKENS = 10
CLS_TOKENS = [f"[CLS_{i + 1}]" for i in range(NUM_CLS_TOKENS)]  # Used when each subtoken should have its own embedding
SEP_TOKEN = "[SEP]"
MASK_STRING = "[MASK_STRING]"  # internally replaces all hardcoded string values in a code snippet
MASK_NUMBER = "[MASK_NUMBER]"  # internally replaces all hardcoded numerical values
MASK_METHOD_NAME = "[MASK_METHOD_NAME]"  # Masks the original method name that is to be predicted
INDENT_TOKEN = "[INDENT]"  # Represents that everything afterwards should be indented by one additional level
DEDENT_TOKEN = "[DEDENT]"  # Represents that everything afterwards i indented by one level less
NUM_SUB_TOKENS = 5  # In how many subtokens one token should be split into, .e.g., get_db_connection => [get, db,
# connection, PAD, PAD]
NUM_SUB_TOKENS_METHOD_NAME = 6  # Maximum number of subtokens for the label, i.e., the method name that is to be predicted
MAX_NUM_TOKENS = 512  # Any Snippet that produces more than the specified amount of tokens will be discarded

# =============================================================================
# Preprocess stage 2
# =============================================================================
BIN_PADDING = 0  # Padding value to be used for discrete distance matrices when #unique values < NUM_BINS
VOCAB_SIZE_TOKENS = 32000  # How many sub-tokens should appear in the final vocabulary. Everything else will be
# treated like <unk>

# Special symbols for the node and token type vocabularies
SPECIAL_SYMBOLS_NODE_TOKEN_TYPES = [
    UNKNOWN_TOKEN,  # 0
    EOS_TOKEN,  # 1
    CLS_TOKEN  # 2
]
SPECIAL_SYMBOLS_NODE_TOKEN_TYPES = dict(
    zip(SPECIAL_SYMBOLS_NODE_TOKEN_TYPES, range(len(SPECIAL_SYMBOLS_NODE_TOKEN_TYPES))))

# =============================================================================
# Vocabulary
# =============================================================================

# Special symbols for the main token vocabulary
SPECIAL_SYMBOLS = [
    UNKNOWN_TOKEN,  # 0
    SOS_TOKEN,  # 1
    EOS_TOKEN,  # 2
    PAD_TOKEN,  # 3
    CLS_TOKEN,  # ...
    SEP_TOKEN,
    "[MASK]",
    "<eod>",
    "<eop>",
    MASK_STRING,
    MASK_NUMBER,
    MASK_METHOD_NAME,
    INDENT_TOKEN,
    DEDENT_TOKEN
]
SPECIAL_SYMBOLS.extend(CLS_TOKENS)
SPECIAL_SYMBOLS = dict(zip(SPECIAL_SYMBOLS, range(len(SPECIAL_SYMBOLS))))

from enum import Enum


class AttentionType(Enum):
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    ADDITIVE = "additive"
    MULTIHEAD = "multihead"

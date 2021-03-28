import ast
import json

import jsonpickle
import yaml


def _restore(flat):
    """
    Restore more complex data that Python's json can't handle (e.g. Numpy arrays).
    Copied from sacred.serializer for performance reasons.
    """
    return jsonpickle.decode(json.dumps(flat), keys=True)


def _convert_value(value):
    """
    Parse string as python literal if possible and fallback to string.
    Copied from sacred.arg_parser for performance reasons.
    """

    try:
        return _restore(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        # use as string if nothing else worked
        return value


def _convert_values(val):
    if isinstance(val, dict):
        for key, inner_val in val.items():
            val[key] = _convert_values(inner_val)
    elif isinstance(val, list):
        for i, inner_val in enumerate(val):
            val[i] = _convert_values(inner_val)
    elif isinstance(val, str):
        return _convert_value(val)
    return val


def read_config(config_path):
    with open(config_path, 'r') as conf:
        config_dict = _convert_values(yaml.load(conf, Loader=yaml.FullLoader))

    return config_dict


def parse_command(config):
    config_strings = [f'{key}="{val}"' if type(val) != str else f'{key}="\'{val}\'"' for key, val in config.items() if
                      not key == 'experiment_setup']
    exe = config['experiment_setup']['executable']
    cmd = f"PYTHONPATH=$(pwd) python {exe} with {' '.join(config_strings)}"
    return exe, cmd

#!/usr/bin/env python3


import re

import vizdoom as vzd


def _get_object_methods(_object):
    object_methods = [
        method_name
        for method_name in dir(_object)
        if callable(getattr(_object, method_name)) and not method_name.startswith("_")
    ]
    return object_methods


def _check_object_docstrings(_object):
    object_methods = _get_object_methods(_object)

    for method in object_methods:
        method_doc = getattr(_object, method).__doc__
        assert method_doc is not None, f"Method {method} has no docstring"

        if method.startswith("set_") and method not in {
            "set_action",
            "set_button_max_value",
            "set_config",
        }:
            search_result = re.search(r"Default value: (.+)", method_doc)
            assert search_result is not None, f"Method {method} has no default value"
            # default_value = search_result.group(1).strip(".").strip("`")
            search_result = re.search(r"Config key: ``(.+)``", method_doc)
            assert search_result is not None, f"Method {method} has no config key"
            config_keys = search_result.group(1).split("/")
            config_keys = [key.strip("`") for key in config_keys if key.strip()]
            if len(config_keys) > 1:
                assert (
                    config_keys[1].replace("_", "").lower() == config_keys[0].lower()
                ), f"Config keys do not match: {config_keys}"
            assert (
                method[4:] in config_keys
            ), f"{method} name does not match its config key: {config_keys}"


def test_docstrings():
    print("Testing all docstrings ...")
    _check_object_docstrings(vzd.DoomGame)


if __name__ == "__main__":
    test_docstrings()

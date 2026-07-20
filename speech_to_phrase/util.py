"""Utility methods."""

import re
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

yaml = YAML(typ="safe")

yaml_output = YAML()
yaml_output.explicit_start = True
yaml_output.default_flow_style = False
yaml_output.indent(sequence=4, offset=2)


def get_language_family(language: str) -> str:
    """Get language family (en_US -> en)."""
    return re.split("[-_]", language, maxsplit=1)[0]


def quote_strings(data: Any) -> Any:
    """Double quote value strings in data for YAML."""
    if isinstance(data, str):
        return DoubleQuotedScalarString(data)

    if isinstance(data, list):
        return [quote_strings(item) for item in data]

    if isinstance(data, dict):
        return {key: quote_strings(value) for key, value in data.items()}

    return data

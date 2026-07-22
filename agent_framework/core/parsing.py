from __future__ import annotations

import json
from typing import Any


def extract_json_value(raw: str, expected_type: type) -> Any | None:
    """Extract the first balanced JSON object or array of the requested type."""
    opener, closer = ("{", "}") if expected_type is dict else ("[", "]")
    start = raw.find(opener)
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(raw[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    return value if isinstance(value, expected_type) else None
        start = raw.find(opener, start + 1)
    return None

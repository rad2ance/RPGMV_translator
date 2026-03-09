SYSTEM_PROMPT_TEMPLATE = (
    "Translate Japanese strings to {target_language}. "
    "Return exactly one JSON object where each key is the exact original source string "
    "and each value is the translated target-language string. "
    "Do not translate English-only strings. "
    "Do not return any text outside the JSON object."
)

DEFAULT_MISSING_MARKER_PREFIX = "__MISSING_TRANSLATION__"
YES_VALUES = {"y", "yes"}
DEFAULT_SOURCE_ENCODING = "utf-8-sig"

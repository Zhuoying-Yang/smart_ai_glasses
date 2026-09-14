from enum import Enum


class Action(str, Enum):
    SKIP = "SKIP"
    MEMORY_ONLY = "MEMORY_ONLY"

    SMALL_1F = "SMALL_1F"
    SMALL_MULTI = "SMALL_MULTI"

    LARGE_1F = "LARGE_1F"
    LARGE_MULTI = "LARGE_MULTI"

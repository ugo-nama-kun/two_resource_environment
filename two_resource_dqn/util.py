from enum import Enum, auto


class ActionType(Enum):
    NONE = 0
    FORWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    EAT_BEHAVIOR = auto()

from enum import Enum, unique

@unique
class PropertyType(Enum):
    unreachability = 1
    memory_safety = 2
    termination = 3

@unique
class Status(Enum):
    true = 1
    false = 2
    unknown = 3
from dataclasses import dataclass
from enum import IntEnum, auto

class OpType(IntEnum):
    PUSH = auto() # Operand: int
    PUSH_EXTERN = auto() # Operand: str
    CALL = auto() # Operand: int
    RET = auto() # Operand: bool (true = return a value, false = no return value)

OpOperand = int | str | bool

@dataclass
class Op:
    type: OpType
    operand: OpOperand


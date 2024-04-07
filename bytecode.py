from typing import Optional
from dataclasses import dataclass
from enum import IntEnum, auto

class OpType(IntEnum):
    PUSH = auto() # Operand: int
    PUSH_EXTERN = auto() # Operand: str
    CALL = auto() # Operand: int
    RET = auto() # Operand: bool (true = return a value, false = no return value)
    LOAD = auto() # Operand: int (size)
    STORE = auto() # Operand: int (size)
    ADD = auto() # Operand: None
    JMP_IF_ZERO = auto() # Operand: int
    JMP = auto() # Operand: int

OpOperand = Optional[int | str | bool]

@dataclass
class Op:
    type: OpType
    operand: OpOperand = None


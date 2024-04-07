#!/usr/bin/env python3
from typing import Optional

import sys
import shlex
import subprocess
from enum import StrEnum, auto
from dataclasses import dataclass
from abc import ABC

from pycparser import CParser
from pycparser.c_ast import *

from bytecode import Op, OpType

# TODO: handle consts
class CType(ABC):
    ...

class BasicCTypeKind(StrEnum):
    VOID = auto()
    INT = auto()
    CHAR = auto()

    @classmethod
    def from_name(cls, name):
        return cls[name.upper()]

@dataclass
class BasicCType:
    kind: BasicCTypeKind

@dataclass
class CPtrType:
    inner: BasicCType

def type_from_ast(node):
    match node:
        case TypeDecl(type=inner):
            return type_from_ast(inner)
        case IdentifierType(names=names):
            # TODO: handle multiple names (like 'unsigned int x')
            # TODO: handle devilish type names (like 'unsigned x')
            name = names[-1]
            return BasicCType(BasicCTypeKind.from_name(name))
        case PtrDecl(type=TypeDecl(type=type_)):
            inner = type_from_ast(type_)
            return CPtrType(inner)
        case _:
            assert False, f"{node.__class__.__name__} type not implemented yet"
    exit(69)

@dataclass
class CFunc:
    params: list[CType]
    return_type: CType
    body: Optional[int] = None

class BytecodeGenerator(NodeVisitor):
    def __init__(self):
        self.ops = []
        self.funcs = {}
        self.init_mem = bytearray()

    def visit_FileAST(self, node):
        super().generic_visit(node)

    def visit_Decl(self, node):
        super().generic_visit(node)

    def visit_FuncDecl(self, node):
        params = []
        for param in node.args.params:
            params.append(type_from_ast(param.type))
        return_type = type_from_ast(node.type.type)
        ret = self.funcs[node.type.declname] = CFunc(params, return_type)
        return ret

    def visit_FuncDef(self, node):
        func = self.visit(node.decl.type)
        ip = len(self.ops)
        func.body = ip
        self.visit(node.body)

    def visit_Constant(self, node):
        match node.type:
            case "string":
                actual_buf = node.value.encode("utf-8").decode("unicode-escape").encode("utf-8")[1:-1]
                ptr = len(self.init_mem)
                self.init_mem.extend(actual_buf)
                self.init_mem.append(0)
                self.ops.append(Op(OpType.PUSH, ptr))
            case "int":
                self.ops.append(Op(OpType.PUSH, int(node.value)))
            case t:
                assert False, f"Unsupported constant type {t!r}"

    def visit_FuncCall(self, node):
        for expr in reversed(node.args.exprs):
            self.visit(expr)
        name = node.name.name
        func = self.funcs[name]
        if func.body is None:
            self.ops.append(Op(OpType.PUSH_EXTERN, name))
        else:
            self.ops.append(Op(OpType.PUSH, func.body))
        self.ops.append(Op(OpType.CALL, len(node.args.exprs)))

    def visit_Return(self, node):
        if node.expr is not None:
            self.visit(node.expr)
        self.ops.append(Op(OpType.RET, node.expr is not None))

    def visit_Compound(self, node):
        for stmt in node.block_items:
            self.visit(stmt)

    def generic_visit(self, node):
        assert False, f"Node {node.__class__.__name__} is not implemented yet"

def builtin_puts(stack, memory):
    if len(stack) < 1:
        print("Not enough arguments for puts", file=sys.stderr)
        return
    ptr = stack.pop()
    while memory[ptr] != 0:
        print(chr(memory[ptr]), end="")
        ptr += 1
    print()

BUILTINS = {"puts": builtin_puts}

def run_bc(generator):
    ip = generator.funcs["main"].body
    stack = []
    memory = generator.init_mem
    ops = generator.ops
    while ip < len(ops):
        op = ops[ip]
        ip += 1
        match op.type:
            case OpType.PUSH:
                stack.append(op.operand)
            case OpType.PUSH_EXTERN:
                stack.append(BUILTINS[op.operand])
            case OpType.CALL:
                func = stack.pop()
                if isinstance(func, CFunc):
                    assert False, "Not implemented yet"
                else:
                    func(stack, memory)
            case OpType.RET:
                pass
            case t:
                assert False, f"Unimplemented op type {t!r}"

def run(command):
    subprocess.run(command)

def main(argv):
    program_name = argv.pop(0)
    if not argv:
        print(f"Usage: {program_name} <input.c>")
        sys.exit(1)

    input_file = argv.pop(0)
    preprocessed_file = "preprocessed.c"
    run(["cpp", "-P", input_file, "-o", preprocessed_file])

    with open(preprocessed_file, "r") as f:
        preprocessed_code = f.read()

    parser = CParser()
    ast = parser.parse(preprocessed_code, filename=preprocessed_file)

    generator = BytecodeGenerator()
    generator.visit(ast)
    run_bc(generator)

if __name__ == "__main__":
    main(sys.argv)

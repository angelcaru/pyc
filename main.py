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
    @property
    def size(self):
        ...

    def __add__(self, other):
        match (self, other):
            case (BasicCType(BasicCTypeKind.INT), BasicCType(BasicCTypeKind.INT)):
                return BasicCType(BasicCTypeKind.INT)
            case (CPtrType(inner), BasicCType(BasicCTypeKind.INT)):
                return CPtrType(inner)
            case (BasicCType(BasicCTypeKind.INT), CPtrType(inner)):
                return CPtrType(inner)
            case (CPtrType(inner1), CPtrType(inner2)):
                assert False, "illegal operation (TODO: properly report errors)"
            case _:
                assert False

class BasicCTypeKind(StrEnum):
    VOID = auto()
    INT = auto()
    CHAR = auto()

    @classmethod
    def from_name(cls, name):
        return cls[name.upper()]

@dataclass
class BasicCType(CType):
    kind: BasicCTypeKind

    @property
    def size(self):
        match self.kind:
            case BasicCTypeKind.VOID:
                return 0
            case BasicCTypeKind.INT:
                return 4
            case BasicCTypeKind.CHAR:
                return 1

@dataclass
class CPtrType(CType):
    inner: BasicCType

    @property
    def size(self):
        return 8

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

@dataclass
class CGlobal:
    type: CType
    addr: int

class BytecodeGenerator(NodeVisitor):
    def __init__(self):
        self.ops = []
        self.funcs = {}
        self.globals = {}
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
        self.ops.append(Op(OpType.RET, False))

    def visit_Constant(self, node):
        match node.type:
            case "string":
                actual_buf = node.value.encode("utf-8").decode("unicode-escape").encode("utf-8")[1:-1]
                ptr = len(self.init_mem)
                self.init_mem.extend(actual_buf)
                self.init_mem.append(0)
                self.ops.append(Op(OpType.PUSH, ptr))
                return CPtrType(BasicCType(BasicCTypeKind.CHAR))
            case "int":
                self.ops.append(Op(OpType.PUSH, int(node.value)))
                return BasicCType(BasicCTypeKind.INT)
            case t:
                assert False, f"Unsupported constant type {t!r}"

    def visit_FuncCall(self, node):
        args = node.args
        exprs = args.exprs if args is not None else []
        for expr in reversed(exprs):
            self.visit(expr)
        name = node.name.name
        func = self.funcs[name]
        if func.body is None:
            self.ops.append(Op(OpType.PUSH_EXTERN, name))
        else:
            self.ops.append(Op(OpType.PUSH, func.body))
        self.ops.append(Op(OpType.CALL, len(exprs)))
        # TODO: handle return type
        return BasicCType(BasicCTypeKind.VOID)

    def visit_Return(self, node):
        if node.expr is not None:
            self.visit(node.expr)
        self.ops.append(Op(OpType.RET, node.expr is not None))

    def visit_Assignment(self, node):
        match node.op:
            case "=":
                match node.lvalue:
                    case ID(name=name):
                        var = self.globals[name]
                        typ = var.type
                        self.ops.append(Op(OpType.PUSH, var.addr))
                    case _:
                        assert False, f"Unsupported lvalue {node.lvalue!r}"
                self.visit(node.rvalue)
                self.ops.append(Op(OpType.STORE, typ.size))
                return typ
            case t:
                assert False, f"Unsupported assignment op {t!r}"

    def visit_ID(self, node):
        name = node.name
        var = self.globals[name]
        typ = var.type
        self.ops.append(Op(OpType.PUSH, var.addr))
        self.ops.append(Op(OpType.LOAD, typ.size))
        return typ

    def visit_UnaryOp(self, node):
        match node.op:
            case "*":
                typ = self.visit(node.expr)
                if not isinstance(typ, CPtrType):
                    assert False, f"dereference of non-pointer (TODO: properly report errors)"
                self.ops.append(Op(OpType.LOAD, typ.inner.size))
                return typ.inner
            case "&":
                if not isinstance(node.expr, ID):
                    assert False, f"address of non-variable (TODO: properly report errors)"
                var = self.globals[node.expr.name]
                self.ops.append(Op(OpType.PUSH, var.addr))
                return CPtrType(var.type)
            case t:
                assert False, f"Unsupported unary op {t!r}"

    def visit_BinaryOp(self, node):
        typ1 = self.visit(node.left)
        typ2 = self.visit(node.right)
        match node.op:
            case "+":
                self.ops.append(Op(OpType.ADD))
                return typ1 + typ2
            case t:
                assert False, f"Unsupported binary op {t!r}"

    def visit_While(self, node):
        cond_ip = len(self.ops)
        self.visit(node.cond)
        jmp_op = Op(OpType.JMP_IF_ZERO, 0)
        self.ops.append(jmp_op)

        stmt_ip = len(self.ops)
        self.visit(node.stmt)
        self.ops.append(Op(OpType.JMP, cond_ip))

        jmp_op.operand = len(self.ops)

    def visit_TypeDecl(self, node):
        type_ = type_from_ast(node.type)
        # TODO: handle `const`
        self.globals[node.declname] = CGlobal(type_, len(self.init_mem))
        self.init_mem.extend([0] * type_.size)

    def visit_PtrDecl(self, node):
        type_ = type_from_ast(node)
        self.globals[node.type.declname] = CGlobal(type_, len(self.init_mem))
        self.init_mem.extend([0] * type_.size)

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

def builtin_putd(stack, memory):
    if len(stack) < 1:
        print("Not enough arguments for putd", file=sys.stderr)
        return
    value = stack.pop()
    print(value)

BUILTINS = {"puts": builtin_puts, "putd": builtin_putd}

def run_bc(generator):
    ip = generator.funcs["main"].body
    stack = []
    ret_stack = []
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
                if isinstance(func, int):
                    ret_stack.append(ip)
                    ip = func
                else:
                    func(stack, memory)
            case OpType.RET:
                if len(ret_stack) == 0:
                    exit(stack.pop())
                else:
                    ip = ret_stack.pop()
            case OpType.STORE:
                size = op.operand
                value = stack.pop()
                ptr = stack.pop()
                memory[ptr:ptr+size] = value.to_bytes(length=size, byteorder="little", signed=True)
            case OpType.LOAD:
                size = op.operand
                ptr = stack.pop()
                value = int.from_bytes(memory[ptr:ptr+size], byteorder="little", signed=True)
                stack.append(value)
            case OpType.ADD:
                right = stack.pop()
                left = stack.pop()
                stack.append(left + right)
            case OpType.JMP_IF_ZERO:
                if stack.pop() == 0:
                    ip = op.operand
            case OpType.JMP:
                ip = op.operand
            case t:
                assert False, f"Unimplemented op type {t!r}"

def run(command):
    subprocess.run(command)

DEBUG = False
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
    if DEBUG:
        print(generator.init_mem)
        for op in generator.ops:
            typ = op.type
            print(typ.name, op.operand, f"({generator.init_mem[op.operand:]})" if typ == OpType.PUSH else "", sep="\t")
    run_bc(generator)

if __name__ == "__main__":
    main(sys.argv)

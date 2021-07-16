import ast

from .clike import CLikeTranspiler
from .inference import get_inferred_nim_type
from .plugins import (
    ATTR_DISPATCH_TABLE,
    CLASS_DISPATCH_TABLE,
    FUNC_DISPATCH_TABLE,
    FUNC_USINGS_MAP,
    MODULE_DISPATCH_TABLE,
    DISPATCH_MAP,
    SMALL_DISPATCH_MAP,
    SMALL_USINGS_MAP,
)

from py2many.analysis import get_id, is_mutable, is_void_function
from py2many.clike import class_for_typename
from py2many.declaration_extractor import DeclarationExtractor
from py2many.exceptions import AstClassUsedBeforeDeclaration
from py2many.tracer import is_list, defined_before

from typing import List


class NimNoneCompareRewriter(ast.NodeTransformer):
    def visit_Compare(self, node):
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        if (
            isinstance(right, ast.Constant)
            and right.value is None
            and isinstance(left, ast.Constant)
            and isinstance(left.value, int)
        ):
            # Convert None to 0 to compare vs int
            right.value = 0
        return node


class NimTranspiler(CLikeTranspiler):
    NAME = "nim"

    CONTAINER_TYPE_MAP = {
        "List": "seq",
        "Dict": "Table",
        "Set": "set",
        "Optional": "Option",
    }

    def __init__(self, indent=2):
        super().__init__()
        self._headers = set([])
        self._indent = " " * indent
        self._default_type = "var"
        self._container_type_map = self.CONTAINER_TYPE_MAP
        if "math" in self._ignored_module_set:
            self._ignored_module_set.remove("math")
        self._dispatch_map = DISPATCH_MAP
        self._small_dispatch_map = SMALL_DISPATCH_MAP
        self._small_usings_map = SMALL_USINGS_MAP
        self._func_dispatch_table = FUNC_DISPATCH_TABLE
        self._attr_dispatch_table = ATTR_DISPATCH_TABLE
        self._func_usings_map = FUNC_USINGS_MAP

    def indent(self, code, level=1):
        return self._indent * level + code

    def usings(self):
        usings = sorted(list(set(self._usings)))
        uses = "\n".join(f"import {mod}" for mod in usings)
        return uses

    def _combine_value_index(self, value_type, index_type) -> str:
        return f"{value_type}[{index_type}]"

    def comment(self, text):
        return f"# {text}\n"

    def visit_FunctionDef(self, node) -> str:
        body = "\n".join([self.indent(self.visit(n)) for n in node.body])
        typenames, args = self.visit(node.args)

        is_python_main = getattr(node, "python_main", False)

        args_list = []
        if len(args) and hasattr(node, "self_type"):
            typenames[0] = node.self_type

        for i in range(len(args)):
            typename = typenames[i]
            arg = args[i]

            args_list.append(f"{arg}: {typename}".format(arg, typename))

        return_type = ""
        if not is_void_function(node):
            if node.returns:
                typename = self._typename_from_annotation(node, attr="returns")
                return_type = f": {typename}"
            else:
                return_type = ""

        args = ", ".join(args_list)
        funcdef = f"proc {node.name}({args}){return_type} ="
        maybe_main = ""
        if is_python_main:
            maybe_main = "\nmain()"
        return f"{funcdef}\n{body}\n{maybe_main}"

    def visit_Return(self, node) -> str:
        if node.value:
            ret = self.visit(node.value)
            fndef = None
            for scope in node.scopes:
                if isinstance(scope, ast.FunctionDef):
                    fndef = scope
                    break
            if fndef:
                return_type = self._typename_from_annotation(fndef, attr="returns")
                value_type = get_inferred_nim_type(node.value)
                if return_type != value_type and value_type is not None:
                    return f"return {return_type}({ret})"
            return f"return {ret}"
        return "return"

    def visit_arg(self, node):
        id = get_id(node)
        if id == "self":
            return (None, "self")
        typename = "T"
        if node.annotation:
            # This works only for arguments, for all other cases, use container_types
            mutable = is_mutable(node.scopes, id)
            use_open_array = isinstance(node.annotation, ast.Subscript) and not mutable
            typename = self._typename_from_annotation(node)
            if use_open_array:
                typename = typename.replace("seq", "openArray")
            if mutable:
                typename = f"var {typename}"
        return (typename, id)

    def visit_Lambda(self, node) -> str:
        typenames, args = self.visit(node.args)
        # HACK: to pass unit tests. TODO: infer types
        typenames = ["int"] * len(args)
        return_type = "int"
        args = [f"{name}: {typename}" for name, typename in zip(args, typenames)]
        args_string = ", ".join(args)
        body = self.visit(node.body)
        return f"proc({args_string}):{return_type} = return {body}"

    def visit_Attribute(self, node) -> str:
        attr = node.attr

        value_id = self.visit(node.value)

        if value_id == "sys":
            if attr == "argv":
                self._usings.add("os")
                return "(@[getAppFilename()] & commandLineParams())"

        if is_list(node.value):
            if node.attr == "append":
                attr = "add"
        if not value_id:
            value_id = ""

        ret = f"{value_id}.{attr}"
        if ret in self._attr_dispatch_table:
            ret = self._attr_dispatch_table[ret]
            return ret(self, node, value_id, attr)
        return ret

    def _visit_object_literal(self, node, fname: str, fndef: ast.ClassDef) -> str:
        vargs = []  # visited args
        if not hasattr(fndef, "declarations"):
            raise AstClassUsedBeforeDeclaration(fndef, node)
        if node.args:
            for arg, decl in zip(node.args, fndef.declarations.keys()):
                arg = self.visit(arg)
                vargs += [f"{decl}: {arg}"]
        if node.keywords:
            for kw in node.keywords:
                value = self.visit(kw.value)
                vargs += [f"{kw.arg}: {value}"]
        args = ", ".join(vargs)
        return f"{fname}({args})"

    def visit_Call(self, node) -> str:
        fname = self.visit(node.func)
        fndef = node.scopes.find(fname)

        if isinstance(fndef, ast.ClassDef):
            return self._visit_object_literal(node, fname, fndef)

        vargs = []

        if node.args:
            vargs += [self.visit(a) for a in node.args]
        if node.keywords:
            vargs += [self.visit(kw.value) for kw in node.keywords]

        ret = self._dispatch(node, fname, vargs)
        if ret is not None:
            return ret
        if vargs:
            args = ", ".join(vargs)
        else:
            args = ""
        return f"{fname}({args})"

    def visit_For(self, node) -> str:
        target = self.visit(node.target)
        it = self.visit(node.iter)
        buf = []
        buf.append(f"for {target} in {it}:")
        buf.extend(
            [self.indent(self.visit(c), level=node.level + 1) for c in node.body]
        )
        return "\n".join(buf)

    def visit_While(self, node) -> str:
        buf = []
        buf.append("while {0}:".format(self.visit(node.test)))
        buf.extend(
            [self.indent(self.visit(n), level=node.level + 1) for n in node.body]
        )
        return "\n".join(buf)

    def visit_Str(self, node) -> str:
        return "" + super().visit_Str(node) + ""

    def visit_Bytes(self, node) -> str:
        bytes_str = "{0}".format(node.s)
        return bytes_str.replace("'", '"')  # replace single quote with double quote

    def visit_Name(self, node) -> str:
        if node.id == "None":
            return "nil"
        else:
            return super().visit_Name(node)

    def visit_NameConstant(self, node) -> str:
        if node.value is None:
            return "nil"
        else:
            return super().visit_NameConstant(node)

    def visit_If(self, node) -> str:
        body_vars = set([get_id(v) for v in node.scopes[-1].body_vars])
        orelse_vars = set([get_id(v) for v in node.scopes[-1].orelse_vars])
        node.common_vars = body_vars.intersection(orelse_vars)

        body = "\n".join(
            [
                self.indent(self.visit(child), level=node.level + 1)
                for child in node.body
            ]
        )
        orelse = "\n".join(
            [
                self.indent(self.visit(child), level=node.level + 1)
                for child in node.orelse
            ]
        )
        test = self.visit(node.test)
        if node.orelse:
            orelse = self.indent(f"else:\n{orelse}", level=node.level)
        else:
            orelse = ""
        return f"if {test}:\n{body}\n{orelse}"

    def visit_UnaryOp(self, node) -> str:
        if isinstance(node.op, ast.USub):
            if isinstance(node.operand, (ast.Call, ast.Num)):
                # Shortcut if parenthesis are not needed
                return "-{0}".format(self.visit(node.operand))
            else:
                return "-({0})".format(self.visit(node.operand))
        else:
            return super().visit_UnaryOp(node)

    def visit_BinOp(self, node) -> str:
        if (
            isinstance(node.left, ast.List)
            and isinstance(node.op, ast.Mult)
            and isinstance(node.right, ast.Num)
        ):
            return "std::vector ({0},{1})".format(
                self.visit(node.right), self.visit(node.left.elts[0])
            )
        else:
            return super().visit_BinOp(node)

    def visit_ClassDef(self, node) -> str:
        extractor = DeclarationExtractor(NimTranspiler())
        extractor.visit(node)
        declarations = node.declarations = extractor.get_declarations()
        node.class_assignments = extractor.class_assignments
        ret = super().visit_ClassDef(node)
        if ret is not None:
            return ret

        decorators = [get_id(d) for d in node.decorator_list]
        decorators = [
            class_for_typename(t, None, self._imported_names) for t in decorators
        ]
        for d in decorators:
            if d in CLASS_DISPATCH_TABLE:
                ret = CLASS_DISPATCH_TABLE[d](self, node)
                if ret is not None:
                    return ret

        fields = []
        index = 0
        for declaration, typename in declarations.items():
            if typename == None:
                typename = "ST{0}".format(index)
                index += 1
            fields.append(f"{declaration}: {typename}")

        for b in node.body:
            if isinstance(b, ast.FunctionDef):
                b.self_type = node.name

        object_def = "type\n"
        object_def += self.indent(f"{node.name} = object\n", level=node.level + 1)
        object_def += "\n".join([self.indent(f, level=node.level + 2) for f in fields])
        body = [self.visit(b) for b in node.body]
        body = "\n".join(body)
        return f"{object_def}\n{body}\n"

    def visit_IntEnum(self, node) -> str:
        fields = []
        for member, var in node.class_assignments.items():
            var = self.visit(var)
            if var == "auto()":
                fields.append(f"{member},")
            else:
                fields.append(f"{member} = {var},")
        fields = "\n".join([self.indent(f) for f in fields])
        return f"type {node.name} = enum\n{fields}\n\n"

    def visit_IntFlag(self, node) -> str:
        fields = []
        for member, var in node.class_assignments.items():
            var = self.visit(var)
            if var == "auto()":
                fields.append(f"{member},")
            else:
                fields.append(f"{member} = {var},")
        fields = "\n".join([self.indent(f, level=2) for f in fields])
        # flags = self.indent(f"{node.name}Flags = set[{node.name}]")
        return "\n".join(
            [
                "type",
                self.indent(f"{node.name} = enum"),
                f"{fields}",
                # f"{flags}",
                "",
            ]
        )

    def visit_StrEnum(self, node) -> str:
        fields = []
        for member, var in node.class_assignments.items():
            var = self.visit(var)
            if var == "auto()":
                fields.append(f"{member},")
            else:
                fields.append(f"{member} = {var},")
        fields = "\n".join([self.indent(f) for f in fields])
        return f"type {node.name} = enum\n{fields}\n\n"

    def _import(self, name: str) -> str:
        return f"import {name}"

    def _import_from(self, module_name: str, names: List[str]) -> str:
        names = ", ".join(names)
        if len(names) == 1:
            # TODO: make this more generic so it works for len(names) > 1
            name = names[0]
            lookup = f"{module_name}.{name}"
            if lookup in MODULE_DISPATCH_TABLE:
                nim_module_name, nim_name = MODULE_DISPATCH_TABLE[lookup]
                return f"from {nim_module_name} import {nim_name}"
        return f"from {module_name} import {names}"

    def visit_List(self, node) -> str:
        elements = [self.visit(e) for e in node.elts]
        elements = ", ".join(elements)
        return f"@[{elements}]"

    def visit_Set(self, node) -> str:
        self._usings.add("sets")
        elements = [self.visit(e) for e in node.elts]
        elements_str = ", ".join(elements)
        return f"toHashSet([{elements_str}])"

    def visit_Dict(self, node) -> str:
        self._usings.add("tables")
        keys = [self.visit(k) for k in node.keys]
        values = [self.visit(k) for k in node.values]
        kv_pairs = ", ".join([f"{k}: {v}" for k, v in zip(keys, values)])
        return f"{{{kv_pairs}}}.newTable"

    def visit_Subscript(self, node) -> str:
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if hasattr(node, "is_annotation"):
            if value in self.CONTAINER_TYPE_MAP:
                value = self.CONTAINER_TYPE_MAP[value]
            if value == "Tuple":
                return f"({index})"
            return f"{value}[{index}]"
        return f"{value}[{index}]"

    def visit_Index(self, node) -> str:
        return self.visit(node.value)

    def visit_Slice(self, node) -> str:
        lower = ""
        if node.lower:
            lower = self.visit(node.lower)
        upper = ""
        if node.upper:
            upper = self.visit(node.upper)

        return "{0}..{1}".format(lower, upper)

    def visit_Tuple(self, node) -> str:
        elts = [self.visit(e) for e in node.elts]
        elts = ", ".join(elts)
        if hasattr(node, "is_annotation"):
            return elts
        return "({0})".format(elts)

    def visit_Try(self, node, finallybody=None) -> str:
        buf = self.visit_unsupported_body(node, "try_dummy", node.body)

        for handler in node.handlers:
            buf += self.visit(handler)
        # buf.append("\n".join(excepts));

        if finallybody:
            buf += self.visit_unsupported_body(node, "finally_dummy", finallybody)

        return "\n".join(buf)

    def visit_ExceptHandler(self, node) -> str:
        exception_type = ""
        if node.type:
            exception_type = self.visit(node.type)
        name = "except!({0})".format(exception_type)
        body = self.visit_unsupported_body(node, name, node.body)
        return body

    def visit_Assert(self, node) -> str:
        return "assert({0})".format(self.visit(node.test))

    def visit_AnnAssign(self, node) -> str:
        target, type_str, val = super().visit_AnnAssign(node)
        kw = "var" if is_mutable(node.scopes, target) else "let"
        if type_str == self._default_type:
            return f"{kw} {target} = {val}"
        return f"{kw} {target}: {type_str} = {val}"

    def visit_Assign(self, node) -> str:
        lines = [self._visit_AssignOne(node, target) for target in node.targets]
        if len(lines) > 1:
            line0 = lines[0]
            lines = [self.indent(line, node.level) for line in lines]
            lines[0] = line0  # parent node is going to add indentation
        return "\n".join(lines)

    def _visit_AssignOne(self, node, target) -> str:
        kw = "var" if is_mutable(node.scopes, get_id(target)) else "let"

        if isinstance(target, ast.Tuple):
            elts = [self.visit(e) for e in target.elts]
            elts_str = ", ".join(elts)
            value = self.visit(node.value)
            return f"{kw} ({elts_str}) = {value}"

        if isinstance(node.scopes[-1], ast.If):
            outer_if = node.scopes[-1]
            target_id = self.visit(target)
            if target_id in outer_if.common_vars:
                value = self.visit(node.value)
                return f"{kw} {target_id} = {value}"

        if isinstance(target, ast.Subscript) or isinstance(target, ast.Attribute):
            target = self.visit(target)
            value = self.visit(node.value)
            return f"{target} = {value}"

        definition = node.scopes.parent_scopes.find(get_id(target))
        if definition is None:
            definition = node.scopes.find(get_id(target))
        if isinstance(target, ast.Name) and defined_before(definition, node):
            target = self.visit(target)
            value = self.visit(node.value)
            return f"{target} = {value}"
        else:
            target = self.visit(target)
            value = self.visit(node.value)

            return f"{kw} {target} = {value}"

    def visit_Delete(self, node) -> str:
        target = node.targets[0]
        return "{0}.drop()".format(self.visit(target))

    def visit_Raise(self, node) -> str:
        if node.exc is not None:
            return "raise!({0}); //unsupported".format(self.visit(node.exc))
        # This handles the case where `raise` is used without
        # specifying the exception.
        return "raise!(); //unsupported"

    def visit_Await(self, node) -> str:
        return "await!({0})".format(self.visit(node.value))

    def visit_AsyncFunctionDef(self, node) -> str:
        return "#[async]\n{0}".format(self.visit_FunctionDef(node))

    def visit_Yield(self, node) -> str:
        if node.value is not None:
            value = self.visit(node.value)
            return f"yield {value}"
        return "yield nil"

    def visit_Print(self, node) -> str:
        buf = []
        for n in node.values:
            value = self.visit(n)
            buf.append('println("{{:?}}",{0})'.format(value))
        return "\n".join(buf)

    def visit_GeneratorExp(self, node) -> str:
        elt = self.visit(node.elt)
        generator = node.generators[0]
        target = self.visit(generator.target)
        iter = self.visit(generator.iter)

        # HACK for dictionary iterators to work
        if not iter.endswith("keys()") or iter.endswith("values()"):
            iter += ".iter()"

        map_str = ".map(|{0}| {1})".format(target, elt)
        filter_str = ""
        if generator.ifs:
            filter_str = ".cloned().filter(|&{0}| {1})".format(
                target, self.visit(generator.ifs[0])
            )

        return "{0}{1}{2}.collect::<Vec<_>>()".format(iter, filter_str, map_str)

    def visit_ListComp(self, node) -> str:
        return self.visit_GeneratorExp(node)  # right now they are the same

    def visit_Global(self, node) -> str:
        return "//global {0}".format(", ".join(node.names))

    def visit_Starred(self, node) -> str:
        return "starred!({0})/*unsupported*/".format(self.visit(node.value))

    def visit_IfExp(self, node) -> str:
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        test = self.visit(node.test)
        return f"if {test}: {body} else: {orelse}"

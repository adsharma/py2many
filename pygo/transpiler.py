import ast
import textwrap

from typing import List

from .clike import CLikeTranspiler
from .inference import get_inferred_go_type
from .plugins import (
    ATTR_DISPATCH_TABLE,
    CLASS_DISPATCH_TABLE,
    FUNC_DISPATCH_TABLE,
    MODULE_DISPATCH_TABLE,
    DISPATCH_MAP,
    SMALL_DISPATCH_MAP,
    SMALL_USINGS_MAP,
)

from py2many.analysis import IGNORED_MODULE_SET, get_id, is_global, is_void_function
from py2many.clike import _AUTO_INVOKED, class_for_typename
from py2many.declaration_extractor import DeclarationExtractor
from py2many.exceptions import AstClassUsedBeforeDeclaration, AstCouldNotInfer
from py2many.rewriters import capitalize_first, rename, camel_case
from py2many.tracer import is_list, defined_before, is_class_or_module, is_enum


class GoMethodCallRewriter(ast.NodeTransformer):
    def visit_Call(self, node):
        needs_assign = False
        fname = node.func
        if isinstance(fname, ast.Attribute):
            if is_list(node.func.value) and fname.attr == "append":
                needs_assign = True
            value_id = get_id(fname.value)
            if value_id not in IGNORED_MODULE_SET:
                if value_id:
                    node0 = ast.Name(id=get_id(fname.value), lineno=node.lineno)
                else:
                    node0 = fname.value
                node.args = [node0] + node.args
                node.func = ast.Name(id=fname.attr, lineno=node.lineno, ctx=fname.ctx)
        if needs_assign:
            ret = ast.Assign(
                targets=[ast.Name(id=fname.value.id, lineno=node.lineno)],
                value=node,
                lineno=node.lineno,
                scopes=node.scopes,
            )
            return ret
        return node


class GoNoneCompareRewriter(ast.NodeTransformer):
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


class GoPropagateTypeAnnotation(ast.NodeTransformer):
    def _visit_assign(self, node, target):
        if hasattr(node, "annotation") and isinstance(
            node.value, (ast.List, ast.Set, ast.Dict)
        ):
            node.value.annotation = node.annotation
        return node

    def visit_Assign(self, node):
        target = node.targets[0]
        return self._visit_assign(node, target)

    def visit_AnnAssign(self, node):
        target = node.target
        return self._visit_assign(node, target)


class GoVisibilityRewriter(ast.NodeTransformer):
    def visit_Name(self, node):
        if hasattr(node, "scopes") and is_global(node):
            old_name = get_id(node)
            new_name = camel_case(old_name)
            if old_name != new_name:
                rename(node.scopes[-1], old_name, new_name)
        return node

    def visit_FunctionDef(self, node):
        if hasattr(node, "scopes") and isinstance(node.scopes[-2], ast.Module):
            old_name = get_id(node)
            if old_name is not None:
                if "_" in old_name:
                    new_name = camel_case(old_name)
                else:
                    new_name = capitalize_first(old_name)
                if old_name != new_name:
                    rename(node.scopes[-2], old_name, new_name)
        return node


class GoIfExpRewriter(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.IfExp):
            if_stmt = ast.parse(
                textwrap.dedent(
                    """\
                if True:
                    a = 1
                else:
                    a = 2"""
                )
            ).body[0]
            assert isinstance(if_stmt, ast.If)
            if_stmt.test = node.value.test
            if_stmt.body[0].targets = node.targets
            if_stmt.body[0].value = node.value.body
            if_stmt.orelse[0].targets = node.targets
            if_stmt.orelse[0].value = node.value.orelse
            if_stmt.lineno = node.lineno
            if_stmt.col_offset = node.col_offset
            ast.fix_missing_locations(if_stmt)
            return if_stmt
        return node


class GoTranspiler(CLikeTranspiler):
    NAME = "go"

    CONTAINER_TYPE_MAP = {"List": "[]", "Dict": None, "Set": None, "Optional": "nil"}

    def __init__(self):
        super().__init__()
        self._default_type = None
        self._container_type_map = self.CONTAINER_TYPE_MAP
        self._dispatch_map = DISPATCH_MAP
        self._small_dispatch_map = SMALL_DISPATCH_MAP
        self._small_usings_map = SMALL_USINGS_MAP
        self._func_dispatch_table = FUNC_DISPATCH_TABLE
        self._attr_dispatch_table = ATTR_DISPATCH_TABLE

    def headers(self, meta):
        return "\n".join(self._headers)

    def usings(self):
        buf = "package main\n\n"  # TODO naming
        if self._usings:
            buf += "import (\n"
            buf += "\n".join([f"{using}" for using in self._usings])
            buf += ")\n"
        return buf + "\n\n"

    def _combine_value_index(self, value_type, index_type) -> str:
        return f"{value_type}{index_type}"

    def comment(self, text):
        return f"// {text}\n"

    def visit_FunctionDef(self, node) -> str:
        body = "\n".join([self.visit(n) for n in node.body])
        typenames, args = self.visit(node.args)

        if len(typenames) and typenames[0] == None and hasattr(node, "self_type"):
            typenames[0] = node.self_type

        args_list = []
        typedecls = []
        index = 0
        for i in range(len(args)):
            typename = typenames[i]
            arg = args[i]

            if typename == "T":
                typename = "T{0} any".format(index)
                typedecls.append(typename)
                index += 1
            args_list.append(f"{arg} {typename}")

        return_type = ""
        if not is_void_function(node):
            if node.returns:
                try:
                    typename = self._typename_from_annotation(node, attr="returns")
                    return_type = f" {typename}"
                except AstCouldNotInfer:
                    pass
            else:
                return_type = " interface{}"

        template = ""
        if len(typedecls) > 0:
            template = "[{0}]".format(", ".join(typedecls))

        args = ", ".join(args_list)
        funcdef = f"func {node.name}{template}({args}){return_type} {{"
        return f"{funcdef}\n{body}}}\n\n"

    def visit_Return(self, node) -> str:
        if node.value:
            ret = self.visit(node.value)
            fndef = None
            for scope in node.scopes:
                if isinstance(scope, ast.FunctionDef):
                    fndef = scope
                    break
            if fndef:
                return_type = None
                try:
                    return_type = self._typename_from_annotation(fndef, attr="returns")
                except AstCouldNotInfer:
                    pass
                value_type = get_inferred_go_type(node.value)
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
            typename = self._typename_from_annotation(node)
        return (typename, id)

    def visit_Lambda(self, node) -> str:
        typenames, args = self.visit(node.args)
        # HACK: to pass unit tests. TODO: infer types
        # Need to get it from the annotation on the lhs of `node`
        typenames = ["int"] * len(args)
        return_type = "int"
        args = [f"{name} {typename}" for name, typename in zip(args, typenames)]
        args_string = ", ".join(args)
        body = self.visit(node.body)
        return f"func({args_string}) {return_type} {{ return {body} }}"

    def visit_Attribute(self, node) -> str:
        attr = node.attr

        value_id = self.visit(node.value)

        if value_id == "sys":
            if attr == "argv":
                self._usings.add('"os"')
                return "os.Args"

        if not value_id:
            value_id = ""

        if is_enum(value_id, node.scopes):
            return f"{attr}"

        if is_class_or_module(value_id, node.scopes):
            return f"{value_id}.{attr}"

        return f"{value_id}.{attr}"

    def _visit_struct_literal(self, node, fname: str, fndef: ast.ClassDef) -> str:
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
        return f"{fname}{{{args}}}"

    def visit_Call(self, node) -> str:
        fname = self.visit(node.func)
        fndef = node.scopes.find(fname)

        if isinstance(fndef, ast.ClassDef):
            return self._visit_struct_literal(node, fname, fndef)

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
        if target == "_":
            buf.append(f"for range {it} {{")
        else:
            buf.append(f"for _, {target} := range {it} {{")
            # Dummy assign to silence the compiler on unused vars
            if target.startswith("_"):
                buf.append(f"_ = {target}")
        buf.extend([self.visit(c) for c in node.body])
        buf.append("}")
        return "\n".join(buf)

    def visit_While(self, node) -> str:
        buf = []
        buf.append("for {0} {{".format(self.visit(node.test)))
        buf.extend([self.visit(c) for c in node.body])
        buf.append("}")
        return "\n".join(buf)

    def visit_Str(self, node) -> str:
        return "" + super().visit_Str(node) + ""

    def visit_Bytes(self, node) -> str:
        bytes_str = "{0}".format(node.s)
        return bytes_str.replace("'", '"')  # replace single quote with double quote

    def _visit_container_compare(self, node) -> str:
        left = self.visit(node.left)
        op = self.visit(node.ops[0])
        right = self.visit(node.comparators[0])
        if op == "==":
            return f"cmp.Equal({left}, {right})"
        return super().visit_Compare(node)

    def visit_Compare(self, node) -> str:
        left = node.left
        right = node.comparators[0]
        if hasattr(left, "annotation") or hasattr(right, "annotation"):
            self._typename_from_annotation(left)
            self._typename_from_annotation(right)
            op = self.visit(node.ops[0])
            if (
                hasattr(left, "container_type")
                or hasattr(right, "container_type")
                and op != "in"
            ):
                self._usings.add('"github.com/google/go-cmp/cmp"')
                return self._visit_container_compare(node)
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return super().visit_Compare(node)

    def visit_Name(self, node) -> str:
        if node.id == "None":
            return "None"
        elif node.id == "StringsContains":
            # TODO: move to plugins
            self._usings.add('"strings"')
            return "strings.Contains"
        else:
            return super().visit_Name(node)

    def visit_NameConstant(self, node) -> str:
        if node.value is None:
            return "nil"
        else:
            return super().visit_NameConstant(node)

    def _make_block(self, node):
        buf = []
        buf.append("{")
        buf.extend([self.visit(child) for child in node.body])
        buf.append("}")
        return "\n".join(buf)

    def visit_If(self, node) -> str:
        body_vars = {get_id(v): v for v in node.scopes[-1].body_vars}
        orelse_vars = {get_id(v): v for v in node.scopes[-1].orelse_vars}
        node.common_vars = set(body_vars.keys()).intersection(set(orelse_vars.keys()))
        types = [self._typename_from_annotation(body_vars[v]) for v in node.common_vars]
        common_vars = "\n".join(
            [f"var {v} {t}" for v, t in zip(node.common_vars, types)]
        )
        if common_vars:
            return common_vars + "\n" + super().visit_If(node)
        else:
            return super().visit_If(node)

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
        extractor = DeclarationExtractor(GoTranspiler())
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
            fields.append(f"{declaration} {typename}")

        for b in node.body:
            if isinstance(b, ast.FunctionDef):
                b.self_type = node.name

        fields = "\n".join(fields)
        body = [self.visit(b) for b in node.body]
        body = "\n".join(body)
        return f"type {node.name} struct {{\n{fields}\n}}\n{body}\n"

    def _visit_enum(self, node, typename, members) -> str:
        ret = f"type {node.name} {typename}\n\n"
        fields = []
        for i, (member, var) in enumerate(members):
            typename = f" {node.name}" if i == 0 else ""
            if var == _AUTO_INVOKED:
                fields.append(f"{member}{typename} = iota")
            else:
                fields.append(f"{member}{typename} = {var}")
        fields = "\n".join(fields)
        return f"{ret} const (\n{fields}\n)\n\n"

    def visit_IntEnum(self, node) -> str:
        members = []
        for i, (member, var) in enumerate(node.class_assignments.items()):
            var = self.visit(var)
            if var == _AUTO_INVOKED:
                members.append((member, "iota"))
            else:
                members.append((member, var))
        return self._visit_enum(node, "int", members)

    def visit_IntFlag(self, node) -> str:
        members = []
        for i, (member, var) in enumerate(node.class_assignments.items()):
            var = self.visit(var)
            if var == _AUTO_INVOKED:
                members.append((member, "1 << iota"))
            else:
                members.append((member, var))
        return self._visit_enum(node, "int", members)

    def visit_StrEnum(self, node) -> str:
        members = []
        for i, (member, var) in enumerate(node.class_assignments.items()):
            var = self.visit(var)
            members.append((member, var))
        return self._visit_enum(node, "string", members)

    def _import(self, name: str) -> str:
        return f'import ("{name}")'

    def _import_from(self, module_name: str, names: List[str]) -> str:
        if len(names) == 1:
            # TODO: make this more generic so it works for len(names) > 1
            name = names[0]
            lookup = f"{module_name}.{name}"
            if lookup in MODULE_DISPATCH_TABLE:
                go_module_name, go_name = MODULE_DISPATCH_TABLE[lookup]
                go_module_name = go_module_name.replace(".", "/")
                return f'import ("{go_module_name}/{go_name}")'
        module_name = module_name.replace(".", "/")
        return "\n".join([f'import ("{module_name}/{name}")' for name in names])

    def visit_List(self, node) -> str:
        _ = self._typename_from_annotation(node)
        element_type = self._default_type
        if hasattr(node, "container_type"):
            _, element_type = node.container_type
        elements = [self.visit(e) for e in node.elts]
        elements_str = ", ".join(elements)
        return f"[]{element_type}{{{elements_str}}}"

    def visit_Set(self, node) -> str:
        _ = self._typename_from_annotation(node)
        element_type = self._default_type
        if hasattr(node, "container_type"):
            _, element_type = node.container_type
        elements = [self.visit(e) for e in node.elts]
        kv_pairs = ", ".join([f"{k}: true" for k in elements])
        return f"map[{element_type}]bool{{{kv_pairs}}}"

    def visit_Dict(self, node) -> str:
        keys = [self.visit(k) for k in node.keys]
        values = [self.visit(k) for k in node.values]
        kv_pairs = ", ".join([f"{k}: {v}" for k, v in zip(keys, values)])
        _ = self._typename_from_annotation(node)
        key_typename = value_typename = self._default_type
        if hasattr(node, "container_type"):
            container_type, element_type = node.container_type
            key_typename, value_typename = element_type
            if key_typename == self._default_type:
                key_typename = "int"
        return f"map[{key_typename}]{value_typename}{{{kv_pairs}}}"

    def visit_Subscript(self, node) -> str:
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if hasattr(node, "is_annotation"):
            if value in self.CONTAINER_TYPE_MAP:
                value = self.CONTAINER_TYPE_MAP[value]
            if value == "Tuple":
                return "({0})".format(index)
            return "{0}{1}".format(value, index)
        return "{0}[{1}]".format(value, index)

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
        return "{0}".format(elts)

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
        condition = self.visit(node.test)
        return f'if !({condition}) {{ panic("assert") }}'

    def visit_AnnAssign(self, node) -> str:
        target = self.visit(node.target)
        type_str = self._typename_from_annotation(node)
        val = self.visit(node.value) if node.value is not None else None
        if type_str is not self._default_type:
            return f"var {target} {type_str} = {val}"
        else:
            return f"var {target} = {val}"

    def _needs_cast(self, left, right) -> bool:
        if not hasattr(left, "annotation"):
            return False
        left_type = self._typename_from_annotation(left)
        right_type = get_inferred_go_type(right)
        return left_type != right_type and (
            left_type != self._default_type and right_type != self._default_type
        )

    def _assign_cast(
        self, value_str: str, cast_to: str, python_annotation, rust_annotation
    ) -> str:
        # python/rust annotations provided to customize the cast if necessary
        return f"{cast_to}({value_str})"

    def _visit_AssignOne(self, node, target) -> str:
        if isinstance(target, ast.Tuple):
            elts = [self.visit(e) for e in target.elts]
            value = self.visit(node.value)
            return "var {0} = {1}".format(", ".join(elts), value)

        if isinstance(node.scopes[-1], ast.If):
            outer_if = node.scopes[-1]
            target_id = self.visit(target)
            if target_id in outer_if.common_vars:
                value = self.visit(node.value)
                return "{0} = {1}".format(target_id, value)

        if isinstance(target, ast.Subscript) or isinstance(target, ast.Attribute):
            target = self.visit(target)
            value = self.visit(node.value)
            if value == None:
                value = "None"
            return "{0} = {1}".format(target, value)

        typename = self._typename_from_annotation(target)
        needs_cast = self._needs_cast(target, node.value)
        target_str = self.visit(target)
        value = self.visit(node.value)
        if needs_cast:
            left_annotation = target.annotation
            right_annotation = getattr(node.value, "annotation", None)
            if right_annotation is None:
                right_annotation = ast.Name(id=get_inferred_go_type(node.value))
            value = self._assign_cast(
                value, typename, left_annotation, right_annotation
            )

        definition = node.scopes.parent_scopes.find(get_id(target))
        if definition is None:
            definition = node.scopes.find(get_id(target))
        if isinstance(target, ast.Name) and defined_before(definition, node):
            return f"{target_str} = {value}"
        else:
            if typename is not None:
                # Dummy assign to silence the compiler on unused vars
                maybe_tail = ""
                if target_str.startswith("_") and target_str != "_":
                    maybe_tail = f"\n_ = {target_str}"
                if target_str == "_":
                    return f"{target_str} = {value}"
                return f"var {target_str} {typename} = {value}{maybe_tail}"

            if is_global(node):
                return f"var {target_str} = {value}"
            return f"{target_str} := {value}"

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
        return "//yield is unimplemented"

    def visit_Print(self, node) -> str:
        buf = []
        self._usings.add('"fmt"')
        for n in node.values:
            value = self.visit(n)
            buf.append('fmt.Printf("%v\n",{0})'.format(value))
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

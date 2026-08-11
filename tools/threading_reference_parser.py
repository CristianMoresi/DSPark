#!/usr/bin/env python3
"""Dependency-free C++ declaration parser for threading reference readouts.

The parser is intentionally focused: it finds public member functions whose
declared return type is a const lvalue reference and whose every return
statement directly names accessible member storage or one of its subobjects.
Definitions may be in class or inline at namespace scope. It tokenizes C++
rather than matching one source spelling, resolves named namespace/class alias
scope and binds parameters and block locals before falling back to a member.
"""

from dataclasses import dataclass, field, replace
import re


_GRAPH_TRAVERSAL_DEPTH_LIMIT = 64
_SELECTED_ALIAS_DEPTH_REASON = (
    "selected alias traversal exceeds the bounded depth limit")
_INHERITANCE_DEPTH_REASON = (
    "inheritance traversal exceeds the bounded depth limit")
_INHERITANCE_DEPTH_SENTINEL = "<unsupported-inheritance-depth>"


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    line: int


@dataclass(frozen=True)
class ReferenceAccessor:
    name: str
    documentation: str
    class_documentation: str
    class_definition: str
    line: int
    access: str
    accessor_owner: tuple = ()
    return_bindings: tuple = ()
    binding_identity_is_uniform: bool = False
    qualifier_target: tuple = ()
    member_declaring_owner: tuple = ()
    member_name: str = ""


@dataclass(frozen=True)
class ReferenceParseDiagnostic:
    """A const-reference definition that the bounded parser cannot bind."""

    line: int
    message: str


@dataclass(frozen=True)
class ClassRegion:
    opening: int
    closing: int
    default_access: str
    declaration: int
    name_index: int
    name: str
    base_tokens: tuple
    tag: str = "class"
    name_indices: tuple = ()
    template_id_name_indices: tuple = ()
    qualified_absolute: bool = False


@dataclass(frozen=True)
class BaseEdge:
    """One resolved same-header inheritance edge."""

    derived_class: tuple
    base_class: tuple
    spelling: tuple
    access: str
    declaration: int
    virtual: bool


@dataclass(frozen=True)
class DataMember:
    """One data-member declaration with its exact owning class."""

    declaring_class: tuple
    name: str
    access: str
    declaration: int


@dataclass(frozen=True)
class QualifiedMemberBinding:
    """Classification of one bounded class-qualified return expression."""

    disposition: str
    qualifier_target: tuple = ()
    member_declaring_owner: tuple = ()
    member_name: str = ""


@dataclass(frozen=True)
class ReturnBinding:
    """One exact return-expression classification and source position."""

    disposition: str
    qualifier_target: tuple = ()
    member_declaring_owner: tuple = ()
    member_name: str = ""
    source_offset: int = 0
    source_line: int = 0


@dataclass(frozen=True)
class NamespaceRegion:
    opening: int
    closing: int
    declaration: int
    path: tuple
    name_indices: tuple = ()


@dataclass(frozen=True)
class FunctionSignature:
    name: str
    const_reference: bool
    return_type: tuple
    parameter_types: tuple
    member_const: bool
    member_volatile: bool
    ref_qualifier: str
    name_start: int
    parameters_open: int
    parameters_close: int


@dataclass(frozen=True)
class FunctionDeclaratorStructure:
    """Token ranges of one supported function declarator."""

    name: str
    name_start: int
    parameters_open: int
    parameters_close: int
    trailing_return: int = -1
    declarator_start: int = -1


@dataclass(frozen=True)
class DeclaratorNameCursor:
    """One backward declarator-name decision and its exact token boundaries."""

    name: str
    name_start: int
    name_end: int
    parameter_open: int
    post_name_attributes: tuple = ()
    declarator_start: int = -1
    wrapper_open: int = -1
    wrapper_close: int = -1


@dataclass(frozen=True)
class AliasResolutionEnvironment:
    type_aliases: dict
    namespace_aliases: dict
    known_namespaces: frozenset
    namespace_declarations: dict
    namespace_alias_cache: dict
    classes_by_identity: dict
    class_identities: dict
    class_declarations: dict
    type_alias_cache: dict
    complete_class_scopes: dict
    base_edges: dict
    declaration_graph: object = None
    source_tokens: tuple = ()


@dataclass(frozen=True)
class LexicalScope:
    """One exact lexical lifetime in a bounded declaration graph."""

    scope_id: int
    kind: str
    parent_scope: int
    opening: int
    closing: int
    canonical_namespace: tuple = ()
    canonical_class: tuple = ()
    callable_owner: int = -1
    declaration_ids: tuple = ()


@dataclass(frozen=True)
class TypeDeclaration:
    """One named type, alias, using declaration, or namespace fact."""

    declaration_id: int
    name: str
    kind: str
    canonical_identity: tuple
    canonical_target: tuple
    declaration_start: int
    point_of_declaration: int
    body_open: int
    body_close: int
    parent_scope: int
    lifetime_end: int
    target_tokens: tuple = ()
    rhs_use_position: int = -1
    access: str = "public"
    dependent: bool = False
    complete_class_eligible: bool = False


@dataclass(frozen=True)
class LookupContext:
    """The exact C++-lookup boundary supplied by one parser consumer."""

    starting_scope: int
    use_offset: int
    purpose: str
    complete_class: bool = False
    owner_class: tuple = ()
    absolute: bool = False


@dataclass(frozen=True)
class LookupResult:
    """One explicit central lookup result; absence never implies fallback."""

    status: str
    declaration_id: int = -1
    canonical_identity: tuple = ()
    declaration_kind: str = ""
    provenance: tuple = ()
    reason: str = ""
    candidate_declaration_ids: tuple = ()


@dataclass(frozen=True)
class QualifiedComponent:
    """One unconsumed qualified-id component with original coordinates."""

    name: str
    token_range: tuple = ()
    has_template_id: bool = False


@dataclass(frozen=True)
class QualifiedLookupResult:
    """One authoritative declaration-selected qualified-name result."""

    status: str
    selected_declaration_id: int
    selected_declaration_kind: str = ""
    canonical_identity: tuple = ()
    alias_target_declaration_id: int = -1
    canonical_alias_target: tuple = ()
    remaining_components: tuple = ()
    use_offset: int = -1
    starting_scope_id: int = -1
    access: str = "public"
    purpose: str = ""
    absolute: bool = False
    provenance: tuple = ()
    reason: str = ""
    selection_origin: str = ""

    @property
    def declaration_id(self):
        """Compatibility projection for existing one-component consumers."""
        return self.selected_declaration_id

    @property
    def declaration_kind(self):
        """Compatibility projection for existing one-component consumers."""
        if self.selection_origin == "injected-base":
            return "injected_base_class_name"
        return self.selected_declaration_kind


@dataclass(frozen=True)
class ClassKeyClassification:
    """Exhaustive classification of one ``class``/``struct``/``union`` token."""

    token_index: int
    key: str
    status: str
    name_range: tuple = ()
    qualified_components: tuple = ()
    canonical_identity: tuple = ()
    point_of_declaration: int = -1
    parent_scope: int = -1
    terminator_range: tuple = ()
    body_range: tuple = ()
    reason: str = ""


@dataclass(frozen=True)
class DataNameDeclaration:
    """One graph-owned data name with an exact source lifetime."""

    declaration_id: int
    name: str
    kind: str
    point_of_declaration: int
    lifetime_begin: int
    lifetime_end: int
    parent_scope: int
    callable_owner: int
    control_id: int = -1
    header_range: tuple = ()
    controlled_statement_range: tuple = ()


@dataclass(frozen=True)
class ControlStatementDescriptor:
    """One control declaration and the full controlled-statement extent."""

    control_id: int
    kind: str
    keyword_offset: int
    header_open: int
    header_close: int
    controlled_statement_open: int
    controlled_statement_end: int
    parent_control: int = -1
    data_declaration_ids: tuple = ()


@dataclass(frozen=True)
class DeclarationGraph:
    """One source-wide lexical graph shared by every supported lookup."""

    source_size: int
    scopes: tuple
    declarations: tuple
    root_scope: int
    namespace_regions: tuple
    class_regions: tuple
    nonlocal_class_regions: tuple
    local_class_regions: tuple
    class_identities: dict
    class_declaration_points: dict
    declarations_by_scope_name: dict
    namespace_scopes: dict
    class_key_classifications: tuple = ()
    data_name_declarations: tuple = ()
    control_statements: tuple = ()
    lookup_cache: dict = field(default_factory=dict, compare=False, repr=False)
    qualified_lookup_cache: dict = field(
        default_factory=dict, compare=False, repr=False)
    qualified_lookup_results: list = field(
        default_factory=list, compare=False, repr=False)


@dataclass
class _DeclarationLookupIndex:
    """Lightweight lookup view over facts awaiting one final graph freeze."""

    source_size: int
    scopes: tuple
    declarations: tuple
    root_scope: int
    namespace_regions: tuple
    class_regions: tuple
    nonlocal_class_regions: tuple
    local_class_regions: tuple
    class_identities: dict
    class_declaration_points: dict
    declarations_by_scope_name: dict
    namespace_scopes: dict
    data_name_declarations: tuple = ()
    control_statements: tuple = ()
    lookup_cache: dict = field(default_factory=dict)
    qualified_lookup_cache: dict = field(default_factory=dict)
    qualified_lookup_results: list = field(default_factory=list)


@dataclass(frozen=True)
class _StructuralJoinDeclaration:
    """One lightweight semantic declaration used before graph freezing."""

    declaration_id: int
    name: str
    kind: str
    canonical_identity: tuple
    parent_identity: tuple
    point_of_declaration: int
    declaration_start: int = -1
    target_tokens: tuple = ()
    rhs_use_position: int = -1
    access: str = "public"
    dependent: bool = False


@dataclass(frozen=True)
class AliasContext:
    environment: AliasResolutionEnvironment
    current_scope: tuple = ()
    use_position: int = 0
    complete_class_scopes: tuple = ()


@dataclass(frozen=True)
class NamespaceAliasTarget:
    scope: tuple
    absolute: bool
    parts: tuple
    declaration: int
    declaration_id: int = -1
    declaration_graph: object = field(
        default=None, compare=False, repr=False)


@dataclass(frozen=True)
class TypeAliasTarget:
    scope: tuple
    rhs: tuple
    declaration: int
    access: str = "public"
    declaration_id: int = -1
    declaration_graph: object = field(
        default=None, compare=False, repr=False)


@dataclass(frozen=True)
class IdentityAssociationCase:
    label: str
    source: str
    expected_count: int
    expected_marker: bool
    expected_line: int
    expected_access: str
    deletion_anchor: str = ""


@dataclass(frozen=True)
class QualifiedIdentityCase:
    label: str
    source: str
    expected_line: int
    accessor_owner: tuple
    qualifier_target: tuple
    member_declaring_owner: tuple
    member_name: str
    deletion_anchor: str


_MULTI_PUNCTUATION = (
    "<=>", "->*", "...", "::", "->", "&&", "[[", "]]", "<=", ">=",
    "==", "!=", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=",
    "|=", "^=", "<<", ">>", "##", ".*",
)

_NON_FUNCTION_NAMES = {
    "alignas", "alignof", "catch", "decltype", "for", "if", "noexcept", "requires",
    "sizeof", "static_assert", "switch", "typeid", "while",
}

_RETURN_SPECIFIERS = {
    "consteval", "constexpr", "explicit", "extern", "friend", "inline",
    "static", "virtual",
}

_PARAMETER_TYPE_KEYWORDS = {
    "auto", "bool", "char", "char8_t", "char16_t", "char32_t",
    "decltype", "double", "float", "int", "long", "short", "signed",
    "unsigned", "void", "wchar_t",
}

_PARAMETER_NON_NAMES = _PARAMETER_TYPE_KEYWORDS | {
    "class", "const", "enum", "register", "restrict", "struct",
    "typename", "union", "volatile",
}

_CLASS_TYPE_DECLARATION_KINDS = frozenset(
    kind
    for key in ("class", "struct", "union")
    for kind in (key + "_definition", key + "_forward")
) | {"local_class", "local_class_forward"}

_SELECTED_TYPE_ALIAS_PROJECTION = object()


def _is_identifier_start(char):
    return char == "_" or char.isalpha()


def _is_identifier_continue(char):
    return char == "_" or char.isalnum()


def _raw_string_end(text, start):
    for prefix in ("u8R\"", "uR\"", "UR\"", "LR\"", "R\""):
        if not text.startswith(prefix, start):
            continue
        delimiter_start = start + len(prefix)
        opening = text.find("(", delimiter_start, delimiter_start + 17)
        if opening < 0:
            return None
        delimiter = text[delimiter_start:opening]
        terminator = ")" + delimiter + "\""
        closing = text.find(terminator, opening + 1)
        return len(text) if closing < 0 else closing + len(terminator)
    return None


def tokenize_cpp(text):
    """Tokenize C++ while discarding comments and preprocessor directives."""
    tokens = []
    i = 0
    line = 1
    line_start = 0
    size = len(text)
    while i < size:
        char = text[i]
        if char.isspace():
            if char == "\n":
                line += 1
                line_start = i + 1
            i += 1
            continue

        if text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = size if end < 0 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = size if end < 0 else end + 2
            line += text.count("\n", i, end)
            last_newline = text.rfind("\n", i, end)
            if last_newline >= 0:
                line_start = last_newline + 1
            i = end
            continue

        if char == "#" and not text[line_start:i].strip():
            end = i
            while True:
                newline = text.find("\n", end)
                if newline < 0:
                    end = size
                    break
                logical = text[end:newline].rstrip()
                end = newline + 1
                line += 1
                line_start = end
                if not logical.endswith("\\"):
                    break
            i = end
            continue

        raw_end = _raw_string_end(text, i)
        if raw_end is not None:
            tokens.append(Token("<literal>", i, raw_end, line))
            line += text.count("\n", i, raw_end)
            last_newline = text.rfind("\n", i, raw_end)
            if last_newline >= 0:
                line_start = last_newline + 1
            i = raw_end
            continue

        if char in ("\"", "'"):
            quote = char
            end = i + 1
            while end < size:
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == quote:
                    end += 1
                    break
                end += 1
            tokens.append(Token("<literal>", i, min(end, size), line))
            line += text.count("\n", i, min(end, size))
            i = min(end, size)
            continue

        if _is_identifier_start(char):
            end = i + 1
            while end < size and _is_identifier_continue(text[end]):
                end += 1
            tokens.append(Token(text[i:end], i, end, line))
            i = end
            continue

        if char.isdigit():
            end = i + 1
            while end < size and (text[end].isalnum() or text[end] in "._'"):
                end += 1
            tokens.append(Token(text[i:end], i, end, line))
            i = end
            continue

        punct = next((value for value in _MULTI_PUNCTUATION
                      if text.startswith(value, i)), char)
        tokens.append(Token(punct, i, i + len(punct), line))
        i += len(punct)
    return tokens


def _pair_map(tokens, opening, closing):
    stack = []
    pairs = {}
    for index, token in enumerate(tokens):
        if token.text == opening:
            stack.append(index)
        elif token.text == closing and stack:
            left = stack.pop()
            pairs[left] = index
            pairs[index] = left
    return pairs


def _bounded_attribute_token_indices(attribute_pairs):
    """Return every token enclosed by a balanced C++ attribute group."""
    return frozenset(
        index
        for opening, closing in attribute_pairs.items()
        if opening < closing
        for index in range(opening, closing + 1)
    )


def _class_head_name(tokens, declaration, attribute_pairs, paren_pairs):
    """Return a bounded class-head name, its template-ids, and its end."""
    cursor = declaration + 1
    while cursor < len(tokens):
        if tokens[cursor].text == "[[":
            closing = attribute_pairs.get(cursor)
            if closing is None:
                return (), (), len(tokens), False
            cursor = closing + 1
            continue
        if (tokens[cursor].text == "alignas"
                and cursor + 1 < len(tokens)
                and tokens[cursor + 1].text == "("):
            closing = paren_pairs.get(cursor + 1)
            if closing is None:
                return (), (), len(tokens), False
            cursor = closing + 1
            continue
        break

    absolute = cursor < len(tokens) and tokens[cursor].text == "::"
    if absolute:
        cursor += 1
    names = []
    template_ids = []
    while cursor < len(tokens):
        if (not tokens[cursor].text
                or not _is_identifier_start(tokens[cursor].text[0])):
            return (), (), cursor, absolute
        names.append(cursor)
        cursor += 1
        if cursor < len(tokens) and tokens[cursor].text == "<":
            template_ids.append(names[-1])
            depth = 0
            while cursor < len(tokens):
                value = tokens[cursor].text
                if value == "<":
                    depth += 1
                elif value == ">":
                    depth -= 1
                elif value == ">>" and depth >= 2:
                    depth -= 2
                cursor += 1
                if depth == 0:
                    break
            if depth:
                return (), (), cursor, absolute
        if cursor >= len(tokens) or tokens[cursor].text != "::":
            break
        cursor += 1
        if cursor < len(tokens) and tokens[cursor].text == "template":
            cursor += 1
    return tuple(names), tuple(template_ids), cursor, absolute


def _class_name_index(tokens, declaration, attribute_pairs, paren_pairs):
    """Return the terminal identifier in a bounded class-head name."""
    names, _, _, _ = _class_head_name(
        tokens, declaration, attribute_pairs, paren_pairs)
    return names[-1] if names else len(tokens)


def _class_definition_opening(tokens, after_name, attribute_pairs):
    """Return a class-head's definition brace, never a declarator body."""
    cursor = after_name
    while cursor < len(tokens):
        if tokens[cursor].text == "[[":
            closing = attribute_pairs.get(cursor)
            if closing is None:
                return None
            cursor = closing + 1
            continue
        if tokens[cursor].text == "final":
            cursor += 1
            continue
        break
    if cursor >= len(tokens):
        return None
    if tokens[cursor].text == "{":
        return cursor
    if tokens[cursor].text != ":":
        return None

    cursor += 1
    parens = brackets = angles = 0
    while cursor < len(tokens):
        value = tokens[cursor].text
        if value == "[[":
            closing = attribute_pairs.get(cursor)
            if closing is None:
                return None
            cursor = closing + 1
            continue
        if value == "(":
            parens += 1
        elif value == ")":
            if parens == 0:
                return None
            parens -= 1
        elif value == "[":
            brackets += 1
        elif value == "]":
            if brackets == 0:
                return None
            brackets -= 1
        elif value == "<":
            angles += 1
        elif value == ">" and angles:
            angles -= 1
        elif value == ">>" and angles >= 2:
            angles -= 2
        elif parens == brackets == angles == 0:
            if value == "{":
                return cursor
            if value in (";", "=", ")"):
                return None
        cursor += 1
    return None


def _class_regions(tokens, brace_pairs, attribute_pairs=None,
                   paren_pairs=None):
    if attribute_pairs is None:
        attribute_pairs = _pair_map(tokens, "[[", "]]")
    if paren_pairs is None:
        paren_pairs = _pair_map(tokens, "(", ")")
    attribute_indices = _bounded_attribute_token_indices(attribute_pairs)
    regions = []
    for index, token in enumerate(tokens):
        if index in attribute_indices:
            continue
        if token.text not in ("class", "struct", "union"):
            continue
        if index > 0 and tokens[index - 1].text == "enum":
            continue
        (name_indices, template_id_name_indices, after_name,
         qualified_absolute) = _class_head_name(
            tokens, index, attribute_pairs, paren_pairs)
        name_index = _class_name_index(
            tokens, index, attribute_pairs, paren_pairs)
        if not name_indices or name_index != name_indices[-1]:
            continue
        opening = _class_definition_opening(
            tokens, after_name, attribute_pairs)
        if opening is None:
            continue
        closing = brace_pairs.get(opening)
        if closing is not None:
            regions.append(ClassRegion(
                opening=opening,
                closing=closing,
                default_access=("private" if token.text == "class"
                                else "public"),
                declaration=index,
                name_index=name_index,
                name=tokens[name_index].text,
                base_tokens=tuple(tokens[after_name:opening]),
                tag=token.text,
                name_indices=name_indices,
                template_id_name_indices=template_id_name_indices,
                qualified_absolute=qualified_absolute,
            ))
    return regions


def _namespace_regions(tokens, brace_pairs, attribute_pairs=None):
    """Return namespace bodies keyed by their qualified namespace identity.

    Separate definitions of one named namespace share one path, while sibling
    and anonymous namespaces retain independent scopes. Nested namespace
    syntax (``namespace outer::inner``) and ordinary nesting map to the same
    identity.
    """
    if attribute_pairs is None:
        attribute_pairs = _pair_map(tokens, "[[", "]]")
    attribute_indices = _bounded_attribute_token_indices(attribute_pairs)
    raw_regions = []
    for index, token in enumerate(tokens):
        if index in attribute_indices:
            continue
        if token.text != "namespace":
            continue
        if index > 0 and tokens[index - 1].text == "using":
            continue
        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor].text not in ("{", ";"):
            cursor += 1
        if cursor >= len(tokens) or tokens[cursor].text != "{":
            continue
        closing = brace_pairs.get(cursor)
        if closing is not None:
            names = []
            name_index = index + 1
            while name_index < cursor:
                if tokens[name_index].text == "[[":
                    attribute_close = attribute_pairs.get(name_index)
                    if attribute_close is None or attribute_close >= cursor:
                        names = []
                        break
                    name_index = attribute_close + 1
                    continue
                if (tokens[name_index].text != "inline"
                        and tokens[name_index].text != "::"
                        and _is_identifier_start(
                            tokens[name_index].text[0])):
                    names.append(name_index)
                name_index += 1
            name_indices = tuple(names)
            names = tuple(tokens[name_index].text
                          for name_index in name_indices)
            raw_regions.append(
                (cursor, closing, index, names, name_indices))

    regions = []
    for opening, closing, declaration, names, name_indices in sorted(
            raw_regions):
        parents = [region for region in regions
                   if region.opening < opening and closing < region.closing]
        parent_path = min(parents,
                          key=lambda region: region.closing - region.opening).path \
            if parents else ()
        own_path = names if names else ("<anonymous@{}>".format(opening),)
        regions.append(NamespaceRegion(
            opening=opening,
            closing=closing,
            declaration=declaration,
            path=parent_path + own_path,
            name_indices=name_indices,
        ))
    return regions


def _graph_scope_for_offset(graph, offset):
    """Return the innermost graph scope containing one source offset."""
    candidates = [
        scope for scope in graph.scopes
        if scope.opening <= offset < scope.closing
    ]
    if not candidates:
        return graph.root_scope
    return max(candidates, key=lambda scope: scope.opening).scope_id


def _scope_descriptor_kind(tokens, opening, bracket_pairs, paren_pairs):
    """Classify a non-declaration brace without treating shape as identity."""
    for capture_open, capture_close in bracket_pairs.items():
        if capture_open > capture_close or capture_close >= opening:
            continue
        if (capture_open > 0 and tokens[capture_open - 1].text not in {
                "=", "(", "{", ",", ";", "return", ":"}):
            continue
        cursor = capture_close + 1
        if cursor < opening and tokens[cursor].text == "(":
            close = paren_pairs.get(cursor)
            if close is None or close >= opening:
                continue
            cursor = close + 1
        while cursor < opening and tokens[cursor].text not in ("{", ";"):
            cursor += 1
        if cursor == opening:
            return "lambda"

    start = opening - 1
    while start >= 0 and tokens[start].text not in (";", "{", "}"):
        start -= 1
    signature = tokens[start + 1:opening]
    if _function_declarator_structure(signature) is not None:
        return "function"
    return "ordinary_block"


def _top_level_comma_groups(values):
    """Split a declaration list without entering balanced subexpressions."""
    groups = []
    start = 0
    parens = brackets = braces = angles = attributes = 0
    for index, token in enumerate(values):
        value = token.text
        if value == "[[":
            attributes += 1
        elif value == "]]" and attributes:
            attributes -= 1
        elif not attributes:
            if value == "(":
                parens += 1
            elif value == ")" and parens:
                parens -= 1
            elif value == "[":
                brackets += 1
            elif value == "]" and brackets:
                brackets -= 1
            elif value == "{":
                braces += 1
            elif value == "}" and braces:
                braces -= 1
            else:
                angles = _angle_depth_change(value, angles)
        if (value == "," and not any((
                parens, brackets, braces, angles, attributes))):
            groups.append(tuple(values[start:index]))
            start = index + 1
    groups.append(tuple(values[start:]))
    return tuple(group for group in groups if group)


def _without_attribute_groups(values):
    """Remove balanced declarator attributes while retaining token identity."""
    output = []
    depth = 0
    for token in values:
        if token.text == "[[":
            depth += 1
        elif token.text == "]]" and depth:
            depth -= 1
        elif not depth:
            output.append(token)
    return tuple(output) if depth == 0 else ()


def _typedef_declarators(values):
    """Return bounded alias names and RHS tokens for one typedef list."""
    groups = tuple(
        _without_attribute_groups(group)
        for group in _top_level_comma_groups(values)
    )
    if any(not group for group in groups):
        return ()
    if not groups:
        return ()
    first = groups[0]
    first_name = next((
        index for index in range(len(first) - 1, -1, -1)
        if first[index].text
        and _is_identifier_start(first[index].text[0])
        and first[index].text not in _PARAMETER_NON_NAMES
    ), None)
    if first_name is None:
        return ()
    declarator_start = next((
        index for index in range(first_name)
        if first[index].text in ("*", "&", "&&", "(", "[")
    ), first_name)
    common = tuple(first[:declarator_start])
    if not common:
        return ()

    output = []
    for group_index, group in enumerate(groups):
        name_index = next((
            index for index in range(len(group) - 1, -1, -1)
            if group[index].text
            and _is_identifier_start(group[index].text[0])
            and group[index].text not in _PARAMETER_NON_NAMES
        ), None)
        if name_index is None:
            return ()
        if group_index == 0:
            prefix = group[declarator_start:name_index]
        else:
            prefix = group[:name_index]
        target = common + tuple(prefix) + tuple(group[name_index + 1:])
        output.append((group[name_index], target))
    return tuple(output)


def _class_key_shape_map(tokens, regions, attribute_pairs, paren_pairs):
    """Classify every class-key before any declaration fact is admitted."""
    attribute_indices = _bounded_attribute_token_indices(attribute_pairs)
    regions_by_declaration = {
        region.declaration: region for region in regions
    }

    def template_parameter(index, after_name):
        for boundary in range(index - 1, -1, -1):
            if tokens[boundary].text in (";", "{", "}"):
                break
            opening = boundary + 1
            if (tokens[boundary].text != "template"
                    or opening >= index or tokens[opening].text != "<"):
                continue
            angles = 1
            parens = brackets = braces = attributes = 0
            component_start = opening + 1
            cursor = component_start
            while cursor < index and angles:
                value = tokens[cursor].text
                if value == "[[":
                    attributes += 1
                elif value == "]]" and attributes:
                    attributes -= 1
                elif not attributes:
                    parens += (value == "(") - (value == ")")
                    brackets += (value == "[") - (value == "]")
                    braces += (value == "{") - (value == "}")
                    if parens == brackets == braces == 0:
                        if value == "<":
                            angles += 1
                        elif value == ">":
                            angles -= 1
                        elif value == ">>" and angles >= 2:
                            angles -= 2
                        elif value == "," and angles == 1:
                            component_start = cursor + 1
                cursor += 1
            return (
                angles == 1 and component_start == index
                and after_name < len(tokens)
                and tokens[after_name].text in (",", ">", "=", "...")
            )
        return False

    def standalone_context(index):
        previous = index - 1
        if previous >= 0 and tokens[previous].text == "]]":
            opening = attribute_pairs.get(previous)
            if opening is None:
                return False
            previous = opening - 1
        if previous < 0 or tokens[previous].text in (";", "{", "}", ":"):
            return True
        if tokens[previous].text not in (">", ">>"):
            return False
        depth = 0
        cursor = previous
        while cursor >= 0:
            value = tokens[cursor].text
            if value == ">":
                depth += 1
            elif value == ">>":
                depth += 2
            elif value == "<":
                depth -= 1
                if depth == 0:
                    return cursor > 0 and tokens[cursor - 1].text == "template"
            cursor -= 1
        return False

    output = {}
    for index, token in enumerate(tokens):
        if token.text not in ("class", "struct", "union"):
            continue
        names, template_ids, after_name, absolute = _class_head_name(
            tokens, index, attribute_pairs, paren_pairs)
        region = regions_by_declaration.get(index)
        if index in attribute_indices:
            status = "attribute-content"
            reason = "class-key inside attribute"
        elif index > 0 and tokens[index - 1].text == "enum":
            status = "enum-class-component"
            reason = "enum class component"
        elif template_parameter(index, after_name):
            status = "template-parameter"
            reason = "template type parameter"
        elif region is not None:
            status = "definition-class-head"
            reason = ""
        elif index > 0 and tokens[index - 1].text == "friend":
            status = "unsupported"
            reason = (
                "friend elaborated declaration is outside the bounded grammar")
        else:
            cursor = after_name
            while cursor < len(tokens) and tokens[cursor].text == "[[":
                closing = attribute_pairs.get(cursor)
                if closing is None:
                    cursor = len(tokens)
                    break
                cursor = closing + 1
            if (names and cursor < len(tokens)
                    and tokens[cursor].text == ";"
                    and standalone_context(index)):
                status = "standalone-forward-declaration"
                reason = ""
            elif names:
                status = "elaborated-type-use"
                reason = ""
            else:
                status = "unsupported"
                reason = "class-key has no bounded declarative name"
        output[index] = {
            "status": status,
            "reason": reason,
            "names": tuple(names),
            "template_id_names": tuple(template_ids),
            "after_name": after_name,
            "region": region,
            "absolute": absolute,
        }
    return output


def _control_statement_end(tokens, start, paren_pairs, brace_pairs):
    """Return the exclusive token end of one complete controlled statement."""
    if start >= len(tokens):
        return start
    value = tokens[start].text
    if value == "{":
        closing = brace_pairs.get(start)
        return len(tokens) if closing is None else closing + 1
    if value in ("for", "if", "switch", "while"):
        opening = start + 1
        while opening < len(tokens) and tokens[opening].text != "(":
            opening += 1
        closing = paren_pairs.get(opening)
        if closing is None:
            return start + 1
        extent = _control_statement_end(
            tokens, closing + 1, paren_pairs, brace_pairs)
        if value == "if" and extent < len(tokens) \
                and tokens[extent].text == "else":
            extent = _control_statement_end(
                tokens, extent + 1, paren_pairs, brace_pairs)
        return extent
    if value == "do":
        extent = _control_statement_end(
            tokens, start + 1, paren_pairs, brace_pairs)
        if extent < len(tokens) and tokens[extent].text == "while":
            opening = extent + 1
            while opening < len(tokens) and tokens[opening].text != "(":
                opening += 1
            closing = paren_pairs.get(opening)
            if closing is not None:
                extent = closing + 1
                if extent < len(tokens) and tokens[extent].text == ";":
                    extent += 1
        return extent

    parens = brackets = braces = attributes = 0
    cursor = start
    while cursor < len(tokens):
        current = tokens[cursor].text
        if current == "[[":
            attributes += 1
        elif current == "]]" and attributes:
            attributes -= 1
        elif not attributes:
            parens += (current == "(") - (current == ")")
            brackets += (current == "[") - (current == "]")
            braces += (current == "{") - (current == "}")
            if (current == ";"
                    and parens == brackets == braces == 0):
                return cursor + 1
        cursor += 1
    return len(tokens)


def _control_header_delimiters(tokens, start, end):
    """Return top-level ``;`` and ``:`` tokens in one control header."""
    output = []
    parens = brackets = braces = attributes = 0
    for index in range(start, end):
        value = tokens[index].text
        if value == "[[":
            attributes += 1
        elif value == "]]" and attributes:
            attributes -= 1
        elif not attributes:
            if (value in (";", ":")
                    and parens == brackets == braces == 0):
                output.append(index)
            parens += (value == "(") - (value == ")")
            brackets += (value == "[") - (value == "]")
            braces += (value == "{") - (value == "}")
    return tuple(output)


def _control_data_names(
        tokens, start, end, attribute_pairs, bracket_pairs,
        declaration_context=None):
    """Return every bounded ordinary or structured control-declaration name."""
    if start >= end:
        return ()
    cursor = start
    parens = brackets = braces = 0
    while cursor < end:
        value = tokens[cursor].text
        if value == "[[":
            closing = attribute_pairs.get(cursor)
            if closing is None or closing >= end:
                return ()
            cursor = closing + 1
            continue
        prefix = tokens[start:cursor]
        prefix_name = _ordinary_parameter_name_index(
            [token.text for token in prefix])
        structured_prefix = (
            (prefix_name is None or prefix_name < 0)
            and any(token.text == "auto" for token in prefix)
            and all(token.text in {
                "auto", "const", "volatile", "&", "&&", "static",
                "thread_local", "[[", "]]",
            } or (token.text and _is_identifier_start(token.text[0]))
                    for token in prefix)
        )
        if (value == "[" and parens == brackets == braces == 0
                and structured_prefix):
            closing = bracket_pairs.get(cursor)
            if closing is None or closing >= end:
                return ()
            components = _top_level_comma_groups(
                tokens[cursor + 1:closing])
            names = []
            for component in components:
                bounded = _without_attribute_groups(component)
                candidates = [
                    token for token in bounded
                    if token.text
                    and _is_identifier_start(token.text[0])
                    and token.text not in _PARAMETER_NON_NAMES
                ]
                if len(candidates) != 1:
                    return ()
                names.append(candidates[0])
            return tuple(names)
        if (value in ("=", "{")
                and parens == brackets == braces == 0):
            break
        parens += (value == "(") - (value == ")")
        brackets += (value == "[") - (value == "]")
        braces += (value == "{") - (value == "}")
        cursor += 1
    return _ordinary_data_name_tokens(
        tokens[start:end], declaration_context)


def _build_control_declarations(
        tokens, scopes, parent_scope_for, paren_pairs, brace_pairs,
        bracket_pairs, attribute_pairs, declaration_graph,
        declaration_environment):
    """Build graph-owned control declarations with full statement extents."""
    raw_controls = []
    for index, token in enumerate(tokens):
        if token.text not in ("for", "if", "switch"):
            continue
        opening = index + 1
        if (token.text == "if" and opening < len(tokens)
                and tokens[opening].text == "constexpr"):
            opening += 1
        if opening >= len(tokens) or tokens[opening].text != "(":
            continue
        closing = paren_pairs.get(opening)
        if closing is None or closing <= opening + 1:
            continue
        delimiters = _control_header_delimiters(
            tokens, opening + 1, closing)
        semicolons = tuple(
            item for item in delimiters if tokens[item].text == ";")
        colons = tuple(
            item for item in delimiters if tokens[item].text == ":")
        declaration_slices = []
        if token.text == "for" and len(semicolons) == 2:
            kind = "classic-for"
            declaration_slices.append((
                kind, opening + 1, semicolons[0]))
        elif (token.text == "for" and len(semicolons) <= 1
              and len(colons) == 1
              and (not semicolons or semicolons[0] < colons[0])):
            kind = "range-for"
            range_start = opening + 1
            if semicolons:
                declaration_slices.append((
                    "range-for-init", range_start, semicolons[0]))
                range_start = semicolons[0] + 1
            declaration_slices.append((
                kind, range_start, colons[0]))
        elif token.text in ("if", "switch") and len(semicolons) == 1:
            kind = token.text + "-init"
            declaration_slices.extend((
                (kind, opening + 1, semicolons[0]),
                (token.text + "-condition", semicolons[0] + 1, closing),
            ))
        elif token.text in ("if", "switch") and not semicolons:
            kind = token.text + "-condition"
            declaration_slices.append((kind, opening + 1, closing))
        else:
            continue
        names = tuple(
            (declaration_kind, name)
            for declaration_kind, declaration_start, declaration_end
            in declaration_slices
            for name in _control_data_names(
                tokens, declaration_start, declaration_end,
                attribute_pairs, bracket_pairs,
                (
                    declaration_graph, declaration_environment,
                    parent_scope_for(tokens[declaration_start].start),
                    tokens[declaration_start].start,
                ))
        )
        if not names:
            continue
        statement_start = closing + 1
        statement_end = _control_statement_end(
            tokens, statement_start, paren_pairs, brace_pairs)
        if (token.text == "if" and statement_end < len(tokens)
                and tokens[statement_end].text == "else"):
            statement_end = _control_statement_end(
                tokens, statement_end + 1, paren_pairs, brace_pairs)
        if statement_start >= len(tokens) or statement_end <= statement_start:
            continue
        raw_controls.append({
            "kind": kind,
            "keyword_index": index,
            "opening": opening,
            "closing": closing,
            "statement_start": statement_start,
            "statement_end": statement_end,
            "names": names,
        })

    controls = []
    data_names = []
    for control_id, raw in enumerate(raw_controls):
        keyword = tokens[raw["keyword_index"]]
        statement_start = raw["statement_start"]
        statement_end = raw["statement_end"]
        parent_control = -1
        containing = [
            (candidate_id, candidate)
            for candidate_id, candidate in enumerate(raw_controls)
            if candidate_id != control_id
            and candidate["statement_start"] <= raw["keyword_index"]
            < candidate["statement_end"]
        ]
        if containing:
            parent_control = min(
                containing,
                key=lambda item: (
                    item[1]["statement_end"] - item[1]["statement_start"],
                    -item[1]["statement_start"],
                ),
            )[0]
        parent_scope = parent_scope_for(keyword.start)
        callable_owner = scopes[parent_scope].callable_owner
        data_ids = []
        for declaration_kind, name in raw["names"]:
            data_id = len(data_names)
            data_ids.append(data_id)
            data_names.append(DataNameDeclaration(
                declaration_id=data_id,
                name=name.text,
                kind=declaration_kind,
                point_of_declaration=name.end,
                lifetime_begin=name.end,
                lifetime_end=tokens[statement_end - 1].end,
                parent_scope=parent_scope,
                callable_owner=callable_owner,
                control_id=control_id,
                header_range=(
                    tokens[raw["opening"]].start,
                    tokens[raw["closing"]].end,
                ),
                controlled_statement_range=(
                    tokens[statement_start].start,
                    tokens[statement_end - 1].end,
                ),
            ))
        controls.append(ControlStatementDescriptor(
            control_id=control_id,
            kind=raw["kind"],
            keyword_offset=keyword.start,
            header_open=tokens[raw["opening"]].start,
            header_close=tokens[raw["closing"]].start,
            controlled_statement_open=tokens[statement_start].start,
            controlled_statement_end=tokens[statement_end - 1].end,
            parent_control=parent_control,
            data_declaration_ids=tuple(data_ids),
        ))
    return tuple(data_names), tuple(controls)


def _declaration_prefix_is_type(prefix, name_index, context):
    """Distinguish a bounded declaration prefix from an expression prefix."""
    if context is None:
        return True
    before_name = tuple(prefix[:name_index])
    values = [token.text for token in before_name]
    declarator_start = next((
        index for index, value in enumerate(values)
        if value in {"*", "&", "&&", "(", "["}
    ), len(values))
    type_tokens = tuple(
        token for token in before_name[:declarator_start]
        if token.text not in {
            "const", "consteval", "constexpr", "extern", "inline",
            "mutable", "register", "static", "thread_local", "volatile",
        }
    )
    type_values = tuple(token.text for token in type_tokens)
    if (type_values
            and all(value in _PARAMETER_TYPE_KEYWORDS
                    for value in type_values)):
        return True
    if type_values[:1] in {
            ("class",), ("enum",), ("struct",), ("union",)}:
        elaborated = _simple_qualified_type(type_values[1:])
        return elaborated is not None and bool(type_values[1:])
    parsed = _simple_qualified_type(
        type_values)
    if not type_tokens:
        return False
    if parsed is None:
        return (
            declarator_start == len(values)
            and _template_qualified_type_skeleton(type_values) is not None
        )
    parts, absolute = parsed
    graph, environment, parent_scope, use_offset = context
    owner_class = next((
        tuple(scope.canonical_class)
        for scope in _graph_scope_chain(graph, parent_scope)
        if scope.canonical_class
    ), ())
    result = _resolve_qualified_name(
        _qualified_component_records(
            parts, type_tokens[0].start, environment.source_tokens),
        LookupContext(
            starting_scope=parent_scope,
            use_offset=use_offset,
            purpose="data-declaration-type",
            complete_class=bool(owner_class),
            owner_class=owner_class,
            absolute=absolute,
        ),
        environment,
        _record=False,
    )
    graph_declarations = graph.declarations
    provenance_is_type = any(
        isinstance(declaration_id, int)
        and 0 <= declaration_id < len(graph_declarations)
        and (
            graph_declarations[declaration_id].kind
            in {"using_alias", "typedef_alias"}
            or graph_declarations[declaration_id].kind
            in _CLASS_TYPE_DECLARATION_KINDS
        )
        for declaration_id in result.provenance
    )
    resolved_type = (
        result.selected_declaration_kind in {"using_alias", "typedef_alias"}
        or provenance_is_type
    )
    if resolved_type:
        return True
    if declarator_start != len(values):
        return False
    # Two adjacent names are declaration syntax even when the bounded graph
    # cannot resolve an imported type. Balanced template-ids receive the same
    # conservative shadow treatment; they must never manufacture a member
    # return positive merely because their type identity is unsupported.
    return (parsed is not None
            or _template_qualified_type_skeleton(type_values) is not None)


def _ordinary_data_name_tokens(statement, declaration_context=None):
    """Return ordinary declarator names from one bounded declaration."""
    if not statement or statement[0].text == "(":
        return ()
    first_identifier = next((
        token.text for token in statement
        if token.text and _is_identifier_start(token.text[0])
    ), "")
    if first_identifier in {
            "asm", "break", "case", "catch", "class", "co_await",
            "co_return", "concept", "continue", "default", "delete",
            "do", "else", "enum", "for", "friend", "goto", "if",
            "namespace", "new", "return", "static_assert", "struct",
            "switch", "template", "throw", "try", "typedef", "union",
            "using", "while"}:
        return ()
    names = []
    declaration_seen = False
    for declarator in _top_level_comma_groups(statement):
        ordinary = _without_attribute_groups(declarator)
        if not ordinary:
            return ()
        prefix = []
        parens = brackets = braces = angles = 0
        for token in ordinary:
            value = token.text
            if (value in ("=", "{")
                    and parens == brackets == braces == angles == 0):
                break
            prefix.append(token)
            parens += (value == "(") - (value == ")")
            brackets += (value == "[") - (value == "]")
            braces += (value == "{") - (value == "}")
            angles = _angle_depth_change(value, angles)
        values = [token.text for token in prefix]
        if (any(token.text in (".", "->", "->*") for token in prefix)
                or _function_declarator_structure(prefix) is not None):
            return ()
        name_index = _ordinary_parameter_name_index(values)
        if name_index is not None and name_index >= 0:
            if (not declaration_seen
                    and not _declaration_prefix_is_type(
                        prefix, name_index, declaration_context)):
                return ()
            names.append(prefix[name_index])
            declaration_seen = True
            continue
        candidates = [
            token for token in prefix
            if token.text and _is_identifier_start(token.text[0])
            and token.text not in _PARAMETER_NON_NAMES
            and token.text not in _RETURN_SPECIFIERS
        ]
        if declaration_seen and len(candidates) == 1:
            names.append(candidates[0])
            continue
        return ()
    return tuple(names)


def _statement_start(tokens, semicolon):
    """Find one statement start without stopping inside balanced syntax."""
    parens = brackets = braces = attributes = 0
    cursor = semicolon - 1
    while cursor >= 0:
        value = tokens[cursor].text
        if value == "]]":
            attributes += 1
        elif value == "[[" and attributes:
            attributes -= 1
        elif not attributes:
            if value == ")":
                parens += 1
            elif value == "(" and parens:
                parens -= 1
            elif value == "]":
                brackets += 1
            elif value == "[" and brackets:
                brackets -= 1
            elif value == "}":
                braces += 1
            elif value == "{" and braces:
                braces -= 1
            elif (value in (";", "{", "}")
                  and parens == brackets == braces == 0):
                return cursor + 1
        cursor -= 1
    return 0


def _build_graph_data_declarations(
        tokens, scopes, parent_scope_for, control_data, controls,
        declaration_graph, declaration_environment):
    """Add parameter and ordinary-block data facts to control facts."""
    data_names = list(control_data)

    opening_tokens = {
        token.end: index for index, token in enumerate(tokens)
        if token.text == "{"
    }
    for scope in scopes:
        if scope.kind not in ("function", "lambda"):
            continue
        opening = opening_tokens.get(scope.opening)
        if opening is None:
            continue
        start = opening - 1
        while start >= 0 and tokens[start].text not in (";", "{", "}"):
            start -= 1
        signature = tokens[start + 1:opening]
        structure = _function_declarator_structure(signature)
        if structure is None:
            continue
        parameters = signature[
            structure.parameters_open + 1:structure.parameters_close]
        for parameter in _top_level_comma_groups(parameters):
            for name in _ordinary_data_name_tokens(parameter):
                data_names.append(DataNameDeclaration(
                    declaration_id=len(data_names),
                    name=name.text,
                    kind="parameter",
                    point_of_declaration=name.end,
                    lifetime_begin=name.end,
                    lifetime_end=scope.closing,
                    parent_scope=scope.scope_id,
                    callable_owner=scope.callable_owner,
                ))

    control_headers = tuple(
        (control.header_open, control.header_close) for control in controls)
    for index, token in enumerate(tokens):
        if token.text != ";" or any(
                opening < token.start < closing
                for opening, closing in control_headers):
            continue
        parent_scope = parent_scope_for(token.start)
        scope = scopes[parent_scope]
        if scope.kind not in ("function", "lambda", "ordinary_block"):
            continue
        start = _statement_start(tokens, index)
        for name in _ordinary_data_name_tokens(
                tokens[start:index], (
                    declaration_graph, declaration_environment, parent_scope,
                    tokens[start].start if start < index else token.start,
                )):
            data_names.append(DataNameDeclaration(
                declaration_id=len(data_names),
                name=name.text,
                kind="block-local",
                point_of_declaration=name.end,
                lifetime_begin=name.end,
                lifetime_end=scope.closing,
                parent_scope=parent_scope,
                callable_owner=scope.callable_owner,
            ))
    return tuple(data_names)


def _structural_lookup_environment(graph, tokens):
    """Project one declaration graph into the central lookup environment."""
    type_aliases = {}
    namespace_aliases = {}
    for declaration in graph.declarations:
        parent = graph.scopes[declaration.parent_scope]
        if parent.kind == "nonlocal_class":
            semantic_scope = tuple(parent.canonical_class)
        elif parent.kind == "namespace_fragment":
            semantic_scope = tuple(parent.canonical_namespace)
        elif parent.kind == "global":
            semantic_scope = ()
        else:
            continue
        key = semantic_scope + (declaration.name,)
        if declaration.kind in {"using_alias", "typedef_alias"}:
            type_aliases.setdefault(key, []).append(TypeAliasTarget(
                semantic_scope, declaration.target_tokens,
                declaration.rhs_use_position, declaration.access,
                declaration.declaration_id, graph,
            ))
        elif declaration.kind == "namespace_alias":
            values = list(declaration.target_tokens)
            absolute = bool(values and values[0] == "::")
            if absolute:
                values.pop(0)
            parts = tuple(values[::2])
            if (parts and len(values) == len(parts) * 2 - 1
                    and all(values[index] == "::"
                            for index in range(1, len(values), 2))):
                namespace_aliases.setdefault(key, []).append(
                    NamespaceAliasTarget(
                        semantic_scope, absolute, parts,
                        declaration.rhs_use_position,
                        declaration.declaration_id, graph,
                    ))
    classes = {
        tuple(identity): region
        for region, identity in graph.class_identities.items()
        if (region in graph.nonlocal_class_regions
            or region in graph.local_class_regions)
    }
    complete_scopes = {
        identity: tuple(
            identity[:length]
            for length in range(len(identity), 0, -1)
            if identity[:length] in classes
        )
        for identity in classes
    }
    return AliasResolutionEnvironment(
        type_aliases={key: tuple(values)
                      for key, values in type_aliases.items()},
        namespace_aliases={key: tuple(values)
                           for key, values in namespace_aliases.items()},
        known_namespaces=frozenset(
            set(graph.namespace_scopes) | {(), ("std",)}),
        namespace_declarations={},
        namespace_alias_cache={},
        classes_by_identity=classes,
        class_identities=graph.class_identities,
        class_declarations=graph.class_declaration_points,
        type_alias_cache={},
        complete_class_scopes=complete_scopes,
        base_edges={},
        declaration_graph=graph,
        source_tokens=tuple(tokens),
    )


def _populate_structural_base_edges(environment):
    """Resolve same-header base edges needed by class-key access checks."""
    graph = environment.declaration_graph
    edges = environment.base_edges
    for region in sorted(
            graph.nonlocal_class_regions,
            key=lambda item: item.declaration):
        derived = tuple(graph.class_identities[region])
        if derived not in environment.classes_by_identity:
            continue
        resolved_edges = []
        for (values, inheritance_access, is_virtual,
             base_use_position) in _base_specifications(
                region.base_tokens, region.default_access):
            resolved = _resolve_base_class(
                values, derived[:-1], base_use_position,
                environment.type_aliases, environment.namespace_aliases,
                environment.known_namespaces,
                environment.namespace_alias_cache,
                environment.classes_by_identity,
                environment.type_alias_cache,
                environment.complete_class_scopes.get(derived, ()),
                edges,
                declaration_graph=graph,
                alias_environment=environment,
            )
            if resolved is None:
                continue
            resolved_edges.append(BaseEdge(
                derived, tuple(graph.class_identities[resolved]),
                tuple(values), inheritance_access,
                base_use_position, is_virtual,
            ))
        edges[derived] = tuple(resolved_edges)


def _member_signature_owner_ranges(graph, tokens, environment):
    """Associate out-of-class signature ranges with their exact owner."""
    opening_tokens = {
        token.end: index for index, token in enumerate(tokens)
        if token.text == "{"
    }
    output = []
    for scope in graph.scopes:
        if scope.kind != "function":
            continue
        opening = opening_tokens.get(scope.opening)
        if opening is None:
            continue
        start = opening - 1
        while start >= 0 and tokens[start].text not in (";", "{", "}"):
            start -= 1
        signature = tokens[start + 1:opening]
        structure = _function_declarator_structure(signature)
        if structure is None:
            continue
        parts, absolute, has_template_id, owner_start = \
            _qualified_owner_components(signature, structure.name_start)
        if not parts or has_template_id:
            continue
        first = next((
            token for token in signature[owner_start:structure.name_start]
            if token.text == parts[0]
        ), None)
        if first is None:
            continue
        context = LookupContext(
            starting_scope=_graph_scope_for_offset(graph, first.start),
            use_offset=first.start,
            purpose="member-definition-owner",
            absolute=absolute,
        )
        result = _resolve_qualified_name(
            _qualified_component_records(
                parts, first.start, environment.source_tokens),
            context, environment, _record=False,
        )
        identity = tuple(result.canonical_identity)
        if (result.status == "found"
                and identity in environment.classes_by_identity):
            output.append((
                tokens[start + 1].start, scope.closing,
                identity, result, scope.scope_id, scope.opening,
            ))
    return tuple(output)


def _build_declaration_graph_pass(
        text, tokens, brace_pairs, qualified_identities=None):
    """Build the one lexical declaration graph consumed by Tier-G lookup."""
    qualified_identities = dict(qualified_identities or {})
    paren_pairs = _pair_map(tokens, "(", ")")
    attribute_pairs = _pair_map(tokens, "[[", "]]")
    attribute_indices = _bounded_attribute_token_indices(attribute_pairs)
    namespace_regions = tuple(_namespace_regions(
        tokens, brace_pairs, attribute_pairs))
    all_class_regions = tuple(_class_regions(
        tokens, brace_pairs, attribute_pairs, paren_pairs))
    class_key_shapes = _class_key_shape_map(
        tokens, all_class_regions, attribute_pairs, paren_pairs)
    class_openings = {region.opening for region in all_class_regions}
    namespace_openings = {region.opening for region in namespace_regions}
    bracket_pairs = _pair_map(tokens, "[", "]")

    local_regions = []
    nonlocal_regions = []
    for region in all_class_regions:
        enclosing_other = any(
            opening < region.opening < brace_pairs[opening]
            and opening not in class_openings
            and opening not in namespace_openings
            for opening in brace_pairs
            if tokens[opening].text == "{"
        )
        (local_regions if enclosing_other else nonlocal_regions).append(region)
    syntactic_local_regions = frozenset(local_regions)

    def namespace_for_region(region):
        namespaces = [
            candidate for candidate in namespace_regions
            if candidate.opening < region.opening
            and region.closing < candidate.closing
        ]
        return (
            min(namespaces, key=lambda item: item.closing - item.opening).path
            if namespaces else ()
        )

    def namespace_for_index(index):
        namespaces = [
            candidate for candidate in namespace_regions
            if candidate.opening < index < candidate.closing
        ]
        return (
            min(namespaces, key=lambda item: item.closing - item.opening).path
            if namespaces else ()
        )

    nonlocal_region_by_declaration = {
        region.declaration: region for region in nonlocal_regions
    }

    def containing_nonlocal_region(index):
        owners = [
            region for region in nonlocal_regions
            if region.opening < index < region.closing
        ]
        return (min(owners, key=lambda item: item.closing - item.opening)
                if owners else None)

    def inside_ordinary_block(index):
        return any(
            opening < index < brace_pairs[opening]
            and opening not in class_openings
            and opening not in namespace_openings
            for opening in brace_pairs
            if tokens[opening].text == "{")

    class_identities = {}
    resolved_forward_indices = set()

    event_indices = tuple(
        index for index, shape in class_key_shapes.items()
        if shape["status"] in (
            "definition-class-head", "standalone-forward-declaration")
    )
    while True:
        changed = False
        for index in event_indices:
            region = nonlocal_region_by_declaration.get(index)
            if region is not None and region not in class_identities:
                spelling = tuple(
                    tokens[name].text for name in region.name_indices)
                owner = containing_nonlocal_region(region.opening)
                if len(spelling) == 1:
                    if owner is not None:
                        if owner not in class_identities:
                            continue
                        identity = class_identities[owner] + spelling
                    else:
                        identity = (tuple(namespace_for_region(region))
                                    + spelling)
                else:
                    identity = qualified_identities.get(index)
                    if identity is None:
                        continue
                class_identities[region] = identity
                changed = True
                continue

            if index in resolved_forward_indices or inside_ordinary_block(index):
                continue
            names, _, after_name, _ = _class_head_name(
                tokens, index, attribute_pairs, paren_pairs)
            cursor = after_name
            while cursor < len(tokens) and tokens[cursor].text == "[[":
                closing = attribute_pairs.get(cursor)
                if closing is None:
                    cursor = len(tokens)
                    break
                cursor = closing + 1
            if (not names or cursor >= len(tokens)
                    or tokens[cursor].text != ";"):
                continue
            spelling = tuple(tokens[name].text for name in names)
            owner = containing_nonlocal_region(index)
            if owner is not None:
                if owner not in class_identities or len(spelling) != 1:
                    continue
                identity = class_identities[owner] + spelling
            elif len(spelling) == 1:
                identity = tuple(namespace_for_index(index)) + spelling
            else:
                identity = qualified_identities.get(index)
                if identity is None:
                    continue
            resolved_forward_indices.add(index)
            changed = True
        if not changed:
            break

    unresolved_regions = [
        region for region in nonlocal_regions
        if region not in class_identities
    ]
    for region in unresolved_regions:
        nonlocal_regions.remove(region)
        local_regions.append(region)
        class_identities[region] = (
            "<unsupported-qualified@{}>".format(
                tokens[region.declaration].start),
            region.name,
        )
    for region in local_regions:
        class_identities.setdefault(region, (
            "<local@{}>".format(tokens[region.declaration].start), region.name
        ))

    descriptors = []
    for region in namespace_regions:
        introduced_count = len(region.name_indices)
        parent_length = len(region.path) - introduced_count
        for component, name_index in enumerate(region.name_indices[:-1], 1):
            descriptors.append((
                name_index,
                region.closing,
                "namespace_prefix_fragment",
                NamespaceRegion(
                    opening=name_index,
                    closing=region.closing,
                    declaration=region.declaration,
                    path=region.path[:parent_length + component],
                    name_indices=region.name_indices[:component],
                ),
            ))
        descriptors.append((
            region.opening, region.closing, "namespace_fragment", region
        ))
    for region in all_class_regions:
        descriptors.append((
            region.opening,
            region.closing,
            "local_class" if region in local_regions else "nonlocal_class",
            region,
        ))
    for opening, closing in brace_pairs.items():
        if (tokens[opening].text != "{" or opening in class_openings
                or opening in namespace_openings):
            continue
        descriptors.append((
            opening, closing,
            _scope_descriptor_kind(
                tokens, opening, bracket_pairs, paren_pairs),
            None,
        ))
    descriptors.sort(key=lambda item: (item[0], -item[1]))

    scopes = [LexicalScope(
        scope_id=0,
        kind="global",
        parent_scope=-1,
        opening=0,
        closing=len(text) + 1,
    )]
    for opening, closing, kind, owner in descriptors:
        open_offset = tokens[opening].end
        close_offset = (
            tokens[closing].end
            if kind == "namespace_prefix_fragment"
            else tokens[closing].start
        )
        parents = [
            scope for scope in scopes
            if scope.opening <= tokens[opening].start
            and tokens[closing].end <= scope.closing
        ]
        parent = max(parents, key=lambda item: item.opening).scope_id
        canonical_namespace = (
            owner.path
            if kind in ("namespace_fragment", "namespace_prefix_fragment")
            else ()
        )
        canonical_class = (
            class_identities[owner]
            if kind in ("nonlocal_class", "local_class") else ()
        )
        scope_id = len(scopes)
        if kind in ("function", "lambda"):
            callable_owner = scope_id
        elif kind == "ordinary_block":
            callable_owner = scopes[parent].callable_owner
        else:
            callable_owner = -1
        scopes.append(LexicalScope(
            scope_id=scope_id,
            kind=("namespace_fragment"
                  if kind == "namespace_prefix_fragment" else kind),
            parent_scope=parent,
            opening=open_offset,
            closing=close_offset,
            canonical_namespace=tuple(canonical_namespace),
            canonical_class=tuple(canonical_class),
            callable_owner=callable_owner,
        ))

    def parent_scope_for(offset):
        candidates = [
            scope for scope in scopes
            if scope.opening <= offset < scope.closing
        ]
        return max(candidates, key=lambda item: item.opening).scope_id

    def lifetime_for(scope_id):
        scope = scopes[scope_id]
        if scope.kind in ("global", "namespace_fragment"):
            return len(text) + 1
        return scope.closing

    brace_token_by_scope_opening = {
        token.end: index for index, token in enumerate(tokens)
        if token.text == "{"
    }

    def logical_scope_identity(scope_id):
        chain = []
        active = set()
        while scope_id >= 0 and scope_id not in active:
            active.add(scope_id)
            chain.append(scopes[scope_id])
            scope_id = scopes[scope_id].parent_scope
        identity = ()
        callable_names = []
        for scope in reversed(chain):
            if scope.canonical_class:
                identity = tuple(scope.canonical_class)
                callable_names = []
            elif scope.canonical_namespace and not identity:
                identity = tuple(scope.canonical_namespace)
            if scope.kind not in ("function", "lambda"):
                continue
            opening_index = brace_token_by_scope_opening.get(scope.opening)
            if opening_index is None:
                continue
            start = opening_index - 1
            while start >= 0 and tokens[start].text not in (";", "{", "}"):
                start -= 1
            structure = _function_declarator_structure(
                tokens[start + 1:opening_index])
            callable_names.append(
                structure.name if structure is not None else "<callable>")
        return identity + tuple(callable_names)

    local_identity_replacements = {}
    for region in syntactic_local_regions:
        old_identity = tuple(class_identities[region])
        parent_scope = parent_scope_for(tokens[region.declaration].start)
        identity = (
            logical_scope_identity(parent_scope)
            + (tokens[region.name_index].text,)
        )
        class_identities[region] = identity
        local_identity_replacements[old_identity] = identity
    if local_identity_replacements:
        scopes = [replace(
            scope,
            canonical_class=local_identity_replacements.get(
                tuple(scope.canonical_class), scope.canonical_class),
        ) for scope in scopes]

    declarations = []

    def add_declaration(name, kind, canonical_identity, canonical_target,
                        declaration_start, point, body_open, body_close,
                        parent_scope, target_tokens=(), rhs_position=-1,
                        access="public", dependent=False,
                        complete_eligible=False):
        declarations.append(TypeDeclaration(
            declaration_id=len(declarations),
            name=name,
            kind=kind,
            canonical_identity=tuple(canonical_identity),
            canonical_target=tuple(canonical_target),
            declaration_start=declaration_start,
            point_of_declaration=point,
            body_open=body_open,
            body_close=body_close,
            parent_scope=parent_scope,
            lifetime_end=lifetime_for(parent_scope),
            target_tokens=tuple(target_tokens),
            rhs_use_position=rhs_position,
            access=access,
            dependent=dependent,
            complete_class_eligible=complete_eligible,
        ))

    add_declaration(
        "std", "implicit_global_namespace", ("std",), ("std",),
        0, 0, -1, -1, 0,
    )

    for region in namespace_regions:
        if not region.name_indices:
            continue
        parent_length = len(region.path) - len(region.name_indices)
        for component, name_index in enumerate(region.name_indices, 1):
            name_token = tokens[name_index]
            identity = region.path[:parent_length + component]
            parent_scope = parent_scope_for(name_token.start)
            add_declaration(
                name_token.text, "namespace_declaration", identity, identity,
                tokens[region.declaration].start, name_token.end,
                tokens[region.opening].start, tokens[region.closing].end,
                parent_scope,
            )

    region_by_declaration = {
        region.declaration: region for region in all_class_regions
    }

    def class_access_at(scope_id, declaration_index):
        scope = scopes[scope_id]
        if not scope.canonical_class:
            return "public"
        candidates = [
            region for region in all_class_regions
            if class_identities[region] == scope.canonical_class
            and region.opening < declaration_index < region.closing
        ]
        if not candidates:
            return "public"
        region = min(
            candidates, key=lambda item: item.closing - item.opening)
        access = region.default_access
        cursor = region.opening + 1
        while cursor < declaration_index:
            if cursor in attribute_indices:
                cursor += 1
                continue
            if tokens[cursor].text == "{":
                closing = brace_pairs.get(cursor)
                if closing is not None and closing < declaration_index:
                    cursor = closing + 1
                    continue
            if (tokens[cursor].text in ("public", "protected", "private")
                    and cursor + 1 < declaration_index
                    and tokens[cursor + 1].text == ":"):
                access = tokens[cursor].text
                cursor += 2
                continue
            cursor += 1
        return access

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index in attribute_indices:
            index += 1
            continue
        if token.text not in ("class", "struct", "union"):
            index += 1
            continue
        shape = class_key_shapes[index]
        if shape["status"] not in (
                "definition-class-head", "standalone-forward-declaration"):
            index += 1
            continue
        name_indices = shape["names"]
        after_name = shape["after_name"]
        if not name_indices:
            index += 1
            continue
        name_index = name_indices[-1]
        region = region_by_declaration.get(index)
        if (region is None and after_name < len(tokens)
                and tokens[after_name].text in (">", ",", "=")):
            index = after_name + 1
            continue
        definition = region is not None
        parent_scope = parent_scope_for(token.start)
        parent = scopes[parent_scope]
        qualified_identity = (
            qualified_identities.get(index)
            if len(name_indices) > 1 else None
        )
        if qualified_identity is not None:
            identity = tuple(qualified_identity)
            owner_identity = identity[:-1]
            owner_scopes = [
                scope for scope in scopes
                if (tuple(scope.canonical_class) == owner_identity
                    or tuple(scope.canonical_namespace) == owner_identity)
                and scope.opening <= token.start
            ]
            if not owner_scopes:
                index = after_name
                continue
            local = False
        elif len(name_indices) > 1:
            index = after_name
            continue
        elif region is not None:
            identity = class_identities[region]
            local = region in local_regions
        elif parent.kind in ("function", "ordinary_block", "lambda", "local_class"):
            identity = (
                logical_scope_identity(parent_scope)
                + (tokens[name_index].text,)
            )
            local = True
        elif parent.canonical_class:
            identity = parent.canonical_class + (tokens[name_index].text,)
            local = False
        elif parent.canonical_namespace:
            identity = parent.canonical_namespace + (tokens[name_index].text,)
            local = False
        else:
            identity = (tokens[name_index].text,)
            local = False
        if definition:
            kind = "local_class" if local else token.text + "_definition"
            body_open = tokens[region.opening].start
            body_close = tokens[region.closing].end
        else:
            kind = ("local_class_forward" if local
                    else token.text + "_forward")
            body_open = body_close = -1
        add_declaration(
            tokens[name_index].text, kind, identity, identity,
            token.start, tokens[name_index].end, body_open, body_close,
            parent_scope,
            access=class_access_at(parent_scope, index),
            complete_eligible=definition and not local,
        )
        index = (region.opening + 1 if definition else after_name)

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index in attribute_indices:
            index += 1
            continue
        if token.text == "namespace" and index + 3 < len(tokens):
            if (tokens[index + 1].text
                    and _is_identifier_start(tokens[index + 1].text[0])
                    and tokens[index + 2].text == "="):
                end = index + 3
                while end < len(tokens) and tokens[end].text != ";":
                    end += 1
                parent_scope = parent_scope_for(token.start)
                target = tuple(part.text for part in tokens[index + 3:end])
                add_declaration(
                    tokens[index + 1].text, "namespace_alias", (), (),
                    token.start, tokens[index + 1].end, -1, -1,
                    parent_scope, target, tokens[index + 3].start,
                )
                index = end + 1
                continue
        if token.text == "using" and index + 1 < len(tokens):
            end = index + 1
            while end < len(tokens) and tokens[end].text != ";":
                end += 1
            if end >= len(tokens):
                break
            parent_scope = parent_scope_for(token.start)
            values = tokens[index + 1:end]
            ordinary_values = _without_attribute_groups(values)
            access = class_access_at(parent_scope, index)
            if (len(ordinary_values) >= 2 and ordinary_values[0].text
                    and _is_identifier_start(ordinary_values[0].text[0])
                    and ordinary_values[1].text == "="):
                target = tuple(part.text for part in ordinary_values[2:])
                add_declaration(
                    ordinary_values[0].text, "using_alias", (), (),
                    token.start, ordinary_values[0].end, -1, -1,
                    parent_scope, target,
                    (ordinary_values[2].start if len(ordinary_values) > 2
                     else ordinary_values[0].end),
                    access=access,
                    dependent=any(part in ("typename", "template", "<")
                                  for part in target),
                    complete_eligible=(
                        scopes[parent_scope].kind == "nonlocal_class"),
                )
            elif (values
                  and values[0].text not in ("enum", "namespace")):
                for raw_declarator in _top_level_comma_groups(values):
                    declarator = _without_attribute_groups(raw_declarator)
                    if not declarator:
                        continue
                    introduced = next((
                        part for part in reversed(declarator)
                        if part.text and _is_identifier_start(part.text[0])
                        and part.text not in ("enum", "typename")
                    ), None)
                    if introduced is None:
                        continue
                    target = tuple(part.text for part in declarator)
                    add_declaration(
                        introduced.text, "using_declaration", (), (),
                        token.start, introduced.end, -1, -1, parent_scope,
                        target, declarator[0].start,
                        access=access,
                        dependent="typename" in target,
                        complete_eligible=(
                            scopes[parent_scope].kind == "nonlocal_class"),
                    )
            index = end + 1
            continue
        if token.text == "typedef":
            end = index + 1
            while end < len(tokens) and tokens[end].text != ";":
                end += 1
            parent_scope = parent_scope_for(token.start)
            access = class_access_at(parent_scope, index)
            for name_token, target_tokens in _typedef_declarators(
                    tokens[index + 1:end]):
                target = tuple(part.text for part in target_tokens)
                add_declaration(
                    name_token.text, "typedef_alias", (), (), token.start,
                    name_token.end, -1, -1, parent_scope, target,
                    tokens[index + 1].start if index + 1 < end else token.end,
                    access=access,
                    dependent=any(part in ("typename", "template", "<")
                                  for part in target),
                    complete_eligible=(
                        scopes[parent_scope].kind == "nonlocal_class"),
                )
            index = end + 1
            continue
        index += 1

    declarations_by_scope_name = {}
    class_declaration_points = {}
    for declaration in declarations:
        declarations_by_scope_name.setdefault(
            (declaration.parent_scope, declaration.name), []).append(
                declaration.declaration_id)
        if declaration.kind.endswith(("_definition", "_forward")) or (
                declaration.kind in ("local_class", "local_class_forward")):
            class_declaration_points.setdefault(
                declaration.canonical_identity, []).append(
                    declaration.point_of_declaration)
    declarations_by_scope_name = {
        key: tuple(sorted(values, key=lambda declaration_id:
                          declarations[declaration_id].point_of_declaration))
        for key, values in declarations_by_scope_name.items()
    }
    class_declaration_points = {
        identity: tuple(sorted(points))
        for identity, points in class_declaration_points.items()
    }

    scope_declarations = {scope.scope_id: [] for scope in scopes}
    for declaration in declarations:
        scope_declarations[declaration.parent_scope].append(
            declaration.declaration_id)
    scopes = tuple(LexicalScope(
        scope_id=scope.scope_id,
        kind=scope.kind,
        parent_scope=scope.parent_scope,
        opening=scope.opening,
        closing=scope.closing,
        canonical_namespace=scope.canonical_namespace,
        canonical_class=scope.canonical_class,
        callable_owner=scope.callable_owner,
        declaration_ids=tuple(scope_declarations[scope.scope_id]),
    ) for scope in scopes)
    namespace_scopes = {}
    for scope in scopes:
        if scope.canonical_namespace:
            namespace_scopes.setdefault(
                scope.canonical_namespace, []).append(scope.scope_id)
    namespace_scopes = {
        identity: tuple(scope_ids)
        for identity, scope_ids in namespace_scopes.items()
    }
    preliminary_graph = _DeclarationLookupIndex(
        source_size=len(text),
        scopes=scopes,
        declarations=tuple(declarations),
        root_scope=0,
        namespace_regions=namespace_regions,
        class_regions=all_class_regions,
        nonlocal_class_regions=tuple(nonlocal_regions),
        local_class_regions=tuple(local_regions),
        class_identities=class_identities,
        class_declaration_points=class_declaration_points,
        declarations_by_scope_name=declarations_by_scope_name,
        namespace_scopes=namespace_scopes,
    )
    structural_environment = _structural_lookup_environment(
        preliminary_graph, tokens)
    _populate_structural_base_edges(structural_environment)
    control_data_declarations, control_statements = \
        _build_control_declarations(
            tokens, scopes, parent_scope_for, paren_pairs, brace_pairs,
            bracket_pairs, attribute_pairs, preliminary_graph,
            structural_environment)
    data_name_declarations = _build_graph_data_declarations(
        tokens, scopes, parent_scope_for, control_data_declarations,
        control_statements, preliminary_graph, structural_environment)
    preliminary_graph.data_name_declarations = data_name_declarations
    preliminary_graph.control_statements = control_statements
    signature_owner_ranges = _member_signature_owner_ranges(
        preliminary_graph, tokens, structural_environment)

    class_key_classifications = []
    for token_index, shape in sorted(class_key_shapes.items()):
        token = tokens[token_index]
        names = shape["names"]
        terminal = tokens[names[-1]] if names else None
        spelling = tuple(tokens[item].text for item in names)
        status = shape["status"]
        reason = shape["reason"]
        parent_scope = parent_scope_for(token.start)
        canonical_identity = ()
        point = -1
        body_range = ()
        terminator_range = ()
        region = shape["region"]
        if status == "definition-class-head" and region is not None:
            canonical_identity = tuple(class_identities[region])
            point = terminal.end
            body_range = (
                tokens[region.opening].start, tokens[region.closing].end)
        elif status == "standalone-forward-declaration" and terminal is not None:
            declaration = next((
                item for item in declarations
                if item.declaration_start == token.start
                and item.name == terminal.text
                and item.kind.endswith("_forward")
            ), None)
            if declaration is not None:
                canonical_identity = tuple(declaration.canonical_identity)
                point = declaration.point_of_declaration
            cursor = shape["after_name"]
            while cursor < len(tokens) and tokens[cursor].text == "[[":
                closing = attribute_pairs.get(cursor)
                if closing is None:
                    break
                cursor = closing + 1
            if cursor < len(tokens) and tokens[cursor].text == ";":
                terminator_range = (tokens[cursor].start, tokens[cursor].end)
        elif status == "elaborated-type-use" and terminal is not None:
            use_offset = tokens[names[0]].start
            template_id_names = frozenset(shape["template_id_names"])
            owner_class = next((
                tuple(scope.canonical_class)
                for scope in _graph_scope_chain(
                    preliminary_graph, parent_scope)
                if scope.canonical_class
            ), ())
            signature_owner = next((
                identity
                for (start, end, identity, _, callable_scope,
                     body_open) in signature_owner_ranges
                if start <= token.start < end
                and (token.start < body_open
                     or preliminary_graph.scopes[
                         parent_scope].callable_owner == callable_scope)
            ), ())
            if signature_owner:
                owner_class = tuple(signature_owner)
            result = _resolve_qualified_name(
                tuple(QualifiedComponent(
                    name=tokens[name].text,
                    token_range=(tokens[name].start, tokens[name].end),
                    has_template_id=name in template_id_names,
                ) for name in names),
                LookupContext(
                    starting_scope=parent_scope,
                    use_offset=use_offset,
                    purpose="elaborated-type-use",
                    complete_class=bool(owner_class),
                    owner_class=owner_class,
                    absolute=shape["absolute"],
                ),
                structural_environment,
                _record=False,
            )
            provenance_is_class = any(
                isinstance(declaration_id, int)
                and 0 <= declaration_id < len(declarations)
                and declarations[declaration_id].kind
                in _CLASS_TYPE_DECLARATION_KINDS
                for declaration_id in result.provenance
            )
            if (result.status == "found"
                    and (result.selected_declaration_kind
                         in _CLASS_TYPE_DECLARATION_KINDS
                         or provenance_is_class)):
                canonical_identity = tuple(result.canonical_identity)
            if not canonical_identity:
                status = "unsupported"
                reason = (
                    "unresolved elaborated type is outside the bounded graph")
        qualified_components = (
            canonical_identity
            if status in ("definition-class-head",
                          "standalone-forward-declaration")
            else spelling if status == "elaborated-type-use" else ()
        )
        class_key_classifications.append(ClassKeyClassification(
            token_index=token_index,
            key=token.text,
            status=status,
            name_range=(terminal.start, terminal.end) if terminal else (),
            qualified_components=qualified_components,
            canonical_identity=canonical_identity,
            point_of_declaration=point,
            parent_scope=parent_scope,
            terminator_range=terminator_range,
            body_range=body_range,
            reason=reason,
        ))

    return DeclarationGraph(
        source_size=preliminary_graph.source_size,
        scopes=preliminary_graph.scopes,
        declarations=preliminary_graph.declarations,
        root_scope=preliminary_graph.root_scope,
        namespace_regions=preliminary_graph.namespace_regions,
        class_regions=preliminary_graph.class_regions,
        nonlocal_class_regions=preliminary_graph.nonlocal_class_regions,
        local_class_regions=preliminary_graph.local_class_regions,
        class_identities=preliminary_graph.class_identities,
        class_declaration_points=preliminary_graph.class_declaration_points,
        declarations_by_scope_name=preliminary_graph.declarations_by_scope_name,
        namespace_scopes=preliminary_graph.namespace_scopes,
        class_key_classifications=tuple(class_key_classifications),
        data_name_declarations=preliminary_graph.data_name_declarations,
        control_statements=preliminary_graph.control_statements,
    )


def _qualified_class_join_events(tokens, brace_pairs):
    """Return qualified class definitions/redeclarations needing an owner."""
    paren_pairs = _pair_map(tokens, "(", ")")
    attribute_pairs = _pair_map(tokens, "[[", "]]")
    regions = tuple(_class_regions(
        tokens, brace_pairs, attribute_pairs, paren_pairs))
    shapes = _class_key_shape_map(
        tokens, regions, attribute_pairs, paren_pairs)
    return tuple(
        (
            token_index,
            tuple(QualifiedComponent(
                name=tokens[name_index].text,
                token_range=(tokens[name_index].start,
                             tokens[name_index].end),
                has_template_id=(
                    name_index in shape["template_id_names"]),
            ) for name_index in shape["names"]),
            shape["absolute"],
        )
        for token_index, shape in sorted(shapes.items())
        if len(shape["names"]) > 1
        and shape["status"] in {
            "definition-class-head", "standalone-forward-declaration"}
    )


def _structural_join_lookup_index(declarations, source_size, tokens):
    """Adapt lightweight structural facts to the one central resolver."""
    namespace_identities = {
        tuple(item.canonical_identity) for item in declarations
        if item.kind == "namespace_declaration"
    }
    class_identities = {
        tuple(item.canonical_identity) for item in declarations
        if item.kind.endswith(("_definition", "_forward"))
    }
    semantic_identities = {()}
    for identity in namespace_identities | class_identities:
        semantic_identities.update(
            identity[:length] for length in range(1, len(identity) + 1))
    ordered_identities = sorted(
        semantic_identities - {()}, key=lambda item: (len(item), item))
    scope_id_by_identity = {(): 0}
    scopes = [LexicalScope(
        0, "global", -1, 0, source_size + 1,
    )]
    for identity in ordered_identities:
        parent = scope_id_by_identity[identity[:-1]]
        is_class = identity in class_identities
        scope_id = len(scopes)
        scope_id_by_identity[identity] = scope_id
        scopes.append(LexicalScope(
            scope_id=scope_id,
            kind="nonlocal_class" if is_class else "namespace_fragment",
            parent_scope=parent,
            opening=0,
            closing=source_size + 1,
            canonical_namespace=() if is_class else identity,
            canonical_class=identity if is_class else (),
        ))

    typed_declarations = []
    for item in declarations:
        parent_scope = scope_id_by_identity.get(item.parent_identity)
        if parent_scope is None:
            continue
        typed_declarations.append(TypeDeclaration(
            declaration_id=len(typed_declarations),
            name=item.name,
            kind=item.kind,
            canonical_identity=item.canonical_identity,
            canonical_target=item.canonical_identity,
            declaration_start=item.declaration_start,
            point_of_declaration=item.point_of_declaration,
            body_open=-1,
            body_close=-1,
            parent_scope=parent_scope,
            lifetime_end=source_size + 1,
            target_tokens=item.target_tokens,
            rhs_use_position=item.rhs_use_position,
            access=item.access,
            dependent=item.dependent,
            complete_class_eligible=bool(item.parent_identity
                                         in class_identities),
        ))
    declarations_by_scope_name = {}
    for declaration in typed_declarations:
        declarations_by_scope_name.setdefault(
            (declaration.parent_scope, declaration.name), []).append(
                declaration.declaration_id)
    declarations_by_scope_name = {
        key: tuple(values)
        for key, values in declarations_by_scope_name.items()
    }
    scope_declarations = {scope.scope_id: [] for scope in scopes}
    for declaration in typed_declarations:
        scope_declarations[declaration.parent_scope].append(
            declaration.declaration_id)
    scopes = tuple(replace(
        scope, declaration_ids=tuple(scope_declarations[scope.scope_id]))
        for scope in scopes)
    namespace_scopes = {
        identity: (scope_id_by_identity[identity],)
        for identity in namespace_identities
    }
    class_points = {
        identity: tuple(sorted(
            declaration.point_of_declaration
            for declaration in typed_declarations
            if tuple(declaration.canonical_identity) == identity
            and declaration.kind.endswith(("_definition", "_forward"))))
        for identity in class_identities
    }
    region_keys = tuple(class_identities)
    graph = _DeclarationLookupIndex(
        source_size=source_size,
        scopes=scopes,
        declarations=tuple(typed_declarations),
        root_scope=0,
        namespace_regions=(),
        class_regions=region_keys,
        nonlocal_class_regions=region_keys,
        local_class_regions=(),
        class_identities={identity: identity for identity in region_keys},
        class_declaration_points=class_points,
        declarations_by_scope_name=declarations_by_scope_name,
        namespace_scopes=namespace_scopes,
    )
    return graph, _structural_lookup_environment(graph, tokens), \
        scope_id_by_identity


def _select_qualified_class_identities(tokens, brace_pairs):
    """Resolve qualified class joins in a lightweight declaration phase."""
    paren_pairs = _pair_map(tokens, "(", ")")
    attribute_pairs = _pair_map(tokens, "[[", "]]")
    namespace_regions = tuple(_namespace_regions(
        tokens, brace_pairs, attribute_pairs))
    class_regions = tuple(_class_regions(
        tokens, brace_pairs, attribute_pairs, paren_pairs))
    shapes = _class_key_shape_map(
        tokens, class_regions, attribute_pairs, paren_pairs)
    class_openings = {region.opening for region in class_regions}
    namespace_openings = {region.opening for region in namespace_regions}
    nonlocal_regions = tuple(
        region for region in class_regions
        if not any(
            opening < region.opening < brace_pairs[opening]
            and opening not in class_openings
            and opening not in namespace_openings
            for opening in brace_pairs
            if tokens[opening].text == "{"
        )
    )
    region_by_declaration = {
        region.declaration: region for region in nonlocal_regions
    }

    def namespace_for_index(index):
        candidates = [
            region for region in namespace_regions
            if region.opening < index < region.closing
        ]
        return (
            min(candidates, key=lambda region:
                region.closing - region.opening).path
            if candidates else ()
        )

    def containing_region(index):
        candidates = [
            region for region in nonlocal_regions
            if region.opening < index < region.closing
        ]
        return (
            min(candidates, key=lambda region:
                region.closing - region.opening)
            if candidates else None
        )

    def inside_ordinary_block(index):
        return any(
            opening < index < brace_pairs[opening]
            and opening not in class_openings
            and opening not in namespace_openings
            for opening in brace_pairs
            if tokens[opening].text == "{"
        )

    declarations = []

    def add_declaration(
            name, kind, identity, parent, point, declaration_start=-1,
            target_tokens=(), rhs_use_position=-1, dependent=False):
        declarations.append(_StructuralJoinDeclaration(
            len(declarations), name, kind, tuple(identity), tuple(parent),
            point, declaration_start, tuple(target_tokens),
            rhs_use_position, dependent=dependent,
        ))

    for region in namespace_regions:
        introduced = len(region.name_indices)
        parent_length = len(region.path) - introduced
        for component, name_index in enumerate(region.name_indices, 1):
            identity = tuple(region.path[:parent_length + component])
            add_declaration(
                tokens[name_index].text, "namespace_declaration",
                identity, identity[:-1], tokens[name_index].end,
                tokens[region.declaration].start,
            )

    alias_records = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (token.text == "namespace" and index + 3 < len(tokens)
                and tokens[index + 1].text
                and _is_identifier_start(tokens[index + 1].text[0])
                and tokens[index + 2].text == "="):
            end = index + 3
            while end < len(tokens) and tokens[end].text != ";":
                end += 1
            alias_records.append((
                index, tokens[index + 1], "namespace_alias",
                tuple(part.text for part in tokens[index + 3:end]),
                tokens[index + 3].start,
                False,
            ))
            index = end + 1
            continue
        if token.text == "using" and index + 1 < len(tokens):
            end = index + 1
            while end < len(tokens) and tokens[end].text != ";":
                end += 1
            values = _without_attribute_groups(tokens[index + 1:end])
            if (len(values) >= 2 and values[0].text
                    and _is_identifier_start(values[0].text[0])
                    and values[1].text == "="):
                target = tuple(part.text for part in values[2:])
                alias_records.append((
                    index, values[0], "using_alias", target,
                    values[2].start if len(values) > 2 else values[0].end,
                    any(part in ("typename", "template", "<")
                        for part in target),
                ))
            index = end + 1
            continue
        if token.text == "typedef":
            end = index + 1
            while end < len(tokens) and tokens[end].text != ";":
                end += 1
            for name_token, target_tokens in _typedef_declarators(
                    tokens[index + 1:end]):
                target = tuple(part.text for part in target_tokens)
                alias_records.append((
                    index, name_token, "typedef_alias", target,
                    tokens[index + 1].start if index + 1 < end else token.end,
                    any(part in ("typename", "template", "<")
                        for part in target),
                ))
            index = end + 1
            continue
        index += 1

    class_identities = {}
    qualified_identities = {}
    added_regions = set()
    added_forwards = set()
    added_aliases = set()
    join_results = {}
    while True:
        changed = False
        for region in nonlocal_regions:
            if region in added_regions:
                continue
            shape = shapes[region.declaration]
            names = shape["names"]
            if len(names) != 1:
                continue
            owner = containing_region(region.opening)
            if owner is not None:
                parent = class_identities.get(owner)
                if parent is None:
                    continue
            else:
                parent = tuple(namespace_for_index(region.declaration))
            identity = tuple(parent) + (tokens[names[-1]].text,)
            class_identities[region] = identity
            add_declaration(
                tokens[names[-1]].text,
                tokens[region.declaration].text + "_definition",
                identity, identity[:-1], tokens[names[-1]].end,
                tokens[region.declaration].start,
            )
            added_regions.add(region)
            changed = True

        for token_index, shape in sorted(shapes.items()):
            if (token_index in added_forwards
                    or shape["status"] != "standalone-forward-declaration"
                    or len(shape["names"]) != 1
                    or inside_ordinary_block(token_index)):
                continue
            owner = containing_region(token_index)
            if owner is not None:
                parent = class_identities.get(owner)
                if parent is None:
                    continue
            else:
                parent = tuple(namespace_for_index(token_index))
            terminal = tokens[shape["names"][-1]]
            identity = tuple(parent) + (terminal.text,)
            add_declaration(
                terminal.text, tokens[token_index].text + "_forward",
                identity, identity[:-1], terminal.end,
                tokens[token_index].start,
            )
            added_forwards.add(token_index)
            changed = True

        for (token_index, name_token, kind, target_tokens,
             rhs_use_position, dependent) in alias_records:
            if token_index in added_aliases or inside_ordinary_block(token_index):
                continue
            owner = containing_region(token_index)
            if owner is not None:
                parent = class_identities.get(owner)
                if parent is None:
                    continue
            else:
                parent = tuple(namespace_for_index(token_index))
            add_declaration(
                name_token.text, kind, (), parent, name_token.end,
                tokens[token_index].start, target_tokens,
                rhs_use_position, dependent,
            )
            added_aliases.add(token_index)
            changed = True

        join_graph, join_environment, join_scopes = \
            _structural_join_lookup_index(
                declarations, tokens[-1].end if tokens else 0, tokens)
        for token_index, components, absolute in _qualified_class_join_events(
                tokens, brace_pairs):
            if token_index in qualified_identities:
                continue
            owner = containing_region(token_index)
            if owner is not None:
                starting_identity = class_identities.get(owner)
                if starting_identity is None:
                    continue
            else:
                starting_identity = tuple(namespace_for_index(token_index))
            result = _resolve_qualified_name(
                components,
                LookupContext(
                    starting_scope=join_scopes[tuple(starting_identity)],
                    use_offset=components[0].token_range[0],
                    purpose="qualified-class-join",
                    absolute=absolute,
                ),
                join_environment,
                _record=False,
            )
            terminal_id = (
                result.provenance[-1] if result.provenance else -1
            )
            terminal = (
                join_graph.declarations[terminal_id]
                if isinstance(terminal_id, int)
                and 0 <= terminal_id < len(join_graph.declarations)
                else None
            )
            if (result.status != "found" or terminal is None
                    or not terminal.kind.endswith(
                        ("_definition", "_forward"))):
                join_results[token_index] = result
                continue
            identity = tuple(result.canonical_identity)
            qualified_identities[token_index] = identity
            join_results[token_index] = result
            shape = shapes[token_index]
            region = region_by_declaration.get(token_index)
            if region is not None:
                class_identities[region] = identity
                added_regions.add(region)
                kind = tokens[token_index].text + "_definition"
            else:
                added_forwards.add(token_index)
                kind = tokens[token_index].text + "_forward"
            terminal_token = tokens[shape["names"][-1]]
            add_declaration(
                terminal_token.text, kind, identity, identity[:-1],
                terminal_token.end, tokens[token_index].start,
            )
            changed = True
        if not changed:
            break
    return qualified_identities, join_results


def _qualified_class_join_results(graph, tokens, events, record):
    """Resolve class joins through central lookup and exact declaration ids."""
    environment = _structural_lookup_environment(graph, tokens)
    identities = {}
    results = {}
    for token_index, components, absolute in events:
        use_offset = components[0].token_range[0]
        starting_scope = _graph_scope_for_offset(graph, use_offset)
        owner_class = next((
            tuple(scope.canonical_class)
            for scope in _graph_scope_chain(graph, starting_scope)
            if scope.canonical_class
        ), ())
        result = _resolve_qualified_name(
            components,
            LookupContext(
                starting_scope=starting_scope,
                use_offset=use_offset,
                purpose="qualified-class-join",
                complete_class=bool(owner_class),
                owner_class=owner_class,
                absolute=absolute,
            ),
            environment,
            _record=record,
        )
        results[token_index] = result
        identity = tuple(result.canonical_identity)
        terminal_selected = any(
            isinstance(declaration_id, int)
            and 0 <= declaration_id < len(graph.declarations)
            and graph.declarations[declaration_id].canonical_identity
            == identity
            and (
                graph.declarations[declaration_id].kind.endswith(
                    ("_definition", "_forward"))
                or graph.declarations[declaration_id].kind
                in {"local_class", "local_class_forward"}
            )
            for declaration_id in result.provenance
        )
        if result.status == "found" and identity and terminal_selected:
            identities[token_index] = identity
    return identities, results


def _build_declaration_graph(text, tokens, brace_pairs):
    """Build one graph through declaration-id-selected structural phases."""
    events = _qualified_class_join_events(tokens, brace_pairs)
    qualified_identities, _ = _select_qualified_class_identities(
        tokens, brace_pairs)
    graph = _build_declaration_graph_pass(
        text, tokens, brace_pairs, qualified_identities)
    if events:
        _qualified_class_join_results(
            graph, tokens, events, record=True)
    return graph


def _graph_scope_chain(graph, scope_id):
    """Return lexical scopes from one exact use outward to global scope."""
    output = []
    active = set()
    while scope_id >= 0 and scope_id not in active:
        active.add(scope_id)
        scope = graph.scopes[scope_id]
        output.append(scope)
        scope_id = scope.parent_scope
    return tuple(output)


def _declaration_visible_in_context(declaration, scope, context):
    if not (declaration.point_of_declaration <= context.use_offset
            < declaration.lifetime_end):
        persistent_member = (
            scope.kind == "nonlocal_class"
            and declaration.point_of_declaration <= context.use_offset
        )
        complete_scope = (
            context.complete_class
            and bool(scope.canonical_class)
            and bool(context.owner_class)
            and (
                tuple(scope.canonical_class) == tuple(context.owner_class)
                or tuple(context.owner_class[:len(scope.canonical_class)])
                    == tuple(scope.canonical_class)
            )
            and declaration.complete_class_eligible
        )
        if not (persistent_member or complete_scope):
            return False
    return True


def _lookup_result_from_declarations(graph, declaration_ids, provenance):
    declarations = tuple(graph.declarations[item] for item in declaration_ids)
    if not declarations:
        return LookupResult("not-found")
    first = declarations[0]
    identities = tuple(item.canonical_identity for item in declarations)
    namespace_kinds = {
        "implicit_global_namespace", "namespace_declaration"
    }
    coalesced_namespace = (
        all(item.kind in namespace_kinds for item in declarations)
        and all(identity == identities[0] for identity in identities)
    )
    coalesced_type = (
        bool(identities[0])
        and all(identity == identities[0] for identity in identities)
        and all(item.kind in _CLASS_TYPE_DECLARATION_KINDS
                for item in declarations)
    )
    if (len(declarations) > 1
            and not (coalesced_namespace or coalesced_type)):
        return LookupResult(
            "ambiguous", reason="multiple visible declarations",
            provenance=tuple(provenance),
            candidate_declaration_ids=tuple(declaration_ids),
        )
    chosen = declarations[-1]
    return LookupResult(
        "found",
        declaration_id=chosen.declaration_id,
        canonical_identity=chosen.canonical_identity,
        declaration_kind=chosen.kind,
        provenance=tuple(provenance),
    )


def _lookup_named_type(graph, name, context):
    """Perform the central point/lifetime-aware bounded named-type lookup."""
    cache_key = name, context
    if cache_key in graph.lookup_cache:
        return graph.lookup_cache[cache_key]
    if context.absolute:
        scope_chain = (graph.scopes[graph.root_scope],)
    else:
        starting_scope = context.starting_scope
        if starting_scope < 0:
            starting_scope = _graph_scope_for_offset(
                graph, context.use_offset)
        scope_chain = list(_graph_scope_chain(graph, starting_scope))
        owner_scopes = [
            scope for scope in graph.scopes
            if context.complete_class
            and context.purpose != "canonical-class-identity"
            and scope.kind == "nonlocal_class"
            and scope.canonical_class == tuple(context.owner_class)
        ]
        if (owner_scopes
                and not any(scope.scope_id == owner_scopes[0].scope_id
                            for scope in scope_chain)):
            local_prefix = []
            for scope in scope_chain:
                if scope.kind not in {
                        "function", "lambda", "ordinary_block"}:
                    break
                local_prefix.append(scope)
            scope_chain = local_prefix + list(_graph_scope_chain(
                graph, owner_scopes[0].scope_id))
        scope_chain = tuple(scope_chain)

    for scope in scope_chain:
        scope_ids = (scope.scope_id,)
        if scope.kind == "namespace_fragment" and scope.canonical_namespace:
            scope_ids = graph.namespace_scopes.get(
                scope.canonical_namespace, scope_ids)
        visible = []
        for declaration_scope in scope_ids:
            for declaration_id in graph.declarations_by_scope_name.get(
                    (declaration_scope, name), ()):
                declaration = graph.declarations[declaration_id]
                if _declaration_visible_in_context(
                        declaration, graph.scopes[declaration_scope], context):
                    visible.append(declaration_id)
        if visible:
            visible.sort(key=lambda declaration_id:
                         graph.declarations[declaration_id]
                         .point_of_declaration)
            result = _lookup_result_from_declarations(
                graph, visible,
                ("scope", scope.scope_id, "purpose", context.purpose),
            )
            graph.lookup_cache[cache_key] = result
            return result
        if context.purpose != "canonical-class-identity":
            canonical_visible = []
            for declaration_scope in scope_ids:
                for declaration_id in graph.declarations_by_scope_name.get(
                        (declaration_scope, name), ()):
                    declaration = graph.declarations[declaration_id]
                    if (declaration.kind not in _CLASS_TYPE_DECLARATION_KINDS
                            or not declaration.canonical_identity):
                        continue
                    if _class_identity_visible(
                            graph, declaration.canonical_identity,
                            context.use_offset):
                        canonical_visible.append(declaration_id)
            if canonical_visible:
                canonical_visible.sort(key=lambda declaration_id:
                                       graph.declarations[declaration_id]
                                       .point_of_declaration)
                result = _lookup_result_from_declarations(
                    graph, canonical_visible,
                    ("scope", scope.scope_id,
                     "purpose", context.purpose),
                )
                graph.lookup_cache[cache_key] = result
                return result
    result = LookupResult("not-found")
    graph.lookup_cache[cache_key] = result
    return result


def _qualified_component_records(parts, use_offset, source_tokens=()):
    """Attach observed coordinates and template-id shape to a spelling."""
    parts = tuple(parts)
    if not source_tokens:
        return tuple(QualifiedComponent(name=part) for part in parts)
    first = next((
        index for index, token in enumerate(source_tokens)
        if token.start >= use_offset and token.text == parts[0]
    ), None)
    if first is None:
        return tuple(QualifiedComponent(name=part) for part in parts)
    output = []
    cursor = first
    for part_index, part in enumerate(parts):
        if cursor >= len(source_tokens) or source_tokens[cursor].text != part:
            return tuple(QualifiedComponent(name=item) for item in parts)
        token = source_tokens[cursor]
        cursor += 1
        has_template_id = (
            cursor < len(source_tokens) and source_tokens[cursor].text == "<")
        output.append(QualifiedComponent(
            name=part, token_range=(token.start, token.end),
            has_template_id=has_template_id))
        if has_template_id:
            depth = 0
            while cursor < len(source_tokens):
                value = source_tokens[cursor].text
                if value == "<":
                    depth += 1
                elif value == ">":
                    depth -= 1
                elif value == ">>" and depth >= 2:
                    depth -= 2
                cursor += 1
                if depth == 0:
                    break
            if depth:
                return tuple(
                    QualifiedComponent(name=item) for item in parts)
        if part_index + 1 < len(parts):
            if (cursor >= len(source_tokens)
                    or source_tokens[cursor].text != "::"):
                return tuple(
                    QualifiedComponent(name=item) for item in parts)
            cursor += 1
            if (cursor < len(source_tokens)
                    and source_tokens[cursor].text == "template"):
                cursor += 1
    return tuple(output)


def _qualified_parent_scope_ids(graph, identity):
    """Return only scopes owned by one already-resolved qualification."""
    identity = tuple(identity)
    if not identity:
        return (graph.root_scope,)
    class_scopes = tuple(
        scope.scope_id for scope in graph.scopes
        if scope.kind in ("nonlocal_class", "local_class")
        and scope.canonical_class == identity
    )
    if class_scopes:
        return class_scopes
    return tuple(graph.namespace_scopes.get(identity, ()))


def _qualified_member_lookup(graph, identity, name, context):
    """Select one member without restarting unqualified lookup."""
    visible = []
    scope_ids = _qualified_parent_scope_ids(graph, identity)
    for scope_id in scope_ids:
        scope = graph.scopes[scope_id]
        for declaration_id in graph.declarations_by_scope_name.get(
                (scope_id, name), ()):
            declaration = graph.declarations[declaration_id]
            if _declaration_visible_in_context(declaration, scope, context):
                visible.append(declaration_id)
    visible = sorted(set(visible), key=lambda declaration_id:
                     graph.declarations[declaration_id]
                     .point_of_declaration)
    return _lookup_result_from_declarations(
        graph, visible,
        ("qualified-parent", tuple(identity), "purpose", context.purpose),
    )


def _inherited_named_type_lookup(name, context, environment):
    """Select inherited type facts by declaration id before outer scopes."""
    owner = tuple(context.owner_class)
    graph = environment.declaration_graph
    if not owner or graph is None:
        return LookupResult("not-found")
    candidates = []
    unsupported = False
    depth_exhausted = False

    def visit(identity, path, active):
        nonlocal unsupported, depth_exhausted
        if identity in active:
            unsupported = True
            return
        scope_ids = _qualified_parent_scope_ids(graph, identity)
        direct = []
        for scope_id in scope_ids:
            scope = graph.scopes[scope_id]
            for declaration_id in graph.declarations_by_scope_name.get(
                    (scope_id, name), ()):
                declaration = graph.declarations[declaration_id]
                if _declaration_visible_in_context(
                        declaration, scope, context):
                    direct.append(declaration_id)
        if direct:
            candidates.extend((item, tuple(path)) for item in direct)
            return
        edges = environment.base_edges.get(identity, ())
        if len(path) >= _GRAPH_TRAVERSAL_DEPTH_LIMIT:
            if edges:
                unsupported = True
                depth_exhausted = True
            return
        for edge in edges:
            nested_path = tuple(path) + (edge,)
            if edge.virtual:
                unsupported = True
            visit(tuple(edge.base_class), nested_path, active + (identity,))

    visit(owner, (), ())
    if depth_exhausted:
        return LookupResult(
            "unsupported", reason=_INHERITANCE_DEPTH_REASON,
            candidate_declaration_ids=tuple(
                item[0] for item in candidates),
        )
    if not candidates:
        return LookupResult(
            "unsupported" if unsupported else "not-found",
            reason=("inherited type path is unsupported"
                    if unsupported else ""),
        )
    declaration_ids = tuple(item[0] for item in candidates)
    paths = tuple(item[1] for item in candidates)
    if (unsupported or len(set(declaration_ids)) != 1
            or len(candidates) != 1
            or not _path_accessible_from_owner(paths[0])):
        return LookupResult(
            "ambiguous" if len(candidates) > 1 else "unsupported",
            reason="inherited type is not uniquely accessible",
            candidate_declaration_ids=declaration_ids,
        )
    declaration = graph.declarations[declaration_ids[0]]
    return LookupResult(
        "found", declaration_id=declaration.declaration_id,
        canonical_identity=declaration.canonical_identity,
        declaration_kind=declaration.kind,
        provenance=("inherited-declaration",)
        + tuple(edge.base_class for edge in paths[0]),
    )


def _lookup_injected_base_name(name, aliases):
    """Select one injected base-class declaration through exact ancestry."""
    environment = aliases.environment
    graph = environment.declaration_graph
    if not aliases.complete_class_scopes or graph is None:
        return LookupResult("not-found")
    owner = tuple(aliases.complete_class_scopes[0])
    candidates = set()
    visited = set()
    unsupported = False

    def visit(identity, depth, active):
        nonlocal unsupported
        if identity in active:
            unsupported = True
            return
        if identity in visited:
            return
        visited.add(identity)
        edges = environment.base_edges.get(identity, ())
        if depth >= _GRAPH_TRAVERSAL_DEPTH_LIMIT:
            unsupported = unsupported or bool(edges)
            return
        for edge in edges:
            if edge.base_class and edge.base_class[-1] == name:
                candidates.add(tuple(edge.base_class))
            visit(tuple(edge.base_class), depth + 1,
                  active + (identity,))

    visit(owner, 0, ())
    if unsupported:
        return LookupResult(
            "unsupported", reason=_INHERITANCE_DEPTH_REASON)
    if not candidates:
        return LookupResult("not-found")
    if len(candidates) != 1:
        return LookupResult(
            "ambiguous", reason="multiple injected base identities")
    candidate = next(iter(candidates))
    relation, _ = _owner_or_unique_ancestor_relation(
        owner, candidate, environment.base_edges)
    if relation != "ancestor":
        return LookupResult(
            "ambiguous" if relation == "ambiguous" else "unsupported",
            canonical_identity=candidate,
            reason="injected base path is not uniquely accessible",
        )
    declaration_id = next((
        declaration.declaration_id
        for declaration in reversed(graph.declarations)
        if declaration.canonical_identity == candidate
        and declaration.kind.endswith("_definition")
    ), -1)
    if declaration_id < 0:
        return LookupResult(
            "unsupported", canonical_identity=candidate,
            reason="injected base declaration is unavailable",
        )
    declaration = graph.declarations[declaration_id]
    return LookupResult(
        "found", declaration_id=declaration_id,
        canonical_identity=candidate,
        declaration_kind=declaration.kind,
        provenance=(declaration_id,),
    )


def _qualified_declaration_accessible(declaration, context, environment):
    """Apply bounded member access after declaration selection."""
    graph = environment.declaration_graph if environment is not None else None
    if graph is None:
        return False
    if context.purpose in {
            "qualified-class-join", "member-definition-owner"}:
        # An out-of-class member definition is associated with its declared
        # owner; private/protected member-type access is valid in that context.
        return True
    if declaration.kind in {"using_alias", "typedef_alias"}:
        targets = tuple(
            target
            for alias_targets in environment.type_aliases.values()
            for target in alias_targets
            if target.declaration_id == declaration.declaration_id
        )
        if targets:
            owner = tuple(context.owner_class)
            access_scopes = environment.complete_class_scopes.get(
                owner, (owner,) if owner else ())
            return _type_alias_targets_accessible(
                targets, access_scopes, environment.classes_by_identity,
                environment.base_edges)
    if declaration.access == "public":
        return True
    parent = graph.scopes[declaration.parent_scope]
    declaring_class = tuple(parent.canonical_class)
    owner = tuple(context.owner_class)
    if declaring_class and declaring_class == owner:
        return True
    if (declaring_class and len(owner) > len(declaring_class)
            and owner[:len(declaring_class)] == declaring_class):
        return True
    if declaration.access != "protected" or not owner or environment is None:
        return False
    relation, _ = _owner_or_unique_ancestor_relation(
        owner, declaring_class, environment.base_edges)
    return relation in ("owner", "ancestor")


def _selected_alias_target(
        declaration, environment, active=(), alias_depth=0):
    """Follow only one selected alias declaration at its exact RHS point."""
    if declaration.declaration_id in active:
        return None, "alias cycle"
    graph = environment.declaration_graph
    path_alias_depth = sum(
        isinstance(declaration_id, int)
        and 0 <= declaration_id < len(graph.declarations)
        and graph.declarations[declaration_id].kind
        in {"using_alias", "typedef_alias", "namespace_alias"}
        for declaration_id in active
    )
    alias_depth = max(alias_depth, path_alias_depth)
    if alias_depth >= _GRAPH_TRAVERSAL_DEPTH_LIMIT:
        return None, _SELECTED_ALIAS_DEPTH_REASON
    if declaration.dependent:
        return None, "dependent alias target is unsupported"
    parsed = _simple_qualified_type(declaration.target_tokens)
    if parsed is None:
        return None, "alias target is outside the bounded grammar"
    target_parts, target_absolute = parsed
    target_scope = graph.scopes[declaration.parent_scope]
    owner = tuple(target_scope.canonical_class)
    target_context = LookupContext(
        starting_scope=declaration.parent_scope,
        use_offset=declaration.rhs_use_position,
        purpose="alias-rhs",
        complete_class=bool(owner),
        owner_class=owner,
        absolute=target_absolute,
    )
    target_components = _qualified_component_records(
        target_parts,
        declaration.rhs_use_position + (2 if target_absolute else 0),
        environment.source_tokens)
    result = _resolve_qualified_name(
        target_components, target_context, environment,
        _active=active + (declaration.declaration_id,),
        _alias_depth=alias_depth + 1,
        _record=False,
    )
    if declaration.kind in {"using_alias", "typedef_alias"}:
        projection_target = TypeAliasTarget(
            scope=(tuple(target_scope.canonical_class)
                   or tuple(target_scope.canonical_namespace)),
            rhs=tuple(declaration.target_tokens),
            declaration=declaration.rhs_use_position,
            access=declaration.access,
            declaration_id=declaration.declaration_id,
            declaration_graph=result,
        )
        projection_key, _ = _qualified_type_alias_key(
            (), False, (), {}, {}, declaration.rhs_use_position,
            frozenset(), _SELECTED_TYPE_ALIAS_PROJECTION,
        )
        if (projection_key is None or _resolve_type_alias_class(
                _SELECTED_TYPE_ALIAS_PROJECTION, projection_target,
                {}, {}, frozenset(), {}, {}, {},
        ) is None):
            return None, "selected type-alias projection is unavailable"
    return result, ""


def _selected_type_alias_identity(declaration, environment, active=()):
    """Canonicalize one selected type-alias RHS without aggregate lookup."""
    if declaration.declaration_id in active:
        return None
    targets = tuple(
        target
        for alias_targets in environment.type_aliases.values()
        for target in alias_targets
        if target.declaration_id == declaration.declaration_id
    )
    if len(targets) != 1:
        return None
    target = targets[0]
    context = AliasContext(
        environment, target.scope, target.declaration)
    expanded = tuple(_expand_aliases(
        target.rhs, context,
        active + (declaration.declaration_id,)))
    return _canonical_expanded_type_identity(
        expanded, context, parameter=False)


def _resolve_qualified_name(
        components, context, environment, *, _active=(), _alias_depth=0,
        _record=True):
    """Resolve a qualified-id once from one frozen first declaration id."""
    graph = environment.declaration_graph
    normalized = tuple(
        item if isinstance(item, QualifiedComponent)
        else QualifiedComponent(str(item))
        for item in components
    )
    if not normalized:
        return QualifiedLookupResult(
            "unsupported", -2, use_offset=context.use_offset,
            starting_scope_id=context.starting_scope,
            purpose=context.purpose, absolute=context.absolute,
            reason="qualified-id has no component",
        )
    cache_key = normalized, context
    if not _active and cache_key in graph.qualified_lookup_cache:
        cached = graph.qualified_lookup_cache[cache_key]
        if _record and cached not in graph.qualified_lookup_results:
            graph.qualified_lookup_results.append(cached)
        return cached

    first_component = normalized[0]
    remainder = normalized[1:]
    selection_origin = ""
    first = _lookup_named_type(
        graph, first_component.name, context)
    if context.owner_class and not context.absolute:
        selected_ids = (
            (first.declaration_id,) if first.declaration_id >= 0
            else tuple(first.candidate_declaration_ids)
        )
        selected_scopes = tuple(
            graph.scopes[graph.declarations[item].parent_scope]
            for item in selected_ids
        )
        nearer_than_inheritance = bool(selected_scopes) and all(
            scope.kind in {"function", "lambda", "ordinary_block"}
            or tuple(scope.canonical_class) == tuple(context.owner_class)
            for scope in selected_scopes
        )
        if not nearer_than_inheritance:
            owner = tuple(context.owner_class)
            injected = _lookup_injected_base_name(
                first_component.name,
                AliasContext(
                    environment, (), context.use_offset,
                    environment.complete_class_scopes.get(
                        owner, (owner,))),
            )
            if injected.status != "not-found":
                first = injected
                selection_origin = "injected-base"
            else:
                inherited = _inherited_named_type_lookup(
                    first_component.name, context, environment)
                if inherited.status != "not-found":
                    first = inherited

    def finish(status, selected_id, selected_kind="", canonical=(),
               alias_target_id=-1, alias_target=(), access="public",
               provenance=(), reason=""):
        result = QualifiedLookupResult(
            status=status,
            selected_declaration_id=selected_id,
            selected_declaration_kind=selected_kind,
            canonical_identity=tuple(canonical),
            alias_target_declaration_id=alias_target_id,
            canonical_alias_target=tuple(alias_target),
            remaining_components=remainder,
            use_offset=context.use_offset,
            starting_scope_id=context.starting_scope,
            access=access,
            purpose=context.purpose,
            absolute=context.absolute,
            provenance=tuple(provenance),
            reason=reason,
            selection_origin=selection_origin,
        )
        if not _active:
            graph.qualified_lookup_cache[cache_key] = result
            if _record:
                graph.qualified_lookup_results.append(result)
        return result

    alias_kinds = {"using_alias", "typedef_alias", "namespace_alias"}
    if first.status == "ambiguous" and first.candidate_declaration_ids:
        candidates = tuple(
            graph.declarations[declaration_id]
            for declaration_id in first.candidate_declaration_ids
        )
        parent_scopes = tuple(
            graph.scopes[item.parent_scope] for item in candidates)
        semantic_parents = tuple(
            ("namespace",) + tuple(scope.canonical_namespace)
            if scope.kind == "namespace_fragment"
            else ("class",) + tuple(scope.canonical_class)
            if scope.kind in {"nonlocal_class", "local_class"}
            else (scope.kind, scope.scope_id)
            for scope in parent_scopes
        )
        type_alias_family = all(
            item.kind in {"using_alias", "typedef_alias"}
            for item in candidates)
        namespace_alias_family = all(
            item.kind == "namespace_alias" for item in candidates)
        same_parent = all(
            parent == semantic_parents[0] for parent in semantic_parents)
        if type_alias_family and same_parent:
            resolved_identities = tuple(
                _selected_type_alias_identity(item, environment, _active)
                for item in candidates
            )
            aliases_equivalent = (
                all(item is not None for item in resolved_identities)
                and all(item == resolved_identities[0]
                        for item in resolved_identities)
            )
        elif namespace_alias_family and same_parent:
            resolved_aliases = tuple(
                _selected_alias_target(
                    item, environment, _active, _alias_depth)[0]
                for item in candidates
            )
            aliases_equivalent = (
                all(item is not None and item.status == "found"
                    for item in resolved_aliases)
                and all(tuple(item.canonical_identity)
                        == tuple(resolved_aliases[0].canonical_identity)
                        for item in resolved_aliases)
            )
        else:
            aliases_equivalent = False
        if aliases_equivalent:
            chosen = candidates[-1]
            first = LookupResult(
                "found", declaration_id=chosen.declaration_id,
                declaration_kind=chosen.kind,
                provenance=("equivalent-alias-declarations",)
                + tuple(item.declaration_id for item in candidates),
            )

    if first.status == "not-found":
        return finish("not-found", -1)
    if first.status != "found":
        return finish(
            first.status, -2, reason=(first.reason
                                      or "first component is ambiguous"))

    selected = graph.declarations[first.declaration_id]
    selected_id = selected.declaration_id
    selected_kind = selected.kind
    selected_access = selected.access
    provenance = [selected_id]
    if not _qualified_declaration_accessible(
            selected, context, environment):
        return finish(
            "unsupported", selected_id, selected_kind,
            access=selected_access, provenance=provenance,
            reason="selected declaration is inaccessible",
        )
    if selected.kind == "using_declaration":
        return finish(
            "unsupported", selected_id, selected_kind,
            access=selected_access, provenance=provenance,
            reason="found unsupported declaration; outward fallback forbidden",
        )

    current = selected
    current_identity = tuple(current.canonical_identity)
    alias_target_id = -1
    canonical_alias_target = ()

    if current.kind in alias_kinds:
        target, alias_reason = _selected_alias_target(
            current, environment, _active, _alias_depth)
        if target is None or target.status != "found":
            return finish(
                ("unsupported"
                 if target is None or target.status == "not-found"
                 else target.status),
                selected_id, selected_kind,
                access=selected_access,
                provenance=(provenance + ([] if target is None
                                          else list(target.provenance))),
                reason=(alias_reason or target.reason
                        or "selected alias target is unresolved"),
            )
        provenance.extend(target.provenance)
        current_identity = tuple(target.canonical_identity)
        alias_target_id = target.selected_declaration_id
        canonical_alias_target = current_identity

    terminal_access = selected_access
    for component in remainder:
        member_context = LookupContext(
            starting_scope=context.starting_scope,
            use_offset=context.use_offset,
            purpose=context.purpose,
            complete_class=context.complete_class,
            owner_class=context.owner_class,
            absolute=True,
        )
        member = _qualified_member_lookup(
            graph, current_identity, component.name, member_context)
        if member.status != "found":
            return finish(
                "unsupported" if member.status == "not-found"
                else member.status,
                selected_id, selected_kind,
                canonical=current_identity,
                alias_target_id=alias_target_id,
                alias_target=canonical_alias_target,
                access=terminal_access,
                provenance=provenance,
                reason=(member.reason
                        or "qualified component cannot be selected"),
            )
        current = graph.declarations[member.declaration_id]
        terminal_access = current.access
        provenance.append(current.declaration_id)
        if not _qualified_declaration_accessible(
                current, context, environment):
            return finish(
                "unsupported", selected_id, selected_kind,
                alias_target_id=alias_target_id,
                alias_target=canonical_alias_target,
                access=terminal_access,
                provenance=provenance,
                reason="cannot bind uniquely and accessibly",
            )
        if current.kind == "using_declaration":
            return finish(
                "unsupported", selected_id, selected_kind,
                alias_target_id=alias_target_id,
                alias_target=canonical_alias_target,
                access=terminal_access,
                provenance=provenance,
                reason="found unsupported declaration; no component fallback",
            )
        if current.kind in alias_kinds:
            target, alias_reason = _selected_alias_target(
                current, environment, _active + tuple(provenance[:-1]),
                _alias_depth)
            if target is None or target.status != "found":
                return finish(
                    ("unsupported"
                     if target is None or target.status == "not-found"
                     else target.status),
                    selected_id, selected_kind,
                    alias_target_id=alias_target_id,
                    alias_target=canonical_alias_target,
                    access=terminal_access,
                    provenance=(provenance + ([] if target is None
                                              else list(target.provenance))),
                    reason=(alias_reason or target.reason
                            or "qualified alias target is unresolved"),
                )
            provenance.extend(target.provenance)
            current_identity = tuple(target.canonical_identity)
        else:
            current_identity = tuple(current.canonical_identity)

    if not current_identity:
        return finish(
            "unsupported", selected_id, selected_kind,
            alias_target_id=alias_target_id,
            alias_target=canonical_alias_target,
            access=terminal_access, provenance=provenance,
            reason="selected declaration has no canonical identity",
        )
    return finish(
        "found", selected_id, selected_kind,
        canonical=current_identity,
        alias_target_id=alias_target_id,
        alias_target=canonical_alias_target,
        access=terminal_access, provenance=provenance,
    )


def _class_identity_visible(graph, identity, use_position,
                            complete_class_scopes=()):
    """Resolve one canonical class identity through the shared lookup."""
    identity = tuple(identity)
    if not identity:
        return False
    parent_identity = identity[:-1]
    parent_scopes = [
        scope.scope_id for scope in graph.scopes
        if scope.kind == "nonlocal_class"
        and scope.canonical_class == parent_identity
    ]
    if not parent_scopes:
        parent_scopes = list(graph.namespace_scopes.get(
            parent_identity, ()))
    if not parent_scopes and not parent_identity:
        parent_scopes = [graph.root_scope]
    owner = (tuple(complete_class_scopes[0])
             if complete_class_scopes else ())
    for parent_scope in parent_scopes:
        result = _lookup_named_type(
            graph,
            identity[-1],
            LookupContext(
                starting_scope=parent_scope,
                use_offset=use_position,
                purpose="canonical-class-identity",
                complete_class=bool(complete_class_scopes),
                owner_class=owner,
            ),
        )
        if (result.status == "found"
                and result.canonical_identity == identity):
            return True
    return False


def _visible_alias_targets(targets, use_position, complete_class_scopes=()):
    """Return graph-backed alias declarations visible in one context."""
    output = []
    for target in targets:
        graph = target.declaration_graph
        if graph is None or target.declaration_id < 0:
            if (target.declaration < use_position
                    or target.scope in complete_class_scopes):
                output.append(target)
            continue
        declaration = graph.declarations[target.declaration_id]
        scope = graph.scopes[declaration.parent_scope]
        context = LookupContext(
            starting_scope=declaration.parent_scope,
            use_offset=use_position,
            purpose="alias",
            complete_class=bool(complete_class_scopes),
            owner_class=(tuple(complete_class_scopes[0])
                         if complete_class_scopes else ()),
        )
        if _declaration_visible_in_context(declaration, scope, context):
            output.append(target)
    return tuple(output)


def _namespace_alias_prefix(parts, scope, aliases, use_position,
                            maximum=None, absolute=False):
    if maximum is None:
        maximum = len(parts)
    for prefix_length in range(min(maximum, len(parts)), 0, -1):
        prefix = tuple(parts[:prefix_length])
        scope_lengths = (0,) if absolute else range(len(scope), -1, -1)
        for scope_length in scope_lengths:
            key = tuple(scope[:scope_length]) + prefix
            if _visible_alias_targets(
                    aliases.get(key, ()), use_position):
                return key, prefix_length
    return None


def _resolve_namespace_alias(key, use_position, aliases, known_namespaces,
                             cache, active=()):
    cache_key = key, use_position
    if cache_key in cache:
        return cache[cache_key]
    if cache_key in active:
        return None
    targets = _visible_alias_targets(aliases.get(key, ()), use_position)
    if not targets:
        cache[cache_key] = None
        return None

    resolutions = []
    for target in targets:
        match = _namespace_alias_prefix(
            target.parts, target.scope, aliases, target.declaration,
            absolute=target.absolute)
        if match is not None:
            nested_key, consumed = match
            base = _resolve_namespace_alias(
                nested_key, target.declaration, aliases, known_namespaces,
                cache, active + (cache_key,))
            candidate = (None if base is None
                         else base + target.parts[consumed:])
            resolution = (candidate if candidate in known_namespaces
                          else None)
        else:
            candidates = ((target.parts,) if target.absolute else (
                tuple(target.scope[:length]) + target.parts
                for length in range(len(target.scope), -1, -1)))
            resolution = next((candidate for candidate in candidates
                               if candidate in known_namespaces), None)
        resolutions.append(resolution)
    resolved = (resolutions[0] if resolutions and resolutions[0] is not None
                and all(item == resolutions[0] for item in resolutions)
                else None)
    cache[cache_key] = resolved
    return resolved


def _resolve_aliased_class(parts, absolute, scope, namespace_aliases,
                           use_position, known_namespaces, alias_cache,
                           classes_by_identity, declaration_graph=None,
                           complete_class_scopes=()):
    def visible(identity):
        return (declaration_graph is None
                or _class_identity_visible(
                    declaration_graph, identity, use_position,
                    complete_class_scopes))

    match = _namespace_alias_prefix(
        parts, scope, namespace_aliases, use_position,
        maximum=max(0, len(parts) - 1), absolute=absolute)
    if match is not None:
        key, consumed = match
        namespace = _resolve_namespace_alias(
            key, use_position, namespace_aliases, known_namespaces,
            alias_cache)
        if namespace is None:
            return None
        identity = namespace + tuple(parts[consumed:])
        return classes_by_identity.get(identity) if visible(identity) else None
    if absolute:
        identity = tuple(parts)
        return classes_by_identity.get(identity) if visible(identity) else None
    return _resolve_class(
        parts, scope, classes_by_identity, use_position,
        declaration_graph, complete_class_scopes)


def _strip_attributes(values):
    result = []
    depth = 0
    for value in values:
        if value == "[[":
            depth += 1
        elif value == "]]" and depth:
            depth -= 1
        elif depth == 0:
            result.append(value)
    return result


def _stable_class_alias_spelling(expanded, canonical):
    """Replace a resolved class base with its stable absolute identity.

    An alias RHS is interpreted at its declaration point.  Retaining only its
    lexical spelling would let a nearer class or namespace alias reinterpret
    that spelling when the enclosing function signature is parsed later.
    Preserve all declarator modifiers, but anchor a same-header class base to
    the absolute identity already established at the declaration.
    """
    if (not canonical or canonical[0] != "type"
            or not canonical[1]
            or canonical[1][0] != "same-header-class"):
        return expanded
    identity = canonical[1][1:]
    if not identity or not all(
            part and _is_identifier_start(part[0]) for part in identity):
        return expanded
    ordinary = _canonical_ordinary_type(expanded, parameter=False)
    if not ordinary or ordinary[0] != "type":
        return expanded
    base = ordinary[1]
    if not base:
        return expanded
    for start in range(len(expanded) - len(base) + 1):
        if tuple(expanded[start:start + len(base)]) != base:
            continue
        absolute = ["::"]
        for index, part in enumerate(identity):
            if index:
                absolute.append("::")
            absolute.append(part)
        return expanded[:start] + tuple(absolute) + expanded[start + len(base):]
    return expanded


def _alias_replacement(parts, absolute, aliases, active):
    environment = aliases.environment
    graph = environment.declaration_graph
    if graph is not None:
        result = _resolve_qualified_name(
            _qualified_component_records(
                parts, aliases.use_position + (2 if absolute else 0),
                environment.source_tokens),
            LookupContext(
                starting_scope=_graph_scope_for_offset(
                    graph, aliases.use_position),
                use_offset=aliases.use_position,
                purpose="signature-type",
                complete_class=bool(aliases.complete_class_scopes),
                owner_class=(tuple(aliases.complete_class_scopes[0])
                             if aliases.complete_class_scopes else ()),
                absolute=absolute,
            ),
            environment,
        )
        selected_id = next((
            declaration_id for declaration_id in reversed(result.provenance)
            if isinstance(declaration_id, int)
            and declaration_id >= 0
            and graph.declarations[declaration_id].kind
            in {"using_alias", "typedef_alias"}
        ), -1)
        if selected_id < 0 and result.selected_declaration_id >= 0:
            candidate = graph.declarations[result.selected_declaration_id]
            if candidate.kind in {"using_alias", "typedef_alias"}:
                selected_id = candidate.declaration_id
        if selected_id < 0:
            return None
        selected = graph.declarations[selected_id]
        targets = tuple(
            target
            for alias_targets in environment.type_aliases.values()
            for target in alias_targets
            if target.declaration_id == selected_id
        )
        access_scopes = (
            aliases.complete_class_scopes
            or environment.complete_class_scopes.get(
                aliases.current_scope, ()))
        if (len(targets) != 1 or not _type_alias_targets_accessible(
                targets, access_scopes, environment.classes_by_identity,
                environment.base_edges)):
            return None
        target = targets[0]
        identity = ("selected-alias", selected_id)
        if identity in active:
            return None
        target_context = AliasContext(
            environment, target.scope, target.declaration)
        expanded = tuple(_expand_aliases(
            target.rhs, target_context, active + (identity,)))
        canonical = _canonical_expanded_type_identity(
            expanded, target_context, parameter=False)
        return _stable_class_alias_spelling(expanded, canonical)

    key, targets = _qualified_type_alias_key(
        parts, absolute, aliases.current_scope,
        environment.type_aliases, environment.namespace_aliases,
        aliases.use_position, environment.known_namespaces,
        environment.namespace_alias_cache, aliases.complete_class_scopes)
    access_scopes = (aliases.complete_class_scopes
                     or environment.complete_class_scopes.get(
                         aliases.current_scope, ()))
    if (key is None or not _type_alias_targets_accessible(
            targets, access_scopes, environment.classes_by_identity,
            environment.base_edges)):
        return None
    resolutions = []
    for target in targets:
        identity = key, target.declaration
        if identity in active:
            return None
        target_context = AliasContext(
            aliases.environment, target.scope, target.declaration)
        expanded = tuple(_expand_aliases(
            target.rhs, target_context, active + (identity,)))
        canonical = _canonical_expanded_type_identity(
            expanded, target_context, parameter=False)
        resolutions.append((
            canonical,
            _stable_class_alias_spelling(expanded, canonical),
        ))
    if (not resolutions
            or any(item[0] != resolutions[0][0] for item in resolutions)):
        return None
    return resolutions[-1][1]


def _expand_aliases(values, aliases, active=()):
    def append_replacement(output, replacement):
        # In ``const Alias`` the qualifier applies to the alias as a whole,
        # not to the first token of its replacement. Move an immediately
        # preceding cv sequence behind the replacement so ``const Pointer``
        # expands like ``Pointer const`` (for Pointer = int*, int* const).
        prefix_cv = []
        while output and output[-1] in ("const", "volatile"):
            prefix_cv.insert(0, output.pop())
        output.extend(replacement)
        output.extend(prefix_cv)

    expanded = list(values)
    for _ in range(16):
        changed = False
        output = []
        index = 0
        while index < len(expanded):
            absolute = expanded[index] == "::"
            first = index + 1 if absolute else index
            if (first < len(expanded)
                    and _is_identifier_start(expanded[first][0])):
                parts = [expanded[first]]
                end = first + 1
                while (end + 1 < len(expanded) and expanded[end] == "::"
                       and _is_identifier_start(expanded[end + 1][0])):
                    parts.append(expanded[end + 1])
                    end += 2
                if len(parts) > 1 or absolute:
                    replacement = _alias_replacement(
                        parts, absolute, aliases, active)
                    if replacement is not None:
                        append_replacement(output, replacement)
                        changed = True
                    else:
                        output.extend(expanded[index:end])
                    index = end
                    continue
                replacement = _alias_replacement(
                    parts, False, aliases, active)
                if replacement is not None:
                    append_replacement(output, replacement)
                    changed = True
                else:
                    output.append(parts[0])
                index += 1
                continue
            output.append(expanded[index])
            index += 1
        expanded = output
        if not changed:
            break
    return expanded


def _is_const_lvalue_reference(values, aliases):
    values = [value for value in _strip_attributes(values)
              if value not in _RETURN_SPECIFIERS]
    values = _expand_aliases(values, aliases)
    return "&" in values and "&&" not in values and "const" in values


def _strip_leading_template_headers(values):
    """Remove function-template headers that are not part of a return type."""
    values = list(values)
    while len(values) >= 2 and values[:2] == ["template", "<"]:
        depth = 0
        cursor = 1
        while cursor < len(values):
            depth = _angle_depth_change(values[cursor], depth)
            cursor += 1
            if depth == 0:
                break
        if depth:
            break
        values = values[cursor:]
    return values


def _reverse_template_component(signature, closing):
    """Return ``(identifier-index, identifier)`` for one template-id."""
    depth = 0
    cursor = closing
    while cursor >= 0:
        value = signature[cursor].text
        if value == ">":
            depth += 1
        elif value == ">>":
            depth += 2
        elif value == "<":
            depth -= 1
            if depth == 0:
                identifier = cursor - 1
                if (identifier >= 0
                        and _is_identifier_start(
                            signature[identifier].text[0])):
                    return identifier, signature[identifier].text
                return None
            if depth < 0:
                return None
        cursor -= 1
    return None


def _qualified_owner_components(signature, name_start):
    """Parse simple/template-id components before a qualified function name.

    Template arguments remain deliberately opaque. Their presence is returned
    so callers can fail closed without treating a class-template definition as
    an ordinary class association.
    """
    separator = name_start - 1
    if separator < 0 or signature[separator].text != "::":
        return (), False, False, name_start

    reversed_parts = []
    has_template_id = False
    cursor = separator - 1
    leftmost = name_start
    absolute = False
    while cursor >= 0:
        value = signature[cursor].text
        if value and _is_identifier_start(value[0]):
            component_start = cursor
            component_name = value
        elif value in (">", ">>"):
            component = _reverse_template_component(signature, cursor)
            if component is None:
                break
            component_start, component_name = component
            has_template_id = True
        else:
            break

        reversed_parts.append(component_name)
        leftmost = component_start
        separator = component_start - 1
        if separator < 0 or signature[separator].text != "::":
            break
        previous = separator - 1
        if (previous >= 0
                and (signature[previous].text in (">", ">>")
                     or _is_identifier_start(signature[previous].text[0]))):
            cursor = previous
            continue
        absolute = True
        leftmost = separator
        break

    if not reversed_parts:
        return (), False, False, name_start
    declarator_start = leftmost
    if (declarator_start > 0
            and signature[declarator_start - 1].text == "("):
        declarator_start -= 1
    return (tuple(reversed(reversed_parts)), absolute, has_template_id,
            declarator_start)


def _qualified_owner_start(signature, name_start):
    """Return the first token of the owner preceding a qualified function."""
    return _qualified_owner_components(signature, name_start)[3]


def _angle_depth_change(value, depth):
    if value == "<":
        return depth + 1
    if value == ">" and depth:
        return depth - 1
    if value == ">>" and depth >= 2:
        return depth - 2
    return depth


def _array_suffix(values):
    """Split ordinary declarator array suffixes from their element type."""
    parens = angles = 0
    opening = None
    for index, value in enumerate(values):
        if value == "(" and angles == 0:
            parens += 1
        elif value == ")" and angles == 0:
            parens -= 1
        elif parens == 0:
            angles = _angle_depth_change(value, angles)
        if value == "[" and parens == angles == 0:
            opening = index
            break
    if opening is None:
        return list(values), ()

    groups = []
    cursor = opening
    while cursor < len(values):
        if values[cursor] != "[":
            return None
        closing = cursor + 1
        depth = 1
        while closing < len(values) and depth:
            depth += (values[closing] == "[") - (values[closing] == "]")
            closing += 1
        if depth:
            return None
        groups.append(tuple(values[cursor + 1:closing - 1]))
        cursor = closing
    return list(values[:opening]), tuple(groups)


def _canonical_cv(values):
    """C++ cv qualification is an idempotent set at each type level."""
    return tuple(qualifier for qualifier in ("const", "volatile")
                 if qualifier in values)


def _canonical_ordinary_type(values, parameter):
    """Build a bounded structural identity for ordinary C++ declarators.

    Modifiers are stored from the base type outwards. This makes the C++
    parameter adjustments explicit: an outer array becomes a pointer, and cv
    is removed only from the outermost non-reference parameter type. Pointee,
    referred-to and inner-pointer cv therefore remain load-bearing.
    """
    split = _array_suffix(values)
    if split is None:
        return ("raw", tuple(values))
    core, arrays = split

    base = []
    base_cv = []
    modifiers = []
    index = 0
    angles = parens = 0
    while index < len(core):
        value = core[index]
        if value == "(" and angles == 0:
            parens += 1
        elif value == ")" and angles == 0:
            parens -= 1
        elif parens == 0:
            angles = _angle_depth_change(value, angles)
        if parens < 0:
            return ("raw", tuple(values))
        if parens == angles == 0 and value in ("*", "&", "&&"):
            break
        if parens == angles == 0 and value in ("const", "volatile"):
            base_cv.append(value)
        else:
            base.append(value)
        index += 1

    if parens or angles or not base:
        return ("raw", tuple(values))

    while index < len(core):
        value = core[index]
        if value in ("&", "&&"):
            if index + 1 != len(core):
                return ("raw", tuple(values))
            modifiers.append(("lref" if value == "&" else "rref",))
            index += 1
            continue
        if value != "*":
            return ("raw", tuple(values))
        index += 1
        pointer_cv = []
        while index < len(core) and core[index] in ("const", "volatile"):
            pointer_cv.append(core[index])
            index += 1
        modifiers.append(("pointer", _canonical_cv(pointer_cv)))

    canonical_base_cv = _canonical_cv(base_cv)

    if arrays:
        # Tokens spell dimensions outermost first. Declarator modifiers are
        # represented base-outwards, so retain inner extents in reverse order.
        for extent in reversed(arrays[1:] if parameter else arrays):
            modifiers.append(("array", extent))
        if parameter:
            modifiers.append(("pointer", ()))

    if parameter:
        if modifiers and modifiers[-1][0] == "pointer":
            modifiers[-1] = ("pointer", ())
        elif not modifiers:
            canonical_base_cv = ()

    return ("type", tuple(base), canonical_base_cv, tuple(modifiers))


def _canonical_expanded_type_identity(values, aliases, parameter):
    """Canonicalize an expanded type, including same-header class names."""
    canonical = _canonical_ordinary_type(values, parameter)
    if not canonical or canonical[0] != "type":
        return canonical
    parsed = _simple_qualified_type(canonical[1])
    if parsed is None:
        return canonical
    parts, absolute = parsed
    environment = aliases.environment
    graph = environment.declaration_graph
    starting_scope = _graph_scope_for_offset(graph, aliases.use_position)
    result = _resolve_qualified_name(
        _qualified_component_records(
            parts, aliases.use_position + (2 if absolute else 0),
            environment.source_tokens),
        LookupContext(
            starting_scope=starting_scope,
            use_offset=aliases.use_position,
            purpose="type-canonicalization",
            complete_class=bool(aliases.complete_class_scopes),
            owner_class=(tuple(aliases.complete_class_scopes[0])
                         if aliases.complete_class_scopes else ()),
            absolute=absolute,
        ),
        environment,
    )
    if result.status != "found":
        return canonical
    identity = tuple(result.canonical_identity)
    if identity not in environment.classes_by_identity:
        return canonical
    return (
        canonical[0], ("same-header-class",) + tuple(identity),
        canonical[2], canonical[3],
    )


def _canonical_type_tokens(values, aliases, parameter=False):
    values = [value for value in _strip_attributes(values)
              if value not in _RETURN_SPECIFIERS]
    expanded = _expand_aliases(values, aliases)
    return _canonical_expanded_type_identity(expanded, aliases, parameter)


def _ordinary_parameter_name_index(values):
    """Find the declarator name in the supported ordinary parameter subset.

    Only a top-level ordinary name is removed. A nested declarator stays in
    the key verbatim, and a sole user-defined type is never mistaken for a
    name.
    """
    parens = brackets = angles = 0
    candidates = []
    for index, value in enumerate(values):
        if value == "(":
            parens += 1
        elif value == ")":
            parens -= 1
        elif value == "[":
            brackets += 1
        elif value == "]":
            brackets -= 1
        elif value == "<":
            angles += 1
        elif value == ">" and angles:
            angles -= 1
        elif value == ">>" and angles >= 2:
            angles -= 2
        elif (parens == brackets == angles == 0 and value
              and _is_identifier_start(value[0])
              and value not in _PARAMETER_NON_NAMES
              and (index == 0 or values[index - 1] != "::")
              and (index + 1 == len(values) or values[index + 1] != "::")):
            candidates.append(index)

    if parens or brackets or angles:
        return None
    if len(candidates) >= 2:
        return candidates[-1]
    if not candidates:
        return -1

    candidate = candidates[0]
    prefix = values[:candidate]
    if candidate and values[candidate - 1] in {
            "class", "enum", "struct", "typename"}:
        return -1
    has_type_evidence = any(
        value in _PARAMETER_TYPE_KEYWORDS
        or value in ("*", "&", "&&", ">", ">>", ")")
        or (value and _is_identifier_start(value[0])
            and value not in {"const", "volatile", "typename"})
        for value in prefix
    )
    return candidate if has_type_evidence else -1


def _canonical_parameter_types(signature, parameter_open, parameter_close,
                               aliases):
    parameters = signature[parameter_open + 1:parameter_close]
    parts = _top_level_parts(parameters, ",")
    if len(parts) == 1 and not parts[0]:
        return ()

    output = []
    for parameter in parts:
        default_parts = _top_level_parts(parameter, "=")
        parameter_tokens = default_parts[0]
        values = _strip_attributes(
            [token.text for token in parameter_tokens])
        if not values:
            return None
        parameter_aliases = _alias_context_at(
            aliases, parameter_tokens[0].start)
        angles = 0
        nested_function_pointer = False
        for index, value in enumerate(values[:-1]):
            angles = _angle_depth_change(value, angles)
            if (value == "(" and angles == 0
                    and values[index + 1] == "*"):
                nested_function_pointer = True
                break
        if nested_function_pointer:
            return None
        name_index = _ordinary_parameter_name_index(values)
        if name_index is None:
            return None
        if name_index >= 0:
            del values[name_index]
        canonical = _canonical_type_tokens(
            values, parameter_aliases, parameter=True)
        if not canonical:
            return None
        output.append(canonical)
    if (len(output) == 1 and output[0]
            == ("type", ("void",), (), ())):
        return ()
    return tuple(output)


def _member_qualifiers(signature, parameter_close):
    member_const = False
    member_volatile = False
    ref_qualifier = ""
    parens = brackets = attributes = 0
    for token in signature[parameter_close + 1:]:
        value = token.text
        if value == "[[":
            attributes += 1
            continue
        if value == "]]" and attributes:
            attributes -= 1
            continue
        if attributes:
            continue
        parens += (value == "(") - (value == ")")
        brackets += (value == "[") - (value == "]")
        if parens or brackets:
            continue
        if value in ("->", "requires"):
            break
        if value == "const":
            member_const = True
        elif value == "volatile":
            member_volatile = True
        elif value in ("&", "&&"):
            ref_qualifier = value
    return member_const, member_volatile, ref_qualifier


def _signature_key(details):
    return (
        details.name,
        details.parameter_types,
        details.member_const,
        details.member_volatile,
        details.ref_qualifier,
        details.return_type,
        details.const_reference,
    )


def _matching_declarations(declarations, owner, details):
    """Return declarations with the exact normalized function identity."""
    expected = _signature_key(details)
    return tuple(
        item for item in declarations[owner].get(details.name, ())
        if _signature_key(item[0]) == expected
    )


def _backward_declarator_name_cursor(
        signature, parameter_open, paren_pairs, attribute_pairs=None):
    """Walk from parameters to one supported name across complete attributes.

    Only immediately adjacent, balanced post-name attribute groups may be
    crossed.  Their token ranges remain available to structural callers; no
    leading or sibling attribute can be consumed by this cursor.
    """
    if attribute_pairs is None:
        attribute_pairs = _pair_map(signature, "[[", "]]")
    previous = parameter_open - 1
    crossed = []
    while previous >= 0 and signature[previous].text == "]]":
        opening = attribute_pairs.get(previous)
        if opening is None or opening >= previous:
            return None
        crossed.append((opening, previous))
        previous = opening - 1
    if previous < 0:
        return None

    name = None
    name_start = previous
    name_end = previous
    declarator_start = previous
    wrapper_open = -1
    wrapper_close = -1
    if _is_identifier_start(signature[previous].text[0]):
        possible = signature[previous].text
        if (possible not in _NON_FUNCTION_NAMES
                and not (possible == "operator"
                         and paren_pairs.get(parameter_open)
                         == parameter_open + 1)):
            name = possible
    elif signature[previous].text == ")":
        left = paren_pairs.get(previous)
        if left is not None:
            inside = [part.text for part in _without_attribute_groups(
                signature[left + 1:previous])]
            if (left > 0 and signature[left - 1].text == "operator"
                    and not inside):
                name = "operator()"
                name_start = left - 1
                name_end = previous
                declarator_start = name_start
            elif (len(inside) == 1
                  and _is_identifier_start(inside[0][0])):
                name = inside[0]
                name_start = next(
                    index for index in range(left + 1, previous)
                    if signature[index].text == name)
                name_end = name_start
                declarator_start = left
                wrapper_open = left
                wrapper_close = previous
                crossed.extend(
                    (opening, closing)
                    for opening, closing in attribute_pairs.items()
                    if opening < closing
                    and name_start < opening < closing < previous
                )
            elif (len(inside) >= 3 and inside[-2] == "::"
                  and _is_identifier_start(inside[-1][0])):
                name = inside[-1]
                name_start = next(
                    index for index in range(previous - 1, left, -1)
                    if signature[index].text == name)
                name_end = name_start
                declarator_start = left
                wrapper_open = left
                wrapper_close = previous
                crossed.extend(
                    (opening, closing)
                    for opening, closing in attribute_pairs.items()
                    if opening < closing
                    and name_start < opening < closing < previous
                )
    elif (previous >= 2
          and signature[previous - 2].text == "operator"
          and signature[previous - 1].text == "["
          and signature[previous].text == "]"):
        name = "operator[]"
        name_start = previous - 2
        name_end = previous
        declarator_start = name_start
    if name is None:
        return None
    return DeclaratorNameCursor(
        name=name,
        name_start=name_start,
        name_end=name_end,
        parameter_open=parameter_open,
        post_name_attributes=tuple(sorted(set(crossed))),
        declarator_start=declarator_start,
        wrapper_open=wrapper_open,
        wrapper_close=wrapper_close,
    )


def _function_declarator_structure(signature):
    """Segment a supported declarator before resolving any type tokens."""
    if not signature:
        return None
    local_pairs = _pair_map(signature, "(", ")")
    candidates = []
    depth = 0
    attribute_depth = 0
    for index, token in enumerate(signature):
        value = token.text
        if value == "[[":
            attribute_depth += 1
            continue
        if value == "]]" and attribute_depth:
            attribute_depth -= 1
            continue
        if attribute_depth:
            continue
        if value == "(" and depth == 0:
            cursor = _backward_declarator_name_cursor(
                signature, index, local_pairs)
            if cursor is not None:
                closing = local_pairs.get(index)
                if closing is not None:
                    candidates.append((
                        cursor.name, cursor.name_start,
                        cursor.declarator_start, index, closing,
                    ))
        depth += (value == "(") - (value == ")")
    if not candidates:
        return None

    selected = candidates[-1]
    nesting = 0
    initializer_colon = None
    for index, token in enumerate(signature):
        value = token.text
        nesting += (value in ("(", "[", "{"))
        nesting -= (value in (")", "]", "}"))
        if value == ":" and nesting == 0:
            initializer_colon = index
            break
    if initializer_colon is not None:
        before_initializer = [candidate for candidate in candidates
                              if candidate[4] < initializer_colon]
        if before_initializer:
            selected = before_initializer[-1]

    (name, name_start, declarator_start,
     parameter_open, parameter_close) = selected
    trailing = -1
    qualifier_depth = 0
    for index in range(parameter_close + 1, len(signature)):
        value = signature[index].text
        qualifier_depth += (value == "(") - (value == ")")
        if value == "->" and qualifier_depth == 0:
            trailing = index
            break
    return FunctionDeclaratorStructure(
        name=name,
        name_start=name_start,
        parameters_open=parameter_open,
        parameters_close=parameter_close,
        trailing_return=trailing,
        declarator_start=declarator_start,
    )


def _function_signature_details(signature, aliases, structure=None,
                                leading_return_aliases=None,
                                trailing_return_aliases=None):
    """Return supported declaration details under token-range contexts."""
    if structure is None:
        structure = _function_declarator_structure(signature)
    if structure is None:
        return None

    name = structure.name
    name_start = structure.name_start
    parameter_open = structure.parameters_open
    parameter_close = structure.parameters_close
    trailing = structure.trailing_return
    if trailing >= 0:
        return_values = [token.text for token in signature[trailing + 1:]]
        if "requires" in return_values:
            return_values = return_values[:return_values.index("requires")]
        return_aliases = trailing_return_aliases or aliases
    else:
        owner_start = _qualified_owner_start(signature, name_start)
        return_end = owner_start
        if structure.declarator_start >= 0:
            return_end = min(return_end, structure.declarator_start)
        return_values = [token.text for token in signature[:return_end]]
        return_aliases = leading_return_aliases or aliases
    return_values = _strip_leading_template_headers(return_values)
    parameter_types = _canonical_parameter_types(
        signature, parameter_open, parameter_close, aliases)
    if parameter_types is None:
        return None
    member_const, member_volatile, ref_qualifier = _member_qualifiers(
        signature, parameter_close)
    return FunctionSignature(
        name=name,
        const_reference=_is_const_lvalue_reference(
            return_values, return_aliases),
        return_type=_canonical_type_tokens(return_values, return_aliases),
        parameter_types=parameter_types,
        member_const=member_const,
        member_volatile=member_volatile,
        ref_qualifier=ref_qualifier,
        name_start=name_start,
        parameters_open=parameter_open,
        parameters_close=parameter_close,
    )


def _potential_const_reference_signature(signature, aliases,
                                         require_qualified=False):
    """Recognize a const-reference function before parameter parsing.

    This intentionally does less than ``_function_signature_details``. It is
    used only to turn an otherwise invisible unsupported definition into a
    fail-closed diagnostic; it never creates a census entry.
    """
    pairs = _pair_map(signature, "(", ")")
    depth = 0
    for index, token in enumerate(signature):
        value = token.text
        if value == "(" and depth == 0 and index:
            cursor = _backward_declarator_name_cursor(
                signature, index, pairs)
            if cursor is None:
                depth += 1
                continue
            name = cursor.name
            name_start = cursor.name_start
            qualified = (name_start > 0
                         and signature[name_start - 1].text == "::")
            if require_qualified and not qualified:
                depth += 1
                continue
            owner_start = _qualified_owner_start(signature, name_start)
            return_end = min(owner_start, cursor.declarator_start)
            return_values = [part.text for part in signature[:return_end]]
            closing = pairs.get(index)
            if closing is not None:
                qualifier_depth = 0
                for cursor in range(closing + 1, len(signature)):
                    current = signature[cursor].text
                    qualifier_depth += (current == "(") - (current == ")")
                    if current == "->" and qualifier_depth == 0:
                        return_values = [
                            part.text for part in signature[cursor + 1:]
                        ]
                        if "requires" in return_values:
                            return_values = return_values[
                                :return_values.index("requires")]
                        break
            if _is_const_lvalue_reference(return_values, aliases):
                return name
        depth += (value == "(") - (value == ")")
    return None


def _function_signature(signature, aliases):
    """Return ``(name, return-is-const-reference)`` or ``None``."""
    details = _function_signature_details(signature, aliases)
    if details is None:
        return None
    return details.name, details.const_reference


def _strip_outer_parentheses(values):
    values = list(values)
    while len(values) >= 2 and values[0] == "(" and values[-1] == ")":
        depth = 0
        closes_at_end = False
        for index, value in enumerate(values):
            depth += (value == "(") - (value == ")")
            if depth == 0:
                closes_at_end = index == len(values) - 1
                break
        if not closes_at_end:
            break
        values = values[1:-1]
    return values


def _member_subobject_suffix(values, cursor):
    """Accept only indexing and field selection after a member root."""
    while cursor < len(values):
        if values[cursor] == "[":
            depth = 1
            cursor += 1
            while cursor < len(values) and depth:
                depth += (values[cursor] == "[") - (values[cursor] == "]")
                cursor += 1
            if depth:
                return False
        elif values[cursor] == ".":
            cursor += 1
            if (cursor >= len(values)
                    or not _is_identifier_start(values[cursor][0])):
                return False
            cursor += 1
        else:
            return False
    return True


def _std_get_argument(values):
    """Return the argument of an exact standard ``get`` wrapper."""
    if values[:4] == ["std", "::", "get", "<"]:
        selector_open = 3
    elif values[:5] == ["::", "std", "::", "get", "<"]:
        selector_open = 4
    else:
        return None

    selector_start = selector_open + 1
    cursor = selector_start
    angle = 1
    parens = brackets = braces = 0
    selector_close = None
    while cursor < len(values):
        value = values[cursor]
        if (value == "," and angle == 1
                and parens == brackets == braces == 0):
            return None
        if value == "<":
            angle += 1
        elif value == ">":
            angle -= 1
        elif value == ">>":
            if angle < 2:
                return None
            angle -= 2
        elif value == "(":
            parens += 1
        elif value == ")":
            parens -= 1
        elif value == "[":
            brackets += 1
        elif value == "]":
            brackets -= 1
        elif value == "{":
            braces += 1
        elif value == "}":
            braces -= 1
        if min(angle, parens, brackets, braces) < 0:
            return None
        if angle == 0:
            selector_close = cursor
            break
        cursor += 1
    if (selector_close is None or selector_close == selector_start
            or parens or brackets or braces):
        return None

    call_open = selector_close + 1
    if call_open >= len(values) or values[call_open] != "(":
        return None
    depth = 1
    call_close = None
    cursor = call_open + 1
    while cursor < len(values):
        depth += (values[cursor] == "(") - (values[cursor] == ")")
        if depth < 0:
            return None
        if depth == 0:
            call_close = cursor
            break
        cursor += 1
    if (call_close != len(values) - 1
            or call_close == call_open + 1):
        return None
    return values[call_open + 1:call_close]


def _expression_qualifier_is_block_type(values, block_type_aliases):
    """Whether a simple qualifier is a graph-backed live block type fact."""
    parsed = _simple_qualified_type(values)
    if parsed is None:
        return False
    parts, absolute = parsed
    return (
        not absolute
        and len(parts) == 1
        and parts[0] in block_type_aliases
    )


def _relative_std_names_global_namespace(aliases):
    """Whether unqualified ``std`` is unshadowed at this use point."""
    environment = aliases.environment
    graph = environment.declaration_graph
    if graph is None:
        return _namespace_alias_prefix(
            ("std",), aliases.current_scope,
            environment.namespace_aliases, aliases.use_position,
            maximum=1) is None
    result = _resolve_expression_qualifier_result(("std",), aliases)
    names_global = (
        result.status == "found"
        and result.canonical_identity == ("std",)
        and result.declaration_kind in {
            "implicit_global_namespace", "namespace_declaration"
        }
    )
    return names_global


def _standard_get_provenance_is_exact(values, aliases):
    """Require every admitted ``std::get`` wrapper to name global std."""
    values = _strip_outer_parentheses(values)
    argument = _std_get_argument(values)
    if argument is None:
        return True
    absolute = values[:5] == ["::", "std", "::", "get", "<"]
    if (not absolute
            and not _relative_std_names_global_namespace(aliases)):
        return False
    return _standard_get_provenance_is_exact(argument, aliases)


def _direct_member_root(values, member_names, shadow_names=()):
    values = _strip_outer_parentheses(values)
    if not values:
        return None

    # std::get<I>(member_) is a direct member subobject, not an arbitrary call.
    get_argument = _std_get_argument(values)
    if get_argument is not None:
        return _direct_member_root(
            get_argument, member_names, shadow_names)

    cursor = 0
    explicit_this = False
    if values[:2] == ["this", "->"]:
        cursor = 2
        explicit_this = True
    if cursor >= len(values) or not _is_identifier_start(values[cursor][0]):
        return None
    root = values[cursor]
    if root not in member_names or (not explicit_this and root in shadow_names):
        return None
    return root if _member_subobject_suffix(values, cursor + 1) else None


def _direct_member_expression(values, member_names, shadow_names=()):
    return _direct_member_root(values, member_names, shadow_names) is not None


def _qualified_base_member_candidate(values, unresolved_base_types):
    """Recognize a qualified-id rooted in one exact unresolved base spelling."""
    values = _strip_outer_parentheses(values)
    get_argument = _std_get_argument(values)
    if get_argument is not None:
        return _qualified_base_member_candidate(
            get_argument, unresolved_base_types)
    for base in unresolved_base_types:
        base = list(base)
        cursor = len(base)
        if (not base or values[:cursor] != base
                or cursor + 1 >= len(values)
                or values[cursor] != "::"
                or not _is_identifier_start(values[cursor + 1][0])):
            continue
        if _member_subobject_suffix(values, cursor + 2):
            return True
    return False


def _direct_member_candidate(values, shadow_names=(),
                             unresolved_base_types=()):
    """Whether an expression has supported member syntax for some root.

    This does not establish ownership. It is used only to fail closed when a
    class has an unresolved base and a direct member-like return could name
    inherited storage that the bounded parser cannot prove.
    """
    candidates = {
        value for value in values
        if value and _is_identifier_start(value[0])
        and value not in {"get", "std", "this"}
    }
    return (any(_direct_member_expression(values, {candidate}, shadow_names)
                for candidate in candidates)
            or _qualified_base_member_candidate(
                values, unresolved_base_types))


def _local_class_body(body, index, brace_pairs):
    if body[index].text not in ("class", "struct", "union"):
        return None
    if index + 1 >= len(body) or not body[index + 1].text:
        return None
    name = body[index + 1].text
    if not _is_identifier_start(name[0]):
        return None
    cursor = index + 2
    while cursor < len(body) and body[cursor].text not in ("{", ";"):
        cursor += 1
    if cursor >= len(body) or body[cursor].text != "{":
        return None
    closing = brace_pairs.get(cursor)
    return None if closing is None else (name, cursor, closing)


def _lambda_body(body, index, bracket_pairs, paren_pairs, brace_pairs):
    if body[index].text != "[":
        return None
    if index > 0 and body[index - 1].text not in {
            "=", "(", "{", ",", ";", "return", ":"}:
        return None
    capture_close = bracket_pairs.get(index)
    if capture_close is None:
        return None
    cursor = capture_close + 1
    if cursor < len(body) and body[cursor].text == "(":
        parameters_close = paren_pairs.get(cursor)
        if parameters_close is None:
            return None
        cursor = parameters_close + 1
    while cursor < len(body) and body[cursor].text not in ("{", ";"):
        cursor += 1
    if cursor >= len(body) or body[cursor].text != "{":
        return None
    closing = brace_pairs.get(cursor)
    return None if closing is None else (cursor, closing)


def _declared_block_type_alias_names(declarations):
    """Project block alias names from exact declaration-graph facts."""
    return {
        declaration.name for declaration in declarations
        if declaration.kind in ("using_alias", "typedef_alias")
    }


def _return_expressions_with_bindings(body, parameter_names, aliases):
    """Return each expression with bindings and its exact lexical position.

    This is the binding distinction the policy needs: an unqualified spelling
    names the nearest graph-owned parameter/local/control fact first, while
    ``this->`` bypasses that lookup. Macro-generated and lambda bodies remain
    outside the supported callable subset.
    """
    # Retain the historical argument boundary for registered mutation gates;
    # the declaration graph is now the sole parameter-name authority.
    del parameter_names
    expressions = []
    brace_pairs = _pair_map(body, "{", "}")
    bracket_pairs = _pair_map(body, "[", "]")
    paren_pairs = _pair_map(body, "(", ")")
    graph = aliases.environment.declaration_graph
    target_callable = -1
    if graph is not None and body:
        candidates = [
            scope for scope in graph.scopes
            if scope.kind == "function"
            and scope.opening <= body[0].start
            and body[-1].end <= scope.closing
        ]
        if candidates:
            target_callable = max(
                candidates, key=lambda scope: scope.opening).scope_id
    block_type_declarations = tuple(
        declaration for declaration in (
            graph.declarations if graph is not None else ())
        if declaration.kind in ("using_alias", "typedef_alias")
        and graph.scopes[declaration.parent_scope].kind
        in ("function", "lambda", "ordinary_block")
    )
    index = 0
    while index < len(body):
        value = body[index].text
        local_class = _local_class_body(body, index, brace_pairs)
        if local_class is not None:
            _, opening, closing = local_class
            nested_scope = (_graph_scope_for_offset(
                graph, body[opening].end) if graph is not None else -1)
            if (target_callable < 0
                    or graph.scopes[nested_scope].callable_owner
                    != target_callable):
                index = closing + 1
                continue
        lambda_body = _lambda_body(
            body, index, bracket_pairs, paren_pairs, brace_pairs)
        if lambda_body is not None:
            opening, closing = lambda_body
            nested_scope = (_graph_scope_for_offset(
                graph, body[opening].end) if graph is not None else -1)
            if (target_callable < 0
                    or graph.scopes[nested_scope].callable_owner
                    != target_callable):
                index = closing + 1
                continue
        if value == "return":
            cursor = index + 1
            parens = brackets = braces = 0
            while cursor < len(body):
                current = body[cursor].text
                parens += (current == "(") - (current == ")")
                brackets += (current == "[") - (current == "]")
                braces += (current == "{") - (current == "}")
                if current == ";" and parens == brackets == braces == 0:
                    break
                cursor += 1
            return_scope = (_graph_scope_for_offset(
                graph, body[index].start) if graph is not None else -1)
            if (target_callable >= 0
                    and graph.scopes[return_scope].callable_owner
                    != target_callable):
                index = cursor + 1
                continue
            shadows = {
                declaration.name
                for declaration in (
                    graph.data_name_declarations if graph is not None else ())
                if declaration.callable_owner == target_callable
                and declaration.point_of_declaration <= body[index].start
                < declaration.lifetime_end
            }
            active_scope_ids = {
                scope.scope_id for scope in _graph_scope_chain(
                    graph, return_scope)
            } if graph is not None else set()
            block_types = _declared_block_type_alias_names(tuple(
                declaration for declaration in block_type_declarations
                if declaration.parent_scope in active_scope_ids
                and declaration.point_of_declaration <= body[index].start
                < declaration.lifetime_end
            ))
            expressions.append((
                [token.text for token in body[index + 1:cursor]], shadows,
                body[index].start, block_types))
            index = cursor + 1
            continue
        index += 1
    return expressions


def _preceding_doc(text, start):
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[:line_start].splitlines()
    output = []
    while prefix:
        line = prefix[-1]
        stripped = line.lstrip()
        if stripped.startswith(("/**", "/*", "*", "//")) or not stripped:
            output.append(prefix.pop())
        else:
            break
    return "\n".join(reversed(output))


def _preceding_class_doc(text, start):
    """Return the Doxygen block owning a class declaration.

    A class template's documentation precedes its ``template <...>`` line,
    rather than the ``class`` token.  Only template declarations may separate
    the block from the class; an unrelated declaration therefore cannot lend
    its comment to a later class.
    """
    opening = text.rfind("/**", 0, start)
    if opening < 0:
        return ""
    closing = text.find("*/", opening, start)
    if closing < 0:
        return ""
    closing += 2
    between = text[closing:start]
    if not re.fullmatch(r"\s*(?:template\s*<[^;{}]*>\s*)*", between,
                        flags=re.S):
        return ""
    return text[opening:closing]


def _top_level_parts(tokens, delimiter):
    parts = []
    start = 0
    parens = brackets = braces = angles = attributes = 0
    for index, token in enumerate(tokens):
        value = token.text
        if value == "[[":
            attributes += 1
        elif value == "]]" and attributes:
            attributes -= 1
        elif attributes == 0:
            parens += (value == "(") - (value == ")")
            brackets += (value == "[") - (value == "]")
            braces += (value == "{") - (value == "}")
            angles += (value == "<") - (value == ">")
            if value == ">>" and angles >= 2:
                angles -= 2
            if (value == delimiter and parens == brackets == braces == angles == 0):
                parts.append(tokens[start:index])
                start = index + 1
    parts.append(tokens[start:])
    return parts


def _declared_data_names(statement, aliases):
    if not statement:
        return set()
    first_identifier = next((token.text for token in statement
                             if _is_identifier_start(token.text[0])), "")
    if first_identifier in {
            "class", "concept", "enum", "friend", "static_assert", "struct",
            "template", "typedef", "using"}:
        return set()
    if _function_signature(statement, aliases) is not None:
        return set()

    names = set()
    for declarator in _top_level_parts(statement, ","):
        prefix = []
        parens = brackets = angles = attributes = 0
        for token in declarator:
            value = token.text
            if value == "[[":
                attributes += 1
            elif value == "]]" and attributes:
                attributes -= 1
            elif attributes == 0:
                parens += (value == "(") - (value == ")")
                brackets += (value == "[") - (value == "]")
                angles += (value == "<") - (value == ">")
                if value == ">>" and angles >= 2:
                    angles -= 2
                if value in ("=", "{") and parens == brackets == angles == 0:
                    break
            prefix.append(token)
        identifiers = [token.text for token in prefix
                       if _is_identifier_start(token.text[0])
                       and token.text not in _RETURN_SPECIFIERS
                       and token.text not in {"const", "mutable", "typename", "volatile"}]
        if identifiers:
            names.add(identifiers[-1])
    return names


def _member_names_by_access(tokens, opening, closing, brace_pairs, aliases,
                            default_access, class_identity):
    names = set()
    inherited_accessible = set()
    declarations = []
    access = default_access
    statement_start = opening + 1
    index = statement_start
    while index < closing:
        value = tokens[index].text
        if (value in ("public", "private", "protected")
                and index + 1 < closing and tokens[index + 1].text == ":"):
            access = value
            statement_start = index + 2
            index += 2
            continue
        if value == ";":
            statement = tokens[statement_start:index]
            context = _alias_context_at(
                aliases, statement[0].start if statement
                else tokens[index].start)
            declared = _declared_data_names(statement, context)
            names.update(declared)
            if access != "private":
                inherited_accessible.update(declared)
            declaration = (statement[0].start if statement
                           else tokens[index].start)
            declarations.extend(
                DataMember(tuple(class_identity), name, access, declaration)
                for name in sorted(declared))
            statement_start = index + 1
            index += 1
            continue
        if value == "{":
            body_close = brace_pairs.get(index)
            if body_close is None or body_close > closing:
                index += 1
                continue
            signature = tokens[statement_start:index]
            context = _alias_context_at(
                aliases, signature[0].start if signature
                else tokens[index].start)
            if _function_signature(signature, context) is not None:
                statement_start = body_close + 1
            index = body_close + 1
            continue
        index += 1
    return names, inherited_accessible, tuple(declarations)


def _context_for_namespace(path, environment, use_position):
    return AliasContext(environment, tuple(path), use_position)


def _alias_context_at(context, use_position):
    return AliasContext(
        context.environment, context.current_scope, use_position,
        context.complete_class_scopes)


def _member_scope_alias_context(context, use_position, owner_identity):
    """Apply complete-class lookup after a qualified member declarator."""
    complete_scopes = context.environment.complete_class_scopes.get(
        tuple(owner_identity), (tuple(owner_identity),))
    return AliasContext(
        context.environment, context.current_scope, use_position,
        complete_scopes)


def _member_body_alias_context(context, use_position, owner_identity):
    """Apply complete-class lookup inside a member-function body."""
    return _member_scope_alias_context(
        context, use_position, owner_identity)


def _definition_signature_details(signature, structure, namespace_context,
                                  owner_context, owner_identity):
    """Resolve each out-of-class signature range in its C++ context."""
    leading_position = signature[0].start
    parameter_index = min(
        structure.parameters_open + 1, len(signature) - 1)
    parameter_context = _member_scope_alias_context(
        owner_context, signature[parameter_index].start, owner_identity)
    if structure.trailing_return >= 0:
        trailing_index = min(
            structure.trailing_return + 1, len(signature) - 1)
    else:
        trailing_index = structure.name_start
    trailing_context = _member_scope_alias_context(
        owner_context, signature[trailing_index].start, owner_identity)
    return _function_signature_details(
        signature,
        parameter_context,
        structure,
        leading_return_aliases=_alias_context_at(
            namespace_context, leading_position),
        trailing_return_aliases=trailing_context,
    )


def _base_specifications(base_tokens, default_access):
    """Return base spellings with their declared inheritance properties."""
    values = list(base_tokens)
    if not values or ":" not in [token.text for token in values]:
        return []
    colon = next(index for index, token in enumerate(values)
                 if token.text == ":")
    output = []
    for base in _top_level_parts(values[colon + 1:], ","):
        filtered = _strip_attributes([token.text for token in base])
        access = next((value for value in filtered if value in {
            "public", "protected", "private"}), default_access)
        is_virtual = "virtual" in filtered
        spelling = tuple(value for value in filtered if value not in {
            "public", "protected", "private", "virtual"})
        if spelling:
            first = next((token for token in base
                          if token.text not in {
                              "public", "protected", "private", "virtual",
                              "[[", "]]"}), base[0])
            output.append((spelling, access, is_virtual, first.start))
    return output


def _resolve_class(parts, scope, classes_by_identity, use_position=None,
                   declaration_graph=None, complete_class_scopes=()):
    for length in range(len(scope), -1, -1):
        candidate = scope[:length] + tuple(parts)
        if (candidate in classes_by_identity
                and (declaration_graph is None or use_position is None
                     or _class_identity_visible(
                         declaration_graph, candidate, use_position,
                         complete_class_scopes))):
            return classes_by_identity[candidate]
    return None


def _simple_qualified_type(values):
    """Parse a non-dependent named type, retaining absolute qualification."""
    values = list(values)
    while values and values[0] in ("class", "struct", "union"):
        values.pop(0)
    absolute = bool(values and values[0] == "::")
    if absolute:
        values.pop(0)
    if not values:
        return None
    parts = values[::2]
    separators = values[1::2]
    if (len(values) != len(parts) + len(separators)
            or not all(part and _is_identifier_start(part[0])
                       for part in parts)
            or not all(separator == "::" for separator in separators)):
        return None
    return tuple(parts), absolute


def _type_alias_key(parts, absolute, scope, aliases, use_position,
                    complete_class_scopes=()):
    scope_lengths = (0,) if absolute else range(len(scope), -1, -1)
    for length in scope_lengths:
        key = tuple(scope[:length]) + tuple(parts)
        targets = _visible_alias_targets(
            aliases.get(key, ()), use_position, complete_class_scopes)
        if targets:
            return key, targets
    return None, ()


def _qualified_type_alias_key(parts, absolute, scope, type_aliases,
                              namespace_aliases, use_position,
                              known_namespaces, namespace_alias_cache,
                              complete_class_scopes=()):
    """Find a type alias after positional namespace-alias composition."""
    if namespace_alias_cache is _SELECTED_TYPE_ALIAS_PROJECTION:
        # Retain the pre-centralization projection seam without performing a
        # second lookup; the selected declaration remains authoritative.
        return _SELECTED_TYPE_ALIAS_PROJECTION, ()
    key, targets = _type_alias_key(
        parts, absolute, scope, type_aliases, use_position,
        complete_class_scopes)
    if key is not None or len(parts) < 2:
        return key, targets

    match = _namespace_alias_prefix(
        parts, scope, namespace_aliases, use_position,
        maximum=len(parts) - 1, absolute=absolute)
    if match is None:
        return None, ()
    namespace_key, consumed = match
    namespace = _resolve_namespace_alias(
        namespace_key, use_position, namespace_aliases, known_namespaces,
        namespace_alias_cache)
    if namespace is None:
        return None, ()
    key = tuple(namespace) + tuple(parts[consumed:])
    targets = _visible_alias_targets(
        type_aliases.get(key, ()), use_position, complete_class_scopes)
    return (key, targets) if targets else (None, ())


def _type_alias_targets_accessible(targets, access_scopes,
                                   classes_by_identity, base_edges):
    """Check class-member alias access without assuming friendship."""
    for target in targets:
        declaring_class = tuple(target.scope)
        if declaring_class not in classes_by_identity:
            continue
        if declaring_class in access_scopes or target.access == "public":
            continue
        if target.access != "protected":
            return False
        if not any(
                _owner_or_unique_ancestor_relation(
                    tuple(scope), declaring_class, base_edges)[0]
                in ("owner", "ancestor")
                for scope in access_scopes):
            return False
    return True


def _resolve_type_alias_class(key, target, type_aliases, namespace_aliases,
                              known_namespaces,
                              namespace_alias_cache, classes_by_identity,
                              cache, declaration_graph=None, active=()):
    if key is _SELECTED_TYPE_ALIAS_PROJECTION:
        # Preserve the registered class-region projection mutation seam
        # without repeating lookup: the declaration-selected central result
        # travels in a type-alias-shaped compatibility carrier.
        return (target.declaration_graph
                if isinstance(target, TypeAliasTarget) else None)
    if (declaration_graph is not None
            and not isinstance(declaration_graph, DeclarationGraph)):
        active = declaration_graph
        declaration_graph = None
    cache_key = key, target.declaration
    if cache_key in cache:
        return cache[cache_key]
    if cache_key in active:
        return None
    parsed = _simple_qualified_type(target.rhs)
    if parsed is None:
        cache[cache_key] = None
        return None
    parts, absolute = parsed
    nested_key, nested_targets = _qualified_type_alias_key(
        parts, absolute, target.scope, type_aliases, namespace_aliases,
        target.declaration, known_namespaces, namespace_alias_cache)
    if nested_key is not None:
        resolutions = [
            _resolve_type_alias_class(
                nested_key, nested_target, type_aliases, namespace_aliases,
                known_namespaces, namespace_alias_cache,
                classes_by_identity, cache, declaration_graph,
                active + (cache_key,))
            for nested_target in nested_targets
        ]
        resolved = (
            resolutions[0] if resolutions and resolutions[0] is not None
            and all(item is resolutions[0] for item in resolutions)
            else None
        )
    else:
        resolved = _resolve_aliased_class(
            parts, absolute, target.scope, namespace_aliases,
            target.declaration, known_namespaces, namespace_alias_cache,
            classes_by_identity, declaration_graph)
    cache[cache_key] = resolved
    return resolved


def _resolve_base_class(values, scope, use_position, type_aliases,
                        namespace_aliases,
                        known_namespaces, namespace_alias_cache,
                        classes_by_identity, type_alias_cache,
                        access_scopes=(), base_edges=None,
                        declaration_graph=None,
                        complete_class_scopes=(), alias_environment=None,
                        lookup_purpose="base"):
    parsed = _simple_qualified_type(values)
    if parsed is None:
        return None
    parts, absolute = parsed
    if declaration_graph is not None and alias_environment is not None:
        first_offset = use_position
        result = _resolve_qualified_name(
            _qualified_component_records(
                parts, first_offset + (2 if absolute else 0),
                alias_environment.source_tokens),
            LookupContext(
                starting_scope=_graph_scope_for_offset(
                    declaration_graph, first_offset),
                use_offset=first_offset,
                purpose=lookup_purpose,
                absolute=absolute,
            ),
            alias_environment,
        )
        if result.status != "found":
            return None
        return classes_by_identity.get(tuple(result.canonical_identity))
    alias_key, alias_targets = _qualified_type_alias_key(
        parts, absolute, scope, type_aliases, namespace_aliases,
        use_position, known_namespaces, namespace_alias_cache)
    if alias_key is not None:
        if not _type_alias_targets_accessible(
                alias_targets, access_scopes, classes_by_identity,
                base_edges or {}):
            return None
        resolutions = [
            _resolve_type_alias_class(
                alias_key, alias_target, type_aliases, namespace_aliases,
                known_namespaces, namespace_alias_cache,
                classes_by_identity, type_alias_cache, declaration_graph)
            for alias_target in alias_targets
        ]
        return (
            resolutions[0] if resolutions and resolutions[0] is not None
            and all(item is resolutions[0] for item in resolutions)
            else None
        )
    return _resolve_aliased_class(
        parts, absolute, scope, namespace_aliases, use_position,
        known_namespaces, namespace_alias_cache, classes_by_identity,
        declaration_graph, complete_class_scopes)


def _inherited_alias_candidates(
        owner, name, environment, active=(), depth=0):
    """Find class-member type aliases through bounded base lookup."""
    if owner in active:
        return (), True
    key = tuple(owner) + (name,)
    direct = environment.type_aliases.get(key, ())
    if direct:
        return ((key, direct, ()),), False

    edges = environment.base_edges.get(owner, ())
    if depth >= _GRAPH_TRAVERSAL_DEPTH_LIMIT:
        return (), bool(edges)
    candidates = []
    unsupported = False
    for edge in edges:
        nested, nested_unsupported = _inherited_alias_candidates(
            edge.base_class, name, environment, active + (owner,), depth + 1)
        candidates.extend((nested_key, targets, (edge,) + path)
                          for nested_key, targets, path in nested)
        unsupported = unsupported or nested_unsupported or edge.virtual
    return tuple(candidates), unsupported


def _checked_member_type_alias(key, targets, aliases):
    """Apply one access decision to an already selected member alias."""
    environment = aliases.environment
    if key is None:
        return None, (), "none"
    access_scopes = (
        aliases.complete_class_scopes
        or environment.complete_class_scopes.get(
            aliases.current_scope, ()))
    if not _type_alias_targets_accessible(
            targets, access_scopes, environment.classes_by_identity,
            environment.base_edges):
        return None, (), "unsupported"
    return key, targets, "found"


def _direct_member_type_alias_lookup(parts, absolute, aliases):
    """Lookup aliases declared directly in a complete class scope."""
    environment = aliases.environment

    if absolute or len(parts) != 1 or not aliases.complete_class_scopes:
        return None, (), "none"

    name = parts[0]
    for class_scope in aliases.complete_class_scopes:
        key = tuple(class_scope) + (name,)
        targets = _visible_alias_targets(
            environment.type_aliases.get(key, ()), aliases.use_position,
            aliases.complete_class_scopes)
        if targets:
            return _checked_member_type_alias(key, targets, aliases)

    return None, (), "none"


def _inherited_member_type_alias_lookup(parts, absolute, aliases):
    """Lookup aliases contributed by base scopes after direct members."""
    environment = aliases.environment
    if absolute or len(parts) != 1 or not aliases.complete_class_scopes:
        return None, (), "none"

    name = parts[0]
    for class_scope in aliases.complete_class_scopes:
        inherited = []
        unsupported = False
        owner = tuple(class_scope)
        for edge in environment.base_edges.get(owner, ()):
            nested, nested_unsupported = _inherited_alias_candidates(
                edge.base_class, name, environment, (owner,), 1)
            inherited.extend(
                (nested_key, targets, (edge,) + path)
                for nested_key, targets, path in nested
            )
            unsupported = (
                unsupported or nested_unsupported or edge.virtual)
        if unsupported or len(inherited) > 1:
            return None, (), "unsupported"
        if inherited:
            inherited_key, inherited_targets, path = inherited[0]
            if (not _path_accessible_from_owner(path)
                    or _checked_member_type_alias(
                        inherited_key, inherited_targets, aliases)[2]
                    == "unsupported"):
                return None, (), "unsupported"
            return inherited_key, inherited_targets, "found"

    return None, (), "none"


def _member_type_alias_lookup(parts, absolute, aliases):
    """Lookup direct class aliases, then aliases inherited from bases."""
    direct = _direct_member_type_alias_lookup(parts, absolute, aliases)
    if direct[2] != "none":
        return direct
    return _inherited_member_type_alias_lookup(parts, absolute, aliases)


def _expression_type_alias_lookup(parts, absolute, aliases):
    """Apply member, base-member and enclosing-class alias lookup order."""
    environment = aliases.environment

    member = _member_type_alias_lookup(parts, absolute, aliases)
    if member[2] != "none":
        return member

    def checked(key, targets):
        if key is None:
            return None, (), "none"
        access_scopes = (
            aliases.complete_class_scopes
            or environment.complete_class_scopes.get(
                aliases.current_scope, ()))
        if not _type_alias_targets_accessible(
                targets, access_scopes, environment.classes_by_identity,
                environment.base_edges):
            return None, (), "unsupported"
        return key, targets, "found"

    key, targets = _qualified_type_alias_key(
        parts, absolute, aliases.current_scope,
        environment.type_aliases, environment.namespace_aliases,
        aliases.use_position, environment.known_namespaces,
        environment.namespace_alias_cache, aliases.complete_class_scopes)
    return checked(key, targets)


def _resolved_alias_identity(alias_key, alias_targets, aliases):
    environment = aliases.environment
    resolutions = [
        _resolve_type_alias_class(
            alias_key, target, environment.type_aliases,
            environment.namespace_aliases, environment.known_namespaces,
            environment.namespace_alias_cache,
            environment.classes_by_identity,
            environment.type_alias_cache, environment.declaration_graph)
        for target in alias_targets
    ]
    resolved = (
        resolutions[0] if resolutions and resolutions[0] is not None
        and all(item is resolutions[0] for item in resolutions)
        else None
    )
    return (None if resolved is None
            else environment.class_identities.get(resolved))


def _resolve_expression_qualifier_result(values, aliases):
    """Resolve one bounded class-id through the central lookup boundary."""
    parsed = _simple_qualified_type(values)
    if parsed is None:
        return LookupResult("unsupported", reason="unsupported class-id")
    parts, absolute = parsed
    environment = aliases.environment
    graph = environment.declaration_graph
    if graph is not None:
        starting_scope = _graph_scope_for_offset(graph, aliases.use_position)
        result = _resolve_qualified_name(
            _qualified_component_records(
                parts, aliases.use_position + (2 if absolute else 0),
                environment.source_tokens),
            LookupContext(
                starting_scope=starting_scope,
                use_offset=aliases.use_position,
                purpose="expression-qualifier",
                complete_class=bool(aliases.complete_class_scopes),
                owner_class=(tuple(aliases.complete_class_scopes[0])
                             if aliases.complete_class_scopes else ()),
                absolute=absolute,
            ),
            environment,
        )
        if (result.status == "found" and len(parts) == 1
                and aliases.complete_class_scopes
                and result.selected_declaration_id >= 0):
            declaration = graph.declarations[
                result.selected_declaration_id]
            parent = graph.scopes[declaration.parent_scope]
            if (parent.kind == "nonlocal_class"
                    and tuple(parent.canonical_class)
                    == tuple(aliases.complete_class_scopes[0])):
                # Compatibility projection for retained one-component
                # consumers; the graph keeps declaration-id provenance.
                return replace(
                    result,
                    provenance=(
                        "scope", declaration.parent_scope,
                        "purpose", "direct-member-class",
                    ),
                )
        return result

    alias_key, alias_targets, alias_status = _expression_type_alias_lookup(
        parts, absolute, aliases)
    if alias_status == "unsupported":
        return LookupResult(
            "unsupported", reason="type alias lookup is unsupported")
    if alias_key is not None:
        identity = _resolved_alias_identity(alias_key, alias_targets, aliases)
        if identity is None:
            return LookupResult(
                "ambiguous", reason="type alias target is ambiguous")
        return LookupResult(
            "found", canonical_identity=tuple(identity),
            declaration_kind="type_alias",
            provenance=("type-alias", alias_key),
        )

    resolved = _resolve_aliased_class(
        parts, absolute, aliases.current_scope,
        environment.namespace_aliases, aliases.use_position,
        environment.known_namespaces,
        environment.namespace_alias_cache,
        environment.classes_by_identity, environment.declaration_graph,
        aliases.complete_class_scopes)
    if resolved is None:
        return LookupResult("not-found")
    identity = environment.class_identities.get(resolved)
    return LookupResult(
        "found", canonical_identity=tuple(identity),
        declaration_kind=resolved.tag + "_definition",
        provenance=("lexical-class", tuple(identity)),
    )


def _resolve_expression_qualifier(values, aliases):
    """Project the central qualified result for retained tuple consumers."""
    result = _resolve_expression_qualifier_result(values, aliases)
    return result.canonical_identity if result.status == "found" else None


def _expression_qualifier_is_namespace(values, aliases):
    """Project canonical namespace classification from central lookup."""
    result = _resolve_expression_qualifier_result(values, aliases)
    return (result.status == "found"
            and tuple(result.canonical_identity)
            in aliases.environment.known_namespaces)


def _qualified_member_parts(values):
    """Split direct qualified-member syntax before class-id validation."""
    values = _strip_outer_parentheses(values)
    get_argument = _std_get_argument(values)
    if get_argument is not None:
        return _qualified_member_parts(get_argument)
    if values[:2] == ["this", "->"]:
        values = values[2:]
    separators = [index for index, value in enumerate(values)
                  if value == "::"]
    if not separators:
        return None
    separator = separators[-1]
    member_index = separator + 1
    if (member_index >= len(values)
            or not _is_identifier_start(values[member_index][0])
            or not _member_subobject_suffix(values, member_index + 1)):
        return None
    qualifier = values[:separator]
    if not qualifier:
        return None
    return tuple(qualifier), values[member_index]


def _qualified_member_syntax(values):
    """Return ``(class-id tokens, member)`` for the supported direct form."""
    parsed = _qualified_member_parts(values)
    if parsed is None or _simple_qualified_type(parsed[0]) is None:
        return None
    return parsed


def _template_qualified_type_skeleton(values):
    """Erase balanced template arguments for exclusion classification."""
    values = list(values)
    absolute = bool(values and values[0] == "::")
    if absolute:
        values.pop(0)
    output = ["::"] if absolute else []
    index = 0
    saw_template_id = False
    while index < len(values):
        if not _is_identifier_start(values[index][0]):
            return None
        output.append(values[index])
        index += 1
        if index < len(values) and values[index] == "<":
            saw_template_id = True
            depth = 0
            while index < len(values):
                value = values[index]
                depth = _angle_depth_change(value, depth)
                index += 1
                if depth == 0:
                    break
            if depth:
                return None
        if index == len(values):
            break
        if values[index] != "::":
            return None
        output.append("::")
        index += 1
    return tuple(output) if saw_template_id else None


def _bind_template_qualified_member_exclusion(values, aliases, owner,
                                               base_edges):
    """Recognize template-id ownership only to keep it fail-closed."""
    parsed = _qualified_member_parts(values)
    if parsed is None:
        return QualifiedMemberBinding("not-qualified")
    skeleton = _template_qualified_type_skeleton(parsed[0])
    if skeleton is None:
        return QualifiedMemberBinding("not-qualified")
    lookup = _resolve_expression_qualifier_result(skeleton, aliases)
    if lookup.status != "found":
        return QualifiedMemberBinding(
            "unsupported" if lookup.status in {"unsupported", "ambiguous"}
            else "unresolved", (), (), parsed[1])
    qualifier = tuple(lookup.canonical_identity)
    relation, _ = _owner_or_unique_ancestor_relation(
        owner, qualifier, base_edges)
    if relation == "unrelated":
        return QualifiedMemberBinding(
            "unrelated", tuple(qualifier), (), parsed[1])
    return QualifiedMemberBinding(
        "unsupported", tuple(qualifier), (), parsed[1])


def _inheritance_paths(owner, target, base_edges, active=(), depth=0):
    """Return bounded paths and whether a cycle/depth edge was rejected."""
    if owner == target:
        return ((),), False
    if owner in active:
        return (), True
    edges = base_edges.get(owner, ())
    if depth >= _GRAPH_TRAVERSAL_DEPTH_LIMIT:
        return (), bool(edges)
    paths = []
    unsupported = False
    for edge in edges:
        suffixes, nested_unsupported = _inheritance_paths(
            edge.base_class, target, base_edges, active + (owner,), depth + 1)
        unsupported = unsupported or nested_unsupported
        for suffix in suffixes:
            paths.append((edge,) + suffix)
    return tuple(paths), unsupported


def _path_accessible_from_owner(path):
    """Whether a derived member can traverse this inheritance path."""
    return all(edge.access != "private" or index == 0
               for index, edge in enumerate(path))


def _owner_or_unique_ancestor_relation(owner, target, base_edges):
    """Classify the exact owner/ancestor relationship for a qualifier."""
    if owner == target:
        return "owner", ()
    paths, unsupported = _inheritance_paths(owner, target, base_edges)
    if unsupported:
        return "unsupported", ()
    if not paths:
        return "unrelated", ()
    if len(paths) != 1 or any(edge.virtual for path in paths for edge in path):
        return "ambiguous", ()
    path = paths[0]
    if not _path_accessible_from_owner(path):
        return "inaccessible", path
    return "ancestor", path


def _member_lookup_candidates(owner, name, base_edges, data_members,
                              active=(), depth=0):
    """Perform bounded qualified data-member lookup with ordinary hiding."""
    if owner in active:
        return (), True
    direct = tuple(member for member in data_members.get(owner, ())
                   if member.name == name)
    if direct:
        return tuple((member, ()) for member in direct), len(direct) != 1

    edges = base_edges.get(owner, ())
    if depth >= _GRAPH_TRAVERSAL_DEPTH_LIMIT:
        return (), bool(edges)
    candidates = []
    unsupported = False
    for edge in edges:
        nested, nested_unsupported = _member_lookup_candidates(
            edge.base_class, name, base_edges, data_members,
            active + (owner,), depth + 1)
        candidates.extend((member, (edge,) + path)
                          for member, path in nested)
        unsupported = unsupported or nested_unsupported or edge.virtual
    return tuple(candidates), unsupported


def _bounded_inherited_member_names(
        owner, inheritable_members, base_owners, cache=None,
        active=(), depth=0):
    """Aggregate inherited names without crossing the graph depth bound."""
    if cache is None:
        cache = {}
    cache_key = owner, depth
    if cache_key in cache:
        return set(cache[cache_key])
    if owner in active:
        return {_INHERITANCE_DEPTH_SENTINEL}
    names = set(inheritable_members.get(owner, ()))
    bases = tuple(base_owners.get(owner, ()))
    if depth >= _GRAPH_TRAVERSAL_DEPTH_LIMIT:
        if bases:
            names.add(_INHERITANCE_DEPTH_SENTINEL)
    else:
        for base in bases:
            names.update(_bounded_inherited_member_names(
                base, inheritable_members, base_owners, cache,
                active + (owner,), depth + 1))
    cache[cache_key] = frozenset(names)
    return names


def _select_qualified_data_member(owner, qualifier, name, owner_path,
                                  base_edges, data_members):
    """Select one exact accessible declaration after qualified lookup."""
    candidates, unsupported = _member_lookup_candidates(
        qualifier, name, base_edges, data_members)
    if unsupported or len(candidates) != 1:
        return QualifiedMemberBinding(
            "unsupported", tuple(qualifier), (), name)
    member, lookup_path = candidates[0]
    complete_path = tuple(owner_path) + tuple(lookup_path)
    accessible = (
        (member.access != "private" or member.declaring_class == owner)
        and _path_accessible_from_owner(complete_path)
    )
    if not accessible:
        return QualifiedMemberBinding(
            "unsupported", tuple(qualifier),
            tuple(member.declaring_class), name)
    return QualifiedMemberBinding(
        "bound", tuple(qualifier), tuple(member.declaring_class), name)


def _bind_resolved_qualified_member(values, aliases, owner, base_edges,
                                    data_members, block_type_aliases=()):
    """Bind one supported class-qualified owner/ancestor member expression."""
    parsed = _qualified_member_syntax(values)
    if parsed is None:
        return QualifiedMemberBinding("not-qualified")
    qualifier_values, member_name = parsed
    qualifier = _resolve_expression_qualifier(qualifier_values, aliases)
    if qualifier is None:
        lookup = _resolve_expression_qualifier_result(
            qualifier_values, aliases)
        graph = aliases.environment.declaration_graph
        selected_id = getattr(
            lookup, "selected_declaration_id", lookup.declaration_id)
        if (graph is not None and selected_id >= 0
                and not block_type_aliases):
            selected = graph.declarations[selected_id]
            parent = graph.scopes[selected.parent_scope]
            if (selected.kind in {"using_alias", "typedef_alias"}
                    and parent.kind
                    in {"function", "lambda", "ordinary_block"}):
                return QualifiedMemberBinding(
                    "unresolved", (), (), member_name)
        if (lookup.status == "unsupported"
                and (lookup.provenance[:1] == ("direct-imported-type",)
                     or lookup.declaration_kind == "using_declaration")):
            return QualifiedMemberBinding(
                "unresolved", (), (), member_name)
        if lookup.status in ("unsupported", "ambiguous"):
            return QualifiedMemberBinding(
                "unsupported", (), (), member_name)
        return QualifiedMemberBinding("unresolved", (), (), member_name)
    qualifier = tuple(qualifier)
    if tuple(qualifier) not in aliases.environment.classes_by_identity:
        if _expression_qualifier_is_namespace(qualifier_values, aliases):
            return QualifiedMemberBinding(
                "namespace", (), (), member_name)
        return QualifiedMemberBinding(
            "unsupported", (), (), member_name)
    relation, owner_path = _owner_or_unique_ancestor_relation(
        owner, qualifier, base_edges)
    if relation == "unrelated" and tuple(qualifier[:-1]) == tuple(owner):
        relation = "direct-nested-member"
    if relation == "unrelated":
        return QualifiedMemberBinding(
            "unrelated", tuple(qualifier), (), member_name)
    if relation not in ("owner", "ancestor", "direct-nested-member"):
        return QualifiedMemberBinding(
            "unsupported", tuple(qualifier), (), member_name)
    return _select_qualified_data_member(
        owner, qualifier, member_name, owner_path, base_edges, data_members)


def _classify_storage_expression(values, member_names, shadow_names, aliases,
                                 owner, base_edges, data_members,
                                 unresolved_base_types,
                                 block_type_aliases=()):
    """Share one direct-storage classifier across all definition placements."""
    if not _standard_get_provenance_is_exact(values, aliases):
        return QualifiedMemberBinding("not-member")
    root = _direct_member_root(values, member_names, shadow_names)
    if root is not None:
        relation = _select_qualified_data_member(
            owner, owner, root, (), base_edges, data_members)
        if relation.disposition == "bound":
            return QualifiedMemberBinding(
                "bound", (), relation.member_declaring_owner, root)
        return relation

    qualified_syntax = _qualified_member_syntax(values)
    if qualified_syntax is not None:
        if _expression_qualifier_is_block_type(
                qualified_syntax[0], block_type_aliases):
            return QualifiedMemberBinding(
                "unsupported", (), (), qualified_syntax[1])

    qualified = _bind_resolved_qualified_member(
        values, aliases, owner, base_edges, data_members,
        block_type_aliases)
    if qualified.disposition in (
            "bound", "unsupported", "unrelated", "namespace"):
        return qualified
    if _qualified_base_member_candidate(values, unresolved_base_types):
        return QualifiedMemberBinding("unresolved-base")
    if (unresolved_base_types
            and _direct_member_candidate(
                values, shadow_names, unresolved_base_types)):
        return QualifiedMemberBinding("unresolved-base")
    template_qualified = _bind_template_qualified_member_exclusion(
        values, aliases, owner, base_edges)
    if template_qualified.disposition in ("unsupported", "unrelated"):
        return template_qualified
    if _INHERITANCE_DEPTH_SENTINEL in member_names:
        return QualifiedMemberBinding("unsupported")
    return QualifiedMemberBinding("not-member")


def _classify_return_bindings(bindings):
    """Aggregate exact returns without conflating deliberate negatives."""
    dispositions = tuple(binding.disposition for binding in bindings)
    if dispositions and all(item == "bound" for item in dispositions):
        return "bound"
    if "unresolved-base" in dispositions:
        return "unresolved-base"
    if "unsupported" in dispositions:
        return "unsupported"
    return "deliberate-negative"


def _position_return_binding(binding, source_offset, source_line):
    """Attach one classifier result to its exact return token."""
    return ReturnBinding(
        disposition=binding.disposition,
        qualifier_target=tuple(binding.qualifier_target),
        member_declaring_owner=tuple(binding.member_declaring_owner),
        member_name=binding.member_name,
        source_offset=source_offset,
        source_line=source_line,
    )


def _retain_return_bindings(bindings):
    """Preserve every return record in lexical source order."""
    return tuple(bindings)


def _binding_compatibility_projection(bindings):
    """Project singular identity only for one uniform bound identity."""
    identities = tuple((
        binding.qualifier_target,
        binding.member_declaring_owner,
        binding.member_name,
    ) for binding in bindings)
    uniform = (
        bool(bindings)
        and all(binding.disposition == "bound" for binding in bindings)
        and all(identity == identities[0] for identity in identities)
    )
    if not uniform:
        return False, (), (), ""
    qualifier, member_owner, member_name = identities[0]
    return True, qualifier, member_owner, member_name


def _classify_positioned_return_bindings(
        returns, text, members, aliases, owner_identity, base_edges,
        data_members, unresolved_base_types):
    """Classify and position all returns through one shared path."""
    output = []
    for expression, shadows, source_offset, block_type_aliases in returns:
        lookup_offset = source_offset
        qualified = _qualified_member_syntax(expression)
        if qualified is not None:
            parsed = _simple_qualified_type(qualified[0])
            if parsed is not None:
                parts, absolute = parsed
                statement_end = text.find(";", source_offset)
                if statement_end < 0:
                    statement_end = len(text)
                found = text.find(
                    ("::" if absolute else "") + parts[0],
                    source_offset, statement_end)
                if found >= 0:
                    lookup_offset = found
        classification = _classify_storage_expression(
            expression,
            members,
            shadows,
            _member_body_alias_context(
                aliases, lookup_offset, owner_identity),
            owner_identity,
            base_edges,
            data_members,
            unresolved_base_types,
            block_type_aliases,
        )
        output.append(_position_return_binding(
            classification,
            source_offset,
            text.count("\n", 0, source_offset) + 1,
        ))
    return tuple(output)


def _direct_definition_start(tokens, parent_opening, opening, brace_pairs):
    start = 0 if parent_opening is None else parent_opening + 1
    cursor = start
    while cursor < opening:
        if tokens[cursor].text == "{":
            closing = brace_pairs.get(cursor)
            if closing is not None and closing < opening:
                cursor = closing + 1
                start = cursor
                continue
        if tokens[cursor].text == ";":
            start = cursor + 1
        cursor += 1
    return start


def analyze_public_const_reference_accessors(text):
    """Enumerate accessors and report recognizable unsupported bindings."""
    tokens = tokenize_cpp(text)
    brace_pairs = _pair_map(tokens, "{", "}")
    declaration_graph = _build_declaration_graph(text, tokens, brace_pairs)
    invalid_nonlocal = next((
        region for region in declaration_graph.nonlocal_class_regions
        if any(
            scope.kind == "local_class"
            and tuple(scope.canonical_class)
            == tuple(declaration_graph.class_identities[region])
            and scope.opening == tokens[region.opening].end
            for scope in declaration_graph.scopes
        )
    ), None)
    if invalid_nonlocal is not None:
        return [], [ReferenceParseDiagnostic(
            line=tokens[invalid_nonlocal.declaration].line,
            message="declaration graph class-locality partition is invalid",
        )]
    class_regions = [
        region for region in declaration_graph.nonlocal_class_regions
        if region.tag in ("class", "struct")
        and not region.template_id_name_indices
    ]
    namespace_regions = list(declaration_graph.namespace_regions)

    known_namespaces = {()}
    namespace_declarations = {}
    for declaration in declaration_graph.declarations:
        if declaration.kind != "namespace_declaration":
            continue
        identity = declaration.canonical_identity
        for length in range(1, len(identity) + 1):
            known_namespaces.add(identity[:length])
        namespace_declarations.setdefault(identity, []).append(
            declaration.declaration_start)
    namespace_declarations = {
        path: tuple(sorted(declarations))
        for path, declarations in namespace_declarations.items()
    }
    namespace_alias_cache = {}

    classes_by_identity = {}
    for region in class_regions:
        identity = declaration_graph.class_identities[region]
        classes_by_identity.setdefault(identity, region)

    scoped_type_aliases = {}
    scoped_namespace_aliases = {}
    for declaration in declaration_graph.declarations:
        parent = declaration_graph.scopes[declaration.parent_scope]
        if parent.kind == "nonlocal_class":
            scope = parent.canonical_class
        elif parent.kind == "namespace_fragment":
            scope = parent.canonical_namespace
        elif parent.kind == "global":
            scope = ()
        else:
            continue
        key = tuple(scope) + (declaration.name,)
        if declaration.kind in ("using_alias", "typedef_alias"):
            scoped_type_aliases.setdefault(key, []).append(TypeAliasTarget(
                tuple(scope), declaration.target_tokens,
                declaration.rhs_use_position, declaration.access,
                declaration.declaration_id, declaration_graph,
            ))
        elif declaration.kind == "namespace_alias":
            values = list(declaration.target_tokens)
            absolute = bool(values and values[0] == "::")
            if absolute:
                values.pop(0)
            parts = tuple(values[::2])
            separators = values[1::2]
            valid = (
                bool(parts)
                and len(values) == len(parts) + len(separators)
                and all(part and _is_identifier_start(part[0])
                        for part in parts)
                and all(separator == "::" for separator in separators)
            )
            if valid:
                scoped_namespace_aliases.setdefault(key, []).append(
                    NamespaceAliasTarget(
                        tuple(scope), absolute, parts,
                        declaration.rhs_use_position,
                        declaration.declaration_id, declaration_graph,
                    ))

    scoped_type_aliases = {
        key: tuple(sorted(values, key=lambda item: item.declaration))
        for key, values in scoped_type_aliases.items()
    }
    scoped_namespace_aliases = {
        key: tuple(sorted(values, key=lambda item: item.declaration))
        for key, values in scoped_namespace_aliases.items()
    }
    class_identities = {
        region: declaration_graph.class_identities[region]
        for region in class_regions
    }
    class_declarations = declaration_graph.class_declaration_points
    canonical_class_identities = frozenset(classes_by_identity)
    complete_class_scopes = {
        identity: tuple(
            identity[:length]
            for length in range(len(identity), 0, -1)
            if identity[:length] in canonical_class_identities
        )
        for identity in canonical_class_identities
    }
    type_alias_cache = {}
    alias_base_edges = {}
    alias_environment = AliasResolutionEnvironment(
        type_aliases=scoped_type_aliases,
        namespace_aliases=scoped_namespace_aliases,
        known_namespaces=frozenset(known_namespaces),
        namespace_declarations=namespace_declarations,
        namespace_alias_cache=namespace_alias_cache,
        classes_by_identity=classes_by_identity,
        class_identities=class_identities,
        class_declarations=class_declarations,
        type_alias_cache=type_alias_cache,
        complete_class_scopes=complete_class_scopes,
        base_edges=alias_base_edges,
        declaration_graph=declaration_graph,
        source_tokens=tuple(tokens),
    )

    alias_contexts = {
        region: AliasContext(
            alias_environment, class_identities[region],
            tokens[region.declaration].start)
        for region in class_regions
    }
    own_members = {}
    inheritable_members = {}
    data_members_by_identity = {}
    for region in class_regions:
        own, inheritable, member_declarations = _member_names_by_access(
            tokens, region.opening, region.closing, brace_pairs,
            alias_contexts[region], region.default_access,
            class_identities[region])
        own_members[region] = own
        inheritable_members[region] = inheritable
        data_members_by_identity[class_identities[region]] = \
            member_declarations

    base_regions = {}
    base_edges_by_identity = {}
    unresolved_bases = {}
    for region in class_regions:
        scope = class_identities[region][:-1]
        resolved_bases = []
        resolved_edges = []
        unresolved = []
        for (values, inheritance_access, is_virtual,
             base_use_position) in _base_specifications(
                region.base_tokens, region.default_access):
            resolved = _resolve_base_class(
                values, scope, base_use_position,
                scoped_type_aliases, scoped_namespace_aliases,
                known_namespaces, namespace_alias_cache,
                classes_by_identity, type_alias_cache,
                complete_class_scopes[class_identities[region]],
                base_edges_by_identity,
                declaration_graph=declaration_graph,
                alias_environment=alias_environment)
            if resolved is None:
                unresolved.append(values)
            else:
                resolved_bases.append(resolved)
                resolved_edges.append(BaseEdge(
                    class_identities[region], class_identities[resolved],
                    tuple(values), inheritance_access,
                    base_use_position, is_virtual))
        base_regions[region] = tuple(resolved_bases)
        base_edges_by_identity[class_identities[region]] = \
            tuple(resolved_edges)
        unresolved_bases[region] = tuple(unresolved)
    alias_base_edges.update(base_edges_by_identity)

    inherited_cache = {}

    all_members = {
        region: own_members[region] | set().union(
            *(_bounded_inherited_member_names(
                base, inheritable_members, base_regions, inherited_cache,
                (region,), 1)
              for base in base_regions[region]))
        for region in class_regions
    }

    output = []
    diagnostics = []
    declarations = {region: {} for region in class_regions}

    for region in class_regions:
        opening = region.opening
        closing = region.closing
        aliases = alias_contexts[region]
        members = all_members[region]
        class_documentation = _preceding_class_doc(
            text, tokens[region.declaration].start)
        class_definition = text[tokens[region.declaration].start:
                                tokens[closing].end]
        access = region.default_access
        member_start = opening + 1
        index = opening + 1
        while index < closing:
            value = tokens[index].text
            if (value in ("public", "private", "protected")
                    and index + 1 < closing and tokens[index + 1].text == ":"):
                access = value
                index += 2
                member_start = index
                continue
            if value == ";":
                signature = tokens[member_start:index]
                signature_aliases = _alias_context_at(
                    aliases, signature[0].start if signature
                    else tokens[index].start)
                details = _function_signature_details(
                    signature, signature_aliases)
                if details is not None:
                    first = signature[0] if signature else tokens[index]
                    declarations[region].setdefault(details.name, []).append((
                        details,
                        _preceding_doc(text, first.start),
                        first.line,
                        access,
                    ))
                member_start = index + 1
                index += 1
                continue
            if value != "{":
                index += 1
                continue

            body_close = brace_pairs.get(index)
            if body_close is None or body_close > closing:
                index += 1
                continue
            signature = tokens[member_start:index]
            signature_aliases = _alias_context_at(
                aliases, signature[0].start if signature
                else tokens[index].start)
            details = _function_signature_details(
                signature, signature_aliases)
            if details is not None:
                body = tokens[index + 1:body_close]
                returns = _return_expressions_with_bindings(
                    body, (), signature_aliases)
                eligible = (access == "public" and details.const_reference
                            and returns)
                owner_identity = class_identities[region]
                bindings = _classify_positioned_return_bindings(
                    returns,
                    text,
                    members,
                    aliases,
                    owner_identity,
                    base_edges_by_identity,
                    data_members_by_identity,
                    unresolved_bases[region],
                ) if eligible else ()
                return_classification = (
                    _classify_return_bindings(bindings)
                    if eligible else "ineligible")
                if return_classification == "bound":
                    first = signature[0] if signature else tokens[index]
                    retained_bindings = _retain_return_bindings(bindings)
                    (binding_uniform, qualifier_target,
                     member_declaring_owner,
                     member_name) = _binding_compatibility_projection(
                         retained_bindings)
                    output.append(ReferenceAccessor(
                        name=details.name,
                        documentation=_preceding_doc(text, first.start),
                        class_documentation=class_documentation,
                        class_definition=class_definition,
                        line=first.line,
                        access=access,
                        accessor_owner=owner_identity,
                        return_bindings=retained_bindings,
                        binding_identity_is_uniform=binding_uniform,
                        qualifier_target=qualifier_target,
                        member_declaring_owner=member_declaring_owner,
                        member_name=member_name,
                    ))
                elif return_classification == "unresolved-base":
                    first = signature[0] if signature else tokens[index]
                    diagnostics.append(ReferenceParseDiagnostic(
                        line=first.line,
                        message=(
                            "public const-reference definition {}() returns "
                            "a member-like expression through an unsupported "
                            "or unresolved base; Tier G cannot safely classify "
                            "inherited storage".format(details.name)),
                    ))
                elif return_classification == "unsupported":
                    first = signature[0] if signature else tokens[index]
                    diagnostics.append(ReferenceParseDiagnostic(
                        line=first.line,
                        message=(
                            "public const-reference definition {}() has a "
                            "direct owner/ancestor member return that Tier G "
                            "cannot bind uniquely and accessibly"
                            .format(details.name)),
                    ))
            elif access == "public":
                unsupported_name = _potential_const_reference_signature(
                    signature, signature_aliases)
                if unsupported_name is not None:
                    first = signature[0] if signature else tokens[index]
                    diagnostics.append(ReferenceParseDiagnostic(
                        line=first.line,
                        message=(
                            "public const-reference definition {}() uses an "
                            "unsupported declarator; Tier G cannot safely "
                            "classify it".format(unsupported_name)),
                    ))
            member_start = body_close + 1
            index = body_close + 1

    # Public declarations may keep their inline definition at namespace scope.
    # The declaration owns access, documentation and class Threading context;
    # the definition supplies the return-expression bindings.
    class_openings = {
        region.opening for region in declaration_graph.class_regions
    }
    namespace_by_opening = {
        region.opening: region for region in namespace_regions
    }
    brace_openings = sorted(index for index in brace_pairs
                            if tokens[index].text == "{")
    for opening in brace_openings:
        if opening in class_openings or opening in namespace_by_opening:
            continue
        containing = [candidate for candidate in brace_openings
                      if candidate < opening < brace_pairs.get(candidate, -1)]
        parent = max(containing) if containing else None
        if parent is not None and parent not in namespace_by_opening:
            continue
        start = _direct_definition_start(tokens, parent, opening, brace_pairs)
        signature = tokens[start:opening]
        definition_position = (signature[0].start if signature
                               else tokens[opening].start)
        namespace_path = (namespace_by_opening[parent].path
                          if parent is not None else ())
        namespace_context = _context_for_namespace(
            namespace_path, alias_environment, definition_position)
        structure = _function_declarator_structure(signature)
        if structure is None:
            unsupported_name = _potential_const_reference_signature(
                signature, namespace_context, require_qualified=True)
            if unsupported_name is not None:
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "qualified const-reference definition {}() uses an "
                        "unsupported declarator; Tier G cannot safely "
                        "associate it".format(unsupported_name)),
                ))
            continue
        (owner_parts, owner_absolute, owner_has_template_id,
         owner_start) = _qualified_owner_components(
             signature, structure.name_start)
        if not owner_parts:
            continue
        owner_position = signature[owner_start].start
        owner_values = (["::"] if owner_absolute else [])
        for part_index, part in enumerate(owner_parts):
            if part_index:
                owner_values.append("::")
            owner_values.append(part)
        owner = _resolve_base_class(
            tuple(owner_values), namespace_path, owner_position,
            scoped_type_aliases, scoped_namespace_aliases,
            known_namespaces, namespace_alias_cache,
            classes_by_identity, type_alias_cache,
            declaration_graph=declaration_graph,
            alias_environment=alias_environment,
            lookup_purpose="member-definition-owner")
        if owner_has_template_id:
            if owner is None:
                continue
            owner_identity = class_identities[owner]
            owner_context = _alias_context_at(
                alias_contexts[owner], owner_position)
            details = _definition_signature_details(
                signature, structure, namespace_context, owner_context,
                owner_identity)
            if details is None or not details.const_reference:
                continue
            declared = _matching_declarations(
                declarations, owner, details)
            if (len(declared) == 1 and declared[0][0].const_reference
                    and declared[0][3] == "public"):
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "qualified const-reference definition {}() uses the "
                        "class-template owner {}<...>; Tier G does not "
                        "instantiate template ownership"
                        .format(structure.name, "::".join(owner_parts))),
                ))
            continue
        if owner is None:
            owner_lookup = _resolve_qualified_name(
                _qualified_component_records(
                    owner_parts,
                    owner_position + (2 if owner_absolute else 0),
                    alias_environment.source_tokens),
                LookupContext(
                    starting_scope=_graph_scope_for_offset(
                        declaration_graph, owner_position),
                    use_offset=owner_position,
                    purpose="owner-diagnostic",
                    absolute=owner_absolute,
                ),
                alias_environment,
            )
            namespace_owner = (
                tuple(owner_lookup.canonical_identity)
                if owner_lookup.status == "found"
                and tuple(owner_lookup.canonical_identity)
                in alias_environment.known_namespaces
                else None
            )
            potential_name = _potential_const_reference_signature(
                signature, namespace_context, require_qualified=True)
            if potential_name is not None and namespace_owner is None:
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "const-reference definition {}() has an owner that "
                        "cannot be resolved as a same-header class or namespace "
                        "in its namespace-alias scope"
                        .format(structure.name)),
                ))
            continue
        owner_identity = class_identities[owner]
        owner_context = _alias_context_at(
            alias_contexts[owner], owner_position)
        details = _definition_signature_details(
            signature, structure, namespace_context, owner_context,
            owner_identity)
        aliases = _member_scope_alias_context(
            owner_context, signature[structure.name_start].start,
            owner_identity)
        if details is None:
            if _potential_const_reference_signature(
                    signature, aliases, require_qualified=True) is not None:
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "const-reference definition for resolved owner {} "
                        "uses an unsupported declarator"
                        .format("::".join(owner_parts))),
                ))
            continue
        if not details.const_reference:
            continue
        declared = _matching_declarations(declarations, owner, details)
        if len(declared) != 1:
            first = signature[0] if signature else tokens[opening]
            diagnostics.append(ReferenceParseDiagnostic(
                line=first.line,
                message=(
                    "const-reference definition {}() matched {} declarations; "
                    "Tier G requires one exact owner, name, parameter, return "
                    "and member-qualifier identity".format(
                        details.name, len(declared))),
            ))
            continue
        if not declared[0][0].const_reference:
            continue
        _, documentation, line, access = declared[0]
        if access != "public":
            continue
        body_close = brace_pairs[opening]
        body = tokens[opening + 1:body_close]
        returns = _return_expressions_with_bindings(body, (), aliases)
        members = all_members[owner]
        if not returns:
            continue
        bindings = _classify_positioned_return_bindings(
            returns,
            text,
            members,
            aliases,
            owner_identity,
            base_edges_by_identity,
            data_members_by_identity,
            unresolved_bases[owner],
        )
        return_classification = _classify_return_bindings(bindings)
        if return_classification != "bound":
            if return_classification == "unresolved-base":
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "const-reference definition {}() returns a member-like "
                        "expression through an unsupported or unresolved "
                        "base; Tier G cannot safely classify inherited "
                        "storage"
                        .format(details.name)),
                ))
            elif return_classification == "unsupported":
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "const-reference definition {}() has a direct "
                        "owner/ancestor member return that Tier G cannot bind "
                        "uniquely and accessibly".format(details.name)),
                ))
            continue
        class_documentation = _preceding_class_doc(
            text, tokens[owner.declaration].start)
        class_definition = text[tokens[owner.declaration].start:
                                tokens[owner.closing].end]
        retained_bindings = _retain_return_bindings(bindings)
        (binding_uniform, qualifier_target, member_declaring_owner,
         member_name) = _binding_compatibility_projection(
             retained_bindings)
        output.append(ReferenceAccessor(
            name=details.name,
            documentation=documentation,
            class_documentation=class_documentation,
            class_definition=class_definition,
            line=line,
            access=access,
            accessor_owner=owner_identity,
            return_bindings=retained_bindings,
            binding_identity_is_uniform=binding_uniform,
            qualifier_target=qualifier_target,
            member_declaring_owner=member_declaring_owner,
            member_name=member_name,
        ))
    return output, tuple(diagnostics)


def find_public_const_reference_accessors(text):
    """Enumerate the supported public const-reference member accessors."""
    accessors, _ = analyze_public_const_reference_accessors(text)
    return accessors


def qualified_owner_ancestor_association_cases(marker):
    """Generate the complete qualified owner/ancestor invariant product."""
    target_axes = (
        ("owner", "Owner", "owner"),
        ("literal-base", "Base", "literal-base"),
        ("aliased-base", "Base", "aliased-base"),
        ("grand-base", "Root", "grand-base"),
    )
    qualifier_axes = (
        ("relative", lambda target: target),
        ("absolute", lambda target: "::matrix::" + target),
        ("namespace", lambda target: "matrix::" + target),
        ("type-alias", lambda target: "Exact"),
    )
    shape_axes = (
        ("direct", "int storage_ = 7;",
         "static inline int storage_ = 11;", "{qualifier}::storage_"),
        ("field", "Payload storage_ {};",
         "static inline Payload storage_ {};",
         "{qualifier}::storage_.field"),
        ("index", "int storage_[1] {7};",
         "static inline int storage_[1] {11};",
         "{qualifier}::storage_[(0)]"),
        ("std-get", "std::tuple<int> storage_ {7};",
         "static inline std::tuple<int> storage_ {11};",
         "std::get<0>({qualifier}::storage_)"),
    )
    placement_axes = ("in-class", "namespace-scope")
    method_anchor = "/** {} matched declaration */".format(marker)
    class_doc = (
        "/** Threading: the foreign readout is atomic; this is a {}. */"
        .format(marker))
    sibling_doc = "/** {} sibling declaration */".format(marker)
    cases = []

    for target_label, target_name, target_kind in target_axes:
        for qualifier_label, qualifier_factory in qualifier_axes:
            qualifier = qualifier_factory(target_name)
            for shape_label, member, decoy_member, expression in shape_axes:
                return_expression = expression.format(qualifier=qualifier)
                for placement in placement_axes:
                    lines = [
                        "#include <tuple>",
                        "namespace matrix {",
                        "struct Payload { int field = 7; };",
                        "struct Decoy {{ {} }};".format(decoy_member),
                    ]
                    if target_kind == "literal-base":
                        lines.append(
                            "struct Base {{ protected: {} }};".format(member))
                    elif target_kind == "aliased-base":
                        lines.extend((
                            "struct Base {{ protected: {} }};".format(member),
                            "using Parent = Base;",
                        ))
                    elif target_kind == "grand-base":
                        lines.extend((
                            "struct Root {{ protected: {} }};".format(member),
                            "struct Mid : public Root {};",
                        ))

                    inheritance = {
                        "owner": "",
                        "literal-base": " : public Base",
                        "aliased-base": " : public Parent",
                        "grand-base": " : public Mid",
                    }[target_kind]
                    lines.extend((class_doc, "struct Owner{} {{".format(
                        inheritance)))
                    if target_kind == "owner":
                        lines.extend(("private:", "    " + member))
                    lines.append("public:")
                    lines.append("    " + method_anchor)
                    declaration_line = len(lines) + 1
                    if placement == "in-class":
                        lines.append(
                            "    const int& state() const { return "
                            + return_expression + "; }")
                    else:
                        lines.append("    const int& state() const;")
                    lines.extend((
                        "    " + sibling_doc,
                        "    int sibling() const { return 0; }",
                        "    int getPublished() const { return 0; }",
                        "    using Exact = {};".format(target_name),
                        "};",
                        "}",
                    ))
                    if placement == "namespace-scope":
                        lines.append(
                            "inline const int& matrix::Owner::state() const "
                            "{ return " + return_expression + "; }")
                    lines.append(
                        "int main() { matrix::Owner value; return value.state(); }")
                    source = "\n".join(lines) + "\n"
                    identity = ("matrix", target_name)
                    cases.append(QualifiedIdentityCase(
                        label="-".join((target_label, qualifier_label,
                                        shape_label, placement)),
                        source=source,
                        expected_line=declaration_line,
                        accessor_owner=("matrix", "Owner"),
                        qualifier_target=identity,
                        member_declaring_owner=identity,
                        member_name="storage_",
                        deletion_anchor=method_anchor,
                    ))
    return tuple(cases)


def cpp_identity_association_cases(marker):
    """Production Tier-G C++ identity and namespace-alias fixtures."""
    class_doc = (
        "/** Threading: this is a {}; getPublished is atomic. */"
        .format(marker))
    matched_doc = "/** {} matched declaration */".format(marker)
    sibling_doc = "/** {} sibling declaration */".format(marker)

    def owner_case(declarations, definition, prelude=""):
        return """
{prelude}
{class_doc}
struct Owner {{
public:
{declarations}
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
{definition}
int main() {{ return 0; }}
""".format(prelude=prelude, class_doc=class_doc,
           declarations=declarations, definition=definition)

    cases = [
        IdentityAssociationCase(
            "top-level-parameter-cv-adjustment",
            owner_case(
                """    {sibling}
    const int& state(double sibling) const;
    {matched}
    const int& state(int declared) const;""".format(
                    sibling=sibling_doc, matched=matched_doc),
                """inline const int& Owner::state(const int defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "array-to-pointer-parameter-adjustment",
            owner_case(
                """    {sibling}
    const int& state(double sibling) const;
    {matched}
    const int& state(int declared[3]) const;""".format(
                    sibling=sibling_doc, matched=matched_doc),
                """inline const int& Owner::state(int* defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "equivalent-cv-token-order",
            owner_case(
                """    struct Payload {{ int value = 0; }};
    {sibling}
    const int& state(double sibling) const;
    {matched}
    const int& state(const Payload& declared) const;""".format(
                    sibling=sibling_doc, matched=matched_doc),
                """inline const int& Owner::state(Payload const& defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 10, "public", matched_doc),
        IdentityAssociationCase(
            "namespace-alias-qualified-owner",
            """
namespace actual {{
{class_doc}
struct Owner {{
public:
    {sibling}
    const int& state(int sibling) const;
    {matched}
    const int& state() const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
}}
namespace facade = actual;
inline const int& facade::Owner::state() const {{ return storage_; }}
int main() {{ return 0; }}
""".format(class_doc=class_doc, sibling=sibling_doc,
           matched=matched_doc),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "top-level-pointer-cv-adjustment",
            owner_case(
                """    {sibling}
    const int& state(double sibling) const;
    {matched}
    const int& state(int* declared) const;""".format(
                    sibling=sibling_doc, matched=matched_doc),
                """inline const int& Owner::state(int* const defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "alias-top-level-cv-adjustment",
            owner_case(
                """    using Pointer = int*;
    {sibling}
    const int& state(double sibling) const;
    {matched}
    const int& state(Pointer const declared) const;""".format(
                    sibling=sibling_doc, matched=matched_doc),
                """inline const int& Owner::state(const Pointer defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 10, "public", matched_doc),
        IdentityAssociationCase(
            "pointee-cv-remains-distinct",
            owner_case(
                """    {sibling}
    const int& state(const int* declared) const;
    /** ordinary matched declaration */
    const int& state(int* declared) const;""".format(
                    sibling=sibling_doc),
                """inline const int& Owner::state(int* defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, False, 9, "public"),
        IdentityAssociationCase(
            "nested-array-extent-retained",
            owner_case(
                """    {sibling}
    const int& state(int declared[3][4]) const;
    /** ordinary matched declaration */
    const int& state(int declared[5][6]) const;""".format(
                    sibling=sibling_doc),
                """inline const int& Owner::state(int defined[5][6]) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, False, 9, "public"),
        IdentityAssociationCase(
            "chained-namespace-alias-owner",
            """
namespace actual_chain {{
{class_doc}
struct Owner {{
public:
    {sibling}
    const int& state(int sibling) const;
    {matched}
    const int& state() const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
}}
namespace first_facade = actual_chain;
namespace second_facade = first_facade;
inline const int& second_facade::Owner::state() const {{ return storage_; }}
int main() {{ return 0; }}
""".format(class_doc=class_doc, sibling=sibling_doc,
           matched=matched_doc),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "nested-reopened-namespace-alias-owner",
            """
namespace actual_nested {{
{class_doc}
struct Owner {{
public:
    {sibling}
    const int& state(int sibling) const;
    {matched}
    const int& state() const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
}}
namespace api {{}}
namespace api {{ namespace facade = actual_nested; }}
namespace api {{}}
inline const int& api::facade::Owner::state() const {{ return storage_; }}
int main() {{ return 0; }}
""".format(class_doc=class_doc, sibling=sibling_doc,
           matched=matched_doc),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "absolute-namespace-alias-owner",
            """
namespace actual_absolute {{
{class_doc}
struct Owner {{
public:
    {sibling}
    const int& state(int sibling) const;
    {matched}
    const int& state() const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
}}
namespace absolute_facade = actual_absolute;
inline const int& ::absolute_facade::Owner::state() const
{{
    return storage_;
}}
int main() {{ return 0; }}
""".format(class_doc=class_doc, sibling=sibling_doc,
           matched=matched_doc),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "qualified-parenthesized-member-name",
            owner_case(
                """    {sibling}
    const int& state(double sibling) const;
    {matched}
    const int& state(int declared) const;""".format(
                    sibling=sibling_doc, matched=matched_doc),
                """inline const int& (Owner::state)(int defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "sibling-namespace-alias-does-not-poison",
            """
namespace marked_target {{
{class_doc}
struct Owner {{
public:
    {matched}
    const int& state() const;
private:
    int storage_ = 7;
}};
}}
namespace ordinary_target {{
{class_doc}
struct Owner {{
public:
    /** ordinary matched declaration */
    const int& state() const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
}}
namespace left_scope {{ namespace facade = marked_target; }}
namespace right_scope {{ namespace facade = ordinary_target; }}
inline const int& right_scope::facade::Owner::state() const
{{
    return storage_;
}}
int main() {{ return 0; }}
""".format(class_doc=class_doc, matched=matched_doc),
            1, False, 17, "public"),
    ]

    duplicate_cv_exact = """

/** Threading: stream-owner reference readout; getPublished() is atomic. */
struct Owner {
public:
    using AlreadyConst = const int;
    /** stream-owner reference readout matched */
    const int& state(AlreadyConst& value) const;
    int getPublished() const noexcept { return 0; }
private:
    int storage_ = 7;
};
inline const int& Owner::state(const AlreadyConst& value) const
{
    (void)value;
    return storage_;
}
int main() { return 0; }
""".replace("stream-owner reference readout", marker)
    aliased_base_exact = """
struct Base { protected: int storage_ = 7; };
using Parent = Base;
/** Threading: stream-owner reference readout; getPublished() is atomic. */
struct Owner : Parent {
public:
    /** stream-owner reference readout matched */
    const int& state() const { return storage_; }
    int getPublished() const noexcept { return 0; }
};
int main() { Owner owner; return owner.state(); }
""".replace("stream-owner reference readout", marker)
    exact_anchor = "/** {} matched */".format(marker)
    cases.extend([
        IdentityAssociationCase(
            "already-const-alias-idempotence",
            duplicate_cv_exact, 1, True, 8, "public", exact_anchor),
        IdentityAssociationCase(
            "visible-base-alias-inheritance",
            aliased_base_exact, 1, True, 8, "public", exact_anchor),
        IdentityAssociationCase(
            "already-volatile-alias-idempotence",
            owner_case(
                """    using AlreadyVolatile = volatile int;
    {matched}
    const int& state(AlreadyVolatile& declared) const;""".format(
                    matched=matched_doc),
                """inline const int& Owner::state(
    volatile AlreadyVolatile&) const
{{
    return storage_;
}}"""),
            1, True, 8, "public", matched_doc),
        IdentityAssociationCase(
            "chained-already-cv-alias-idempotence",
            owner_case(
                """    using AlreadyConst = const int;
    using Chained = AlreadyConst;
    {matched}
    const int& state(Chained& declared) const;""".format(
                    matched=matched_doc),
                """inline const int& Owner::state(
    const Chained& defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "alias-pointee-cv-boundary",
            owner_case(
                """    using AlreadyConst = const int;
    {sibling}
    const int& state(AlreadyConst* sibling) const;
    /** ordinary matched declaration */
    const int& state(int* declared) const;""".format(
                    sibling=sibling_doc),
                """inline const int& Owner::state(int* defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, False, 10, "public"),
        IdentityAssociationCase(
            "referred-cv-boundary",
            owner_case(
                """    {sibling}
    const int& state(const int& sibling) const;
    /** ordinary matched declaration */
    const int& state(int& declared) const;""".format(
                    sibling=sibling_doc),
                """inline const int& Owner::state(int& defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, False, 9, "public"),
        IdentityAssociationCase(
            "duplicate-cv-sibling-marker-isolation",
            owner_case(
                """    using AlreadyConst = const int;
    {sibling}
    const int& state(double sibling) const;
    {matched}
    const int& state(AlreadyConst& declared) const;""".format(
                    sibling=sibling_doc, matched=matched_doc),
                """inline const int& Owner::state(
    const AlreadyConst& defined) const
{{
    (void)defined;
    return storage_;
}}"""),
            1, True, 10, "public", matched_doc),
        IdentityAssociationCase(
            "base-alias-sibling-marker-isolation",
            """
struct Base {{ protected: int storage_ = 7; }};
using Parent = Base;
{class_doc}
struct Owner : Parent {{
public:
    {sibling}
    const int& state(int sibling) const;
    {matched}
    const int& state() const {{ return storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
int main() {{ Owner owner; return owner.state(); }}
""".format(class_doc=class_doc, sibling=sibling_doc,
           matched=matched_doc),
            1, True, 10, "public", matched_doc),
        IdentityAssociationCase(
            "chained-visible-base-alias",
            """
struct Base {{ protected: int storage_ = 7; }};
using DirectParent = Base;
using Parent = DirectParent;
{class_doc}
struct Owner : Parent {{
public:
    {matched}
    const int& state() const {{ return storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
int main() {{ Owner owner; return owner.state(); }}
""".format(class_doc=class_doc, matched=matched_doc),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "typedef-visible-base-alias",
            """
struct Base {{ protected: int storage_ = 7; }};
typedef Base Parent;
{class_doc}
struct Owner : Parent {{
public:
    {matched}
    const int& state() const {{ return storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
int main() {{ Owner owner; return owner.state(); }}
""".format(class_doc=class_doc, matched=matched_doc),
            1, True, 8, "public", matched_doc),
        IdentityAssociationCase(
            "reopened-namespace-base-alias",
            """
namespace api {{
struct Base {{ protected: int storage_ = 7; }};
using Parent = Base;
}}
namespace api {{
{class_doc}
struct Owner : Parent {{
public:
    {matched}
    const int& state() const {{ return storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
}}
int main() {{ api::Owner owner; return owner.state(); }}
""".format(class_doc=class_doc, matched=matched_doc),
            1, True, 11, "public", matched_doc),
        IdentityAssociationCase(
            "outer-class-base-alias",
            """
struct Base {{ protected: int storage_ = 7; }};
struct Outer {{
    using Parent = Base;
    {class_doc}
    struct Owner : Parent {{
    public:
        {matched}
        const int& state() const {{ return storage_; }}
        int getPublished() const noexcept {{ return 0; }}
    }};
}};
int main() {{ Outer::Owner owner; return owner.state(); }}
""".format(class_doc=class_doc, matched=matched_doc),
            1, True, 9, "public", matched_doc),
        IdentityAssociationCase(
            "sibling-scope-base-alias-isolation",
            """
namespace left {{
struct Base {{ protected: int wrong_storage_ = 1; }};
using Parent = Base;
}}
namespace right {{
struct Base {{ protected: int storage_ = 7; }};
using Parent = Base;
{class_doc}
struct Owner : Parent {{
public:
    {matched}
    const int& state() const {{ return storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
}}
int main() {{ right::Owner owner; return owner.state(); }}
""".format(class_doc=class_doc, matched=matched_doc),
            1, True, 13, "public", matched_doc),
        IdentityAssociationCase(
            "unresolved-base-own-member-control",
            """
#define DECLARE_EXTERNAL_BASE struct ExternalBase {{}}
DECLARE_EXTERNAL_BASE;
{class_doc}
struct Owner : ExternalBase {{
public:
    {sibling}
    const int& state(int sibling) const;
    {matched}
    const int& state() const {{ return storage_; }}
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
int main() {{ Owner owner; return owner.state(); }}
""".format(class_doc=class_doc, sibling=sibling_doc,
           matched=matched_doc),
            1, True, 10, "public", matched_doc),
    ])
    temporal_identity_sources = (
        (
            "base-alias-late-local-shadow",
            """
struct BaseA { protected: int storage_a_ = 7; };
struct BaseB { protected: int storage_b_ = 9; };
using Parent = BaseA;
namespace ns {
struct Owner : Parent {
public:
    const int& state() const { return storage_a_; }
};
using Parent = BaseB;
}
int main() { ns::Owner owner; return owner.state(); }
""",
            8,
        ),
        (
            "base-alias-prior-local-shadow",
            """
struct BaseA { protected: int storage_a_ = 7; };
struct BaseB { protected: int storage_b_ = 9; };
using Parent = BaseA;
namespace ns {
using Parent = BaseB;
struct Owner : Parent {
public:
    const int& state() const { return storage_b_; }
};
}
int main() { ns::Owner owner; return owner.state(); }
""",
            9,
        ),
        (
            "base-alias-chain-declaration-point",
            """
struct BaseA { protected: int storage_a_ = 7; };
struct BaseB { protected: int storage_b_ = 9; };
using Root = BaseA;
namespace ns {
using Parent = Root;
using Root = BaseB;
struct Owner : Parent {
public:
    const int& state() const { return storage_a_; }
};
}
int main() { ns::Owner owner; return owner.state(); }
""",
            10,
        ),
        (
            "namespace-alias-late-local-shadow",
            """
namespace left { struct Base { protected: int storage_left_ = 7; }; }
namespace right { struct Base { protected: int storage_right_ = 9; }; }
namespace facade = left;
namespace ns {
struct Owner : facade::Base {
public:
    const int& state() const { return storage_left_; }
};
namespace facade = right;
}
int main() { ns::Owner owner; return owner.state(); }
""",
            8,
        ),
        (
            "namespace-alias-prior-local-shadow",
            """
namespace left { struct Base { protected: int storage_left_ = 7; }; }
namespace right { struct Base { protected: int storage_right_ = 9; }; }
namespace facade = left;
namespace ns {
namespace facade = right;
struct Owner : facade::Base {
public:
    const int& state() const { return storage_right_; }
};
}
int main() { ns::Owner owner; return owner.state(); }
""",
            9,
        ),
        (
            "namespace-alias-chain-declaration-point",
            """
namespace left { struct Base { protected: int storage_left_ = 7; }; }
namespace right { struct Base { protected: int storage_right_ = 9; }; }
namespace root_facade = left;
namespace ns {
namespace facade = root_facade;
namespace root_facade = right;
struct Owner : facade::Base {
public:
    const int& state() const { return storage_left_; }
};
}
int main() { ns::Owner owner; return owner.state(); }
""",
            10,
        ),
        (
            "signature-alias-late-local-shadow",
            """
using Param = int;
namespace ns {
struct Owner {
public:
    const int& state(Param value) const;
private:
    int storage_ = 7;
};
using Param = double;
}
inline const int& ns::Owner::state(int value) const
{
    (void)value;
    return storage_;
}
int main() { ns::Owner owner; return owner.state(1); }
""",
            6,
        ),
        (
            "signature-alias-prior-local-shadow",
            """
using Param = int;
namespace ns {
using Param = double;
struct Owner {
public:
    const int& state(Param value) const;
private:
    int storage_ = 7;
};
}
inline const int& ns::Owner::state(double value) const
{
    (void)value;
    return storage_;
}
int main() { ns::Owner owner; return owner.state(1.0); }
""",
            7,
        ),
        (
            "signature-alias-chain-declaration-point",
            """
using Root = int;
namespace ns {
using Param = Root;
using Root = double;
struct Owner {
public:
    const int& state(Param value) const;
private:
    int storage_ = 7;
};
}
inline const int& ns::Owner::state(int value) const
{
    (void)value;
    return storage_;
}
int main() { ns::Owner owner; return owner.state(1); }
""",
            8,
        ),
        (
            "signature-class-alias-declaration-scope",
            """
struct Payload {};
using Param = Payload;
namespace ns {
struct Payload {};
struct Owner {
public:
    const int& state(const Param& value) const;
private:
    int storage_ = 7;
};
}
inline const int& ns::Owner::state(const ::Payload& value) const
{
    (void)value;
    return storage_;
}
int main() { ns::Owner owner; ::Payload value; return owner.state(value); }
""",
            8,
        ),
        (
            "signature-namespace-alias-declaration-scope",
            """
namespace left { struct Payload {}; }
namespace right { struct Payload {}; }
namespace facade = left;
using Param = facade::Payload;
namespace ns {
namespace facade = right;
struct Owner {
public:
    const int& state(const Param& value) const;
private:
    int storage_ = 7;
};
}
inline const int& ns::Owner::state(const left::Payload& value) const
{
    (void)value;
    return storage_;
}
int main() { ns::Owner owner; left::Payload value; return owner.state(value); }
""",
            10,
        ),
        (
            "equivalent-signature-alias-redeclaration",
            """
using Param = const int;
using Param = int const;
struct Owner {
public:
    const int& state(Param& value) const;
private:
    int storage_ = 7;
};
inline const int& Owner::state(const int& value) const
{
    (void)value;
    return storage_;
}
int main() { Owner owner; int value = 0; return owner.state(value); }
""",
            6,
        ),
        (
            "equivalent-signature-absolute-class-redeclaration",
            """
struct Payload {};
using Param = Payload;
using Param = ::Payload;
struct Owner {
public:
    const int& state(const Param& value) const;
private:
    int storage_ = 7;
};
inline const int& Owner::state(const Payload& value) const
{
    (void)value;
    return storage_;
}
int main() { Owner owner; Payload value; return owner.state(value); }
""",
            7,
        ),
        (
            "equivalent-signature-absolute-class-reversed",
            """
struct Payload {};
using Param = Payload;
using Param = ::Payload;
struct Owner {
public:
    const int& state(const Param& value) const;
private:
    int storage_ = 7;
};
inline const int& Owner::state(const ::Payload& value) const
{
    (void)value;
    return storage_;
}
int main() { Owner owner; Payload value; return owner.state(value); }
""",
            7,
        ),
        (
            "equivalent-signature-namespace-class-redeclaration",
            """
namespace actual { struct Payload {}; }
namespace facade = actual;
using Param = actual::Payload;
using Param = facade::Payload;
struct Owner {
public:
    const int& state(const Param& value) const;
private:
    int storage_ = 7;
};
inline const int& Owner::state(const actual::Payload& value) const
{
    (void)value;
    return storage_;
}
int main() { Owner owner; actual::Payload value; return owner.state(value); }
""",
            8,
        ),
        (
            "equivalent-signature-namespace-class-reversed",
            """
namespace actual { struct Payload {}; }
namespace facade = actual;
using Param = actual::Payload;
using Param = facade::Payload;
struct Owner {
public:
    const int& state(const Param& value) const;
private:
    int storage_ = 7;
};
inline const int& Owner::state(const facade::Payload& value) const
{
    (void)value;
    return storage_;
}
int main() { Owner owner; actual::Payload value; return owner.state(value); }
""",
            8,
        ),
        (
            "equivalent-base-alias-redeclaration",
            """
struct Base { protected: int storage_ = 7; };
using Parent = Base;
using Parent = ::Base;
struct Owner : Parent {
public:
    const int& state() const { return storage_; }
};
int main() { Owner owner; return owner.state(); }
""",
            7,
        ),
        (
            "equivalent-namespace-alias-redeclaration",
            """
namespace actual {
struct Owner {
public:
    const int& state() const;
private:
    int storage_ = 7;
};
}
namespace facade = actual;
namespace facade = ::actual;
inline const int& facade::Owner::state() const { return storage_; }
int main() { actual::Owner owner; return owner.state(); }
""",
            5,
        ),
    )
    cases.extend(
        IdentityAssociationCase(label, source, 1, False, line, "public")
        for label, source, line in temporal_identity_sources
    )
    return tuple(cases)


def fail_closed_association_cases(marker):
    """Compiler-valid unsupported/control definitions for diagnostic policy."""
    class_doc = "/** Threading: this is a {}; getPublished is atomic. */".format(
        marker)
    unsupported = """
{class_doc}
struct UnsupportedOwner {{
public:
    /** {marker} matched declaration */
    const int& state(int (*declared)(int)) const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
inline const int& UnsupportedOwner::state(int (*defined)(int)) const
{{
    (void)defined;
    return storage_;
}}
int main() {{ return 0; }}
""".format(class_doc=class_doc, marker=marker)
    value_control = """
struct ValueOwner {
public:
    int state(int (*declared)(int)) const;
};
inline int ValueOwner::state(int (*defined)(int)) const
{
    return defined(0);
}
int identity(int value) { return value; }
int main() { ValueOwner owner; return owner.state(identity); }
"""
    parenthesized = """
{class_doc}
struct ParenthesizedOwner {{
public:
    /** {marker} matched declaration */
    const int& state(int (*declared)()) const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
inline const int& (ParenthesizedOwner::state)(int (*defined)()) const
{{
    (void)defined;
    return storage_;
}}
int main() {{ return 0; }}
""".format(class_doc=class_doc, marker=marker)
    subscript_operator = """
{class_doc}
struct SubscriptOwner {{
public:
    /** {marker} matched declaration */
    const int& operator[](int (*declared)()) const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
inline const int& SubscriptOwner::operator[](int (*defined)()) const
{{
    (void)defined;
    return storage_;
}}
int main() {{ return 0; }}
""".format(class_doc=class_doc, marker=marker)
    call_operator = """
{class_doc}
struct CallOwner {{
public:
    /** {marker} matched declaration */
    const int& operator()(int (*declared)()) const;
    int getPublished() const noexcept {{ return 0; }}
private:
    int storage_ = 7;
}};
inline const int& CallOwner::operator()(int (*defined)()) const
{{
    (void)defined;
    return storage_;
}}
int main() {{ return 0; }}
""".format(class_doc=class_doc, marker=marker)
    unresolved_owner = """
#define DECLARE_EXTERNAL_OWNER \\
struct ExternalOwner { \\
public: \\
    const int& state() const; \\
private: \\
    int storage_ = 7; \\
}
DECLARE_EXTERNAL_OWNER;
inline const int& ExternalOwner::state() const { return storage_; }
int main() { return 0; }
"""
    namespace_free_function = """
namespace utility {
inline int storage = 7;
const int& state();
}
inline const int& utility::state() { return storage; }
int main() { return 0; }
"""
    template_owner = """
template<class T>
/** Threading: stream-owner reference readout; getPublished() is atomic. */
struct Owner {
public:
    /** stream-owner reference readout matched */
    const T& state() const;
    int getPublished() const noexcept { return 0; }
private:
    T storage_ {};
};
template<class T>
inline const T& Owner<T>::state() const { return storage_; }
int main() { Owner<int> owner; return owner.state(); }
""".replace("stream-owner reference readout", marker)
    dependent_base = """
template<class T>
struct Base {{ protected: int storage_ = 7; }};
template<class T>
/** Threading: {marker}; getPublished() is atomic. */
struct DependentOwner : Base<T> {{
public:
    /** {marker} matched */
    const int& state() const {{ return this->storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
int main() {{ DependentOwner<int> owner; return owner.state(); }}
""".format(marker=marker)
    qualified_dependent_base = """
template<class T>
struct Base { protected: int storage_ = 7; };
template<class T>
struct Owner : Base<T> {
public:
    const int& state() const { return Base<T>::storage_; }
};
int main() { Owner<int> owner; return owner.state(); }
"""
    qualified_dependent_subobject = """
template<class T>
struct Base {
protected:
    struct Cell { int value = 7; };
    Cell storage_ {};
};
template<class T>
struct Owner : Base<T> {
public:
    const int& state() const { return Base<T>::storage_.value; }
};
int main() { Owner<int> owner; return owner.state(); }
"""
    qualified_dependent_get_subobject = """
#include <array>
template<class T>
struct Base { protected: std::array<int, 1> storage_ {{7}}; };
template<class T>
struct Owner : Base<T> {
public:
    const int& state() const
    {
        return std::get<0>(Base<T>::storage_);
    }
};
int main() { Owner<int> owner; return owner.state(); }
"""
    qualified_dependent_value_control = """
template<class T>
struct Base { protected: int storage_ = 7; };
template<class T>
struct Owner : Base<T> {
public:
    int state() const { return Base<T>::storage_; }
};
int main() { Owner<int> owner; return owner.state(); }
"""
    qualified_dependent_call_control = """
template<class T>
struct Base {
protected:
    const int& storage() const { return storage_; }
    int storage_ = 7;
};
template<class T>
struct Owner : Base<T> {
public:
    const int& state() const { return Base<T>::storage(); }
};
int main() { Owner<int> owner; return owner.state(); }
"""
    qualified_dependent_free_control = """
namespace utility { inline int storage = 7; }
template<class T>
struct Base {};
template<class T>
struct Owner : Base<T> {
public:
    const int& state() const { return utility::storage; }
};
int main() { Owner<int> owner; return owner.state(); }
"""
    qualified_dependent_private_control = """
template<class T>
struct Base { protected: int storage_ = 7; };
template<class T>
struct Owner : Base<T> {
private:
    const int& state() const { return Base<T>::storage_; }
};
int main() { Owner<int> owner; (void)owner; return 0; }
"""
    ambiguous_base_alias = """
struct LeftBase {{ protected: int wrong_storage_ = 1; }};
struct RightBase {{ protected: int storage_ = 7; }};
#if defined(DSPARK_USE_LEFT_BASE)
using Parent = LeftBase;
#else
using Parent = RightBase;
#endif
/** Threading: {marker}; getPublished() is atomic. */
struct AmbiguousOwner : Parent {{
public:
    /** {marker} matched */
    const int& state() const {{ return this->storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
int main() {{ AmbiguousOwner owner; return owner.state(); }}
""".format(marker=marker)
    template_value_control = """
template<class T>
struct TemplateValueOwner { T state() const; };
template<class T>
inline T TemplateValueOwner<T>::state() const { return T {}; }
int main() { return TemplateValueOwner<int> {}.state(); }
"""
    private_template_control = """
template<class T>
struct PrivateTemplateOwner {
private:
    const T& state() const;
    T storage_ {};
};
template<class T>
inline const T& PrivateTemplateOwner<T>::state() const { return storage_; }
int main() { return 0; }
"""
    private_template_overload_control = """
template<class T>
struct Owner {
public:
    const T& state(int) const;
private:
    const T& state() const;
    T storage_ {};
};
template<class T>
inline const T& Owner<T>::state() const { return storage_; }
int main() { Owner<int> owner; (void)owner; return 0; }
"""
    public_template_overload = """
template<class T>
struct Owner {
private:
    const T& state(int) const;
public:
    const T& state() const;
    T storage_ {};
};
template<class T>
inline const T& Owner<T>::state() const { return storage_; }
int main() { Owner<int> owner; return owner.state(); }
"""
    cyclic_base_alias = """
struct Base {{ protected: int storage_ = 7; }};
#if 0
using First = Second;
using Second = First;
#endif
#define First Base
/** Threading: {marker}; getPublished() is atomic. */
struct CyclicOwner : First {{
public:
    /** {marker} matched */
    const int& state() const {{ return this->storage_; }}
    int getPublished() const noexcept {{ return 0; }}
}};
int main() {{ CyclicOwner owner; return owner.state(); }}
""".format(marker=marker)
    conflicting_signature_alias = """
#if defined(DSPARK_USE_DOUBLE_PARAM)
using Param = double;
#else
using Param = int;
#endif
struct Owner {
public:
    const int& state(Param& value) const;
private:
    int storage_ = 7;
};
inline const int& Owner::state(int& value) const
{
    (void)value;
    return storage_;
}
int main() { Owner owner; int value = 0; return owner.state(value); }
"""
    conflicting_namespace_alias = """
namespace left { struct Base { protected: int storage_left_ = 7; }; }
namespace right { struct Base { protected: int storage_right_ = 9; }; }
#if defined(DSPARK_USE_LEFT_BASE)
namespace facade = left;
#else
namespace facade = right;
#endif
struct Owner : facade::Base {
public:
    const int& state() const { return storage_right_; }
};
int main() { Owner owner; return owner.state(); }
"""
    conflicting_signature_class_alias = """
struct LeftPayload {};
struct RightPayload {};
#if defined(DSPARK_USE_LEFT_PAYLOAD)
using Param = LeftPayload;
#else
using Param = RightPayload;
#endif
struct Owner {
public:
    const int& state(const Param& value) const;
private:
    int storage_ = 7;
};
inline const int& Owner::state(const RightPayload& value) const
{
    (void)value;
    return storage_;
}
int main() { Owner owner; RightPayload value; return owner.state(value); }
"""
    return (
        ("unsupported-const-reference-declarator", unsupported, 0, 1),
        ("unsupported-parenthesized-const-reference-declarator",
         parenthesized, 0, 1),
        ("unsupported-subscript-operator-declarator",
         subscript_operator, 0, 1),
        ("unsupported-call-operator-declarator",
         call_operator, 0, 1),
        ("unresolved-qualified-class-owner", unresolved_owner, 0, 1),
        ("class-template-owner-definition", template_owner, 0, 1),
        ("dependent-base-member", dependent_base, 0, 1),
        ("qualified-dependent-base-member", qualified_dependent_base, 0, 1),
        ("qualified-dependent-base-subobject",
         qualified_dependent_subobject, 0, 1),
        ("qualified-dependent-base-get-subobject",
         qualified_dependent_get_subobject, 0, 1),
        ("ambiguous-base-alias", ambiguous_base_alias, 0, 1),
        ("cyclic-base-alias", cyclic_base_alias, 0, 1),
        ("conflicting-signature-alias",
         conflicting_signature_alias, 0, 1),
        ("conflicting-namespace-alias",
         conflicting_namespace_alias, 0, 1),
        ("conflicting-signature-class-alias",
         conflicting_signature_class_alias, 0, 1),
        ("ordinary-value-definition-control", value_control, 0, 0),
        ("qualified-namespace-free-function-control",
         namespace_free_function, 0, 0),
        ("class-template-value-control", template_value_control, 0, 0),
        ("private-class-template-control", private_template_control, 0, 0),
        ("class-template-private-target-public-overload-control",
         private_template_overload_control, 0, 0),
        ("class-template-public-target-private-overload",
         public_template_overload, 0, 1),
        ("qualified-dependent-base-value-control",
         qualified_dependent_value_control, 0, 0),
        ("qualified-dependent-base-call-control",
         qualified_dependent_call_control, 0, 0),
        ("qualified-dependent-base-free-control",
         qualified_dependent_free_control, 0, 0),
        ("qualified-dependent-base-private-control",
         qualified_dependent_private_control, 0, 0),
    )


def fail_closed_diagnostic_expectations():
    """Exact metadata for the fail-closed paths added to the policy matrix."""
    return {
        "class-template-owner-definition": (
            12, "class-template owner Owner<...>"),
        "class-template-public-target-private-overload": (
            10, "class-template owner Owner<...>"),
        "dependent-base-member": (
            9, "unsupported or unresolved base"),
        "qualified-dependent-base-member": (
            7, "unsupported or unresolved base"),
        "qualified-dependent-base-subobject": (
            11, "unsupported or unresolved base"),
        "qualified-dependent-base-get-subobject": (
            8, "unsupported or unresolved base"),
        "ambiguous-base-alias": (
            13, "unsupported or unresolved base"),
        "cyclic-base-alias": (
            12, "unsupported or unresolved base"),
        "conflicting-signature-alias": (
            13, "matched 0 declarations"),
        "conflicting-namespace-alias": (
            11, "unsupported or unresolved base"),
        "conflicting-signature-class-alias": (
            15, "matched 0 declarations"),
    }


def mutation_matrix_cases(marker):
    """Return the accepted and negative sources used by the production gate."""
    accepted = [
        ("explicit", "const Storage& explicitAccessor() const noexcept", "return storage_;"),
        ("trailing", "auto trailingAccessor() const -> const Storage&", "return storage_;"),
        ("alias", "StorageRef aliasAccessor() const", "return storage_;"),
        ("multiline", "const\nStorage\n&\nmultilineAccessor\n(\n) const", "return storage_;"),
        ("parenthesized", "const Storage& (parenthesizedAccessor)() const", "return storage_;"),
        ("attributes-qualifiers",
         "[[nodiscard]] constexpr const Storage& qualifiedAccessor() const & noexcept",
         "return storage_;"),
        ("this-member", "const Storage& thisAccessor() const", "return this->storage_;"),
        ("this-member-parameter-shadow",
         "const Storage& thisParameterAccessor(const Storage& storage_) const",
         "return this->storage_;"),
        ("this-member-local-shadow",
         "const Storage& thisLocalAccessor() const",
         "static const Storage storage_ {}; return this->storage_;"),
        ("member-subobject", "const float& subobjectAccessor() const", "return storage_[0];"),
        ("operator-subscript", "const float& operator[](unsigned i) const",
         "return storage_[i];"),
        ("operator-call", "const float& operator()(unsigned i) const",
         "return storage_[i];"),
    ]
    near_misses = [
        ("value", "Storage valueAccessor() const", "return storage_;"),
        ("deduced-value", "auto deducedValueAccessor() const", "return storage_;"),
        ("mutable-reference", "Storage& mutableAccessor()", "return storage_;"),
        ("pointer", "const Storage* pointerAccessor() const", "return &storage_;"),
        ("temporary", "const Storage& temporaryAccessor() const", "return makeStorage();"),
        ("suffix-parameter",
         "const Storage& parameterAccessor(const Storage& value_) const",
         "return value_;"),
        ("suffix-local", "const Storage& localAccessor() const",
         "static const Storage value_ {}; return value_;"),
        ("member-name-parameter-shadow",
         "const Storage& memberNameParameterAccessor(const Storage& storage_) const",
         "return storage_;"),
        ("member-name-local-shadow",
         "const Storage& memberNameLocalAccessor() const",
         "static const Storage storage_ {}; return storage_;"),
    ]

    def source_for(signature, body):
        return """
/** Threading: the foreign readout is atomic; this is a %s. */
struct MatrixCase {
public:
    using Storage = std::array<float, 4>;
    using StorageRef = const Storage&;
    /** %s */
    %s { %s }
    int getPublished() const noexcept { return 0; }
private:
    static Storage makeStorage();
    Storage storage_ {};
};
""" % (marker, marker, signature, body)

    def overload_source(owner, sibling_declaration, matched_declaration,
                        definition):
        return """
/** Threading: the foreign readout is atomic; this is a %s. */
struct %s {
    /** %s sibling overload */
    %s
    /** %s */
    %s
    int getPublished() const { return 0; }
private:
    int storage_ = 0;
};
%s
{
    return storage_;
}
""" % (marker, owner, marker, sibling_declaration, marker,
       matched_declaration, definition)

    accepted_sources = [(label, source_for(signature, body))
                        for label, signature, body in accepted]
    accepted_sources.append((
        "class-attribute",
        source_for("const Storage& classAttributeAccessor() const",
                   "return storage_;").replace(
                       "struct MatrixCase", "class [[nodiscard]] MatrixCase", 1)))
    accepted_sources.append((
        "member-without-suffix",
        source_for("const Storage& unsuffixedAccessor() const",
                   "return storage;").replace(
                       "Storage storage_ {};", "Storage storage {};", 1)))
    accepted_sources.extend([
        ("reopened-namespace-alias", """
namespace reopened { using ResultRef = const int&; }
namespace reopened {
/** Threading: the foreign readout is atomic; this is a %s. */
struct ReopenedOwner {
    /** %s */
    ResultRef reopenedAccessor() const { return storage_; }
    int getPublished() const { return 0; }
private:
    int storage_ = 0;
};
}
""" % (marker, marker)),
        ("qualified-namespace-alias", """
namespace result_types { using ResultRef = const int&; }
/** Threading: the foreign readout is atomic; this is a %s. */
struct QualifiedAliasOwner {
    /** %s */
    result_types::ResultRef qualifiedAliasAccessor() const { return storage_; }
    int getPublished() const { return 0; }
private:
    int storage_ = 0;
};
""" % (marker, marker)),
        ("nested-reopened-namespace-alias", """
namespace outer { namespace inner { using ResultRef = const int&; } }
namespace outer::inner {
/** Threading: the foreign readout is atomic; this is a %s. */
struct NestedReopenedOwner {
    /** %s */
    ResultRef nestedReopenedAccessor() const { return storage_; }
    int getPublished() const { return 0; }
private:
    int storage_ = 0;
};
}
""" % (marker, marker)),
        ("out-of-class-inline-definition", """
/** Threading: the foreign readout is atomic; this is a %s. */
struct OutOfClassOwner {
    /** %s */
    const int& outOfClassAccessor() const noexcept;
    int getPublished() const { return 0; }
private:
    int storage_ = 0;
};
inline const int& OutOfClassOwner::outOfClassAccessor() const noexcept
{
    return storage_;
}
""" % (marker, marker)),
        ("out-of-class-arity-overload", overload_source(
            "ArityOverloadOwner",
            "const int& state(int sibling) const;",
            "const int& state() const;",
            "inline const int& ArityOverloadOwner::state() const")),
        ("out-of-class-type-default-name-overload", overload_source(
            "TypeOverloadOwner",
            "const int& state(int sibling) const;",
            "const int& state(double declared = 0.0) const;",
            "inline const int& TypeOverloadOwner::state(double defined) const")),
        ("out-of-class-ref-qualifier-overload", overload_source(
            "RefOverloadOwner",
            "const int& state() const &;",
            "const int& state() const &&;",
            "inline const int& RefOverloadOwner::state() const &&")),
        ("out-of-class-cv-qualifier-overload", overload_source(
            "CvOverloadOwner",
            "const volatile int& state() const;",
            "const volatile int& state() const volatile;",
            "inline const volatile int& CvOverloadOwner::state() const volatile")),
        ("inherited-member", """
struct StorageBase { protected: int storage_ = 0; };
/** Threading: the foreign readout is atomic; this is a %s. */
struct InheritedOwner : StorageBase {
    /** %s */
    const int& inheritedAccessor() const { return storage_; }
    int getPublished() const { return 0; }
};
""" % (marker, marker)),
    ])
    near_miss_sources = [(label, source_for(signature, body))
                         for label, signature, body in near_misses]
    near_miss_sources.extend([
        ("out-of-class-value-return-overload", """
/** Threading: the foreign readout is atomic; this is a %s. */
struct ValueReturnOverloadOwner {
    /** %s sibling overload */
    const int& state(int sibling) const;
    int state() const;
    int getPublished() const { return 0; }
private:
    int storage_ = 0;
};
inline int ValueReturnOverloadOwner::state() const
{
    return storage_;
}
""" % (marker, marker)),
        ("out-of-class-ambiguous-exact-declarations", """
/** Threading: the foreign readout is atomic; this is a %s. */
struct AmbiguousOverloadOwner {
    /** %s first declaration */
    const int& state() const;
    /** %s second declaration */
    const int& state() const;
    int getPublished() const { return 0; }
private:
    int storage_ = 0;
};
inline const int& AmbiguousOverloadOwner::state() const
{
    return storage_;
}
""" % (marker, marker, marker)),
    ])
    return accepted_sources, near_miss_sources


def overload_association_cases(marker):
    """Return exact metadata expectations for out-of-class overloads."""
    return (
        (
            "unoverloaded-control",
            """
/** Threading: the publication is atomic; this is a %s. */
struct Owner {
  /** %s: the reference is stream-owned. */
  const int& state() const;
private:
  int storage_ = 0;
};
inline const int& Owner::state() const { return storage_; }
""" % (marker, marker),
            1, True, 5,
        ),
        (
            "arity-overload-does-not-borrow-marker",
            """
/** Threading: the publication is atomic; this is a %s. */
struct Owner {
  /** %s: marked overload. */
  const int& state(int) const;
  /** Ordinary unmarked overload. */
  const int& state() const;
private:
  int storage_ = 0;
};
inline const int& Owner::state() const { return storage_; }
""" % (marker, marker),
            1, False, 7,
        ),
        (
            "type-overload-does-not-borrow-marker",
            """
/** Threading: the publication is atomic; this is a %s. */
struct Owner {
  /** %s: marked overload. */
  const int& state(int) const;
  /** Ordinary unmarked overload. */
  const int& state(double) const;
private:
  int storage_ = 0;
};
inline const int& Owner::state(double) const { return storage_; }
""" % (marker, marker),
            1, False, 7,
        ),
        (
            "ref-qualifier-overload-does-not-borrow-marker",
            """
/** Threading: the publication is atomic; this is a %s. */
struct Owner {
  /** %s: marked overload. */
  const int& state() const &;
  /** Ordinary unmarked overload. */
  const int& state() const &&;
private:
  int storage_ = 0;
};
inline const int& Owner::state() const && { return storage_; }
""" % (marker, marker),
            1, False, 7,
        ),
    )


def run_mutation_matrix(marker):
    """Exercise every accepted declaration axis and non-candidate near miss."""
    accepted, near_misses = mutation_matrix_cases(marker)
    diagnostics = []
    for label, source in accepted:
        found = find_public_const_reference_accessors(source)
        if len(found) != 1 or marker not in found[0].documentation:
            raise AssertionError("accepted spelling was not classified: " + label)
        deleted = source.replace("/** {} */".format(marker),
                                 "/** same-thread reference */", 1)
        deleted_found = find_public_const_reference_accessors(deleted)
        if len(deleted_found) != 1 or marker in deleted_found[0].documentation:
            raise AssertionError("marker deletion did not become a failure: " + label)
        diagnostics.append("{}: discovered, marked, deletion rejected".format(label))

    for label, source in near_misses:
        if find_public_const_reference_accessors(source):
            raise AssertionError("near miss entered the census: " + label)
        diagnostics.append("{}: excluded".format(label))
    return diagnostics

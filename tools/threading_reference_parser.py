#!/usr/bin/env python3
"""Dependency-free C++ declaration parser for threading reference readouts.

The parser is intentionally focused: it finds public member functions whose
declared return type is a const lvalue reference and whose every return
statement directly names accessible member storage or one of its subobjects.
Definitions may be in class or inline at namespace scope. It tokenizes C++
rather than matching one source spelling, resolves named namespace/class alias
scope and binds parameters and block locals before falling back to a member.
"""

from dataclasses import dataclass
import re


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


@dataclass(frozen=True)
class NamespaceRegion:
    opening: int
    closing: int
    declaration: int
    path: tuple


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
class AliasResolutionEnvironment:
    type_aliases: dict
    namespace_aliases: dict
    known_namespaces: frozenset
    namespace_alias_cache: dict
    classes_by_identity: dict
    class_identities: dict


@dataclass(frozen=True)
class AliasContext:
    environment: AliasResolutionEnvironment
    current_scope: tuple = ()
    use_position: int = 0


@dataclass(frozen=True)
class NamespaceAliasTarget:
    scope: tuple
    absolute: bool
    parts: tuple
    declaration: int


@dataclass(frozen=True)
class TypeAliasTarget:
    scope: tuple
    rhs: tuple
    declaration: int


@dataclass(frozen=True)
class IdentityAssociationCase:
    label: str
    source: str
    expected_count: int
    expected_marker: bool
    expected_line: int
    expected_access: str
    deletion_anchor: str = ""


_MULTI_PUNCTUATION = (
    "<=>", "->*", "...", "::", "->", "&&", "[[", "]]", "<=", ">=",
    "==", "!=", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=",
    "|=", "^=", "<<", ">>", "##", ".*",
)

_NON_FUNCTION_NAMES = {
    "alignof", "catch", "decltype", "for", "if", "noexcept", "requires",
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
    "typename", "volatile",
}


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


def _class_regions(tokens, brace_pairs):
    regions = []
    for index, token in enumerate(tokens):
        if token.text not in ("class", "struct"):
            continue
        if index > 0 and tokens[index - 1].text == "enum":
            continue
        name_index = index + 1
        while name_index < len(tokens):
            if tokens[name_index].text == "[[":
                name_index += 1
                depth = 1
                while name_index < len(tokens) and depth:
                    depth += (tokens[name_index].text == "[[")
                    depth -= (tokens[name_index].text == "]]" )
                    name_index += 1
                continue
            if (tokens[name_index].text == "alignas" and name_index + 1 < len(tokens)
                    and tokens[name_index + 1].text == "("):
                closing = _pair_map(tokens, "(", ")").get(name_index + 1)
                name_index = len(tokens) if closing is None else closing + 1
                continue
            break
        if name_index >= len(tokens) or not _is_identifier_start(tokens[name_index].text[0]):
            continue
        after_name = name_index + 1
        if after_name >= len(tokens) or tokens[after_name].text in (">", ",", "="):
            continue
        cursor = after_name
        while cursor < len(tokens) and tokens[cursor].text not in ("{", ";"):
            cursor += 1
        if cursor >= len(tokens) or tokens[cursor].text != "{":
            continue
        closing = brace_pairs.get(cursor)
        if closing is not None:
            regions.append(ClassRegion(
                opening=cursor,
                closing=closing,
                default_access=("public" if token.text == "struct"
                                else "private"),
                declaration=index,
                name_index=name_index,
                name=tokens[name_index].text,
                base_tokens=tuple(tokens[after_name:cursor]),
            ))
    return regions


def _namespace_regions(tokens, brace_pairs):
    """Return namespace bodies keyed by their qualified namespace identity.

    Separate definitions of one named namespace share one path, while sibling
    and anonymous namespaces retain independent scopes. Nested namespace
    syntax (``namespace outer::inner``) and ordinary nesting map to the same
    identity.
    """
    raw_regions = []
    for index, token in enumerate(tokens):
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
            names = tuple(
                part.text for part in tokens[index + 1:cursor]
                if part.text != "inline"
                and part.text not in ("[[", "]]", "::")
                and _is_identifier_start(part.text[0]))
            raw_regions.append((cursor, closing, index, names))

    regions = []
    for opening, closing, declaration, names in sorted(raw_regions):
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
        ))
    return regions


def _direct_scope_tokens(tokens, opening, closing, brace_pairs):
    """Tokens declared directly in one lexical scope.

    Every nested brace region is skipped.  For a class this excludes method and
    nested-class bodies; for a namespace it excludes classes, functions and
    child namespaces.  That is the crucial distinction between an alias owned
    by this scope and a same-spelled alias owned by a descendant.
    """
    output = []
    index = 0 if opening < 0 else opening + 1
    while index < closing:
        if tokens[index].text == "{":
            body_close = brace_pairs.get(index)
            if body_close is not None and body_close <= closing:
                index = body_close + 1
                continue
        output.append(tokens[index])
        index += 1
    return output


def _region_key(kind, opening, closing):
    return kind, opening, closing


def _type_alias_declarations(tokens, scope):
    """Collect direct type aliases with their lexical declaration points."""
    declarations = []
    index = 0
    while index < len(tokens):
        if tokens[index].text == "using" and index + 2 < len(tokens):
            name = tokens[index + 1].text
            if _is_identifier_start(name[0]) and tokens[index + 2].text == "=":
                end = index + 3
                parens = brackets = braces = 0
                while end < len(tokens):
                    value = tokens[end].text
                    parens += (value == "(") - (value == ")")
                    brackets += (value == "[") - (value == "]")
                    braces += (value == "{") - (value == "}")
                    if value == ";" and parens == brackets == braces == 0:
                        break
                    end += 1
                rhs = tuple(token.text for token in tokens[index + 3:end])
                declarations.append((name, TypeAliasTarget(
                    tuple(scope), rhs, tokens[index].start)))
                index = end
        elif tokens[index].text == "typedef":
            end = index + 1
            while end < len(tokens) and tokens[end].text != ";":
                end += 1
            identifiers = [i for i in range(index + 1, end)
                           if _is_identifier_start(tokens[i].text[0])]
            if identifiers:
                name_index = identifiers[-1]
                name = tokens[name_index].text
                rhs = tuple(token.text for token in tokens[index + 1:name_index])
                declarations.append((name, TypeAliasTarget(
                    tuple(scope), rhs, tokens[index].start)))
                index = end
        index += 1
    return tuple(declarations)


def _namespace_alias_declarations(tokens, scope):
    """Collect direct namespace aliases with their declaration points."""
    declarations = []
    index = 0
    while index + 3 < len(tokens):
        if (tokens[index].text != "namespace"
                or not _is_identifier_start(tokens[index + 1].text[0])
                or tokens[index + 2].text != "="):
            index += 1
            continue
        name = tokens[index + 1].text
        end = index + 3
        while end < len(tokens) and tokens[end].text != ";":
            end += 1
        values = [token.text for token in tokens[index + 3:end]]
        absolute = bool(values and values[0] == "::")
        if absolute:
            values = values[1:]
        parts = values[::2]
        separators = values[1::2]
        valid = (bool(parts)
                 and all(part and _is_identifier_start(part[0])
                         for part in parts)
                 and all(separator == "::" for separator in separators)
                 and len(values) == len(parts) + len(separators))
        if valid:
            declarations.append((name, NamespaceAliasTarget(
                tuple(scope), absolute, tuple(parts), tokens[index].start)))
        index = end + 1
    return tuple(declarations)


def _visible_alias_targets(targets, use_position):
    """Return declarations visible at one lexical use point, in order."""
    return tuple(target for target in targets
                 if target.declaration < use_position)


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
                           classes_by_identity):
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
        return classes_by_identity.get(namespace + tuple(parts[consumed:]))
    if absolute:
        return classes_by_identity.get(tuple(parts))
    return _resolve_class(parts, scope, classes_by_identity)


def _resolve_aliased_namespace(parts, absolute, scope, namespace_aliases,
                               use_position, known_namespaces, alias_cache):
    """Resolve a qualified owner that may be a namespace, not a class."""
    match = _namespace_alias_prefix(
        parts, scope, namespace_aliases, use_position, absolute=absolute)
    if match is not None:
        key, consumed = match
        namespace = _resolve_namespace_alias(
            key, use_position, namespace_aliases, known_namespaces,
            alias_cache)
        candidate = None if namespace is None else (
            namespace + tuple(parts[consumed:]))
        return candidate if candidate in known_namespaces else None
    if absolute:
        candidate = tuple(parts)
        return candidate if candidate in known_namespaces else None
    for length in range(len(scope), -1, -1):
        candidate = tuple(scope[:length]) + tuple(parts)
        if candidate in known_namespaces:
            return candidate
    return None


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
    key, targets = _type_alias_key(
        parts, absolute, aliases.current_scope,
        aliases.environment.type_aliases,
        aliases.use_position)
    if key is None:
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
    owner = _resolve_aliased_class(
        parts, absolute, aliases.current_scope,
        environment.namespace_aliases, aliases.use_position,
        environment.known_namespaces, environment.namespace_alias_cache,
        environment.classes_by_identity)
    if owner is None:
        return canonical
    identity = environment.class_identities.get(owner)
    if identity is None:
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
        values = _strip_attributes(
            [token.text for token in default_parts[0]])
        if not values:
            return None
        name_index = _ordinary_parameter_name_index(values)
        if name_index is None:
            return None
        if name_index >= 0:
            del values[name_index]
        canonical = _canonical_type_tokens(values, aliases, parameter=True)
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


def _function_name_candidate(signature, parameter_open, paren_pairs):
    """Return a supported function name and its declarator start."""
    previous = parameter_open - 1
    if previous < 0:
        return None

    name = None
    name_start = previous
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
            inside = [
                part.text for part in signature[left + 1:previous]
                if part.text not in ("[[", "]]")
            ]
            if (left > 0 and signature[left - 1].text == "operator"
                    and not inside):
                name = "operator()"
                name_start = left - 1
            elif (len(inside) == 1
                  and _is_identifier_start(inside[0][0])):
                name = inside[0]
                name_start = left
            elif (len(inside) >= 3 and inside[-2] == "::"
                  and _is_identifier_start(inside[-1][0])):
                name = inside[-1]
                name_start = previous - 1
    elif (previous >= 2
          and signature[previous - 2].text == "operator"
          and signature[previous - 1].text == "["
          and signature[previous].text == "]"):
        name = "operator[]"
        name_start = previous - 2
    return None if name is None else (name, name_start)


def _function_signature_details(signature, aliases):
    """Return the supported function-declaration details or ``None``."""
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
            candidate = _function_name_candidate(
                signature, index, local_pairs)
            if candidate is not None:
                name, name_start = candidate
                closing = local_pairs.get(index)
                if closing is not None:
                    candidates.append((name, name_start, index, closing))
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
                              if candidate[3] < initializer_colon]
        if before_initializer:
            selected = before_initializer[-1]

    name, name_start, parameter_open, parameter_close = selected
    trailing = None
    qualifier_depth = 0
    for index in range(parameter_close + 1, len(signature)):
        value = signature[index].text
        qualifier_depth += (value == "(") - (value == ")")
        if value == "->" and qualifier_depth == 0:
            trailing = index
            break
    if trailing is not None:
        return_values = [token.text for token in signature[trailing + 1:]]
        if "requires" in return_values:
            return_values = return_values[:return_values.index("requires")]
    else:
        owner_start = _qualified_owner_start(signature, name_start)
        return_values = [token.text for token in signature[:owner_start]]
    return_values = _strip_leading_template_headers(return_values)
    parameter_types = _canonical_parameter_types(
        signature, parameter_open, parameter_close, aliases)
    if parameter_types is None:
        return None
    member_const, member_volatile, ref_qualifier = _member_qualifiers(
        signature, parameter_close)
    return FunctionSignature(
        name=name,
        const_reference=_is_const_lvalue_reference(return_values, aliases),
        return_type=_canonical_type_tokens(return_values, aliases),
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
            candidate = _function_name_candidate(signature, index, pairs)
            if candidate is None:
                depth += 1
                continue
            name, name_start = candidate
            qualified = (name_start > 0
                         and signature[name_start - 1].text == "::")
            if require_qualified and not qualified:
                depth += 1
                continue
            owner_start = _qualified_owner_start(signature, name_start)
            return_values = [part.text for part in signature[:owner_start]]
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
        elif values[cursor] in (".", "->"):
            cursor += 1
            if (cursor >= len(values)
                    or not _is_identifier_start(values[cursor][0])):
                return False
            cursor += 1
        else:
            return False
    return True


def _std_get_argument(values):
    """Return the sole argument of the supported ``std::get<I>`` form."""
    if len(values) < 7 or values[:3] not in (
            ["std", "::", "get"], ["get", "<", "0"]):
        return None
    get_index = 2 if values[:3] == ["std", "::", "get"] else 0
    cursor = get_index + 1
    if cursor >= len(values) or values[cursor] != "<":
        return None
    angle = 1
    cursor += 1
    while cursor < len(values) and angle:
        angle = _angle_depth_change(values[cursor], angle)
        cursor += 1
    if (angle or cursor >= len(values) or values[cursor] != "("
            or values[-1] != ")"):
        return None
    return values[cursor + 1:-1]


def _direct_member_expression(values, member_names, shadow_names=()):
    values = _strip_outer_parentheses(values)
    if not values:
        return False

    # std::get<I>(member_) is a direct member subobject, not an arbitrary call.
    get_argument = _std_get_argument(values)
    if get_argument is not None:
        return _direct_member_expression(
            get_argument, member_names, shadow_names)

    cursor = 0
    explicit_this = False
    if values[:2] == ["this", "->"]:
        cursor = 2
        explicit_this = True
    if cursor >= len(values) or not _is_identifier_start(values[cursor][0]):
        return False
    root = values[cursor]
    if root not in member_names or (not explicit_this and root in shadow_names):
        return False
    return _member_subobject_suffix(values, cursor + 1)


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


def _return_expressions(body):
    expressions = []
    index = 0
    while index < len(body):
        if body[index].text != "return":
            index += 1
            continue
        cursor = index + 1
        parens = brackets = braces = 0
        while cursor < len(body):
            value = body[cursor].text
            parens += (value == "(") - (value == ")")
            brackets += (value == "[") - (value == "]")
            braces += (value == "{") - (value == "}")
            if value == ";" and parens == brackets == braces == 0:
                break
            cursor += 1
        expressions.append([token.text for token in body[index + 1:cursor]])
        index = cursor + 1
    return expressions


def _parameter_names(signature, details, aliases):
    names = set()
    parameters = signature[details.parameters_open + 1:
                           details.parameters_close]
    for parameter in _top_level_parts(parameters, ","):
        names.update(_declared_data_names(parameter, aliases))
    return names


def _looks_like_block_open(tokens, index, brace_kinds):
    if any(not is_block for is_block in brace_kinds):
        return False
    previous = tokens[index - 1].text if index > 0 else ""
    if previous in (")", "]", "else", "try", "do", ";", "{", "}"):
        return True
    if previous in ("=", ",", "(", "["):
        return False
    if previous and (_is_identifier_start(previous[0]) or previous[0].isdigit()
                     or previous in (">", ">>", "<literal>")):
        return False
    return True


def _return_expressions_with_bindings(body, parameter_names, aliases):
    """Return each expression with parameters and visible block locals.

    This is the binding distinction the policy needs: an unqualified spelling
    names the nearest parameter/local first, while ``this->`` bypasses that
    lookup. The supported local subset is ordinary semicolon-terminated data
    declarations in lexical blocks; macro-generated and lambda bodies are not
    interpreted.
    """
    expressions = []
    scopes = [set(parameter_names)]
    brace_kinds = []
    statement_start = 0
    index = 0
    while index < len(body):
        value = body[index].text
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
            shadows = set().union(*scopes)
            expressions.append((
                [token.text for token in body[index + 1:cursor]], shadows))
            statement_start = cursor + 1
            index = cursor + 1
            continue
        if value == "{":
            is_block = _looks_like_block_open(body, index, brace_kinds)
            brace_kinds.append(is_block)
            if is_block:
                scopes.append(set())
                statement_start = index + 1
        elif value == "}" and brace_kinds:
            is_block = brace_kinds.pop()
            if is_block:
                if len(scopes) > 1:
                    scopes.pop()
                statement_start = index + 1
        elif value == ";" and not any(not kind for kind in brace_kinds):
            statement = body[statement_start:index]
            first = statement[0].text if statement else ""
            if first not in {
                    "break", "co_return", "continue", "goto", "if",
                    "return", "static_assert", "switch", "while"}:
                scopes[-1].update(_declared_data_names(statement, aliases))
            statement_start = index + 1
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
                            default_access):
    names = set()
    inherited_accessible = set()
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
    return names, inherited_accessible


def _containing_namespace(region, namespace_regions):
    candidates = [namespace for namespace in namespace_regions
                  if namespace.opening < region.opening
                  and region.closing < namespace.closing]
    if not candidates:
        return ()
    return min(candidates,
               key=lambda namespace: namespace.closing - namespace.opening).path


def _class_ancestors(region, class_regions):
    return sorted(
        (candidate for candidate in class_regions
         if candidate.opening < region.opening
         and region.closing < candidate.closing),
        key=lambda candidate: candidate.opening)


def _class_identity(region, class_regions, namespace_regions):
    namespace = _containing_namespace(region, namespace_regions)
    owners = tuple(ancestor.name
                   for ancestor in _class_ancestors(region, class_regions))
    return namespace + owners + (region.name,)


def _context_for_class(region, class_regions, namespace_regions,
                       environment, use_position):
    return AliasContext(
        environment,
        _class_identity(region, class_regions, namespace_regions),
        use_position,
    )


def _context_for_namespace(path, environment, use_position):
    return AliasContext(environment, tuple(path), use_position)


def _alias_context_at(context, use_position):
    return AliasContext(
        context.environment, context.current_scope, use_position)


def _base_type_values(base_tokens):
    """Return each base type spelling without access/virtual specifiers."""
    values = list(base_tokens)
    if not values or ":" not in [token.text for token in values]:
        return []
    colon = next(index for index, token in enumerate(values)
                 if token.text == ":")
    output = []
    for base in _top_level_parts(values[colon + 1:], ","):
        filtered = _strip_attributes([token.text for token in base])
        filtered = [value for value in filtered if value not in {
            "public", "protected", "private", "virtual"}]
        if filtered:
            output.append(tuple(filtered))
    return output


def _resolve_class(parts, scope, classes_by_identity):
    for length in range(len(scope), -1, -1):
        candidate = scope[:length] + tuple(parts)
        if candidate in classes_by_identity:
            return classes_by_identity[candidate]
    return None


def _simple_qualified_type(values):
    """Parse a non-dependent named type, retaining absolute qualification."""
    values = list(values)
    while values and values[0] in ("class", "struct"):
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


def _type_alias_key(parts, absolute, scope, aliases, use_position):
    scope_lengths = (0,) if absolute else range(len(scope), -1, -1)
    for length in scope_lengths:
        key = tuple(scope[:length]) + tuple(parts)
        targets = _visible_alias_targets(
            aliases.get(key, ()), use_position)
        if targets:
            return key, targets
    return None, ()


def _resolve_type_alias_class(key, target, type_aliases, namespace_aliases,
                              known_namespaces,
                              namespace_alias_cache, classes_by_identity,
                              cache, active=()):
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
    nested_key, nested_targets = _type_alias_key(
        parts, absolute, target.scope, type_aliases, target.declaration)
    if nested_key is not None:
        resolutions = [
            _resolve_type_alias_class(
                nested_key, nested_target, type_aliases, namespace_aliases,
                known_namespaces, namespace_alias_cache,
                classes_by_identity, cache, active + (cache_key,))
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
            classes_by_identity)
    cache[cache_key] = resolved
    return resolved


def _resolve_base_class(values, scope, use_position, type_aliases,
                        namespace_aliases,
                        known_namespaces, namespace_alias_cache,
                        classes_by_identity, type_alias_cache):
    parsed = _simple_qualified_type(values)
    if parsed is None:
        return None
    parts, absolute = parsed
    alias_key, alias_targets = _type_alias_key(
        parts, absolute, scope, type_aliases, use_position)
    if alias_key is not None:
        resolutions = [
            _resolve_type_alias_class(
                alias_key, alias_target, type_aliases, namespace_aliases,
                known_namespaces, namespace_alias_cache,
                classes_by_identity, type_alias_cache)
            for alias_target in alias_targets
        ]
        return (
            resolutions[0] if resolutions and resolutions[0] is not None
            and all(item is resolutions[0] for item in resolutions)
            else None
        )
    return _resolve_aliased_class(
        parts, absolute, scope, namespace_aliases, use_position,
        known_namespaces, namespace_alias_cache, classes_by_identity)


def _qualified_owner(signature, name_start):
    parts, absolute, has_template_id, _ = _qualified_owner_components(
        signature, name_start)
    return parts, absolute, has_template_id


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
    class_regions = _class_regions(tokens, brace_pairs)
    namespace_regions = _namespace_regions(tokens, brace_pairs)

    global_direct_tokens = _direct_scope_tokens(
        tokens, -1, len(tokens), brace_pairs)
    namespace_direct_fragments = {}
    for namespace in namespace_regions:
        direct = _direct_scope_tokens(
            tokens, namespace.opening, namespace.closing, brace_pairs)
        namespace_direct_fragments.setdefault(namespace.path, []).append(
            direct)

    known_namespaces = {()}
    for namespace in namespace_regions:
        for length in range(1, len(namespace.path) + 1):
            known_namespaces.add(namespace.path[:length])
    namespace_alias_cache = {}
    class_direct_tokens = {
        region: _direct_scope_tokens(
            tokens, region.opening, region.closing, brace_pairs)
        for region in class_regions
    }

    classes_by_identity = {}
    for region in class_regions:
        identity = _class_identity(region, class_regions, namespace_regions)
        classes_by_identity.setdefault(identity, region)

    scoped_type_aliases = {}
    scoped_namespace_aliases = {}

    def add_aliases(destination, scope, direct, collector):
        for name, target in collector(direct, scope):
            destination.setdefault(tuple(scope) + (name,), []).append(target)

    add_aliases(
        scoped_type_aliases, (), global_direct_tokens,
        _type_alias_declarations)
    add_aliases(
        scoped_namespace_aliases, (), global_direct_tokens,
        _namespace_alias_declarations)
    for path, fragments in namespace_direct_fragments.items():
        for direct in fragments:
            add_aliases(
                scoped_type_aliases, path, direct,
                _type_alias_declarations)
            add_aliases(
                scoped_namespace_aliases, path, direct,
                _namespace_alias_declarations)
    for region, direct in class_direct_tokens.items():
        add_aliases(
            scoped_type_aliases,
            _class_identity(region, class_regions, namespace_regions),
            direct, _type_alias_declarations)

    scoped_type_aliases = {
        key: tuple(sorted(values, key=lambda item: item.declaration))
        for key, values in scoped_type_aliases.items()
    }
    scoped_namespace_aliases = {
        key: tuple(sorted(values, key=lambda item: item.declaration))
        for key, values in scoped_namespace_aliases.items()
    }
    class_identities = {
        region: _class_identity(region, class_regions, namespace_regions)
        for region in class_regions
    }
    alias_environment = AliasResolutionEnvironment(
        type_aliases=scoped_type_aliases,
        namespace_aliases=scoped_namespace_aliases,
        known_namespaces=frozenset(known_namespaces),
        namespace_alias_cache=namespace_alias_cache,
        classes_by_identity=classes_by_identity,
        class_identities=class_identities,
    )

    alias_contexts = {
        region: _context_for_class(
            region, class_regions, namespace_regions, alias_environment,
            tokens[region.declaration].start)
        for region in class_regions
    }
    own_members = {}
    inheritable_members = {}
    for region in class_regions:
        own, inheritable = _member_names_by_access(
            tokens, region.opening, region.closing, brace_pairs,
            alias_contexts[region], region.default_access)
        own_members[region] = own
        inheritable_members[region] = inheritable

    base_regions = {}
    unresolved_bases = {}
    type_alias_cache = {}
    for region in class_regions:
        scope = _class_identity(region, class_regions, namespace_regions)[:-1]
        resolved_bases = []
        unresolved = []
        for values in _base_type_values(region.base_tokens):
            resolved = _resolve_base_class(
                values, scope, tokens[region.declaration].start,
                scoped_type_aliases, scoped_namespace_aliases,
                known_namespaces, namespace_alias_cache,
                classes_by_identity, type_alias_cache)
            if resolved is None:
                unresolved.append(values)
            else:
                resolved_bases.append(resolved)
        base_regions[region] = tuple(resolved_bases)
        unresolved_bases[region] = tuple(unresolved)

    inherited_cache = {}

    def accessible_from(region, active=()):
        if region in inherited_cache:
            return inherited_cache[region]
        if region in active:
            return set()
        names = set(inheritable_members[region])
        for base in base_regions[region]:
            names.update(accessible_from(base, active + (region,)))
        inherited_cache[region] = names
        return names

    all_members = {
        region: own_members[region] | set().union(
            *(accessible_from(base) for base in base_regions[region]))
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
                parameters = _parameter_names(
                    signature, details, signature_aliases)
                returns = _return_expressions_with_bindings(
                    body, parameters, signature_aliases)
                eligible = (access == "public" and details.const_reference
                            and returns)
                member_bound = (eligible and all(
                    _direct_member_expression(expr, members, shadows)
                    for expr, shadows in returns))
                if member_bound:
                    first = signature[0] if signature else tokens[index]
                    output.append(ReferenceAccessor(
                        name=details.name,
                        documentation=_preceding_doc(text, first.start),
                        class_documentation=class_documentation,
                        class_definition=class_definition,
                        line=first.line,
                        access=access,
                    ))
                elif (eligible and unresolved_bases[region]
                      and all(_direct_member_candidate(
                          expr, shadows, unresolved_bases[region])
                              for expr, shadows in returns)):
                    first = signature[0] if signature else tokens[index]
                    diagnostics.append(ReferenceParseDiagnostic(
                        line=first.line,
                        message=(
                            "public const-reference definition {}() returns "
                            "a member-like expression through an unsupported "
                            "or unresolved base; Tier G cannot safely classify "
                            "inherited storage".format(details.name)),
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
    class_openings = {region.opening for region in class_regions}
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
        preliminary = _function_signature_details(signature, namespace_context)
        if preliminary is None:
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
        owner_parts, owner_absolute, owner_has_template_id = _qualified_owner(
            signature, preliminary.name_start)
        if not owner_parts:
            continue
        owner = _resolve_aliased_class(
            owner_parts, owner_absolute, namespace_path,
            scoped_namespace_aliases, definition_position, known_namespaces,
            namespace_alias_cache, classes_by_identity)
        if owner_has_template_id:
            if owner is None:
                continue
            aliases = _alias_context_at(
                alias_contexts[owner], definition_position)
            details = _function_signature_details(signature, aliases)
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
                        .format(preliminary.name, "::".join(owner_parts))),
                ))
            continue
        if owner is None:
            namespace_owner = _resolve_aliased_namespace(
                owner_parts, owner_absolute, namespace_path,
                scoped_namespace_aliases, definition_position,
                known_namespaces,
                namespace_alias_cache)
            if preliminary.const_reference and namespace_owner is None:
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "const-reference definition {}() has an owner that "
                        "cannot be resolved as a same-header class or namespace "
                        "in its namespace-alias scope"
                        .format(preliminary.name)),
                ))
            continue
        aliases = _alias_context_at(
            alias_contexts[owner], definition_position)
        details = _function_signature_details(signature, aliases)
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
        parameters = _parameter_names(signature, details, aliases)
        returns = _return_expressions_with_bindings(body, parameters, aliases)
        members = all_members[owner]
        if not returns:
            continue
        member_bound = all(
            _direct_member_expression(expr, members, shadows)
            for expr, shadows in returns)
        if not member_bound:
            if (unresolved_bases[owner]
                    and all(_direct_member_candidate(
                        expr, shadows, unresolved_bases[owner])
                            for expr, shadows in returns)):
                first = signature[0] if signature else tokens[opening]
                diagnostics.append(ReferenceParseDiagnostic(
                    line=first.line,
                    message=(
                        "const-reference definition {}() returns a member-like "
                        "expression through an unsupported or unresolved "
                        "base; Tier G cannot safely classify inherited storage"
                        .format(details.name)),
                ))
            continue
        class_documentation = _preceding_class_doc(
            text, tokens[owner.declaration].start)
        class_definition = text[tokens[owner.declaration].start:
                                tokens[owner.closing].end]
        output.append(ReferenceAccessor(
            name=details.name,
            documentation=documentation,
            class_documentation=class_documentation,
            class_definition=class_definition,
            line=line,
            access=access,
        ))
    return output, tuple(diagnostics)


def find_public_const_reference_accessors(text):
    """Enumerate the supported public const-reference member accessors."""
    accessors, _ = analyze_public_const_reference_accessors(text)
    return accessors


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

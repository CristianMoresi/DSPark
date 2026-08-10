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


@dataclass
class AliasContext:
    unqualified: dict
    qualified: dict
    current_scope: tuple = ()


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


def _collect_aliases(tokens):
    aliases = {}
    ambiguous = set()
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
                if name in aliases and aliases[name] != rhs:
                    ambiguous.add(name)
                else:
                    aliases[name] = rhs
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
                if name in aliases and aliases[name] != rhs:
                    ambiguous.add(name)
                else:
                    aliases[name] = rhs
                index = end
        index += 1
    for name in ambiguous:
        aliases.pop(name, None)
    return aliases


def _merge_aliases(alias_maps):
    """Merge repeated declarations, dropping only genuinely ambiguous names."""
    merged = {}
    ambiguous = set()
    for aliases in alias_maps:
        for name, rhs in aliases.items():
            if name in merged and merged[name] != rhs:
                ambiguous.add(name)
            elif name not in ambiguous:
                merged[name] = rhs
    for name in ambiguous:
        merged.pop(name, None)
    return merged


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


def _lookup_qualified_alias(parts, absolute, aliases):
    if absolute:
        return aliases.qualified.get(tuple(parts))
    for length in range(len(aliases.current_scope), -1, -1):
        key = aliases.current_scope[:length] + tuple(parts)
        replacement = aliases.qualified.get(key)
        if replacement is not None:
            return replacement
    return None


def _expand_aliases(values, aliases):
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
                    replacement = _lookup_qualified_alias(parts, absolute,
                                                          aliases)
                    if replacement is not None:
                        output.extend(replacement)
                        changed = True
                    else:
                        output.extend(expanded[index:end])
                    index = end
                    continue
                replacement = aliases.unqualified.get(parts[0])
                if replacement is not None:
                    output.extend(replacement)
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


def _qualified_owner_start(signature, name_start):
    """Return the first token of the owner preceding a qualified function."""
    cursor = name_start - 1
    while cursor >= 1 and signature[cursor].text == "::":
        owner = signature[cursor - 1].text
        if not owner or not _is_identifier_start(owner[0]):
            break
        cursor -= 2
    if cursor >= 0 and signature[cursor].text == "::":
        cursor -= 1
    return cursor + 1


def _canonical_type_tokens(values, aliases):
    values = [value for value in _strip_attributes(values)
              if value not in _RETURN_SPECIFIERS]
    return tuple(_expand_aliases(values, aliases))


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
        canonical = _canonical_type_tokens(values, aliases)
        if not canonical:
            return None
        output.append(canonical)
    if len(output) == 1 and output[0] == ("void",):
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
            previous = index - 1
            name = None
            name_start = previous
            if previous >= 0 and _is_identifier_start(signature[previous].text[0]):
                possible = signature[previous].text
                if possible not in _NON_FUNCTION_NAMES:
                    name = possible
            elif previous >= 0 and signature[previous].text == ")":
                left = local_pairs.get(previous)
                if left is not None:
                    inside = [part.text for part in signature[left + 1:previous]
                              if part.text not in ("[[", "]]")]
                    if len(inside) == 1 and _is_identifier_start(inside[0][0]):
                        name = inside[0]
                        name_start = left
                if (left is not None and left > 0
                        and signature[left - 1].text == "operator"
                        and left + 1 == previous):
                    name = "operator()"
                    name_start = left - 1
            elif (previous >= 2 and signature[previous - 2].text == "operator"
                  and signature[previous - 1].text == "["
                  and signature[previous].text == "]"):
                name = "operator[]"
                name_start = previous - 2
            if name is not None:
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


def _direct_member_expression(values, member_names, shadow_names=()):
    values = _strip_outer_parentheses(values)
    if not values:
        return False

    # std::get<I>(member_) is a direct member subobject, not an arbitrary call.
    if len(values) >= 7 and values[:3] in (["std", "::", "get"], ["get", "<", "0"]):
        get_index = 2 if values[:3] == ["std", "::", "get"] else 0
        cursor = get_index + 1
        if cursor < len(values) and values[cursor] == "<":
            angle = 1
            cursor += 1
            while cursor < len(values) and angle:
                angle += (values[cursor] == "<") - (values[cursor] == ">")
                cursor += 1
            if (angle == 0 and cursor < len(values) and values[cursor] == "("
                    and values[-1] == ")"):
                return _direct_member_expression(values[cursor + 1:-1],
                                                 member_names, shadow_names)

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
    cursor += 1
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
            if cursor >= len(values) or not _is_identifier_start(values[cursor][0]):
                return False
            cursor += 1
        else:
            return False
    return True


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
            declared = _declared_data_names(tokens[statement_start:index], aliases)
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
            if _function_signature(tokens[statement_start:index], aliases) is not None:
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
                       global_aliases, namespace_aliases, class_aliases,
                       qualified_aliases):
    unqualified = dict(global_aliases)
    namespace = _containing_namespace(region, namespace_regions)
    for length in range(1, len(namespace) + 1):
        unqualified.update(namespace_aliases.get(namespace[:length], {}))
    for ancestor in _class_ancestors(region, class_regions):
        unqualified.update(class_aliases.get(ancestor, {}))
    unqualified.update(class_aliases.get(region, {}))
    return AliasContext(
        unqualified=unqualified,
        qualified=qualified_aliases,
        current_scope=_class_identity(region, class_regions, namespace_regions),
    )


def _context_for_namespace(path, global_aliases, namespace_aliases,
                           qualified_aliases):
    unqualified = dict(global_aliases)
    for length in range(1, len(path) + 1):
        unqualified.update(namespace_aliases.get(path[:length], {}))
    return AliasContext(unqualified, qualified_aliases, path)


def _base_names(base_tokens):
    values = list(base_tokens)
    if not values or ":" not in [token.text for token in values]:
        return []
    colon = next(index for index, token in enumerate(values)
                 if token.text == ":")
    output = []
    for base in _top_level_parts(values[colon + 1:], ","):
        filtered = [token.text for token in base
                    if token.text not in {
                        "public", "protected", "private", "virtual",
                        "typename", "template", "[[", "]]"}]
        if "<" in filtered:
            filtered = filtered[:filtered.index("<")]
        identifiers = [value for value in filtered
                       if value and _is_identifier_start(value[0])]
        if identifiers:
            output.append(tuple(identifiers))
    return output


def _resolve_class(parts, scope, classes_by_identity):
    for length in range(len(scope), -1, -1):
        candidate = scope[:length] + tuple(parts)
        if candidate in classes_by_identity:
            return classes_by_identity[candidate]
    return None


def _qualified_owner(signature, name_start):
    cursor = name_start - 1
    reversed_parts = []
    while cursor >= 1 and signature[cursor].text == "::":
        owner = signature[cursor - 1].text
        if not owner or not _is_identifier_start(owner[0]):
            break
        reversed_parts.append(owner)
        cursor -= 2
    return tuple(reversed(reversed_parts))


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


def find_public_const_reference_accessors(text):
    """Enumerate the supported public const-reference member accessors."""
    tokens = tokenize_cpp(text)
    brace_pairs = _pair_map(tokens, "{", "}")
    class_regions = _class_regions(tokens, brace_pairs)
    namespace_regions = _namespace_regions(tokens, brace_pairs)

    global_aliases = _collect_aliases(
        _direct_scope_tokens(tokens, -1, len(tokens), brace_pairs))
    namespace_fragments = {}
    for namespace in namespace_regions:
        namespace_fragments.setdefault(namespace.path, []).append(
            _collect_aliases(_direct_scope_tokens(
                tokens, namespace.opening, namespace.closing, brace_pairs)))
    namespace_aliases = {
        path: _merge_aliases(fragments)
        for path, fragments in namespace_fragments.items()
    }
    class_aliases = {
        region: _collect_aliases(_direct_scope_tokens(
            tokens, region.opening, region.closing, brace_pairs))
        for region in class_regions
    }

    classes_by_identity = {}
    for region in class_regions:
        identity = _class_identity(region, class_regions, namespace_regions)
        classes_by_identity.setdefault(identity, region)

    qualified_aliases = {
        (name,): rhs for name, rhs in global_aliases.items()
    }
    for path, aliases in namespace_aliases.items():
        for name, rhs in aliases.items():
            qualified_aliases[path + (name,)] = rhs
    for region, aliases in class_aliases.items():
        identity = _class_identity(region, class_regions, namespace_regions)
        for name, rhs in aliases.items():
            qualified_aliases[identity + (name,)] = rhs

    alias_contexts = {
        region: _context_for_class(
            region, class_regions, namespace_regions, global_aliases,
            namespace_aliases, class_aliases, qualified_aliases)
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
    for region in class_regions:
        scope = _class_identity(region, class_regions, namespace_regions)[:-1]
        base_regions[region] = tuple(
            resolved for resolved in (
                _resolve_class(parts, scope, classes_by_identity)
                for parts in _base_names(region.base_tokens))
            if resolved is not None)

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
                details = _function_signature_details(signature, aliases)
                if details is not None and access == "public":
                    first = signature[0] if signature else tokens[index]
                    declarations[region].setdefault(details.name, []).append((
                        details,
                        _preceding_doc(text, first.start),
                        first.line,
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
            details = _function_signature_details(signature, aliases)
            if details is not None:
                body = tokens[index + 1:body_close]
                parameters = _parameter_names(signature, details, aliases)
                returns = _return_expressions_with_bindings(
                    body, parameters, aliases)
                if (access == "public" and details.const_reference and returns
                        and all(_direct_member_expression(expr, members, shadows)
                                for expr, shadows in returns)):
                    first = signature[0] if signature else tokens[index]
                    output.append(ReferenceAccessor(
                        name=details.name,
                        documentation=_preceding_doc(text, first.start),
                        class_documentation=class_documentation,
                        class_definition=class_definition,
                        line=first.line,
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
        namespace_path = (namespace_by_opening[parent].path
                          if parent is not None else ())
        namespace_context = _context_for_namespace(
            namespace_path, global_aliases, namespace_aliases,
            qualified_aliases)
        preliminary = _function_signature_details(signature, namespace_context)
        if preliminary is None:
            continue
        owner_parts = _qualified_owner(signature, preliminary.name_start)
        if not owner_parts:
            continue
        owner = _resolve_class(owner_parts, namespace_path,
                               classes_by_identity)
        if owner is None:
            continue
        aliases = alias_contexts[owner]
        details = _function_signature_details(signature, aliases)
        if details is None or not details.const_reference:
            continue
        declared = [
            item for item in declarations[owner].get(details.name, ())
            if _signature_key(item[0]) == _signature_key(details)
        ]
        if len(declared) != 1 or not declared[0][0].const_reference:
            continue
        body_close = brace_pairs[opening]
        body = tokens[opening + 1:body_close]
        parameters = _parameter_names(signature, details, aliases)
        returns = _return_expressions_with_bindings(body, parameters, aliases)
        members = all_members[owner]
        if not returns or not all(
                _direct_member_expression(expr, members, shadows)
                for expr, shadows in returns):
            continue
        _, documentation, line = declared[0]
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
        ))
    return output


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

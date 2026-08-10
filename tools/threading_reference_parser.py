#!/usr/bin/env python3
"""Dependency-free C++ declaration parser for threading reference readouts.

The parser is intentionally focused: it finds public inline member functions
whose declared return type is a const lvalue reference and whose every return
statement directly names member storage or one of its subobjects. It tokenizes
C++ rather than matching one source spelling, so whitespace, attributes,
aliases, trailing returns, parenthesized declarators and member qualifiers do
not change the census.
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
            regions.append((cursor, closing,
                            "public" if token.text == "struct" else "private",
                            index))
    return regions


def _namespace_regions(tokens, brace_pairs):
    """Return lexical namespace bodies as ``(opening, closing, declaration)``.

    Namespace scopes matter to alias lookup just as class scopes do.  Treating
    every non-class alias as global makes two sibling namespaces poison one
    another, and lets a namespace-local shadow erase an otherwise valid global
    accessor from the census.
    """
    regions = []
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
            regions.append((cursor, closing, index))
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


def _expand_aliases(values, aliases):
    expanded = list(values)
    for _ in range(16):
        changed = False
        output = []
        for value in expanded:
            replacement = aliases.get(value)
            if replacement is None:
                output.append(value)
            else:
                output.extend(replacement)
                changed = True
        expanded = output
        if not changed:
            break
    return expanded


def _is_const_lvalue_reference(values, aliases):
    values = [value for value in _strip_attributes(values)
              if value not in _RETURN_SPECIFIERS]
    values = _expand_aliases(values, aliases)
    return "&" in values and "&&" not in values and "const" in values


def _function_signature(signature, aliases):
    """Return (name, return-is-const-reference) or None."""
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

    name, name_start, _, parameter_close = candidates[-1]
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
        return_values = [token.text for token in signature[:name_start]]
    return name, _is_const_lvalue_reference(return_values, aliases)


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


def _direct_member_expression(values, member_names):
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
                return _direct_member_expression(values[cursor + 1:-1], member_names)

    cursor = 0
    if values[:2] == ["this", "->"]:
        cursor = 2
    if cursor >= len(values) or not _is_identifier_start(values[cursor][0]):
        return False
    root = values[cursor]
    if root not in member_names:
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


def _member_names(tokens, opening, closing, brace_pairs, aliases):
    names = set()
    statement_start = opening + 1
    index = statement_start
    while index < closing:
        value = tokens[index].text
        if (value in ("public", "private", "protected")
                and index + 1 < closing and tokens[index + 1].text == ":"):
            statement_start = index + 2
            index += 2
            continue
        if value == ";":
            names.update(_declared_data_names(tokens[statement_start:index], aliases))
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
    return names


def find_public_const_reference_accessors(text):
    """Enumerate public inline accessors returning const member references."""
    tokens = tokenize_cpp(text)
    brace_pairs = _pair_map(tokens, "{", "}")
    class_regions = _class_regions(tokens, brace_pairs)
    namespace_regions = _namespace_regions(tokens, brace_pairs)

    # Alias lookup follows the lexical namespace/class chain.  Aliases are
    # collected only from the region that owns their declaration: a nested
    # class cannot shadow its outer class, and sibling namespaces/classes
    # cannot make one another's aliases ambiguous.
    all_regions = [
        (_region_key("namespace", opening, closing), opening, closing)
        for opening, closing, _ in namespace_regions
    ] + [
        (_region_key("class", opening, closing), opening, closing)
        for opening, closing, _, _ in class_regions
    ]
    scope_aliases = {
        key: _collect_aliases(
            _direct_scope_tokens(tokens, opening, closing, brace_pairs))
        for key, opening, closing in all_regions
    }
    global_aliases = _collect_aliases(
        _direct_scope_tokens(tokens, -1, len(tokens), brace_pairs))
    output = []

    for opening, closing, default_access, declaration in class_regions:
        aliases = dict(global_aliases)
        ancestors = sorted(
            ((region_opening, region_closing, key)
             for key, region_opening, region_closing in all_regions
             if region_opening < opening and closing < region_closing),
            key=lambda item: (item[0], -item[1]))
        for _, _, key in ancestors:
            aliases.update(scope_aliases[key])
        aliases.update(scope_aliases[_region_key("class", opening, closing)])
        members = _member_names(tokens, opening, closing, brace_pairs, aliases)
        class_documentation = _preceding_class_doc(
            text, tokens[declaration].start)
        class_definition = text[tokens[declaration].start:
                                tokens[closing].end]
        access = default_access
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
            function = _function_signature(signature, aliases)
            if function is not None:
                name, const_reference = function
                body = tokens[index + 1:body_close]
                returns = _return_expressions(body)
                if (access == "public" and const_reference and returns
                        and all(_direct_member_expression(expr, members)
                                for expr in returns)):
                    first = signature[0] if signature else tokens[index]
                    output.append(ReferenceAccessor(
                        name=name,
                        documentation=_preceding_doc(text, first.start),
                        class_documentation=class_documentation,
                        class_definition=class_definition,
                        line=first.line,
                    ))
                member_start = body_close + 1
            index = body_close + 1
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
    return (accepted_sources,
            [(label, source_for(signature, body))
             for label, signature, body in near_misses])


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

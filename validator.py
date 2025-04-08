import re
'''The overall algorithm to Check if a einops expression is valid or not (Used GPT to give like a better structure and a shorter expression syntax 
and evaluate some good test cases, which I may have missed out)
For overall
1. Global count of parenthesis
2. Global count of ellipsis
3. Invalid dots
4. Check for the presence of one exact arrow ('->') in the expression.
5. Split the expression based on the arrow, get input and output stuff. 
6. If all right, do an additional check for ellipsis, missed that

For each side 
1. Check for whether an axis is alpha numeric or underscore 
2. For ellipsis, 3 dots exactly. no other doubts
3. Parenthesis should be complete and no nested parenthesis allowed
'''
def validate_arrow(pattern: str) -> bool:
    # Require exactly one arrow.
    return pattern.count('->') == 1

def initial_split(pattern: str) -> tuple:
    if validate_arrow(pattern):
        left, right = map(str.strip, pattern.split("->"))
        return left, right
    else:
        raise ValueError("Expected exactly one arrow ('->') in the expression")

def is_valid_axis(token: str) -> bool:
        return re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', token) is not None

def validate_side_expression(pattern: str) -> bool:
    #Check each 
    tokens = pattern.split()
    if not tokens:
        return False

    for token in tokens:
        if token == '...':
            continue

        # Check for parenthesized expressions.
        elif token.startswith('(') and token.endswith(')'):
            inner = token[1:-1].strip()
            if not inner:
                return False
            if '(' in inner or ')' in inner:
                return False
            if '...' in inner:
                return False

            # Split inner tokens by whitespace.
            inner_tokens = inner.split()
            for it in inner_tokens:
                if not is_valid_axis(it):
                    return False
        else:
            if not is_valid_axis(token):
                return False
            
            if '.' in token:
                return False

    return True

def is_valid_einops_expression(pattern: str) -> bool:
    pattern = pattern.strip()

    if pattern.count('(') != pattern.count(')'):
        return False
    
    if pattern.count('...') > 1:
        return False
    
    if re.search(r'(?<![.])(\.{1,2}|\.{4,})(?![.])', pattern):
        return False


    if '->' in pattern:
        try:
            left, right = initial_split(pattern)
        except ValueError:
            return False
        if ('...' not in left) and ('...' in right):
            return False
        return validate_side_expression(left) and validate_side_expression(right)
    else:
        return validate_side_expression(pattern)

def validate_einops_expression_ellipsis(patterns: tuple) -> bool:
    left, right = patterns
    if is_valid_einops_expression(left) and is_valid_einops_expression(right):
       if left.count('...') == right.count('...'): #an additional check if ellipsis is present on both sides
        return True
    else:
        raise ValueError("Invalid einops expression")

def main_parser(pattern: str) -> bool :
    pattern = pattern.strip()
    try:
        left, right = initial_split(pattern)
    except ValueError as e:
       # print(f"Invalid expression: {e}")
        return False
    try:
        validate_einops_expression_ellipsis((left, right))
        return True
    except Exception as e:
       # print(f"Invalid einops expression: {pattern} - {e}")
        return False



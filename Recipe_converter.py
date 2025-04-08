import re
import numpy as np

def identify_left(pattern: str) -> list:
    """
    Tokenization 
    AST kind of formation:
      - "axis": a variable axis 
      - "literal":number axis 
      - "group": composite axes
      
    Parameters:
        pattern (str): Input pattern string 
        
    Returns:
        list of dict: The Tree
    """
    tokens = pattern.split()
    result = []
    i = 0
    axis_regex = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    while i < len(tokens):
        token = tokens[i]
        if token.startswith('('):
            group_tokens = []
            token_clean = token.lstrip('(')
            group_tokens.append(token_clean)
            i += 1
            group_found = False
            while i < len(tokens):
                token = tokens[i]
                if token.endswith(')'):
                    token_clean = token.rstrip(')')
                    group_tokens.append(token_clean)
                    group_found = True
                    break
                else:
                    group_tokens.append(token)
                i += 1
            if not group_found:
                raise ValueError("Unbalanced parentheses: missing closing ')'")
            group_content = " ".join(group_tokens).strip()
            if not group_content:
                raise ValueError("Empty parentheses are not allowed")
            # For tokens inside a group, we expect each token to be a valid axis name (no literals allowed).
            for inner in group_content.split():
                if inner.isdigit():
                    raise ValueError(f"Literal dimension ({inner}) not allowed within a group")
                if axis_regex.fullmatch(inner) is None:
                    raise ValueError(f"Invalid axis name in group: '{inner}'")
            result.append({'type': 'group', 'content': group_content})
        else:
            if token.isdigit():
                result.append({'type': 'literal', 'content': token})
            else:
                if axis_regex.fullmatch(token) is None:
                    raise ValueError(f"Invalid axis token: '{token}'")
                result.append({'type': 'axis', 'content': token})
        i += 1
    return result

def process_right(right_pattern: str, left_tokens: list) -> list:
    """
    Compares right stuff with the left token by matching indices
    
    
    Parameters:
        right_pattern (str): The right (output) pattern.
        left_tokens (list): List of tokens (as produced by identify_left) from the left pattern.
        
    Returns:
        plan (list of dict): The transformation plan, where each action is described as:
            - For "copy" actions: {'action': 'copy', 'source': left_index, 'name': token}
            - For "merge" actions: {'action': 'merge', 'sources': [list of left indices], 'name': group_content}
            - For an "ellipsis" action: {'action': 'ellipsis', 'source_range': [list of left indices]}

    """
    # Parse the right pattern.
    right_tokens = identify_left(right_pattern)
    
    # Flatten the left tokens into a list.
    left_flat = []
    for token in left_tokens:
        if token['type'] in ['axis', 'literal']:
            left_flat.append(token['content'])
        elif token['type'] == 'group':
            left_flat.extend(token['content'].split())
    
    # Build a mapping from token content to its left index.
    left_map = {}
    for idx, name in enumerate(left_flat):
        if name in left_map:
            raise ValueError(f"Duplicate name '{name}' found in left pattern")
        left_map[name] = idx
    
    plan = []
    used_indices = set()
    
    for token in right_tokens:
        if token['content'] == '...':
            remaining = [i for i in range(len(left_flat)) if i not in used_indices]
            plan.append({'action': 'ellipsis', 'source_range': remaining})
            used_indices.update(remaining)
            continue
        
        if token['type'] in ['axis', 'literal']:
            name = token['content']
            if name not in left_map:
                raise ValueError(f"Token '{name}' from right pattern not found in left pattern")
            idx = left_map[name]
            if idx in used_indices:
                raise ValueError(f"Token '{name}' in left pattern is referenced more than once")
            plan.append({'action': 'copy', 'source': idx, 'name': name})
            used_indices.add(idx)
        elif token['type'] == 'group':
            inner_tokens = token['content'].split()
            indices = []
            for inner in inner_tokens:
                if inner not in left_map:
                    raise ValueError(f"Axis '{inner}' from right group not found in left pattern")
                idx = left_map[inner]
                if idx in used_indices:
                    raise ValueError(f"Axis '{inner}' in left pattern is referenced more than once")
                indices.append(idx)
                used_indices.add(idx)
            plan.append({'action': 'merge', 'sources': indices, 'name': token['content']})
    
    if set(range(len(left_flat))) != used_indices:
        raise ValueError("Not all axes from the left pattern are accounted for in the right pattern")
    
    return plan

def apply_recipe(tensor: np.ndarray, plan: list) -> np.ndarray:
    # Track current axis order and reshape steps
    current_shape = list(tensor.shape)
    n_dim = len(current_shape)
    
    # 1: Handle transposes and repetitions
    transpose_order = []
    merge_actions = []
    ellipsis_indices = []
    
    # Collect transpose order from 'copy' actions
    for step in plan:
        if step['action'] == 'copy':
            transpose_order.append(step['source'])
        elif step['action'] == 'merge':
            merge_actions.append(step)
        elif step['action'] == 'ellipsis':
            ellipsis_indices.extend(step['source_range'])
    
    # Handle ellipsis 
    remaining_indices = [i for i in range(n_dim) 
                        if i not in transpose_order 
                        and i not in ellipsis_indices]
    transpose_order += remaining_indices
    
    # Validate transpose order is a valid permutation
    if sorted(transpose_order) != list(range(n_dim)):
        raise ValueError(f"Invalid transpose order {transpose_order} for {n_dim}D tensor")
    
    # Apply transpose - if needed
    if transpose_order != list(range(n_dim)):
        tensor = tensor.transpose(transpose_order)
        current_shape = list(tensor.shape)
    
    #  Handle merging of axes
    for merge_step in merge_actions:
        sources = merge_step['sources']
        
        # Map original indices to transpose positions obtained from the transpose order
        mapped_sources = [transpose_order.index(idx) for idx in sources]
        mapped_sources.sort()  # Ensure consecutive order
        
        # Verify sources are consecutive after mapping
        if not all(mapped_sources[i+1] == mapped_sources[i]+1 
                for i in range(len(mapped_sources)-1)):
            raise ValueError(f"Non-consecutive axes {mapped_sources} cannot be merged")
        
        # Calculate merged dimension size
        merged_size = 1
        for dim in mapped_sources:
            merged_size *= current_shape[dim]
        
        # Build new shape
        new_shape = (
            list(current_shape[:mapped_sources[0]]) + 
            [merged_size] + 
            list(current_shape[mapped_sources[-1]+1:])
        )
        
        tensor = tensor.reshape(new_shape)
        current_shape = new_shape
    
    return tensor


  
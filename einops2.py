from validator import main_parser
from Recipe_converter import identify_left,process_right,apply_recipe
import numpy as np 
import re

#Something I forgot here is to handle a case where there are other arguments passed like b=2, h=2, So I handled them in the end


def substitute_vars(pattern: str, kwargs: dict) -> str:
    def replace(match):
        var = match.group(0)
        return str(kwargs.get(var, var))  

    return re.sub(r'\b\w+\b', replace, pattern)

def rearrange(tensor:np.ndarray, pattern:str, **kwargs) -> np.ndarray:
    new_pattern= substitute_vars(pattern, kwargs)
    if main_parser(pattern):
        left,right= new_pattern.split('->')
        left_tokens = identify_left(left)
        plan = process_right(right, left_tokens)
        transformed=apply_recipe(tensor,plan)
        return transformed
    else:       
        raise ValueError(f"Invalid einops expression: {pattern}")
    



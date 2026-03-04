import inspect
from functools import wraps
from typing import get_type_hints, get_origin, get_args, Any, Dict, Callable


def tool(func: Callable) -> Callable:
    """
    Decorator that converts a Python function into an OpenAI tool definition.

    The function must have:
    - Type hints for all parameters
    - A docstring with a description on the first line
    - Optional parameter descriptions using :param name: format

    Example:
        @openai_tool
        def get_weather(location: str) -> str:
            '''
            Get current temperature for a given location.

            :param location: City and country e.g. Bogotá, Colombia
            '''
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    docstring = inspect.getdoc(func) or ""
    lines = docstring.split('\n')
    
    # Extract description (all lines before :param, :return, etc.)
    description_lines = []
    param_start_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(':param') or stripped.startswith(':return') or stripped.startswith(':raises'):
            param_start_idx = i
            break
        description_lines.append(stripped)
    
    description = ' '.join(line for line in description_lines if line).strip()

    param_descriptions = {}
    for line in lines[param_start_idx:]:
        line = line.strip()
        if line.startswith(':param '):
            # Format: :param name: description
            parts = line.split(':', 2)  # Split into at most 3 parts
            if len(parts) >= 3:
                param_name = parts[1].replace('param', '').strip()
                param_desc = parts[2].strip()
                param_descriptions[param_name] = param_desc

    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Build parameters schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue

        param_type = type_hints.get(param_name, Any)

        # Convert Python type to JSON schema
        schema = _python_type_to_json_schema(param_type)

        properties[param_name] = {
            **schema,
            "description": param_descriptions.get(param_name, "")
        }

        # Mark as required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Create the tool definition (OpenAI format)
    tool_def = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # Attach the tool definition to the function
    wrapper.tool_definition = tool_def

    return wrapper


def _python_type_to_json_schema(python_type) -> Dict[str, Any]:
    origin = get_origin(python_type)

    if origin is list:
        args = get_args(python_type)
        item_type = args[0] if args else str
        item_schema = _python_type_to_json_schema(item_type)
        return {
            "type": "array",
            "items": item_schema
        }
    elif origin is dict:
        return {"type": "object"}

    # Handle basic types
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    return {"type": type_mapping.get(python_type, "string")}


# Helper function to collect all tool definitions
def get_tool_definitions(*funcs: Callable) -> list[Dict[str, Any]]:
    """
    Extract tool definitions from decorated functions.

    :param funcs: Decorated functions with tool definitions
    :return: List of tool definitions for OpenAI API
    """
    return [func.tool_definition for func in funcs if hasattr(func, 'tool_definition')]

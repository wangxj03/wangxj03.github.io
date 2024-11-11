---
title: "Schema Generation for LLM Function Calling"
date: 2024-11-08T20:39:45-08:00
tags: ["AI"]
author: "Xiaojing Wang"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
disableHLJS: false
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: false
UseHugoToc: true
---

The rise of Large Language Models (LLMs) has opened up exciting possibilities for automation and natural language interfaces. But to unlock their full potential, we need to connect them with external tools — and that's where function calling comes in. In this post, we'll explore how to streamline the process of defining these connections, moving from manual schema writing to automated solutions.

## Tool Function Definitions

When connecting LLMs to external tools, we need two key components: the tool functions themselves and their definitions. Each definition describes the function’s purpose and required parameters. OpenAI's [function calling guide](https://platform.openai.com/docs/guides/function-calling#step-2-describe-your-function-to-the-model-so-it-knows-how-to-call-it) provides examples for creating these schemas in JSON format:

```json
{
  "name": "get_delivery_date",
  "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "description": "The customer's order ID."
      }
    },
    "required": ["order_id"],
    "additionalProperties": false
  }
}
```

## Automating Schema Generation with Python Inspection

While manual schema creation is straightforward for simple functions, maintaining these definitions can become tedious and error-prone as the number or complexity of functions grows. Automating schema generation is a more scalable solution. Python’s built-in `inspect` module allows us to peek into function signatures and automatically generate these schemas. OpenAI's [Swarm](https://github.com/openai/swarm/tree/main) framework offers a reference implementation for this approach.

```python
def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

Swarm’s implementation is a helpful starting point, though it may lack support for more complex parameter types, such as enums. The core concept, however, is sound and extensible.

## Enhanced Schema Generation with Pydantic

[Pydantic](https://docs.pydantic.dev/latest/) offers a powerful alternative to handle more complex types. By defining parameters as Pydantic models, you benefit from automatic type validation, default values, and detailed parameter descriptions, all while generating accurate JSON schemas. Below is an example of using Pydantic to create a JSON schema for a function that retrieves the current temperature.

```python
from typing import Literal

from pydantic import BaseModel, Field


class CurrentTemperature(BaseModel):
    location: str = Field(
        ..., description="Get the current temperature for a specific location"
    )

    unit: Literal["Celsius", "Fahrenheit"] = Field(
        ...,
        description="The temperature unit to use. Infer this from the user's location.",
    )

print(CurrentTemperature.model_json_schema())
```

The resulting JSON schema aligns well with OpenAI's function calling [example](https://platform.openai.com/docs/assistants/tools/function-calling)

```json
{
  "properties": {
    "localtion": {
      "description": "Get the current temperature for a specific location",
      "title": "Localtion",
      "type": "string"
    },
    "unit": {
      "description": "The temperature unit to use. Infer this from the user's location.",
      "enum": ["Celsius", "Fahrenheit"],
      "title": "Unit",
      "type": "string"
    }
  },
  "required": ["localtion", "unit"],
  "title": "CurrentTemperature",
  "type": "object"
}
```

Pydantic’s approach is both readable and maintainable.

## Combining Inspection and Pydantic

We can take a step further by combining both approaches. The following implementation leverages type annotations for type mapping and uses a dynamic Pydantic model to generate the schema:

```python
import inspect
from typing import Any, Callable, Literal

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import create_model


def get_tool_param(func: Callable[..., Any]) -> ChatCompletionToolParam:
    # Get the signature of the function
    sig = inspect.signature(func)

    # Prepare a dictionary to store the fields for the Pydantic model
    model_fields = {}

    # Loop over the function's parameters and extract the type and default value
    for param_name, param in sig.parameters.items():
        # Get the type hint
        param_type = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )

        # Check if the parameter has a default value
        if param.default != inspect.Parameter.empty:
            model_fields[param_name] = (param_type, param.default)
        else:
            model_fields[param_name] = (param_type, ...)

    # Dynamically create a Pydantic model
    model_name = (
        "".join(word.capitalize() for word in func.__name__.split("_")) + "Model"
    )
    model = create_model(model_name, **model_fields)
    schema = model.model_json_schema()

    return ChatCompletionToolParam(
        function=FunctionDefinition(
            name=func.__name__,
            description=(func.__doc__ or "").strip(),
            parameters=schema,
        ),
        type="function",
    )
```

## Conclusion

Leveraging Python’s introspection capabilities alongside Pydantic’s type system allows for automated generation of JSON schemas directly from function signatures. This approach minimizes manual effort, maintains consistency, and strengthens type safety, providing developers with an efficient and scalable way to connect LLMs with external tools.

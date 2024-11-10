---
title: "Schema Generation for Function Calling in Large Language Models"
date: 2024-11-08T20:39:45-08:00
tags: ["AI"]
author: "Xiaojing Wang"
showToc: true
TocOpen: false
draft: true
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

Connecting Large Language Models (LLMs) to external tools through function calling is fundamental to creating effective Agentic AI workflows. In this post, we’ll cover five tips to enhance your experience with function calling.

When calling LLMs, function definitions must be provided as available "tools". Each definition describes the function’s purpose and required parameters. OpenAI's [function calling guide](https://platform.openai.com/docs/guides/function-calling#step-2-describe-your-function-to-the-model-so-it-knows-how-to-call-it) shows how to define this schema in JSON format:

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

Manually creating schemas for simple functions is manageable. However, as functions grow in complexity, maintaining accurate schemas becomes challenging and error-prone—especially when managing multiple tools.

OpenAI's multi-agent framework, [Swarm](https://github.com/openai/swarm/tree/main), showcases an approach to use Python's `inspect` module to introspect the given Python function's parameters and build the schema dynamically.

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

Since Swarm is primarily for educational purposes. The authors haven't really tried very hard to make the type mapping comprehensive. For example, it doesn't support enum parameters. However, the core idea is sound and can be extended. For developers seeking type safety and more control over the schema, they can define a [pydantic](https://docs.pydantic.dev/latest/) model and build the schema from it. For example, the following code snippet shows schema generation for the `get_current_temperature` function:

```python
from typing import Literal

from pydantic import BaseModel, Field


class CurrentTemperature(BaseModel):
    localtion: str = Field(
        ..., description="Get the current temperature for a specific location"
    )

    unit: Literal["Celsius", "Fahrenheit"] = Field(
        ...,
        description="The temperature unit to use. Infer this from the user's location.",
    )

print(CurrentTemperature.model_json_schema())
```

The output JSON matches the `parameters` field in the OpenAI Assistants function calling [example](https://platform.openai.com/docs/assistants/tools/function-calling)

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

Furthermore, we can combine both the `inspect` and `pydantic` approaches. The following code snippet leverages type annotations for type mapping and creates a pydantic model dynamically to build the schema.

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

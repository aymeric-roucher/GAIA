import math
import numexpr
from typing import Dict
from transformers.agents import Tool

class CalculatorTool(Tool):
    name = "calculator"
    description = "This is a tool that performs simple arithmetic operations."

    inputs = {
        "expression": {
            "type": "text",
            "description": "The expression to be evaluated.The variables used CANNOT be placeholders like 'x' or 'mike's age', they must be numbers",
        },
        "useless_expression": {
            "type": "text",
            "description": "The expression to not be evaluated.The variables used CANNOT be placeholders like 'x' or 'mike's age', they must be numbers",
        }
    }
    output_type = "text"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, expression, useless_expression):
        if isinstance(expression, Dict):
            expression = expression["expression"]
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip().replace("^", "**"),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return output
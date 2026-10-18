from langchain_core.tools import tool
import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg 
}

def safe_evaluate(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_evaluate(node.left)
        right = safe_evaluate(node.right)
        return OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_evaluate(node.operand)
        return OPERATORS[type(node.op)](operand)
    else:
        raise TypeError(f"Unsupported mathematical operation: {node}")

@tool
def calculator(expression: str) -> float:
    """
    A simple calculator tool that safely evaluates a mathematical expression.
    """
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_evaluate(tree.body)
        return float(result)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

@tool
def current_datetime() -> str:
    """
    A tool that returns the current date and time as a string.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_student_grades(name: str) -> dict:
    """
    A tool that returns a student's grades based on their name.
    """
    # Sample data for demonstration purposes
    student_grades = {
        "mona": {"Math": 90, "AI": 88},
        "Bob": {"Math": 75, "AI": 80},
        "Charlie": {"Math": 95, "AI": 92}
    }
    
    return student_grades.get(name, {"Error": "Student not found"})

@tool
def save_note(content: str) -> str:
    """
    A tool that saves content to a specified file.
    """
    file_path = "notes.txt"
    try:
        with open(file_path, "a") as file:
            file.write(content + "\n")
        return f"Note saved to {file_path}"
    except Exception as e:
        raise IOError(f"Error saving note: {e}")


ALL_TOOLS = [calculator, current_datetime, get_student_grades, save_note]
# %%
from math import asin, pi, ceil
from random import random, choices, choice
from collections import deque
from subprocess import run, DEVNULL
from os import remove

from lexicons import *


# lexicons



# randomize the output length
MEAN_TEX_EXPR_LENGTH_IN_COMPLEXITY = 4

def get_tex_expr_length_in_complexity() -> int:
    return ceil(MEAN_TEX_EXPR_LENGTH_IN_COMPLEXITY*2/pi*asin(2*random() - 1) + MEAN_TEX_EXPR_LENGTH_IN_COMPLEXITY)


# Syntax node that generate a structure for a TeX expression
# Tuple of: Operation name, Possible precedence values, Canonical precedence value, Number of arguments.
SYNTAX_NODE = [("subscript", (0,), 0, 2), ("add", (1,2), 2, 2), ("subtract", (1,2), 2, 2), ("multiply", (1,2), 1, 2), ("power", (1,), 1, 2), ("juxtapose", (0,), 0, 2),
               ("equality", (4,), 4, 2), ("square root", (3,), 3, 1), ("fraction", (3,), 3, 2)]
               
OUTPUT_DIRECTORY = "./generated/"

# %%

###########################
# Generate TeX Expression #
###########################

def generate_tex_expression() -> str:

    pretree = choices(SYNTAX_NODE, k=get_tex_expr_length_in_complexity())

    # generate tree by choosing the precedence form the possible precedence values
    tree = []
    for operation, precedences, canonical_precedence, num_args in pretree:
        tree.append((operation, choice(precedences), canonical_precedence, num_args))


    def apply_operator(operator, evaluated_stack: deque[str]) -> tuple[str, bool]:
        evaluated_string = ""

        if len(evaluated_stack) < operator[3]:
            # format string, precedence of the expression, tall, subscripted, superscripted
            evaluated_stack.extendleft([("$", 0, False, False, False)]*(operator[3]-len(evaluated_stack)))
        
        operand1 = operand2 = ""
        previous_precedence1 = previous_precedence2 = 0
        tall1 = tall2 = False
        sub1 = sub2 = False
        sup1 = sup2 = False
        if operator[3] == 1:
            operand1, previous_precedence1, tall1, sub1, sup1= evaluated_stack.pop()
        elif operator[3] == 2:
            operand1, previous_precedence1, tall1, sub1, sup1 = evaluated_stack.pop()
            operand2, previous_precedence2, tall2, sub2, sup2 = evaluated_stack.pop()
        else:
            raise ValueError("Unknown Operator")

        if (previous_precedence1 > operator[1]) or (sub1 and operator[0] == "subscript") or (sup1 and operator[0] == "power"):
            if tall1:
                operand1 = r"\left(" + operand1 + r"\right)"
            else:
                operand1 = "(" + operand1 + ")"
        if previous_precedence2 > operator[2]:
            if tall2:
                operand2 = r"\left(" + operand1 + r"\right)"
            else:
                operand2 = "(" + operand2 + ")"
        
        tall = tall1 or tall2
        sub = False
        sup = False
        
        match operator[0]:
            case "subscript":
                evaluated_string = operand1 + "_{" + operand2 + "}"
                sub = True
            case "add":
                evaluated_string = operand1 + " + " + operand2
            case "subtract":
                evaluated_string = operand1 + " - " + operand2
            case "multiply":
                evaluated_string = operand1 + r" \cdot " + operand2
            case "power":
                evaluated_string = operand1 + "^{" + operand2 + "}"
                sup = True
            case "juxtapose":
                evaluated_string = operand1 + " " + operand2
            case "equality":
                evaluated_string = operand1 + " = " + operand2
            case "square root":
                evaluated_string = r"\sqrt{" + operand1 + "}"
            case "fraction":
                evaluated_string = r"\dfrac{" + operand1 + "}{" + operand2 + "}"
                tall = True
            case _:
                raise ValueError("Unknown Operator")

        return evaluated_string, operator[1], tall, sub, sup

            
    # Dijkstra's two stack algorithm
    operator_stack = []
    evaluated_stack = deque()
    format_string = ""
    while tree:
        operator = tree.pop()
        tall_bracket = False
        while not ( not operator_stack or operator_stack[-1][1] > operator[1] ):
            evaluated_stack.append(apply_operator(operator_stack.pop(), evaluated_stack))
        operator_stack.append(operator)

    while operator_stack:
        evaluated_stack.append(apply_operator(operator_stack.pop(), evaluated_stack))

    tex_expression: str = evaluated_stack[-1][0]


    # replace the placeholder $
    num_symbols = tex_expression.count("$")
    symbols = choices(SYMBOLS, k=num_symbols)

    for symbol in symbols:
        tex_expression = tex_expression.replace("$", symbol, 1)
    
    return tex_expression


# %%

###########################
# Generate Image-TeX Pair #
###########################

def generate_png_image(tex_expression: str, file_basename: str) -> None:

    tex_document = r"""
    \documentclass{standalone}
    \usepackage{amsmath}

    \begin{document}
    \( $ \)
    \end{document}
    """.replace("$", tex_expression)

    with open(OUTPUT_DIRECTORY + file_basename + ".tex", "w") as t:
        t.write(tex_document)

    latex_complete = run(["latex", "--output-directory=" + OUTPUT_DIRECTORY , OUTPUT_DIRECTORY + file_basename + ".tex"], stdout=DEVNULL)

    if latex_complete.returncode != 0:
        raise ValueError("TeX is not formatted correctly.")

    dvipng_complete = run(["dvipng", "-D 256", "-o" + OUTPUT_DIRECTORY + file_basename + ".png", OUTPUT_DIRECTORY + file_basename + ".dvi"], stdout=DEVNULL)

    if dvipng_complete.returncode != 0:
        raise ValueError("Something went wrong in processing png image")

    remove(OUTPUT_DIRECTORY + file_basename + ".aux")
    remove(OUTPUT_DIRECTORY + file_basename + ".log")
    remove(OUTPUT_DIRECTORY + file_basename + ".dvi")

    with open("./generated/record.tsv", "a") as r:
        r.write(file_basename + "\t" + tex_expression + "\n")



# %%
if __name__ == "__main__":
    pass
    # generate_png_image(generate_tex_expression(), "7174")
    # generate_png_image(generate_tex_expression(), "17692")
    # generate_png_image(generate_tex_expression(), "26063")
    # generate_png_image(generate_tex_expression(), "41799")
    # generate_png_image(generate_tex_expression(), "59468")
    # for i in range(5000):
    #     generate_png_image(generate_tex_expression(), str(i))
    #     if i % 100 == 0:
    #         print(f"reached {i}th image")
'''
This is a "simple" homework to practice parsing grammars and working with the resulting parse tree.
'''

import lark

# Grammar with added support for modular division and exponentiation
grammar = r"""
    start: sum

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: exponent
        | product "*" exponent  -> mul
        | product "/" exponent  -> div
        | product "%" exponent  -> mod

    ?exponent: atom
        | exponent "**" atom   -> exp

    ?atom: NUMBER           -> number
        | "(" sum ")"       -> paren
        | atom "(" atom ")" -> mul  // for implicit multiplication like 2(3)

    NUMBER: /-?[0-9]+/

    %import common.WS_INLINE
    %ignore WS_INLINE
"""
parser = lark.Lark(grammar)


class Interpreter(lark.visitors.Interpreter):
    '''
    Compute the value of the expression.
    The interpreter class processes nodes "top down",
    starting at the root and recursively evaluating subtrees.

    >>> interpreter = Interpreter()
    >>> interpreter.visit(parser.parse("1"))
    1
    >>> interpreter.visit(parser.parse("-1"))
    -1
    >>> interpreter.visit(parser.parse("1+2"))
    3
    >>> interpreter.visit(parser.parse("1-2"))
    -1
    >>> interpreter.visit(parser.parse("(1+2)*3"))
    9
    >>> interpreter.visit(parser.parse("1+2*3"))
    7
    >>> interpreter.visit(parser.parse("1*2+3"))
    5
    >>> interpreter.visit(parser.parse("1*(2+3)"))
    5
    >>> interpreter.visit(parser.parse("(1*2)+3*4*(5-6)"))
    -10
    >>> interpreter.visit(parser.parse("((1*2)+3*4)*(5-6)"))
    -14
    >>> interpreter.visit(parser.parse("(1*(2+3)*4)*(5-6)"))
    -20
    >>> interpreter.visit(parser.parse("((1*2+(3)*4))*(5-6)"))
    -14

    Modular division:
    >>> interpreter.visit(parser.parse("1%2"))
    1
    >>> interpreter.visit(parser.parse("3%2"))
    1
    >>> interpreter.visit(parser.parse("(1+2)%3"))
    0

    Exponentiation:
    >>> interpreter.visit(parser.parse("2**1"))
    2
    >>> interpreter.visit(parser.parse("2**2"))
    4
    >>> interpreter.visit(parser.parse("2**3"))
    8
    >>> interpreter.visit(parser.parse("1+2**3"))
    9
    >>> interpreter.visit(parser.parse("(1+2)**3"))
    27
    >>> interpreter.visit(parser.parse("1+2**3+4"))
    13
    >>> interpreter.visit(parser.parse("(1+2)**(3+4)"))
    2187
    >>> interpreter.visit(parser.parse("(1+2)**3-4"))
    23
    '''
    def add(self, tree):
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left + right

    def sub(self, tree):
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left - right

    def mul(self, tree):
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left * right

    def div(self, tree):
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left // right

    def mod(self, tree):
        left = self.visit(tree.children[0])
        right = self.visit(tree.children[1])
        return left % right

    def exp(self, tree):
        base = self.visit(tree.children[0])
        exponent = self.visit(tree.children[1])
        return base ** exponent if exponent >= 0 else 0

    def number(self, tree):
        return int(tree.children[0])

    def paren(self, tree):
        return self.visit(tree.children[0])

    def visit_children(self, tree):
        ''' Custom visit_children to avoid list wrapping '''
        children = super().visit_children(tree)
        # If visit_children returned a single item list, return the item directly
        if len(children) == 1:
            return children[0]
        return children


class Simplifier(lark.Transformer):
    '''
    Compute the value of the expression.

    >>> simplifier = Simplifier()
    >>> simplifier.transform(parser.parse("1"))
    1
    >>> simplifier.transform(parser.parse("-1"))
    -1
    >>> simplifier.transform(parser.parse("1+2"))
    3
    >>> simplifier.transform(parser.parse("1-2"))
    -1
    >>> simplifier.transform(parser.parse("(1+2)*3"))
    9
    >>> simplifier.transform(parser.parse("1+2*3"))
    7
    >>> simplifier.transform(parser.parse("1*2+3"))
    5
    >>> simplifier.transform(parser.parse("1*(2+3)"))
    5
    >>> simplifier.transform(parser.parse("(1*2)+3*4*(5-6)"))
    -10
    >>> simplifier.transform(parser.parse("((1*2)+3*4)*(5-6)"))
    -14
    >>> simplifier.transform(parser.parse("(1*(2+3)*4)*(5-6)"))
    -20
    >>> simplifier.transform(parser.parse("((1*2+(3)*4))*(5-6)"))
    -14
    >>> simplifier.transform(parser.parse("1%2"))
    1
    >>> simplifier.transform(parser.parse("3%2"))
    1
    >>> simplifier.transform(parser.parse("(1+2)%3"))
    0
    >>> simplifier.transform(parser.parse("2**1"))
    2
    >>> simplifier.transform(parser.parse("2**2"))
    4
    >>> simplifier.transform(parser.parse("2**3"))
    8
    >>> simplifier.transform(parser.parse("1+2**3"))
    9
    >>> simplifier.transform(parser.parse("(1+2)**3"))
    27
    >>> simplifier.transform(parser.parse("1+2**3+4"))
    13
    >>> simplifier.transform(parser.parse("(1+2)**(3+4)"))
    2187
    >>> simplifier.transform(parser.parse("(1+2)**3-4"))
    23
    '''
    
    def start(self, items):
        return items[0]  # Return the computed value directly

    def add(self, items):
        return items[0] + items[1]

    def sub(self, items):
        return items[0] - items[1]

    def mul(self, items):
        return items[0] * items[1]

    def div(self, items):
        return items[0] // items[1]

    def mod(self, items):
        return items[0] % items[1]

    def exp(self, items):
        base, exponent = items[0], items[1]
        return base ** exponent if exponent >= 0 else 0

    def number(self, items):
        return int(items[0])

    def paren(self, items):
        return items[0]


########################################
# other transformations
########################################


def minify(expr):
    '''
    Minifies code by removing unnecessary whitespace and parentheses.

    >>> minify("1 + 2")
    '1+2'
    >>> minify("1 + ((((2))))")
    '1+2'
    >>> minify("1 + (2*3)")
    '1+2*3'
    >>> minify("1 + (2/3)")
    '1+2/3'
    '''
    class RemoveParentheses(lark.Transformer):
        def paren(self, items):
            return items[0]  # Remove unneeded parentheses

    class ToString(lark.Transformer):
        def add(self, items):
            return f"{items[0]}+{items[1]}"
        def sub(self, items):
            return f"{items[0]}-{items[1]}"
        def mul(self, items):
            return f"{items[0]}*{items[1]}"
        def div(self, items):
            return f"{items[0]}/{items[1]}"
        def mod(self, items):
            return f"{items[0]}%{items[1]}"
        def exp(self, items):
            return f"{items[0]}**{items[1]}"
        def number(self, items):
            return str(items[0])

    tree = parser.parse(expr)
    tree = RemoveParentheses().transform(tree)
    result = ToString().transform(tree)
    return result.children[0] if isinstance(result, lark.Tree) else result


def infix_to_rpn(expr):
    '''
    Converts infix notation to reverse polish notation (RPN).

    >>> infix_to_rpn('1')
    '1'
    >>> infix_to_rpn('1+2')
    '1 2 +'
    >>> infix_to_rpn('1-2')
    '1 2 -'
    >>> infix_to_rpn('(1+2)*3')
    '1 2 + 3 *'
    >>> infix_to_rpn('1+2*3')
    '1 2 3 * +'
    '''
    class RPNTransformer(lark.Transformer):
        def add(self, items):
            return f"{items[0]} {items[1]} +"
        def sub(self, items):
            return f"{items[0]} {items[1]} -"
        def mul(self, items):
            return f"{items[0]} {items[1]} *"
        def div(self, items):
            return f"{items[0]} {items[1]} /"
        def mod(self, items):
            return f"{items[0]} {items[1]} %"
        def exp(self, items):
            return f"{items[0]} {items[1]} **"
        def number(self, items):
            return str(items[0])
        def paren(self, items):
            return items[0]  # Ignore parentheses in RPN

    tree = parser.parse(expr)
    result = RPNTransformer().transform(tree)
    return result.children[0] if isinstance(result, lark.Tree) else result


def eval_rpn(expr):
    '''
    Evaluates an expression written in RPN.

    >>> eval_rpn("1")
    1
    >>> eval_rpn("1 2 +")
    3
    >>> eval_rpn("1 2 -")
    -1
    '''
    tokens = expr.split()
    stack = []
    operators = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a // b,
        '%': lambda a, b: a % b,
        '**': lambda a, b: a ** b
    }
    for token in tokens:
        if token not in operators:
            stack.append(int(token))
        else:
            b, a = stack.pop(), stack.pop()  # Correct order for b, a
            stack.append(operators[token](a, b))
    return stack[0]
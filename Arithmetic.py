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

    FIXME:
    Get all the test cases to pass.
    '''
    
    def add(self, tree):
        return self.visit(tree.children[0]) + self.visit(tree.children[1])

    def sub(self, tree):
        return self.visit(tree.children[0]) - self.visit(tree.children[1])

    def mul(self, tree):
        return self.visit(tree.children[0]) * self.visit(tree.children[1])

    def div(self, tree):
        return self.visit(tree.children[0]) // self.visit(tree.children[1])

    def mod(self, tree):
        return self.visit(tree.children[0]) % self.visit(tree.children[1])

    def exp(self, tree):
        base = self.visit(tree.children[0])
        exponent = self.visit(tree.children[1])
        return base ** exponent if exponent >= 0 else 0

    def number(self, tree):
        return int(tree.children[0])

    def paren(self, tree):
        return self.visit(tree.children[0])


class Simplifier(lark.Transformer):
    '''
    Compute the value of the expression.
    The lark.Transformer class processes nodes "bottom up",
    starting at the leaves and ending at the root.
    In general, the Transformer class is less powerful than the Interpreter class.
    But in the case of simple arithmetic expressions,
    both classes can be used to evaluate the expression.

    FIXME:
    This class contains all of the same test cases as the Interpreter class.
    You should fix all the failing test cases.
    You shouldn't need to make any additional modifications to the grammar beyond what was needed for the interpreter class.
    You should notice that the functions in the lark.Transformer class are simpler to implement because you do not have to manage the recursion yourself.
    '''
    
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
    "Minifying" code is the process of removing unnecessary characters.
    In our arithmetic language, this means removing unnecessary whitespace and unnecessary parentheses.
    It is common to minify code in order to save disk space and bandwidth.
    For example, google penalizes a web site's search ranking if they don't minify their html/javascript code.

    FIXME:
    Implement this function so that the test cases below pass.

    HINT:
    My solution uses two lark.Transformer classes.
    The first one takes an AST and removes any unneeded parentheses.
    The second taks an AST and converts the AST into a string.
    You can solve this problem by calling parser.parse,
    and then applying the two transformers above to the resulting AST.

    NOTE:
    It is important that these types of "syntactic" transformations use the Transformer class and not the Interpreter class.
    If we used the Interpreter class, we could "accidentally do too much computation",
    but the Transformer class's leaf-to-root workflow prevents this class of bug.

    NOTE:
    The test cases below do not require any of the "new" features that you are required to add to the Arithmetic grammar.
    It only uses the features in the starting code.
    '''
    class RemoveParentheses(lark.Transformer):
        def paren(self, items):
            return items[0]
    
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
        
        def paren(self, items):
            return f"({items[0]})"
    
    tree = parser.parse(expr)
    tree = RemoveParentheses().transform(tree)
    return ToString().transform(tree)


def infix_to_rpn(expr):
    '''
    This function takes an expression in standard infix notation and converts it into an expression in reverse polish notation.
    This type of translation task is commonly done by first converting the input expression into an AST (i.e. by calling parser.parse),
    and then simplifying the AST in a leaf-to-root manner (i.e. using the Transformer class).

    HINT:
    If you need help understanding reverse polish notation,
    see the eval_rpn function.
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
    
    tree = parser.parse(expr)
    return RPNTransformer().transform(tree)


def eval_rpn(expr):
    '''
    This function evaluates an expression written in RPN.

    RPN (Reverse Polish Notation) is an alternative syntax for arithmetic.
    It was widely used in the first scientific calculators because it is much easier to parse than standard infix notation.
    For example, parentheses are never needed to disambiguate order of operations.
    Parsing of RPN is so easy, that it is usually done at the same time as evaluation without a separate parsing phase.
    More complicated languages (like the infix language above) are basically always implemented with separate parsing/evaluation phases.

    You can find more details on wikipedia: <https://en.wikipedia.org/wiki/Reverse_Polish_notation>.

    NOTE:
    There is nothing to implement for this function,
    it is only provided as a reference for understanding the infix_to_rpn function.
    '''
    tokens = expr.split()
    stack = []
    operators = {
        '+': lambda a, b: a+b,
        '-': lambda a, b: a-b,
        '*': lambda a, b: a*b,
        '/': lambda a, b: a//b,
        '%': lambda a, b: a%b,
        '**': lambda a, b: a**b
    }
    for token in tokens:
        if token not in operators.keys():
            stack.append(int(token))
        else:
            assert len(stack) >= 2
            v1 = stack.pop()
            v2 = stack.pop()
            stack.append(operators[token](v2, v1))
    assert len(stack) == 1
    return stack[0]
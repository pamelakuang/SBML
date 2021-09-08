# Pamela Kuang
# Note: I've used Professor's Kane in class examples to define my Node class and AST.
import sys

symbolTable = {}
functions = {}

    
class SemanticError(Exception):
    pass
    
    
class Node():
    def __init__(self):
        self.parent = None
    
    def parentCount(self):
        count = 0
        current = self.parent
        while current is not None:
            count += 1
            current = current.parent
        return count


class Negation(Node):
    def __init__(self, child):
        super().__init__()
        self.child = child
    
    def eval(self):
        compat_types = [bool]
        if type(self.child.eval()) in compat_types:
            return not self.child.eval()


class Conjunction(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        compat_types = [bool]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() and self.right.eval()


class Disjunction(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
    
    def eval(self):
        compat_types = [bool]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() or self.right.eval()


class Addition(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        if self.left is not None and self.right is not None:
            left_eval = self.left.eval()
            right_eval = self.right.eval()
            if type(self.left) == String:
                left_eval = left_eval[1:len(left_eval)-1]
            if type(self.right) == String:
                right_eval = right_eval[1:len(right_eval)-1]
            compat_types = [int, float]
            if type(left_eval) in compat_types and type(right_eval) in compat_types:
                return left_eval + right_eval
            elif type(left_eval) == str and type(right_eval) == str:
                return left_eval + right_eval
            elif type(left_eval) == list and type(right_eval) == list:
                return left_eval + right_eval


class Subtraction(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() - self.right.eval()


class Multiplication(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
    
    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() * self.right.eval()


class Division(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return float(self.left.eval() / self.right.eval())


class IntegerDivision(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        if type(self.left.eval()) == int and type(self.right.eval()) == int:
            return int(self.left.eval() / self.right.eval())


class Modulus(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        if type(self.left.eval()) == int and type(self.right.eval()) == int:
            return int(self.left.eval() % self.right.eval())


class Cons(Node):
    def __init__(self, head, tail):
        super().__init__()
        self.head = head
        self.tail = tail

    def eval(self):
        if type(self.tail.eval()) == list:
            return [self.head.eval()] + self.tail.eval()
        #return "SEMANTIC ERROR"


class Membership(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        if type(self.right) == str or type(self.right) == list:
            return self.left.eval() in self.right.eval()


class LessThan(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() < self.right.eval()
        else:
            if type(self.left.eval()) == str and type(self.right.eval()) == str:
                return self.left.eval() < self.right.eval()


class LessThanEqual(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
    
    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() <= self.right.eval()
        else:
            if type(self.left.eval()) == str and type(self.right.eval()) == str:
                return self.left.eval() <= self.right.eval()
       # return "SEMANTIC ERROR"


class GreaterThan(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() > self.right.eval()
        else:
            if type(self.left.eval()) == str and type(self.right.eval()) == str:
                return self.left.eval() > self.right.eval()
        #return "SEMANTIC ERROR"


class GreaterThanEqual(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        
    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() >= self.right.eval()
        else:
            if type(self.left.eval()) == str and type(self.right.eval()) == str:
                return self.left.eval() >= self.right.eval()
        #return "SEMANTIC ERROR"

class NotEqual(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        
    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() != self.right.eval()
        else:
            if type(self.left.eval()) == str and type(self.right.eval()) == str:
                return self.left.eval() != self.right.eval()



class EqualTo(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() == self.right.eval()
        else:
            if type(self.left.eval()) == str and type(self.right.eval()) == str:
                return self.left.eval() == self.right.eval()
        #return "SEMANTIC ERROR"


class Exponentiation(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def eval(self):
        compat_types = [int, float]
        if type(self.left.eval()) in compat_types and type(self.right.eval()) in compat_types:
            return self.left.eval() ** self.right.eval()

class String(Node):
    def __init__(self, val):
        super().__init__()
        self.value = val[1:len(val)-1]

    def eval(self):
        return "\"" + str(self.value) + "\""

    # def __str__(self):
    #     res = "\t" * self.parentCount() + "String"
    #     return res


class Real(Node):
    def __init__(self, val):
        super().__init__()
        self.value = val
    
    def eval(self):
        return self.value
    
    # def __str__(self):
    #     res = "\t" * self.parentCount() + "Real"
    #     return res


class Integer(Node):
    def __init__(self, val):
        super().__init__()
        self.value = val

    def eval(self):
        return self.value

    # def __str__(self):
    #     res = "\t" * self.parentCount() + "Integer"
    #     return res


class Uminus(Node):
    def __init__(self, val):
        super().__init__()
        self.value = val.eval() * -1

    def eval(self):
        return self.value

    # def __str__(self):
    #     res = "\t" * self.parentCount() + "Uminus"
    #     return res


class AST_True(Node):
    def __init__(self):
        super().__init__()
        self.value = True
    
    def eval(self):
        return self.value
    
    # def __str__(self):
    #     res = "\t" * self.parentCount() + "True"
    #     return res


class AST_False(Node):
    def __init__(self):
        super().__init__()
        self.value = False
    
    def eval(self):
        return self.value
    
    # def __str__(self):
    #     res = "\t" * self.parentCount() + "False"
    #     return res


class List(Node):
    def __init__(self, expr, tail):
        super().__init__()
        if expr is None and tail is None:
            self.child = None
            self.tail = None
        elif tail is None:
            self.child = expr
            self.tail = None
            self.child.parent = self
        else:
            self.child = expr
            self.child.parent = self
            self.tail = tail
            self.tail.parent = self

    def eval(self):
        if self.child is None and self.tail is None:
            return []
        elif self.tail is None:
            return [self.child.eval()]
        else:
            return self.child.eval() + [self.tail.eval()]
            #return [self.child.eval()] + self.tail.eval()

    # def __str__(self):
    #     res = "\t" * self.parentCount() + "List"
    #     return res


class Indexing(Node):
    def __init__(self, expression, index):
        super().__init__()
        index_type = type(index.eval())
        expression_type = type(expression.eval())
        if index_type == int and (expression_type == list or expression_type == str):
            if expression_type == str:
                length = len(expression.eval()[1:len(expression.eval())-1])
            else:
                length = len(expression.eval())
            if 0 <= index.eval() < length:
                self.expression = expression
                self.index = index
                self.expression.parent = self
                self.index.parent = self
            else:
                self.expression = None
                self.index = None
        else:
            self.expression = None
            self.index = None

    def eval(self):
        if self.expression is not None and self.index is not None:
            if isinstance(self.expression, List) or isinstance(self.expression, Indexing):
                return (self.expression.eval())[self.index.eval()]
            else:
                temp = self.expression.eval()
                temp = temp[1:len(temp)-1]
                return temp[self.index.eval()]

    # def __str__(self):
    #     if self.expression is None or self.index is None:
    #         return "SEMANTIC ERROR"
    #     res = "\t" * self.parentCount() + "Indexing"
    #     res += "\n" + str(self.expression)
    #     return res


class Tuple(Node):
    def __init__(self, expr, tail):
        super().__init__()
        if tail is None:
            self.child = expr
            self.tail = None
            self.child.parent = self
        else:
            self.child = expr
            self.child.parent = self
            self.tail = tail
            self.tail.parent = self

    def eval(self):
        if self.tail is None:
            return (self.child.eval(),)
        else:
            return (self.child.eval(),) + self.tail.eval()

    def __str__(self):
        res = "\t" * self.parentCount() + "Tuple"
        return res


class TupleIndex(Node):
    def __init__(self, index, tup):
        super().__init__()
        self.tup = tup
        self.index = index
        
    def eval(self):
        # if self.tup is not None and self.index is not None:
        index = self.index.eval()
        tup = self.tup.eval()
        if type(tup) == tuple and type(index) == int:
            if 0 < index <= len(tup):
                return self.tup.eval()[self.index.eval()-1]

    # def __str__(self):
    #     if self.tup is None or self.index is None:
    #         return "SEMANTIC ERROR"
    #     res = "\t" * self.parentCount() + "Tuple Indexing"
    #     res += "\n" + str(self.tup)
    #     return res

class VariableTupleIndex(Node):
    def __init__(self, index, tup):
           super().__init__()
           self.tup = tup
           self.index = index
           
    def eval(self):
        if type(self.index.eval()) == int and type(symbolTable[self.tup]) == tuple:
            if self.tup in symbolTable:
                if 0 < self.index.eval() <= len(symbolTable[self.tup]):
                    return symbolTable[self.tup][self.index.eval()-1]
            else:
                return "SEMANTIC ERROR"
        else:
            return "SEMANTIC ERROR"
        

class VariableList(Node):
    def __init__(self, name, index):
        super().__init__()
        self.name = name
        self.index = index

    def eval(self):
        if type(self.name) is VariableList:
            if 0 <= self.index.eval() < len(symbolTable[self.name.name]):
                return self.name.eval()[self.index.eval()]
        else:
            if self.name in symbolTable:
                if 0 <= self.index.eval() < len(symbolTable[self.name]):
                    if type(symbolTable[self.name]) == str:
                        if 0 <= self.index.eval() < (len(symbolTable[self.name]) - 2):
                            return symbolTable[self.name][self.index.eval() + 1]
                    else:
                        return symbolTable[self.name][self.index.eval()]


class VariableListAssign(Node):
    def __init__(self, name, index, value):
        super().__init__()
        self.name = name
        self.index = index
        self.value = value

    def eval(self):
        if self.name in symbolTable:
            if 0 <= self.index.eval() < len(symbolTable[self.name]):
                symbolTable[self.name][self.index.eval()] = self.value.eval()
            else:
                return "SEMANTIC ERROR"
        else:
            return "SEMANTIC ERROR"


class Variable(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def eval(self):
        if self.name in symbolTable:
            return symbolTable[self.name]
        else:
            return "SEMANTIC ERROR"


class Assignment(Node):
    def __init__(self, lvalue, rvalue):
        super().__init__()
        self.lvalue = lvalue
        self.rvalue = rvalue

    def eval(self):
        symbolTable[self.lvalue] = self.rvalue.eval()


class Print(Node):
    def __init__(self, expr):
        super().__init__()
        self.expression = expr

    def eval(self):
        temp = self.expression.eval()
        if temp is None or temp == "SEMANTIC ERROR":
            return "SEMANTIC ERROR"
        print(temp)


class Block(Node):
    def __init__(self, statements):
        super().__init__()
        self.statements = statements

    def eval(self):
    # block may be empty
        if self.statements is not None:
            for s in self.statements:
                temp = s.eval()
                if temp == "SEMANTIC ERROR":
                    #print("SEMANTIC ERROR")
                    raise SemanticError
                    break


class IfStatement(Node):
    def __init__(self, expr, block):
        super().__init__()
        self.expression = expr
        self.block = block

    def eval(self):
        # if self.expression.eval() == "SEMANTIC ERROR" or self.block.eval() == "SEMANTIC ERROR":
        #     return "SEMANTIC ERROR"
        if self.expression.eval() is True:
            return self.block.eval()


class IfElseStatement(Node):
    def __init__(self, expr, if_block, else_block ):
        super().__init__()
        self.expression = expr
        self.if_block = if_block
        self.else_block = else_block

    def eval(self):
        # if self.expression.eval() == "SEMANTIC ERROR":
        #     return "SEMANTIC ERROR"
        if self.expression.eval() is True:
            return self.if_block.eval()
        else:
            return self.else_block.eval()


class While(Node):
    def __init__(self, condition, block):
        super().__init__()
        self.condition = condition
        self.block = block

    def eval(self):
        while self.condition.eval():
            self.block.eval()


class FunctionDef(Node):
    def __init__(self, fun_name, param, block, return_expr):
        super().__init__()
        self.fun_name = fun_name
        self.param = param
        self.block = block
        self.return_expr = return_expr
        functions[self.fun_name] = self

    def eval(self):
        self.block.eval()


class FunctionCall(Node):
    def __init__(self, fun_name, args):
        super().__init__()
        self.fun_name = fun_name
        self.args = args

    def eval(self):
        global symbolTable
        global functions
        # get the function definition that is being called
        if self.fun_name not in functions:
            return "SEMANTIC ERROR"
        fun = functions[self.fun_name]
        temp = symbolTable
        local_vars = {}
        # params are not empty
        if self.args is not None and fun.param is not None:
            if len(fun.param) != len(self.args):
                return "SEMANTIC ERROR"
            for index in range(len(fun.param)):
                local_vars[fun.param[index]] = self.args[index].eval()
        elif self.args is None and fun.param is None:
            pass
        # num of args dont match num of params
        else:
            return "SEMANTIC ERROR"
        # replace symbolTable with variables in local scope of function def
        symbolTable = local_vars
        fun.eval()
        return_expr = fun.return_expr.eval()
        # return old symbolTable (scope before function call)
        symbolTable = temp
        if return_expr is None:
            return "SEMANTIC ERROR"
        return return_expr


reserved = {
            'True' : 'True',
            'False' : 'False',
            'print' : 'print',
            'if' : 'if',
            'else' : 'else',
            'while' : 'while',
            'andalso' : 'CONJUNCTION',
            'mod' : 'MODULUS',
            'orelse' : 'DISJUNCTION',
            'not' : 'NEGATION',
            'div' : 'INTEGER_DIVISION',
            'in' : 'MEMBERSHIP',
            'fun' : 'FUNCTION',
}

tokens = ['LEFT_PARENTHESIS',
          'RIGHT_PARENTHESIS',
          'EXPONENTIATION',
          'MULTIPLICATION',
          'DIVISION',
          # 'INTEGER_DIVISION',
          # 'MODULUS',
          'ADDITION',
          'SUBTRACTION',
          # 'MEMBERSHIP',
          'CONS',
          # 'NEGATION',
          # 'CONJUNCTION',
          # 'DISJUNCTION',
          'LESS_THAN',
          'LESS_THAN_EQUAL',
          'NOT_EQUAL',
          'EQUAL',
          'GREATER_THAN',
          'GREATER_THAN_EQUAL',
          'INTEGER',
          'REAL',
          'STRING',
          'VARIABLE',
          'ASSIGNMENT',
          'LEFT_BRACKET',
          'RIGHT_BRACKET',
          'COMMA',
          'TUPLE_INDEX',
          'LEFT_BRACE',
          'RIGHT_BRACE',
          'SEMICOLON'
]

tokens += list(reserved.values())

t_LEFT_PARENTHESIS = r'\('
t_RIGHT_PARENTHESIS = r'\)'
t_EXPONENTIATION = r'\*\*'
t_MULTIPLICATION = r'\*'
t_DIVISION = r'/'
# t_INTEGER_DIVISION = r'div'
# t_MODULUS = r'mod'
t_ADDITION = r'\+'
t_SUBTRACTION = r'-'
# t_MEMBERSHIP = r'in'
t_CONS = r'::'
# t_NEGATION = r'not'
# t_CONJUNCTION = r'andalso'
# t_DISJUNCTION = r'orelse'
t_LESS_THAN = r'<'
t_LESS_THAN_EQUAL = r'<='
t_NOT_EQUAL = r'<>'
t_GREATER_THAN = r'>'
t_GREATER_THAN_EQUAL = r'>='
t_EQUAL = r'=='
t_LEFT_BRACKET = r'\['
t_RIGHT_BRACKET = r'\]'
t_COMMA = r','
t_TUPLE_INDEX = r'\#'
t_ASSIGNMENT = r'='
t_LEFT_BRACE = r'\{'
t_RIGHT_BRACE = r'\}'
t_SEMICOLON = r';'


def t_VARIABLE(t):
    r'[a-zA-Z][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'VARIABLE')
    return t


def t_STRING(t):
    r'([\"][^\"\']*[\"])|([\'][^\"\']*[\'])'
    t.value = str(t.value)
    return t


def t_REAL(t):
    r'[-]?[0-9]*[.][0-9]*([e][-]?)?[0-9]*'
    # check case for when t is only a "."
    if t.value != ".":
        try:
            t.value = float(t.value)
        except ValueError:
            t.value = 0
        return t


def t_INTEGER(t):
    r'\d+'
    try:
        t.value = int(t.value)
    except ValueError:
        t.value = 0
    return t


t_ignore = ' \t'


def t_newline(t):
    r'\n+'
    t.lexer.lineno =+ t.value.count("\n")


def t_error(t):
    print("Illegal character '%s', at %d, %d" % (t.value[0], t.lineno, t.lexpos))
    t.lexer.skip(1)


import ply.lex as lex 
lexer = lex.lex(debug = False)


def tokenize(inp):
    lexer.input(inp)
    while True:
        tok = lexer.token()
        if not tok:
            break


precedence = (('left', 'DISJUNCTION'),
            ('left', 'CONJUNCTION'),
            ('left', 'NEGATION'),
            ('left', 'LESS_THAN', 'LESS_THAN_EQUAL', 'NOT_EQUAL', 'EQUAL', 'GREATER_THAN', 'GREATER_THAN_EQUAL'),
            ('right', 'CONS'),
            ('left', 'MEMBERSHIP'),
            ('left', 'ADDITION', 'SUBTRACTION'),
            ('left', 'MULTIPLICATION', 'DIVISION', 'INTEGER_DIVISION', 'MODULUS'),
            ('right', 'UMINUS'),
            ('right', 'EXPONENTIATION'),
            ('left', 'INDEXING'),
            ('left', 'TUPLE_INDEXING'),
            ('left', 'TUPLE_CREATION'),
            ('left', 'PARENTHETICAL_EXPRESSION'),
)


def p_expr_block_exp(p):
    '''expression : BLOCK
                | functions BLOCK'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]


def p_expr_block(p):
    '''BLOCK : LEFT_BRACE block_item_tail RIGHT_BRACE
            | LEFT_BRACE RIGHT_BRACE'''
    if len(p) == 3:
        p[0] = Block(None)
    elif len(p) > 3:
        p[0] = Block(p[2])


def p_block_item_tail(p):
    '''block_item_tail : block_item_tail statement
                    | statement'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) > 2:
        p[0] = p[1] + [p[2]]


def p_expr_statement_fun_call(p):
    'statement : FUNCTION_CALL SEMICOLON'
    p[0] = p[1]


def p_expr_param(p):
    '''param : VARIABLE
            | VARIABLE COMMA param'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) > 2:
        p[0] = [p[1]] + p[3]


def p_expr_nested_fun_def(p):
    '''functions : FUNCTION_DEF
                | functions FUNCTION_DEF'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) > 2:
        p[0] = p[1] + [p[2]]


def p_expr_function_def(p):
    '''FUNCTION_DEF : FUNCTION VARIABLE LEFT_PARENTHESIS param RIGHT_PARENTHESIS ASSIGNMENT BLOCK prop SEMICOLON
                | FUNCTION VARIABLE LEFT_PARENTHESIS RIGHT_PARENTHESIS ASSIGNMENT BLOCK prop SEMICOLON'''
    if len(p) == 10:
        p[0] = FunctionDef(p[2], p[4], p[7], p[8])
    # no params
    elif len(p) == 9:
        p[0] = FunctionDef(p[2], None, p[6], p[7])


def p_expr_function_call(p):
    '''FUNCTION_CALL : VARIABLE LEFT_PARENTHESIS fun_args RIGHT_PARENTHESIS
                    | VARIABLE LEFT_PARENTHESIS RIGHT_PARENTHESIS'''
    if len(p) == 5:
        p[0] = FunctionCall(p[1], p[3])
    # no args
    elif len(p) == 4:
        p[0] = FunctionCall(p[1], None)


def p_expr_fun_args(p):
    '''fun_args : prop
                | prop COMMA fun_args'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) > 2:
        p[0] = [p[1]] + p[3]


def p_expr_nested_var_list(p):
    '''prop : variable LEFT_BRACKET prop RIGHT_BRACKET
            | variable '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = VariableList(p[1], p[3])


def p_expr_variable_list(p):
    '''variable : VARIABLE LEFT_BRACKET prop RIGHT_BRACKET'''
    p[0] = VariableList(p[1], p[3])


def p_expr_variable(p):
    '''prop : VARIABLE'''
    p[0] = Variable(p[1])


def p_expr_print(p):
    'statement : print LEFT_PARENTHESIS prop RIGHT_PARENTHESIS SEMICOLON'
    p[0] = Print(p[3])


def p_expr_prop_fun_call(p):
    'prop : FUNCTION_CALL'
    p[0] = p[1]


def p_expr_if_statement(p):
    'statement : if LEFT_PARENTHESIS prop RIGHT_PARENTHESIS BLOCK'
    p[0] = IfStatement(p[3], p[5])


def p_expr_if_else_statement(p):
    'statement : if LEFT_PARENTHESIS prop RIGHT_PARENTHESIS BLOCK else BLOCK'
    p[0] = IfElseStatement(p[3], p[5], p[7])


def p_expr_while_statement(p):
    'statement : while LEFT_PARENTHESIS prop RIGHT_PARENTHESIS BLOCK'
    p[0] = While(p[3], p[5])


def p_expr_assignment(p):
    '''statement : VARIABLE ASSIGNMENT prop SEMICOLON'''
    p[0] = Assignment(p[1], p[3])


def p_expr_variable_list_assign(p):
    'statement : VARIABLE LEFT_BRACKET prop RIGHT_BRACKET ASSIGNMENT prop SEMICOLON'
    p[0] = VariableListAssign(p[1], p[3], p[6])


def p_expr_variable_tuple_indexing(p):
    'prop : TUPLE_INDEX prop VARIABLE %prec TUPLE_INDEXING'
    p[0] = VariableTupleIndex(p[2], p[3])


def p_expr_disjunction(p):
    'prop : prop DISJUNCTION prop'
    p[0] = Disjunction(p[1], p[3])


def p_expr_conjunction(p):
    'prop : prop CONJUNCTION prop'
    p[0] = Conjunction(p[1], p[3])


def p_expr_negation(p):
    'prop : NEGATION prop'
    p[0] = Negation(p[2])


def p_expr_less_than(p):
    'prop : prop LESS_THAN prop'
    p[0] = LessThan(p[1], p[3])


def p_expr_less_than_equal(p):
    'prop : prop LESS_THAN_EQUAL prop'
    p[0] = LessThanEqual(p[1], p[3])


def p_expr_not_equal(p):
    'prop : prop NOT_EQUAL prop'
    p[0] = NotEqual(p[1], p[3])


def p_expr_greater_than(p):
    'prop : prop GREATER_THAN prop'
    p[0] = GreaterThan(p[1], p[3])


def p_expr_greater_than_equal(p):
    'prop : prop GREATER_THAN_EQUAL prop'
    p[0] = GreaterThanEqual(p[1], p[3])


def p_expr_equal_to(p):
    'prop : prop EQUAL prop'
    p[0] = EqualTo(p[1], p[3])


def p_expr_cons(p):
    'prop : prop CONS prop'
    p[0] = Cons(p[1], p[3])


def p_expr_membership(p):
    'prop : prop MEMBERSHIP prop'
    p[0] = Membership(p[1], p[3])


def p_expr_addition(p):
    'prop : prop ADDITION prop'
    p[0] = Addition(p[1], p[3])


def p_expr_subtraction(p):
    'prop : prop SUBTRACTION prop'
    p[0] = Subtraction(p[1], p[3])


def p_expr_multiplication(p):
    'prop : prop MULTIPLICATION prop'
    p[0] = Multiplication(p[1], p[3])


def p_expr_division(p):
    'prop : prop DIVISION prop'
    p[0] = Division(p[1], p[3])


def p_expr_int_division(p):
    'prop : prop INTEGER_DIVISION prop'
    p[0] = IntegerDivision(p[1], p[3])


def p_expr_exponentiation(p):
    'prop : prop EXPONENTIATION prop'
    p[0] = Exponentiation(p[1], p[3])


def p_expr_modulus(p):
    'prop : prop MODULUS prop'
    p[0] = Modulus(p[1], p[3])


def p_expr_uminus(p):
    'prop : SUBTRACTION prop %prec UMINUS'
    p[0] = Uminus(p[2])


def p_expr_nested_indexing(p):
    '''prop : indexing LEFT_BRACKET prop RIGHT_BRACKET %prec INDEXING
                | indexing %prec INDEXING'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = Indexing(p[1], p[3])
        #p[0] = p[1][p[3]]


def p_expr_indexing(p):
    '''indexing : STRING LEFT_BRACKET prop RIGHT_BRACKET %prec INDEXING
                 | LIST LEFT_BRACKET prop RIGHT_BRACKET %prec INDEXING'''
    if type(p[1]) == str:
        p[1] = String(p[1])
    p[0] = Indexing(p[1], p[3])


def p_arguments(p):
    '''arguments : arguments COMMA prop
                | prop'''
    if len(p) == 2:
        p[0] = List(p[1], None)
    elif len(p) > 2:
        p[0] = List(p[1], p[3])


def p_expr_list_creation(p):
    '''LIST : LEFT_BRACKET arguments RIGHT_BRACKET'''
    p[0] = p[2]


def p_expr_list(p):
    'prop : LIST'
    p[0] = p[1]


def p_expr_empty_list(p):
    'LIST : LEFT_BRACKET RIGHT_BRACKET'
    p[0] = List(None, None)


def p_args(p):
    '''args : prop
            | prop COMMA args'''
    if len(p) > 2:
        p[0] = Tuple(p[1], p[3])
    else:
        p[0] = Tuple(p[1], None)


def p_expr_tuple_indexing(p):
    'prop : TUPLE_INDEX prop LEFT_PARENTHESIS args RIGHT_PARENTHESIS %prec TUPLE_INDEXING'
    p[0] = TupleIndex(p[2], p[4])


def p_expr_tuple_creation(p):
    'prop : LEFT_PARENTHESIS args RIGHT_PARENTHESIS %prec TUPLE_CREATION'
    p[0] = p[2]


def p_expr_parenthetical_expression(p):
    'prop : LEFT_PARENTHESIS prop RIGHT_PARENTHESIS %prec PARENTHETICAL_EXPRESSION'
    p[0] = p[2]


def p_expr_real(p):
    'prop : REAL'
    p[0] = Real(p[1])


def p_expr_integer(p):
    'prop : INTEGER'
    p[0] = Integer(p[1])


def p_expr_string(p):
    'prop : STRING'
    p[0] = String(p[1])


def p_expr_true(p):
    'prop : True'
    p[0] = AST_True()


def p_expr_false(p):
    'prop : False'
    p[0] = AST_False()


def p_error(p):
    #print("Syntax error at '%s' (%d, %d)" % (p.value, p.lineno, p.lexpos))
    print("SYNTAX ERROR")
    sys.exit()


import ply.yacc as yacc
parser = yacc.yacc(debug = False)


def parse(inp):
    result = parser.parse(inp, debug = 0)
    return result


def main():
    if len(sys.argv) != 2:
        print("please enter the name of the script followed by an input file")
    else:
        input_file = sys.argv[1]
        # tokenize(inp)
        try:
            f = open(input_file, "r")
            st = ""
            for x in f:
                st = st + x
            result = parse(st)
            result.eval()
        except SemanticError:
            print("SEMANTIC ERROR")

if __name__ == "__main__":
    main()

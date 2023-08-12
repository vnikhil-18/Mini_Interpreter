INTEGER = 'INTEGER'
REAL = 'REAL'
INTEGER_CONST = 'INTEGER_CONST'
REAL_CONST = 'REAL_CONST'
PLUS = 'PLUS'
IF = 'IF'
MINUS = 'MINUS'
MUL = 'MUL'
INTEGER_DIV = 'INTEGER_DIV'
FLOAT_DIV = 'FLOAT_DIV'
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'
ID = 'ID'
ASSIGN = 'ASSIGN'
begin = 'begin'
end = 'end'
SEMI = 'SEMI'
DOT = 'DOT'
COLON = 'COLON'
COMMA = 'COMMA'
EOF = 'EOF'
TRUE = 'TRUE'
FALSE = 'FALSE'
GREATER_THAN = 'GREATER_THAN'
LESS_THAN = 'LESS_THAN'
EQUALS = 'EQUALS'
NOT_EQUALS = 'NOT_EQUALS'
AND = 'AND'
OR = 'OR'
PROGRAM = 'PROGRAM'
println = 'println'
QUOTES = '"'


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


RESERVED_KEYWORDS = {
    'begin': Token('begin', 'begin'),
    'end': Token('end', 'end'),
    'if': Token('IF', 'if'),
    'elseif': Token('ELSEIF', 'elseif'),
    'else': Token('ELSE', 'else'),
    'println': Token('println', 'println'),
    'True': Token('TRUE', True),
    'False': Token('FALSE', False)
}


class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self, x):
        if x:
            while self.current_char != '=' or self.peek() != '#':
                self.advance()
            self.advance()
        else:
            while self.current_char != '\n':
                self.advance()
            self.advance()

    def number(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        if self.current_char == '.':
            result += self.current_char
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()

            token = Token('REAL_CONST', float(result))
        else:
            token = Token('INTEGER_CONST', int(result))
        return token

    def _id(self):
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result, Token(ID, result))
        return token

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '#':
                if self.peek() == '=':
                    self.skip_comment(True)
                else:
                    self.skip_comment(False)
                self.advance()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '=' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(EQUALS, '==')
            elif self.current_char == '=':
                self.advance()
                return Token(ASSIGN, '=')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '>':
                self.advance()
                return Token(GREATER_THAN, '>')

            if self.current_char == '<':
                self.advance()
                return Token(LESS_THAN, '<')
            if self.current_char == '!' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(NOT_EQUALS, '!')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')
            if self.current_char == '"':
                self.advance()
                return Token(QUOTES, '"')
            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(FLOAT_DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == '&' and self.peek() == '&':
                self.advance()
                self.advance()
                return Token(AND, '&')

            if self.current_char == '|' and self.peek() == '|':
                self.advance()
                self.advance()
                return Token(OR, '|')

            self.error()

        return Token(EOF, None)


class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class BinIF(AST):
    def __init__(self, cNode, nodes, other_stmt):
        self.token = cNode
        self.left = self.nodes = nodes
        self.right = other_stmt
class stamnt(AST):
    def _init_(self):
        self.children=[]

class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class Bool(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value
        self.op = Token(None, 'None')


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Compound(AST):
    def __init__(self):
        self.children = []


class Condition(AST):
    def __init__(self):
        self.children = []


class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Var(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class NoOp(AST):
    pass


class Program(AST):
    def __init__(self, block):
        self.block = block


class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement


class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node


class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        block_node = self.block()
        program_node = Program(block_node)
        return program_node

    def block(self):
        declarations = []
        if self.current_token.type == 'LPAREN' or self.current_token.type == 'begin' or self.current_token.type == 'ID':
            compound_statement_node = self.compound_statement()
            node = Block(declarations, compound_statement_node)
            return node
        elif self.current_token.type == 'IF':
            conditional_statement_node = self.conditional_statement()
            node = Block(declarations, conditional_statement_node)
            return node
        elif self.current_token.type == 'println':
            return self.print_statement()

    def type_spec(self):
        token = self.current_token
        if self.current_token.type == INTEGER:
            self.eat(INTEGER)
        else:
            self.eat(REAL)
        node = Type(token)
        return node

    def compound_statement(self):
        if self.current_token.type == 'begin' or self.current_token.type == 'println':
            self.eat(begin)
            nodes = self.statement_list()
            self.eat(end)
        elif self.current_token.type == 'LPAREN':
            self.eat(LPAREN)
            nodes = self.statement_list()
            self.eat(RPAREN)
        elif self.current_token.type == 'ID':
            nodes = self.statement_list()
        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        node = self.statement()
        results = [node]
        while self.current_token.type == SEMI:
            self.eat(SEMI)
            results.append(self.statement())

        return results

    def statement(self):
        if self.current_token.type == begin:
            node = self.compound_statement()
        if self.current_token.type == ID:
            node = self.assignment_statement()
        elif self.current_token.type == IF:
            node = self.conditional_statement()
        elif self.current_token.type == println:
            node = self.print_statement()
        else:
            node = self.empty()
        return node

    def print_statement(self):
        self.eat('println')
        self.eat('LPAREN')
        if self.current_token == QUOTES:
            self.eat(QUOTES)
        stmt = ""
        while self.current_token.type != RPAREN:
            stmt = stmt + self.current_token.value
            stmt = stmt + ' '
            if self.current_token.type == QUOTES:
                self.eat('"')
            elif self.current_token.type == ID:
                self.eat('ID')

        self.eat('RPAREN')
        return stmt

    def conditional_statement(self):
        self.eat('IF')
        self.eat('LPAREN')
        cNode = self.expr()
        self.eat('RPAREN')
        nodes = self.statement_list()
        other_stmt = self.other()
        if self.current_token == SEMI:
            self.eat(SEMI)
        self.eat('end')
        root = BinIF(cNode, nodes, other_stmt)
        return root

    def other(self):
        if self.current_token.type == 'ELSEIF':
            self.eat('ELSEIF')
            self.eat('LPAREN')
            cNode = self.expr()
            self.eat('RPAREN')
            nodes = self.statement_list()
            other_stmt = self.other()
            root = BinIF(cNode, nodes, other_stmt)
            return root
        if self.current_token.type == 'ELSE':
            self.eat('ELSE')
            nodes = self.statement_list()
            other_stmt = self.other()
            root = BinIF(BinOp(Num(Token(ID, '5')), Token(EQUALS, '=='), Num(Token(ID, '5'))), nodes, other_stmt)
            return root

    def assignment_statement(self):
        left = self.variable()
        token = self.current_token
        print(self.current_token)
        self.eat(ASSIGN)
        if self.current_token.type == 'begin' or self.current_token.type == 'LPAREN':
            right = self.compound_statement()
        else:
            right = self.expr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        node = Var(self.current_token)
        self.eat(ID)
        return node

    def empty(self):
        return NoOp()

    def expr(self):
        node = self.term()
        if self.current_token.type == GREATER_THAN:
            token = self.current_token
            self.eat(GREATER_THAN)
            node = BinOp(left=node, op=token, right=self.term())
        if self.current_token.type == LESS_THAN:
            token = self.current_token
            self.eat(LESS_THAN)
            node = BinOp(left=node, op=token, right=self.term())
        if self.current_token.type == EQUALS:
            token = self.current_token
            self.eat(EQUALS)
            node = BinOp(left=node, op=token, right=self.term())
        if self.current_token.type == NOT_EQUALS:
            token = self.current_token
            self.eat(NOT_EQUALS)
            node = BinOp(left=node, op=token, right=self.term())
        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)

            node = BinOp(left=node, op=token, right=self.term())
        return node

    def term(self):
        node = self.factor()
        while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == INTEGER_DIV:
                self.eat(INTEGER_DIV)
            elif token.type == FLOAT_DIV:
                self.eat(FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def factor(self):
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == INTEGER_CONST:
            self.eat(INTEGER_CONST)
            return Num(token)
        elif token.type == REAL_CONST:
            self.eat(REAL_CONST)
            return Num(token)
        elif token.type == TRUE:
            self.eat(TRUE)
            return Bool(token)
        elif token.type == FALSE:
            self.eat(FALSE)
            return Bool(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        else:
            node = self.variable()
            return node

    def parse(self):
        node = self.program()
        if self.current_token.type != EOF:
            self.error()

        return node


class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser
        self.GLOBAL_SCOPE = {}

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        pass
    def visit_NoneType(self,node):
        pass

    def visit_str(self, node):
        x = True
        for k, v in sorted(self.GLOBAL_SCOPE.items()):
            node = node.translate({ord(' '): None})
            if k == node:
                x = False
                print(v)
        if x:
            print(node.translate({ord('"') : None}))

    def visit_Type(self, node):
        pass

    def visit_BinIF(self, node):
        if self.visit_BinOp(node.token):
            root = Compound()
            for node in node.left:
                root.children.append(node)
            self.visit_Condition(root)
        else:
            self.visit_BinIF(node.right)

    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == FLOAT_DIV:
            return float(self.visit(node.left)) / float(self.visit(node.right))
        if node.op.type == GREATER_THAN:
            return self.visit(node.left) > self.visit(node.right)
        if node.op.type == LESS_THAN:
            return self.visit(node.left) < self.visit(node.right)
        if node.op.type == EQUALS:
            return self.visit(node.left) == self.visit(node.right)
        if node.op.type == NOT_EQUALS:
            return self.visit(node.left) != self.visit(node.right)
        if node.op.type is None:
            return node.value

    def visit_Num(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)

    def visit_Condition(self, node):
        for child in node.children:
            self.visit(child)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_Assign(self, node):
        var_name = node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)

    def visit_Var(self, node):
        var_name = node.value
        var_value = self.GLOBAL_SCOPE.get(var_name)
        if var_value is None:
            raise NameError(repr(var_name))
        else:
            return var_value

    def visit_NoOp(self, node):
        pass

    def interpret(self):
        tree = self.parser.parse()
        if tree is None:
            return ''
        return self.visit(tree)


def main():
    text = input()
    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    interpreter.interpret()
    print("From Symbol Table")
    for k, v in sorted(interpreter.GLOBAL_SCOPE.items()):
        print('{} = {}'.format(k, v))


if __name__ == '__main__':
    main()

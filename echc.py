#!/usr/bin/env python3
# echc.py — Echidna: tiny single-file compiler from .ech -> portable C
# Usage:
#   python3 echc.py hello.ech > hello.c
#   cc hello.c -o hello -lm
# Language:
#   Types: int, double, bool, string, void
#   Decls: extern fn f(...)->T;  fn f(...)->T { ... }
#   Stmts: let name: T (= expr)? ;  name = expr;
#          if (expr) { ... } else { ... }
#          while (expr) { ... }
#          return expr? ;
#          print(expr[, expr...]);   # space-separated, newline
#          expr ;   # call-only stmt
#   Expr: literals (123, 1.23, true, false, "str"), vars, calls
#         unary: - ! ;  binary: * / % + - < <= > >= == != && ||

import sys

# ============================== Lexing ==============================
TT = {
  'EOF':'EOF','ID':'ID','INT':'INT','FLOAT':'FLOAT','STRING':'STRING',
  'LP':'(','RP':')','LB':'{','RB':'}','COMMA':',','SEMI':';','COLON':':',
  'ARROW':'->','ASSIGN':'=','PLUS':'+','MINUS':'-','STAR':'*','SLASH':'/',
  'PERCENT':'%','BANG':'!','LT':'<','LE':'<=','GT':'>','GE':'>=',
  'EQ':'==','NE':'!=','AND':'&&','OR':'||',
}
KEYWORDS = {'fn','extern','let','return','if','else','while',
            'true','false','int','double','bool','string','void','print'}

class Tok:
    def __init__(self,k,v,l,c):
        self.k=k; self.v=v; self.l=l; self.c=c

def lex(src:str):
    i,n,l,c = 0,len(src),1,1
    def peek(k=0):
        j=i+k
        return src[j] if j<n else ''
    def adv():
        nonlocal i,l,c
        ch = peek()
        if ch=='\n': l+=1; c=1
        else: c+=1
        i+=1
        return ch
    def match(s):
        nonlocal i
        if src.startswith(s,i):
            for _ in s: adv()
            return True
        return False
    toks=[]
    while True:
        ch = peek()
        if ch=='':
            toks.append(Tok('EOF','',l,c)); break
        if ch in ' \t\r\n': adv(); continue
        # comments: //... and #...
        if (ch=='/' and peek(1)=='/') or ch=='#':
            while peek() and peek()!='\n': adv()
            continue
        # multi-char ops
        if   match('->'): toks.append(Tok('ARROW','->',l,c)); continue
        if   match('<='): toks.append(Tok('LE','<=',l,c)); continue
        if   match('>='): toks.append(Tok('GE','>=',l,c)); continue
        if   match('=='): toks.append(Tok('EQ','==',l,c)); continue
        if   match('!='): toks.append(Tok('NE','!=',l,c)); continue
        if   match('&&'): toks.append(Tok('AND','&&',l,c)); continue
        if   match('||'): toks.append(Tok('OR','||',l,c)); continue
        # string
        if ch=='"':
            sl,sc=l,c; adv()
            buf=[]
            while True:
                p=peek()
                if p=='': raise SyntaxError(f'Unterminated string at {sl}:{sc}')
                if p=='"': adv(); break
                if p=='\\':
                    adv(); q=peek()
                    if   q=='n': buf.append('\n'); adv()
                    elif q=='t': buf.append('\t'); adv()
                    elif q=='"': buf.append('"');  adv()
                    elif q=='\\': buf.append('\\'); adv()
                    else: raise SyntaxError(f'Bad escape \\{q} at {l}:{c}')
                else:
                    buf.append(adv())
            toks.append(Tok('STRING',''.join(buf),sl,sc)); continue
        # number
        if ch.isdigit():
            sl,sc=l,c; s=adv(); isf=False
            while peek().isdigit(): s+=adv()
            if peek()=='.' and peek(1).isdigit():
                isf=True; s+=adv()
                while peek().isdigit(): s+=adv()
            toks.append(Tok('FLOAT' if isf else 'INT', float(s) if isf else int(s), sl, sc)); continue
        # id / keyword
        if ch.isalpha() or ch=='_':
            sl,sc=l,c; s=adv()
            while peek().isalnum() or peek()=='_': s+=adv()
            toks.append(Tok(s if s in KEYWORDS else 'ID', s, sl, sc)); continue
        # single-char
        single = {'(':'LP',')':'RP','{':'LB','}':'RB',',':'COMMA',';':'SEMI',':':'COLON',
                  '+':'PLUS','-':'MINUS','*':'STAR','/':'SLASH','%':'PERCENT',
                  '!':'BANG','<':'LT','>':'GT','=':'ASSIGN'}
        if ch in single:
            k = single[ch]; adv(); toks.append(Tok(k,ch,l,c)); continue
        raise SyntaxError(f'Lex error near {src[i:i+20]!r}')
    return toks

# ========================== AST & Types ============================
class Type:
    def __init__(self,k): self.k=k
    def __eq__(self,o): return isinstance(o,Type) and self.k==o.k
    def __repr__(self): return self.k
T_INT=Type('int'); T_DBL=Type('double'); T_BOOL=Type('bool'); T_STR=Type('string'); T_VOID=Type('void')
def type_from_name(n): return {'int':T_INT,'double':T_DBL,'bool':T_BOOL,'string':T_STR,'void':T_VOID}[n]

class Node: pass

class Program(Node):
    def __init__(self,decls): self.decls=decls

class Extern(Node):
    def __init__(self,name,params,ret): self.name=name; self.params=params; self.ret=ret

class Func(Node):
    def __init__(self,name,params,ret,body): self.name=name; self.params=params; self.ret=ret; self.body=body

class Param(Node):
    def __init__(self,name,ty): self.name=name; self.ty=ty

class Block(Node):
    def __init__(self,stmts): self.stmts=stmts

class Let(Node):
    def __init__(self,name,ty,expr): self.name=name; self.ty=ty; self.expr=expr

class Assign(Node):
    def __init__(self,name,expr): self.name=name; self.expr=expr

class If(Node):
    def __init__(self,cond,then,els): self.cond=cond; self.then=then; self.els=els

class While(Node):
    def __init__(self,cond,body): self.cond=cond; self.body=body

class Return(Node):
    def __init__(self,expr): self.expr=expr

class Print(Node):
    def __init__(self,args): self.args=args

class ExprStmt(Node):
    def __init__(self,expr): self.expr=expr

# Exprs
class Var(Node):
    def __init__(self,name): self.name=name; self.ty=None

class Lit(Node):
    def __init__(self,val,ty): self.val=val; self.ty=ty

class Call(Node):
    def __init__(self,name,args): self.name=name; self.args=args; self.ty=None

class Unary(Node):
    def __init__(self,op,rhs): self.op=op; self.rhs=rhs; self.ty=None

class Binary(Node):
    def __init__(self,op,lhs,rhs): self.op=op; self.lhs=lhs; self.rhs=rhs; self.ty=None

# ============================= Parser =============================
class Parser:
    def __init__(self,toks): self.toks=toks; self.i=0
    def cur(self): return self.toks[self.i]
    def at(self,k): return self.cur().k==k
    def eat(self,k,msg=None):
        if not self.at(k):
            t=self.cur(); raise SyntaxError(msg or f'Expected {k}, got {t.k} at {t.l}:{t.c}')
        tok=self.cur(); self.i+=1; return tok
    def opt(self,k):
        if self.at(k): return self.eat(k)
        return None

    def parse(self):
        ds=[]
        while not self.at('EOF'):
            if self.at('extern'): ds.append(self.parse_extern())
            else: ds.append(self.parse_func())
        return Program(ds)

    def parse_type(self):
        if self.at('int') or self.at('double') or self.at('bool') or self.at('string') or self.at('void'):
            return type_from_name(self.eat(self.cur().k).v)
        raise SyntaxError(f'Expected type at {self.cur().l}:{self.cur().c}')

    def parse_params(self):
        ps=[]
        if self.at('RP'): return ps
        while True:
            name=self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            ps.append(Param(name,ty))
            if self.opt('COMMA'): continue
            break
        return ps

    def parse_extern(self):
        self.eat('extern'); self.eat('fn'); name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        self.eat('ARROW'); ret=self.parse_type()
        self.eat('SEMI')
        return Extern(name,ps,ret)

    def parse_func(self):
        self.eat('fn'); name=self.eat('ID').v
        self.eat('LP'); ps=self.parse_params(); self.eat('RP')
        ret=T_VOID
        if self.opt('ARROW'): ret=self.parse_type()
        body=self.parse_block()
        return Func(name,ps,ret,body)

    def parse_block(self):
        self.eat('LB'); ss=[]
        while not self.at('RB'): ss.append(self.parse_stmt())
        self.eat('RB'); return Block(ss)

    def parse_stmt(self):
        if self.at('let'):
            self.eat('let'); name=self.eat('ID').v
            self.eat('COLON'); ty=self.parse_type()
            expr=None
            if self.opt('ASSIGN'): expr=self.parse_expr()
            self.eat('SEMI'); return Let(name,ty,expr)
        if self.at('return'):
            self.eat('return'); e=None
            if not self.at('SEMI'): e=self.parse_expr()
            self.eat('SEMI'); return Return(e)
        if self.at('if'):
            self.eat('if'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            then=self.parse_block(); els=None
            if self.opt('else'): els=self.parse_block()
            return If(cond,then,els)
        if self.at('while'):
            self.eat('while'); self.eat('LP'); cond=self.parse_expr(); self.eat('RP')
            body=self.parse_block(); return While(cond,body)
        if self.at('print'):
            self.eat('print'); self.eat('LP')
            args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP'); self.eat('SEMI')
            return Print(args)
        if self.at('ID') and self.toks[self.i+1].k=='ASSIGN':
            name=self.eat('ID').v; self.eat('ASSIGN'); e=self.parse_expr(); self.eat('SEMI')
            return Assign(name,e)
        e=self.parse_expr(); self.eat('SEMI'); return ExprStmt(e)

    # precedence
    def parse_expr(self): return self.parse_or()
    def parse_or(self):
        e=self.parse_and()
        while self.opt('OR'): e=Binary('||',e,self.parse_and())
        return e
    def parse_and(self):
        e=self.parse_eq()
        while self.opt('AND'): e=Binary('&&',e,self.parse_eq())
        return e
    def parse_eq(self):
        e=self.parse_rel()
        while True:
            if self.opt('EQ'): e=Binary('==',e,self.parse_rel())
            elif self.opt('NE'): e=Binary('!=',e,self.parse_rel())
            else: break
        return e
    def parse_rel(self):
        e=self.parse_add()
        while True:
            if self.opt('LE'): e=Binary('<=',e,self.parse_add())
            elif self.opt('GE'): e=Binary('>=',e,self.parse_add())
            elif self.opt('LT'): e=Binary('<',e,self.parse_add())
            elif self.opt('GT'): e=Binary('>',e,self.parse_add())
            else: break
        return e
    def parse_add(self):
        e=self.parse_mul()
        while True:
            if self.opt('PLUS'): e=Binary('+',e,self.parse_mul())
            elif self.opt('MINUS'): e=Binary('-',e,self.parse_mul())
            else: break
        return e
    def parse_mul(self):
        e=self.parse_un()
        while True:
            if self.opt('STAR'): e=Binary('*',e,self.parse_un())
            elif self.opt('SLASH'): e=Binary('/',e,self.parse_un())
            elif self.opt('PERCENT'): e=Binary('%',e,self.parse_un())
            else: break
        return e
    def parse_un(self):
        if self.opt('MINUS'): return Unary('-', self.parse_un())
        if self.opt('BANG'):  return Unary('!', self.parse_un())
        return self.parse_call()
    def parse_call(self):
        e=self.parse_primary()
        while self.at('LP'):
            self.eat('LP'); args=[]
            if not self.at('RP'):
                while True:
                    args.append(self.parse_expr())
                    if self.opt('COMMA'): continue
                    break
            self.eat('RP')
            if isinstance(e,Var): e=Call(e.name,args)
            else: raise SyntaxError(f'Call target must be a name at {self.cur().l}:{self.cur().c}')
        return e
    def parse_primary(self):
        t=self.cur()
        if self.opt('INT'):    return Lit(t.v,T_INT)
        if self.opt('FLOAT'):  return Lit(t.v,T_DBL)
        if self.opt('STRING'): return Lit(t.v,T_STR)
        if self.opt('true'):   return Lit(1,T_BOOL)
        if self.opt('false'):  return Lit(0,T_BOOL)
        if self.opt('ID'):     return Var(t.v)
        if self.opt('LP'):
            e=self.parse_expr(); self.eat('RP'); return e
        raise SyntaxError(f'Unexpected token {t.k} at {t.l}:{t.c}')

# ============================ Semantics ============================
class FuncSig:
    def __init__(self,name,ps,ret):
        self.name=name; self.param_types=ps; self.ret=ret

class Ctx:
    def __init__(self):
        self.funcs={}; self.vars=[]; self.cur=None
    def push(self): self.vars.append({})
    def pop(self): self.vars.pop()
    def decl_var(self,n,t):
        if n in self.vars[-1]: raise TypeError(f'Variable {n} already declared')
        self.vars[-1][n]=t
    def get_var(self,n):
        for d in reversed(self.vars):
            if n in d: return d[n]
        raise TypeError(f'Undeclared variable {n}')
    def decl_fn(self,name,sig): self.funcs[name]=sig
    def get_fn(self,name):
        if name in self.funcs: return self.funcs[name]
        raise TypeError(f'Undeclared function {name}')

def is_num(t): return t in (T_INT,T_DBL)
def can_assign(dst,src): return dst==src or (dst==T_DBL and src==T_INT)

def analyze_expr(e,ctx:Ctx):
    if isinstance(e,Lit): return e.ty
    if isinstance(e,Var):
        e.ty=ctx.get_var(e.name); return e.ty
    if isinstance(e,Unary):
        rt=analyze_expr(e.rhs,ctx)
        if e.op=='-':
            if not is_num(rt): raise TypeError('Unary - expects numeric')
            e.ty=rt; return rt
        if e.op=='!':
            if rt not in (T_BOOL,T_INT,T_DBL): raise TypeError('! expects bool/numeric')
            e.ty=T_BOOL; return T_BOOL
    if isinstance(e,Binary):
        lt=analyze_expr(e.lhs,ctx); rt=analyze_expr(e.rhs,ctx); op=e.op
        if op in ('*','/','+','-'):
            if not (is_num(lt) and is_num(rt)): raise TypeError(f'Op {op} needs numeric')
            e.ty = T_DBL if (lt==T_DBL or rt==T_DBL) else T_INT; return e.ty
        if op=='%':
            if lt!=T_INT or rt!=T_INT: raise TypeError('% requires int')
            e.ty=T_INT; return T_INT
        if op in ('<','<=','>','>='):
            if not (is_num(lt) and is_num(rt)): raise TypeError('Comparison needs numeric')
            e.ty=T_BOOL; return T_BOOL
        if op in ('==','!='):
            if lt!=rt and not (is_num(lt) and is_num(rt)): raise TypeError('==/!= need same or both numeric')
            e.ty=T_BOOL; return T_BOOL
        if op in ('&&','||'):
            if lt not in (T_BOOL,T_INT,T_DBL) or rt not in (T_BOOL,T_INT,T_DBL):
                raise TypeError('Logical ops need bool/numeric')
            e.ty=T_BOOL; return T_BOOL
    if isinstance(e,Call):
        sig=ctx.get_fn(e.name)
        if len(sig.param_types)!=len(e.args): raise TypeError(f'Arity mismatch for {e.name}')
        for i,(pt,a) in enumerate(zip(sig.param_types,e.args),1):
            at=analyze_expr(a,ctx)
            if pt==at: continue
            if pt==T_DBL and at==T_INT: continue
            raise TypeError(f'Arg {i} to {e.name} expects {pt}, got {at}')
        e.ty=sig.ret; return e.ty
    raise TypeError('Unknown expression')

def analyze_stmt(s,ctx:Ctx):
    if isinstance(s,Let):
        if s.expr:
            et=analyze_expr(s.expr,ctx)
            if not can_assign(s.ty,et): raise TypeError(f'Cannot assign {et} to {s.ty}')
        ctx.decl_var(s.name,s.ty)
    elif isinstance(s,Assign):
        vt=ctx.get_var(s.name); et=analyze_expr(s.expr,ctx)
        if not can_assign(vt,et): raise TypeError(f'Cannot assign {et} to {vt}')
    elif isinstance(s,If):
        analyze_expr(s.cond,ctx)
        ctx.push(); [analyze_stmt(x,ctx) for x in s.then.stmts]; ctx.pop()
        if s.els: ctx.push(); [analyze_stmt(x,ctx) for x in s.els.stmts]; ctx.pop()
    elif isinstance(s,While):
        analyze_expr(s.cond,ctx)
        ctx.push(); [analyze_stmt(x,ctx) for x in s.body.stmts]; ctx.pop()
    elif isinstance(s,Return):
        if s.expr is None:
            if ctx.cur.ret!=T_VOID and ctx.cur.name!='main': raise TypeError('Return without value')
        else:
            rt=analyze_expr(s.expr,ctx)
            if ctx.cur.ret==T_DBL and rt==T_INT: pass
            elif ctx.cur.ret!=rt: raise TypeError(f'Return type {rt} != {ctx.cur.ret}')
    elif isinstance(s,Print):
        for a in s.args: analyze_expr(a,ctx)
    elif isinstance(s,ExprStmt):
        if not isinstance(s.expr,Call): raise TypeError('Only function calls allowed as expression statements')
        analyze_expr(s.expr,ctx)

def analyze(prog:Program):
    ctx=Ctx()
    for d in prog.decls:
        if isinstance(d,Extern) or isinstance(d,Func):
            ctx.decl_fn(d.name, FuncSig(d.name,[p.ty for p in d.params], d.ret))
    for d in prog.decls:
        if isinstance(d,Func):
            ctx.cur = ctx.get_fn(d.name)
            ctx.push()
            for p in d.params: ctx.decl_var(p.name,p.ty)
            for st in d.body.stmts: analyze_stmt(st,ctx)
            ctx.pop()
    if 'main' not in ctx.funcs:
        print('// Warning: no main() defined; producing a library C file.', file=sys.stderr)
    return ctx

# ============================= Codegen =============================
def c_type(t:Type):
    if t==T_INT: return 'long long'
    if t==T_DBL: return 'double'
    if t==T_BOOL: return 'bool'
    if t==T_STR: return 'const char*'
    if t==T_VOID: return 'void'
    raise RuntimeError('unknown type')

def c_escape(s:str):
    return s.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\t','\\t')

class CGen:
    def __init__(self,prog:Program,ctx:Ctx):
        self.p=prog; self.ctx=ctx; self.out=[]; self.ind=0
    def emit(self,s=''): self.out.append('  '*self.ind + s)
    def gen(self):
        self.emit('#include <stdio.h>')
        self.emit('#include <math.h>')
        self.emit('#include <stdbool.h>')
        self.emit('#include <string.h>')
        self.emit('')
        self.emit('static inline void __p_int(long long x){ printf("%lld", x); }')
        self.emit('static inline void __p_double(double x){ printf("%.15g", x); }')
        self.emit('static inline void __p_bool(bool x){ printf("%s", x ? "true" : "false"); }')
        self.emit('static inline void __p_string(const char* s){ printf("%s", s); }')
        self.emit('')
        for d in self.p.decls:
            if isinstance(d,Extern): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        for d in self.p.decls:
            if isinstance(d,Func): self.emit(self.sig_to_c(d.name,d.params,d.ret) + ';')
        self.emit('')
        for d in self.p.decls:
            if isinstance(d,Func): self.gen_func(d)
        return '\n'.join(self.out)
    def sig_to_c(self,name,ps,ret):
        if name=='main': return 'int main(void)'
        params = ', '.join(f'{c_type(p.ty)} {p.name}' for p in ps) or 'void'
        return f'{c_type(ret)} {name}({params})'
    def gen_func(self,f:Func):
        self.emit(self.sig_to_c(f.name,f.params,f.ret) + ' {'); self.ind+=1
        for st in f.body.stmts: self.gen_stmt(st)
        if f.name=='main': self.emit('return 0;')
        self.ind-=1; self.emit('}'); self.emit('')
    def gen_stmt(self,s):
        if isinstance(s,Let):
            init = f' = {self.gen_expr(s.expr)}' if s.expr is not None else ''
            self.emit(f'{c_type(s.ty)} {s.name}{init};')
        elif isinstance(s,Assign):
            self.emit(f'{s.name} = {self.gen_expr(s.expr)};')
        elif isinstance(s,If):
            self.emit(f'if ({self.gen_expr(s.cond)}) {{'); self.ind+=1
            for x in s.then.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
            if s.els:
                self.emit('else {'); self.ind+=1
                for x in s.els.stmts: self.gen_stmt(x)
                self.ind-=1; self.emit('}')
        elif isinstance(s,While):
            self.emit(f'while ({self.gen_expr(s.cond)}) {{'); self.ind+=1
            for x in s.body.stmts: self.gen_stmt(x)
            self.ind-=1; self.emit('}')
        elif isinstance(s,Return):
            if s.expr is None: self.emit('return;')
            else: self.emit(f'return {self.gen_expr(s.expr)};')
        elif isinstance(s,Print):
            first=True
            for a in s.args:
                if not first: self.emit('printf(" ");')
                first=False
                ty=self.expr_static_type(a); ce=self.gen_expr(a)
                if   ty==T_INT:   self.emit(f'__p_int({ce});')
                elif ty==T_DBL:   self.emit(f'__p_double({ce});')
                elif ty==T_BOOL:  self.emit(f'__p_bool({ce});')
                elif ty==T_STR:   self.emit(f'__p_string({ce});')
                else:              self.emit(f'__p_string("<?>");')
            self.emit('printf("\\n");')
        elif isinstance(s,ExprStmt):
            self.emit(self.gen_expr(s.expr) + ';')
    def expr_static_type(self,e):
        # Prefer analyzer annotation if present
        if hasattr(e,'ty') and e.ty: return e.ty
        if isinstance(e,Call): return self.ctx.get_fn(e.name).ret
        if isinstance(e,Unary):
            t=self.expr_static_type(e.rhs); return T_BOOL if e.op=='!' else t
        if isinstance(e,Binary):
            if e.op in ('*','/','+','-'): return T_DBL
            if e.op=='%': return T_INT
            return T_BOOL
        return None
    def gen_expr(self,e):
        if isinstance(e,Lit):
            if e.ty==T_INT: return str(int(e.val))
            if e.ty==T_DBL:
                s=repr(float(e.val))
                if '.' not in s and 'e' not in s and 'E' not in s: s+='.0'
                return s
            if e.ty==T_BOOL: return 'true' if e.val else 'false'
            if e.ty==T_STR:  return f'"{c_escape(e.val)}"'
        if isinstance(e,Var): return e.name
        if isinstance(e,Unary): return f'({e.op}{self.gen_expr(e.rhs)})'
        if isinstance(e,Binary): return f'({self.gen_expr(e.lhs)} {e.op} {self.gen_expr(e.rhs)})'
        if isinstance(e,Call):
            args=', '.join(self.gen_expr(a) for a in e.args)
            return f'{e.name}({args})'
        raise RuntimeError('unknown expr')

# ============================== Driver =============================
def compile_source(src:str)->str:
    toks=lex(src)
    ast=Parser(toks).parse()
    ctx=analyze(ast)
    return CGen(ast,ctx).gen()

def main():
    if len(sys.argv)!=2:
        print('Usage: echc.py <input.ech>  # outputs C on stdout', file=sys.stderr)
        sys.exit(1)
    src=open(sys.argv[1],'r',encoding='utf-8').read()
    try:
        sys.stdout.write(compile_source(src))
    except (SyntaxError,TypeError) as e:
        print(f'Compile error: {e}', file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':
    main()



use std::collections::VecDeque;
use std::collections::HashMap;

use rustyline::error::ReadlineError;
use rustyline::Editor;

use atty::Stream;

use itertools::join;



#[derive(Debug)]
enum Error {
    Msg(String),
    EOF,
}

macro_rules! fmt_err {
    ($($arg:tt)*) => { Result::Err(Error::Msg(format!($($arg)*))) }
}




//////////////////////////////////////////////////////////////////////////////////////////

struct RustylineStream {
    line: VecDeque<char>, // Iterator over current line
    rustyline: Editor<()>,

    row: i32,
    col: i32,
}

impl RustylineStream {
    fn new() -> Self {
        Self {
            line: VecDeque::new(),
            rustyline: Editor::<()>::new(),
            row: 0,
            col: 0,
        }
    }
    fn read_line(&mut self) -> Result<(), Error> {
        let line = self.rustyline.readline("> ");
        match line {
            Ok(line) => {
                self.rustyline.add_history_entry(&line);
                // self.line = line.into();
                self.line = line.chars().collect();
                self.line.push_back('\n');
                self.row += 1;
                self.col = -1;
                Ok(())
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => Result::Err(Error::EOF),
            Err(err) => panic!("Error: {:?}", err),
        }
    }

    fn get(&mut self) -> Result<char, Error> {
        if self.line.is_empty() {
            self.read_line()?;
        }

        let c = self.line.pop_front().unwrap();
        self.col += 1;
        Ok(c)
    }

    fn unget(&mut self, c: char) {
        self.line.push_front(c);
    }

    // Line number, column
    fn pos(&self) -> (i32, i32) {
        (self.row, self.col)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
enum Token {
    ParenOpen,
    QuotedOpen,
    ParenClose,
    Int(i32),
    Float(f64),
    Str(String),
    Variable(String),
    True,
    Nil,
    // TODO: char, symbol
}


struct TokenStream {
    stream: RustylineStream,
    next: Option<Token>, }


impl TokenStream {
    fn new() -> Self {
        TokenStream{ 
            stream: RustylineStream::new(),
            next: None,
        }
    }

    fn unget(&mut self, t: Token) {
        assert!(!self.next.is_some());
        self.next = Some(t);
    }

    fn get_str(&mut self) -> Result<String, Error> {
        let mut s = String::new();

        let mut in_escape = false;
        loop {
            let next = self.stream.get()?;

            if in_escape {
                match next {
                    'a' => s.push('\x07'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    '\'' => s.push('\''),
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    _ => return fmt_err!("Unrecognized escape \\{}", next),
                }


                in_escape = false;
            } else if next == '\\' {
                in_escape = true;
            } else if next == '"' {
                if in_escape {
                    return fmt_err!("Unterminated escape sequence");
                }
                return Ok(s);
            } else {
                s.push(next);
            }

        }
    }
    
    fn eat_line_comment(&mut self) -> Result<(), Error> {
        loop {
            let next = self.stream.get()?;
            if next == '\n' {
                return Ok(());
            }
        }
    }


    // bool is true iff token is a string
    fn get_raw(&mut self) -> Result<(String, bool), Error> {

        let mut word = String::new();
        loop {
            let next = self.stream.get()?;
            if next.is_whitespace() {
                if !word.is_empty() {
                    self.stream.unget(next);
                    break;
                }

                continue
            }
            
            match next {
                '(' => {
                    if word.is_empty() || word == "'" {
                        word.push(next);
                        break;
                    }

                    self.stream.unget(next);
                    break;
                },
                ')' => {
                    if word.is_empty() {
                        word.push(next);
                        break;
                    }
                    self.stream.unget(next);
                    break;

                }
                '\'' => {
                    if !word.is_empty() {
                        return fmt_err!("Single quote only allowed at begining of atom");
                    }
                    word.push(next);
                },
                '"' => {
                     if word.is_empty() {
                        return Ok((self.get_str()?, true))
                    } else {
                        return fmt_err!("Unexpected double quote");
                    }
                },
                ';' =>
                    // TODO: Block comments
                    if word.is_empty() {
                        self.eat_line_comment()?;
                    } else {
                        self.stream.unget(next);
                        return Ok((word, false));
                    }
                _ => word.push(next),

            }
        }
        Ok((word, false))
    }


    fn get(&mut self) -> Result<Token, Error> {
        if let Some(x) = self.next.take() {
            self.next = None;
            return Ok(x);
        }
        let (word, is_str) = self.get_raw()?;


        if is_str {
            return Ok(Token::Str(word));
        }

        match word.as_str() {
            "(" => return Ok(Token::ParenOpen),
            "'(" => return Ok(Token::QuotedOpen),
            ")" => return Ok(Token::ParenClose),
            "t" => return Ok(Token::True),
            "Nil" => return Ok(Token::Nil),
            _ => (),
        }
        if let Some(i) = word.parse::<i32>().ok() {
            return Ok(Token::Int(i));
        }

        if let Some(f) = word.parse::<f64>().ok() {
            return Ok(Token::Float(f));
        }

        return Ok(Token::Variable(word))
    }

    fn pos(&self) -> (i32, i32) {
        self.stream.pos()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone)]
enum Atom {
    Int(i32),
    Float(f64),
    Str(String),
    Variable(String),
    True,
    Nil,
    // TODO: char, symbol
}

#[derive(Debug, Clone)]
enum SExpr {
    Atom(Atom),
    List(Vec<SExpr>),
    Quoted(Vec<SExpr>),
}



struct Parser {
    stream: TokenStream,
    depth: i32,
}


impl Parser {
    fn new() -> Self {
        Parser { 
            stream: TokenStream::new(),
            depth: 0,
        }
    }


    fn parse_atom(&mut self) -> SExpr {
        let atom = match self.stream.get().unwrap() {
            Token::ParenOpen => unreachable!(),
            Token::QuotedOpen => unreachable!(),
            Token::ParenClose => unreachable!(),
            Token::Int(i) => Atom::Int(i),
            Token::Float(f) => Atom::Float(f),
            Token::Str(s) => Atom::Str(s),
            Token::Variable(s) => Atom::Variable(s),
            Token::True => Atom::True,
            Token::Nil => Atom::Nil,
        };

        SExpr::Atom(atom)
    }

    fn parse_list(&mut self) -> Result<SExpr, Error> {
        let next = self.stream.get().unwrap();
        let quoted = match next {
            Token::QuotedOpen => true,
            Token::ParenOpen => false,
            _ => unreachable!(),
        };

        let mut list = vec![];
        loop {
            let next = self.stream.get().unwrap();
            if matches!(next, Token::ParenClose) {
                if quoted {
                    return Ok(SExpr::Quoted(list));
                } else {
                    return Ok(SExpr::List(list));
                }
            } else {
                self.stream.unget(next);
                let next = self.parse_sexpr().unwrap();
                list.push(next);
            }
        }
    }

    fn parse_sexpr(&mut self) -> Result<SExpr, Error> {
        let first = self.stream.get()?;

        match first {
            Token::QuotedOpen | Token::ParenOpen => {
                self.stream.unget(first);
                self.depth += 1;
                let list = self.parse_list();
                self.depth -= 1;
                return list;
            },
            Token::ParenClose => return fmt_err!("Unexpected `)'"),
            _ => {
                self.stream.unget(first);
                return Ok(self.parse_atom());
            }


        }
    }

    fn pos(&self) -> (i32, i32) {
        self.stream.pos()
    }

    fn reset(&mut self) {
        self.depth = 0;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////

// Also make sure they're all numeric
fn any_float(vals: &Vec<Value>) -> Result<bool, Error> {
    let mut any = false;
    for val in vals {
        match val {
            Value::Int(_) => continue,
            Value::Float(_) => any = true,
            _ => return fmt_err!("Type error: nonnumeric argument to numeric operator"),
        }
    }
    Ok(any)
}




macro_rules! apply_acc {
    ($_vals:expr, $op:tt, $ident:expr) => {{
        let vals = $_vals;
        if any_float(vals)? {
            let mut acc = $ident as f64;
            for val in vals {
                match val {
                    Value::Int(x) => acc $op (*x as f64),
                    Value::Float(x) => acc $op x,
                    _ => unreachable!(),
                }
            }
            Ok(Value::Float(acc))
        } else {
            let mut acc = $ident;
            for val in vals {
                match val {
                    Value::Int(x) => acc $op x,
                    _ => unreachable!(),
                }
            }
            Ok(Value::Int(acc))
        }
    }};

}

fn builtin_sum(vals: &Vec<Value>) -> Result<Value, Error> {
    apply_acc!(vals, +=, 0)
}

fn builtin_mul(vals: &Vec<Value>) -> Result<Value, Error> {
    apply_acc!(vals, *=, 1)
}

fn builtin_sub(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() == 0 {
        return fmt_err!("Operator `-' requires arguments");
    } else if vals.len() == 1 {
        return match vals[0] {
            Value::Int(x) => Ok(Value::Int(-x)),
            Value::Float(x) => Ok(Value::Float(-x)),
            _ => fmt_err!("Type error"),
        }
    }

    if any_float(vals)? {
        let mut it = vals.iter();
        let mut val = match it.next().unwrap() {
            Value::Int(x) => *x as f64,
            Value::Float(x) => *x,
            _ => unreachable!(),
        };

        while let Some(next) = it.next() {
            val -= match next {
                Value::Int(x) => *x as f64,
                Value::Float(x) => *x,
                _ => unreachable!(),
            }
        }

        Ok(Value::Float(val))
    } else {
        let mut it = vals.iter();
        let mut val = match it.next().unwrap() {
            Value::Int(x) => *x,
            _ => unreachable!(),
        };

        while let Some(next) = it.next() {
            val -= match next {
                Value::Int(x) => *x,
                _ => unreachable!(),
            }
        }

        Ok(Value::Int(val))
    }
}

fn builtin_quot(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() == 0 {
        return fmt_err!("Operator `/' requires arguments");
    } else if vals.len() < 2 {
        return fmt_err!("`/' requires at least 2 arguments (ratios not yet supported)");
        
    }

    if any_float(vals)? {
        let mut it = vals.iter();
        let mut val = match it.next().unwrap() {
            Value::Int(x) => *x as f64,
            Value::Float(x) => *x,
            _ => unreachable!(),
        };

        while let Some(next) = it.next() {
            val /= match next {
                Value::Int(x) => *x as f64,
                Value::Float(x) => *x,
                _ => unreachable!(),
            }
        }

        Ok(Value::Float(val))
    } else {
        let mut it = vals.iter();
        let mut val = match it.next().unwrap() {
            Value::Int(x) => *x,
            _ => unreachable!(),
        };

        while let Some(next) = it.next() {
            val /= match next {
                Value::Int(x) => *x,
                _ => unreachable!(),
            }
        }

        Ok(Value::Int(val))
    }

}

fn builtin_mod(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() == 0 {
        return fmt_err!("Operator `mod' requires arguments");
    } else if vals.len() < 2 {
        return fmt_err!("`mod' requires at least 2 arguments (ratios not yet supported)");
        
    }

    if any_float(vals)? {
        return fmt_err!("`mod' only defined on integers");
    } else {
        let mut it = vals.iter();
        let mut val = match it.next().unwrap() {
            Value::Int(x) => *x,
            _ => unreachable!(),
        };

        while let Some(next) = it.next() {
            val %= match next {
                Value::Int(x) => *x,
                _ => unreachable!(),
            }
        }

        Ok(Value::Int(val))
    }

}
fn builtin_print(vals: &Vec<Value>) -> Result<Value, Error> {
    let s = vals.iter() .map(|x| format!("{}", x));
    println!("{}", join(s, " ")); 
    Ok(Value::Bool(false))
}


fn num_eq(a: &Value, b:&Value) -> Result<bool, Error> {
    let val = match a {
        Value::Int(a) => {
            match b {
                Value::Int(b) => a == b,
                Value::Float(b) => (*a as f64) == *b,
                _ => return fmt_err!("Only numbers are comparable with `='"),
            }
        },
        Value::Float(a) => {
            match b {
                Value::Int(b) => *a == (*b as f64),
                Value::Float(b) => a == b,
                _ => return fmt_err!("Only numbers are comparable with `='"),
            }
        },
        _ => return fmt_err!("Only numbers are comparable with `='"),
    };

    Ok(val)
}

fn builtin_num_eq(vals: &Vec<Value>) -> Result<Value, Error> {
    let mut iter = vals.iter();
    let first = match iter.next() {
        Some(x) => x,
        None => return fmt_err!("`=' missing arguments"),
    };

    while let Some(next) = iter.next() {
        if !num_eq(first, next)? {
            return Ok(Value::Bool(false));
        }
    }

    Ok(Value::Bool(true))
}

fn builtin_num_neq(vals: &Vec<Value>) -> Result<Value, Error> {
    let mut iter = vals.iter();
    let first = match iter.next() {
        Some(x) => x,
        None => return fmt_err!("`=' missing arguments"),
    };


    while let Some(next) = iter.next() {
        if num_eq(first, next)? {
            return Ok(Value::Bool(false));
        }
    }

    Ok(Value::Bool(true))
}

#[derive(Debug, Clone)]
struct Func {
    params: Vec<String>,
    body: SExpr,
}



#[derive(Debug, Clone)]
enum Value {
    Int(i32),
    Float(f64),
    Str(String),
    Func(Func),
    Bool(bool),
    // TODO: fn
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Value::Int(x) => write!(f, "{}", x),
            Value::Float(x) => write!(f, "{}", x),
            Value::Str(x) => write!(f, "{}", x),
            Value::Func(_) => write!(f, "fn"),
            Value::Bool(b) => write!(f, "{}", if *b { "T" } else { "Nil" }),
        }
    }
}

struct Interp {
    parser: Parser,

    let_var: Vec<HashMap<String, Value>>, // Stack
    // TODO: other kinds of variables

    builtins: HashMap<String, fn(&Vec<Value>) -> Result<Value, Error>>,
}



impl Interp {
    fn new() -> Self {
        let mut interp = Interp { 
            parser: Parser::new(), 
            let_var: vec![HashMap::new()], // Global
            builtins: HashMap::new(),
        };

        interp.builtins.insert("+".into(), builtin_sum);
        interp.builtins.insert("-".into(), builtin_sub);
        interp.builtins.insert("*".into(), builtin_mul);
        interp.builtins.insert("/".into(), builtin_quot);
        interp.builtins.insert("mod".into(), builtin_mod);
        interp.builtins.insert("print".into(), builtin_print);
        interp.builtins.insert("=".into(), builtin_num_eq);
        interp.builtins.insert("/=".into(), builtin_num_neq);

        interp
    }

    fn eval_atom(&mut self, atom: &Atom) -> Result<Value, Error> {
        let val = match atom {
            Atom::Int(i) => Value::Int(*i),
            Atom::Float(f) => Value::Float(*f),
            Atom::Str(s) => Value::Str(s.clone()), // TODO: clone nessesary?
            Atom::Variable(s) => {
                if self.let_var.is_empty() {
                    return fmt_err!("Variable `{}' not found", s);
                }
                for vars in self.let_var.iter().rev() {
                    if let Some(v) = vars.get(s) {
                        return Ok(v.clone());
                    }
                }
                return fmt_err!("Variable `{}' not found", s)
            },
            Atom::True => Value::Bool(true),
            Atom::Nil => Value::Bool(false),
        };
        Ok(val)
    }

    fn eval_fun_params(&mut self, expr: &SExpr) -> Result<Vec<String>, Error> {
        let list = match expr {
            SExpr::List(l) => l,
            _ => return fmt_err!("`defun' formal parameters must be a list"),
        };

        
        let mut params = vec![];
        for param in list {
            match param {
                SExpr::Atom(Atom::Variable(s)) => params.push(s.clone()),
                _ => return fmt_err!("`defun' formal parameter must be variable name"),
            }
        }
        Ok(params)
    }


    fn eval_list(&mut self, list: &Vec<SExpr>) -> Result<Value, Error> {
        if list.is_empty() {
            return Ok(Value::Bool(false));
        }

        let mut iter = list.iter();
        let func = match iter.next().unwrap() {
            SExpr::Atom(Atom::Variable(s)) => s,
            func => return fmt_err!("Invalid function {:?}", func),
        };


        // TODO: break these out
        if func == "let" {
            let (params, vals) = match iter.next() {
                Some(bindings) => self.eval_let_bindings(bindings)?,
                _ => return fmt_err!("`let' missing bindings and body"),
            };

            let body = match iter.next() {
                Some(body) => body,
                _ => return fmt_err!("`let' missing body"),
            };
            return self.eval_let_body(&params, &vals, body);
        } else if func == "defun" {
            // TODO: function definitions are now global. If this behavior is kept, disallow defun
            // top level?
            let name = match iter.next() {
                Some(name) => {
                    match name {
                        SExpr::Atom(Atom::Variable(s)) => s,
                        _ => return fmt_err!("`defun' name must be variable name"),
                    }
                },
                _ => return fmt_err!("`defun' missing name, bindings, and body"),
            };
            let params = match iter.next() {
                Some(bindings) => self.eval_fun_params(bindings)?,
                _ => return fmt_err!("`defun' missing bindings and body"),
            };

            let body = match iter.next() {
                Some(body) => body,
                _ => return fmt_err!("`defun` missing body"),
            };

            let func = Func{params, body: body.clone()};
            self.let_var[0].insert(name.clone(), Value::Func(func));
            return Ok(Value::Bool(false));
        } else if func == "if" {
            let cond = match iter.next() {
                Some(cond) => cond,
                _ => return fmt_err!("`if' missing condition"),
            };

            let then = match iter.next() {
                Some(then) => then,
                _ => return fmt_err!("`if' missing then clause"),
            };

            let other = match iter.next() {
                Some(other) => other,
                _ => &SExpr::Atom(Atom::Nil),
            };

            let val = match self.eval_sexpr(cond)? {
                Value::Bool(false) => self.eval_sexpr(other)?,
                _ => self.eval_sexpr(then)?,
            };

            return Ok(val);
        }

        let mut vals = Vec::<_>::new();
        for exp in iter {
            vals.push(self.eval_sexpr(exp)?);
        }

        if let Some(func) = self.builtins.get(func) {
            Ok(func(&vals)?)
        } else if let Some(func) = self.let_var[0].get(func) {
            let func = match func {
                Value::Func(f) => f,
                _ => return fmt_err!("Variable isn't a function"),
            };
            if vals.len() != func.params.len() {
                return fmt_err!("Number of variables not equal to number of formal parameters");
            }

            let func = func.clone(); // TODO: this avoid self.eval_let() reborrowing as mut while the func itself is borrowed in the .get(). Is this nececssary?

            return self.eval_let_body(&func.params, &vals, &func.body);
        } else {
            fmt_err!("Function `{}' not found", func)
        }
            

    }

    fn eval_sexpr(&mut self, expr: &SExpr) -> Result<Value, Error> {
        let val = match expr {
            SExpr::Atom(a) => self.eval_atom(&a)?,
            SExpr::List(l) => self.eval_list(&l)?,
            SExpr::Quoted(_) => todo!(),
        };

        Ok(val)
    }
    // Returns (formal params, values)
    fn eval_let_bindings(&mut self, bindings: &SExpr) -> Result<(Vec<String>, Vec<Value>), Error> {
        let mut params = vec![];
        let mut vals = vec![];
        let list = match bindings {
            SExpr::List(list) => list,
            _ => return fmt_err!("`let' bindings must be a list"),
        };
        for bind in list {
            let pair = match bind {
                SExpr::List(pair) => pair,
                _ => return fmt_err!("`let' bindings must be a list of lists"),
            };

            if pair.len() != 2 {
                return fmt_err!("`let' bindings must be a list of pairs");

            }
             match &pair[0] {
                SExpr::Atom(Atom::Variable(s)) => params.push(s.clone()),
                _ => return fmt_err!("First element of a `let' binding pair must be a vriable name"),

            };
            vals.push(self.eval_sexpr(&pair[1])?);
        }
        Ok((params, vals))
    }

    fn eval_let_body(&mut self, params: &Vec<String>, vals: &Vec<Value>, expr: &SExpr) -> Result<Value, Error> {
        if params.len() != vals.len() {
            return fmt_err!("Number of arguments doesn't match number of formal parameters");
        }
        self.let_var.push(params.iter().zip(vals).map(|(a,b)| (a.clone(), b.clone())).collect());
        let val = self.eval_sexpr(expr)?;
        self.let_var.pop();
        Ok(val)
    }

    fn run(&mut self) {
        loop {
            let err = match self.parser.parse_sexpr() {
                Ok(expr) => match self.eval_sexpr(&expr) {
                    Ok(val) => {
                        if atty::is(Stream::Stdin) {
                            println!("{}", val);
                        }
                        continue;
                    },
                    Err(msg) => msg
                },
                Err(e) => e
            };

            match err {
                Error::Msg(msg) => {
                    let (line, col) = self.parser.pos();
                    eprintln!("Error line {}, col {}: {}", line, col, msg);
                    if atty::is(Stream::Stdin) {
                        self.parser.reset();
                    } else {
                        std::process::exit(1);
                    }
                },
                Error::EOF => {
                    if atty::is(Stream::Stdin) {
                        std::process::exit(0);
                    } else if self.parser.depth > 0{
                        eprintln!("Unexpected EOF");
                        std::process::exit(1);
                    } else {
                        std::process::exit(0);
                    }
                }
            }
        }
    }
}





fn main() {
    let mut interp = Interp::new();
    interp.run();
}


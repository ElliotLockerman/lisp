

use std::collections::HashMap;
use std::rc::Rc;

use itertools::join;

use crate::error::Error;
use super::fmt_err;

use crate::parser::{Parser, Atom, SExpr};
use crate::text_stream::{TextStream, StringStream};

use crate::builtins::{Builtin, BUILTINS};


#[derive(Debug, Clone)]
pub struct Func {
    params: Vec<String>,
    body: SExpr,
}

#[derive(Debug, Clone)]
pub enum ConsLink {
    Tail(Rc<Cons>),
    Nil,
}

impl ConsLink {
    pub fn iter(&self) -> ConsIter {
        ConsIter::new(self)
    }
}

#[derive(Debug, Clone)]
pub struct Cons {
    pub car: Value,
    pub cdr: ConsLink,
}

impl Cons {
    pub fn new(car: Value) -> Rc<Cons> {
        Rc::new(Cons{car, cdr: ConsLink::Nil})
    }
    pub fn link(car: Value, cdr: Rc<Cons>) -> Rc<Cons> {
        Rc::new(Cons{car, cdr: ConsLink::Tail(cdr)})
    }
    pub fn from_raw(car: Value, cdr: ConsLink) -> Rc<Cons> {
        Rc::new(Cons{car, cdr})
    }
}



pub struct ConsIter<'a> {
    next: &'a ConsLink,
}

impl<'a> ConsIter<'a> {
    pub  fn new(curr: &'a ConsLink) -> ConsIter<'a> {
        ConsIter{next: curr}
    }
}

impl<'a> Iterator for ConsIter<'a>{
    // TODO: switch to iterating over &Value
    type Item = &'a Value;
    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            ConsLink::Tail(cons) => {
                self.next = &cons.cdr;
                Some(&cons.car)
            }
            ConsLink::Nil => return None,
        }
    }
}

#[derive(Clone)]
pub enum Value {
    Int(i32),
    Float(f64),
    Str(String),
    Func(Func),
    Builtin(Builtin),
    Bool(bool),
    Cons(ConsLink),
    // TODO: fn
}

impl Value {
    fn to_string(&self) -> String {
        match self {
            Value::Int(x) => x.to_string(),
            Value::Float(x) => x.to_string(),
            Value::Str(x) => x.clone(),
            Value::Func(_) => "fn".into(), 
            Value::Builtin(_) => "builtin".into(), 
            Value::Bool(b) => if *b { "T".to_string() } else { "Nil".to_string() },
            Value::Cons(l) => {
                let elem = l.iter().map(|x| x.to_string());
                format!("({})", join(elem, " "))
            },
        }
    }
}
impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}


pub struct Interp {
    parser: Parser,

    let_var: Vec<HashMap<String, Value>>, // Stack
    // TODO: other kinds of variables

}



impl Interp {
    pub fn new(stream: Box<dyn TextStream>) -> Self {
        let mut interp = Interp { 
            parser: Parser::new(stream), 
            let_var: vec![HashMap::new()], // Global
        };


        interp.load_stdlib();

        interp
    }

    pub fn set_stream(&mut self, stream: Box<dyn TextStream>) -> Box<dyn TextStream> {
        self.parser.set_stream(stream)
    }

    fn lookup_variable(&mut self, s: &String) -> Option<Value> { 
        for vars in self.let_var.iter().rev() {
            if let Some(v) = vars.get(s) {
                return Some(v.clone());
            }
        }

        if let Some(func) = BUILTINS.get(s.as_str()) {
            return Some(Value::Builtin(*func));
        }
        None
    }

    fn eval_atom(&mut self, atom: &Atom) -> Result<Value, Error> {
        let val = match atom {
            Atom::Int(i) => Value::Int(*i),
            Atom::Float(f) => Value::Float(*f),
            Atom::Str(s) => Value::Str(s.clone()), // TODO: clone nessesary?
            Atom::Variable(s) => {
                if let Some(v) = self.lookup_variable(s) {
                    v
                } else {
                    return fmt_err!("Variable `{}' not found", s)
                }
            },
            Atom::True => Value::Bool(true),
            Atom::Nil => Value::Bool(false),
        };
        Ok(val)
    }

    fn eval_fun_params(&mut self, expr: &SExpr) -> Result<Vec<String>, Error> {
        let list = match expr {
            SExpr::Form(l) => l,
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


    fn eval_form(&mut self, list: &Vec<SExpr>) -> Result<Value, Error> {
        if list.is_empty() {
            return Ok(Value::Bool(false));
        }

        let mut iter = list.iter();

        // TODO: evaluate function first and take it from there instead of treating it as a special
        // case(it may be a lambda)
        let func = match iter.next().unwrap() {
            SExpr::Atom(Atom::Variable(s)) => s,
            func => return fmt_err!("Invalid function {:?}", func),
        };


        // Intrinsics
        match func.as_str() {
            "let" => return self.eval_let(iter),
            "defun" => return self.eval_defun(iter),
            "lambda" => return self.eval_lambda(iter),
            "if" => return self.eval_if(iter),
            _ => (),
        }

        // Function call
        let mut vals = Vec::<_>::new();
        for exp in iter {
            vals.push(self.eval_sexpr(exp)?);
        }


        if let Some(func) = BUILTINS.get(func.as_str()) {
            Ok(func(&vals)?)
        } else if let Some(func) = self.lookup_variable(func) {
            let func = match func {
                Value::Func(f) => f,
                Value::Builtin(b) => return b(&vals),
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
            SExpr::Form(f) => self.eval_form(&f)?,
            SExpr::Quoted(_) => todo!(),
        };

        Ok(val)
    }
    // Returns (formal params, values)
    fn eval_let_bindings(&mut self, bindings: &SExpr) -> Result<(Vec<String>, Vec<Value>), Error> {
        let mut params = vec![];
        let mut vals = vec![];
        let list = match bindings {
            SExpr::Form(list) => list,
            _ => return fmt_err!("`let' bindings must be a list"),
        };
        for bind in list {
            let pair = match bind {
                SExpr::Form(pair) => pair,
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

    fn eval_let<'a>(&mut self, mut iter: impl Iterator<Item = &'a SExpr>) -> Result<Value, Error> {
        let (params, vals) = match iter.next() {
            Some(bindings) => self.eval_let_bindings(bindings)?,
            _ => return fmt_err!("`let' missing bindings and body"),
        };

        let body = match iter.next() {
            Some(body) => body,
            _ => return fmt_err!("`let' missing body"),
        };
        return self.eval_let_body(&params, &vals, body);
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

    fn eval_defun<'a>(&mut self, mut iter: impl Iterator<Item = &'a SExpr>) -> Result<Value, Error> {
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
    }

    fn eval_lambda<'a>(&mut self, mut iter: impl Iterator<Item = &'a SExpr>) -> Result<Value, Error> {
        let params = match iter.next() {
            Some(bindings) => self.eval_fun_params(bindings)?,
            _ => return fmt_err!("`defun' missing bindings and body"),
        };

        let body = match iter.next() {
            Some(body) => body,
            _ => return fmt_err!("`defun` missing body"),
        };

        let func = Func{params, body: body.clone()};
        return Ok(Value::Func(func));
    }

    fn eval_if<'a>(&mut self, mut iter: impl Iterator<Item = &'a SExpr>) -> Result<Value, Error> {
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

    pub fn step(&mut self) -> Result<Value, Error> {
            match self.parser.parse_sexpr() {
                Ok(expr) => self.eval_sexpr(&expr),
                Err(e) => Err(e),
            }

    }

    fn load_stdlib(&mut self) {
        // TODO: move to file
        let stdlib = 
        "(defun map (f seq)
          (if seq 
               (cons (f (car seq)) (map f (cdr seq)))))".into();


        let ss = Box::new(StringStream::new(stdlib));
        let old = self.set_stream(ss);
        loop {
            match self.step() {
                Ok(_) => continue,
                Err(Error::EOF) => break,
                Err(Error::Msg(msg)) => {
                    let (line, col) = self.parser.pos();
                    eprintln!("Error in lib line {}, col {}: {}", line, col, msg);
                    std::process::exit(1);
                },
            }
        }
        self.set_stream(old);

    }

    pub fn run(&mut self) {
        loop {
            let err = match self.step() {
                Ok(_) => continue,
                Err(e) => e,
            };
            match err {
                Error::Msg(msg) => {
                    let (line, col) = self.parser.pos();
                    eprintln!("Error line {}, col {}: {}", line, col, msg);
                    if self.parser.is_term() {
                        self.parser.reset();
                    } else {
                        std::process::exit(1);
                    }
                },
                Error::EOF => {
                    if self.parser.is_term() {
                        std::process::exit(0);
                    } else if self.parser.depth() > 0{
                        eprintln!("Unexpected EOF");
                        std::process::exit(1);
                    } else {
                        std::process::exit(0);
                    }
                }
            }
        }
    }

    #[cfg(test)] // Currently used only in tests, might be used more broadly later
    pub fn eval(&mut self, s: String) -> Result<Value, Error> {
        let ss = Box::new(StringStream::new(s));
        let old = self.set_stream(ss);
        let ret = match self.step() {
            Ok(v) => Ok(v),
            Err(e) => {
                self.parser.reset();
                Err(e)
            },
        };

        self.set_stream(old);
        ret 
    }
}




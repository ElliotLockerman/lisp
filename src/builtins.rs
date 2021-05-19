
use std::rc::Rc;

use itertools::join;

use phf::phf_map;

use crate::interp::{Value, Cons};
use crate::error::Error;
use super::fmt_err;


pub type Builtin = fn(&Vec<Value>) -> Result<Value, Error>;

pub static BUILTINS: phf::Map<&'static str, Builtin> = phf_map! {
        // TODO: write macro to setup map automatically
        "+" => builtin_sum,
        "-" => builtin_sub,
        "*" => builtin_mul,
        "/" => builtin_quot,
        "mod" => builtin_mod,
        "print" => builtin_print,
        "=" => builtin_num_eq,
        "/=" => builtin_num_neq,
        ">" => builtin_num_gt,
        "<" => builtin_num_lt,
        ">=" => builtin_num_ge,
        "<=" => builtin_num_le,

        "list" => builtin_list,
        "cons" => builtin_cons,
        "car" => builtin_car,
        "head" => builtin_car,
        "cdr" => builtin_cdr,
        "tail" => builtin_cdr,

        "not" => builtin_not,
};

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

macro_rules! promoting {
    ($a:expr, $op:tt, $b:expr) => { 
        match ($a, $b) {
            (Value::Int(a), Value::Int(b)) => a $op b,
            (Value::Int(a), Value::Float(b)) => (*a as f64) $op *b,
            (Value::Float(a), Value::Int(b)) => *a $op (*b as f64),
            (Value::Float(a), Value::Float(b)) => a $op b,
            _ => return fmt_err!("`{}' is only defined on numbers", stringify!($op)),
        }
    }
}


fn builtin_num_neq(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() != 2 {
        return fmt_err!("`/=' is currently limited to exactly 2 arguments");
    }

    let a = &vals[0];
    let b = &vals[1];

    let val = promoting!(a, !=, b);
    Ok(Value::Bool(val))
}


macro_rules! pairwise_compare {
    ($op:tt, $vals:expr) => {{
        if $vals.len() == 0 {
            return fmt_err!("`{}' missing arguments", stringify!($op));
        }
        let mut good = true;
        for i in 0..$vals.len() - 1 {
            let a = &$vals[i];
            let b = &$vals[i+1];

            good = promoting!(a, $op, b);
            if !good {
                break;
            }

        }

        Ok(Value::Bool(good))
    }}
}
fn builtin_num_eq(vals: &Vec<Value>) -> Result<Value, Error> {
    pairwise_compare!(==, vals)
}

fn builtin_num_gt(vals: &Vec<Value>) -> Result<Value, Error> {
    pairwise_compare!(>, vals)
}

fn builtin_num_lt(vals: &Vec<Value>) -> Result<Value, Error> {
    pairwise_compare!(<, vals)
}

fn builtin_num_ge(vals: &Vec<Value>) -> Result<Value, Error> {
    pairwise_compare!(>=, vals)
}

fn builtin_num_le(vals: &Vec<Value>) -> Result<Value, Error> {
    pairwise_compare!(<=, vals)
}

fn builtin_list(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() == 0 {
        return Ok(Value::Bool(false));
    }
    let mut iter = vals.iter().rev();
    let last = iter.next().unwrap();
    let mut head: Rc<Cons> = Cons::new(last.clone());

    while let Some(next) = iter.next() {
        let prev = head.clone();
        head = Cons::link(next.clone(), prev);
    }

    Ok(Value::Cons(head))
}

fn builtin_cons(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() != 2 {
        return fmt_err!("cons takes exactly 2 arguments");
    }

    let first = vals[0].clone();
    let tail = match &vals[1] {
        Value::Cons(tail) => Some(tail.clone()),
        Value::Bool(false) => None,
        _ => return fmt_err!("cons' second argument must be a cons cell"),
    };
    Ok(Value::Cons(Cons::from_raw(first, tail)))
}

fn builtin_car(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() != 1 {
        return fmt_err!("car takes exactly 1 arguments");
    }

    match &vals[0] {
        Value::Cons(cons) => Ok(cons.car.clone()),
        _ => fmt_err!("car's argument must be a cons cell"),
    }
}

fn builtin_cdr(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() != 1 {
        return fmt_err!("car takes exactly 1 arguments");
    }

    let cons = match &vals[0] {
        Value::Cons(cons) => cons, 
        _ => return fmt_err!("car's argument must be a cons cell"),
    };

    match cons.cdr {
        Some(ref t) => Ok(Value::Cons(t.clone())),
        None => Ok(Value::Bool(false)),
    }
}

fn builtin_not(vals: &Vec<Value>) -> Result<Value, Error> {
    if vals.len() != 1 {
        return fmt_err!("car takes exactly 1 arguments");
    }

    let val = match &vals[0] {
        Value::Bool(false) => true,
        _ => false,
    };

    Ok(Value::Bool(val))
}

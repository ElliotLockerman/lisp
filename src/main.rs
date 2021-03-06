
mod error;
mod text_stream;
mod parser;
mod builtins;

mod tokenizer;
use text_stream::RustylineStream;

mod interp;
use interp::Interp;




fn main() {

    let rl = Box::new(RustylineStream::new());
    let mut interp = Interp::new(rl);
    interp.run();

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_stream::NullStream;
    use interp::Value;
   
    #[test]
    fn sum() {
        let mut interp = Interp::new(Box::new(NullStream::new()));
        let val = match interp.eval("(+ 2 2)".into()).unwrap() {
            Value::Int(i) => i,
            _ => panic!(),
        };
        assert_eq!(val, 4);
    }

    #[test]
    fn fact() {
        let mut interp = Interp::new(Box::new(NullStream::new()));
        let fact = "(defun fact (x) (if (= x 0) 1 (* x (fact (- x 1)))))";
        interp.eval(fact.into()).unwrap();
        let val = match interp.eval("(fact 4)".into()).unwrap() {
            Value::Int(i) => i,
            _ => panic!(),
        };
        assert_eq!(val, 24);
    }

    #[test]
    fn map() {
        let mut interp = Interp::new(Box::new(NullStream::new()));
        interp.eval("(defun inc (a) (+ a 1))".into()).unwrap();
        let val: Vec<i32> = match interp.eval("(map inc (list 1 2 3))".into()).unwrap() {
            Value::Cons(c) => c.iter()
                .map(|x| match x {
                    Value::Int(i) => *i,
                    _ => panic!(),
                })
                .collect(),
            _ => panic!(),
        };

        assert_eq!(val, vec![2, 3, 4]);
    }
    #[test]


    fn lambda() {
        let mut interp = Interp::new(Box::new(NullStream::new()));
        let val: Vec<i32> = match interp.eval("(map (lambda (a) (+ a 1)) (list 1 2 3))".into()).unwrap() {
            Value::Cons(c) => c.iter()
                .map(|x| match x {
                    Value::Int(i) => *i,
                    _ => panic!(),
                })
                .collect(),
            _ => panic!(),
        };

        assert_eq!(val, vec![2, 3, 4]);
    }

}


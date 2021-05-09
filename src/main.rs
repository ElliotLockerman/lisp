
mod error;
mod text_stream;
mod parser;

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
}


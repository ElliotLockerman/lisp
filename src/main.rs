
mod error;
mod text_stream;
mod tokenizer;
mod parser;

mod interp;
use interp::Interp;





fn main() {
    let mut interp = Interp::new();
    interp.run();
}




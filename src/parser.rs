
use crate::text_stream::TextStream;

use crate::tokenizer::{Tokenizer, Token};

use crate::error::Error;
use super::fmt_err;

#[derive(Debug, Clone)]
pub enum Atom {
    Int(i32),
    Float(f64),
    Str(String),
    Variable(String),
    True,
    Nil,
    // TODO: char, symbol
}

#[derive(Debug, Clone)]
pub enum SExpr {
    Atom(Atom),
    Form(Vec<SExpr>),
    Quoted(Vec<SExpr>),
}



pub struct Parser {
    tok: Tokenizer,
    depth: i32,
}


impl Parser {
    pub fn new(stream: Box<dyn TextStream>) -> Self {
        Parser { 
            tok: Tokenizer::new(stream),
            depth: 0,
        }
    }

    pub fn set_stream(&mut self, stream: Box<dyn TextStream>) -> Box<dyn TextStream> {
        self.tok.set_stream(stream)
    }

    fn parse_atom(&mut self) -> SExpr {
        let atom = match self.tok.get().unwrap() {
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
        let next = self.tok.get().unwrap();
        let quoted = match next {
            Token::QuotedOpen => true,
            Token::ParenOpen => false,
            _ => unreachable!(),
        };

        let mut list = vec![];
        loop {
            let next = self.tok.get().unwrap();
            if matches!(next, Token::ParenClose) {
                if quoted {
                    return Ok(SExpr::Quoted(list));
                } else {
                    return Ok(SExpr::Form(list));
                }
            } else {
                self.tok.unget(next);
                let next = self.parse_sexpr().unwrap();
                list.push(next);
            }
        }
    }

    pub fn parse_sexpr(&mut self) -> Result<SExpr, Error> {
        let first = self.tok.get()?;

        match first {
            Token::QuotedOpen | Token::ParenOpen => {
                self.tok.unget(first);
                self.depth += 1;
                let list = self.parse_list();
                self.depth -= 1;
                return list;
            },
            Token::ParenClose => return fmt_err!("Unexpected `)'"),
            _ => {
                self.tok.unget(first);
                return Ok(self.parse_atom());
            }


        }
    }

    pub fn pos(&self) -> (i32, i32) {
        self.tok.pos()
    }

    pub fn reset(&mut self) {
        self.depth = 0;
    }

    pub fn depth(&self) -> i32 {
        self.depth
    }

    pub fn is_term(&self) -> bool {
        self.tok.is_term()
    }
}

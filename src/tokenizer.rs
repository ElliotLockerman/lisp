
use crate::text_stream::TextStream;

use crate::error::Error;
use super::fmt_err;

#[derive(Debug)]
pub enum Token {
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


pub struct Tokenizer {
    stream: Box<dyn TextStream>,
    next: Option<Token>, 
}


impl Tokenizer {
    pub fn new(stream: Box<dyn TextStream>) -> Self {
        Tokenizer{ 
            stream,
            next: None,
        }
    }

    // Returns old text stream
    pub fn set_stream(&mut self, stream: Box<dyn TextStream>) -> Box<dyn TextStream> {
        let old = std::mem::replace(&mut self.stream, stream);
        old
    }

    pub fn unget(&mut self, t: Token) {
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


    pub fn get(&mut self) -> Result<Token, Error> {
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

    pub fn pos(&self) -> (i32, i32) {
        self.stream.pos()
    }

    pub fn is_term(&self) -> bool {
        self.stream.is_term()
    }

}


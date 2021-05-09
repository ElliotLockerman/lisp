
use std::collections::VecDeque;

use rustyline::error::ReadlineError;
use rustyline::Editor;

use atty::Stream;

use crate::error::Error;


pub trait TextStream {
    fn get(&mut self) -> Result<char, Error>;
    fn unget(&mut self, c: char);
    fn pos(&self) -> (i32, i32);
    fn is_term(&self) -> bool;
}

////////////////////////////////////////////////////////////////////////////////

pub struct NullStream {
}

impl NullStream {
    pub fn new() -> Self {
        NullStream{}
    }
}

impl TextStream for NullStream {

    fn get(&mut self) -> Result<char, Error> {
        Err(Error::EOF)
    }

    fn unget(&mut self, _: char) {
        unreachable!();
    }
    
    fn pos(&self) -> (i32, i32) {
        (0, 0)
    }

    fn is_term(&self) -> bool {
        false
    }
}


////////////////////////////////////////////////////////////////////////////////

pub struct RustylineStream {
    line: VecDeque<char>, // Iterator over current line
    rustyline: Editor<()>,

    row: i32,
    col: i32,
}


impl RustylineStream {
    pub fn new() -> Self {
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
}

impl TextStream for RustylineStream {
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

    fn is_term(&self) -> bool {
        atty::is(Stream::Stdin)
    }
}

////////////////////////////////////////////////////////////////////////////////


pub struct StringStream {
    text: VecDeque<char>,

    row: i32,
    col: i32,
}

impl StringStream {
    pub fn new(s: String) -> Self {
        StringStream{
            text: s.chars().collect(),
            row: 0, 
            col: 0
        }
    }
}


impl TextStream for StringStream {
    fn get(&mut self) -> Result<char, Error> {
        let c = match self.text.pop_front() {
            Some(c) => c,
            None => return Err(Error::EOF),
        };

        if c == '\n' {
            self.row += 1;
            self.col += 0;
        }

        Ok(c)
    }
    fn unget(&mut self, c: char) {
        self.text.push_front(c);
    }

    // Line number, column
    fn pos(&self) -> (i32, i32) {
        (self.row, self.col)
    }

    fn is_term(&self) -> bool {
        false
    }
}




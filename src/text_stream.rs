
use std::collections::VecDeque;

use rustyline::error::ReadlineError;
use rustyline::Editor;

use crate::error::Error;

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

    pub fn get(&mut self) -> Result<char, Error> {
        if self.line.is_empty() {
            self.read_line()?;
        }

        let c = self.line.pop_front().unwrap();
        self.col += 1;
        Ok(c)
    }

    pub fn unget(&mut self, c: char) {
        self.line.push_front(c);
    }

    // Line number, column
    pub fn pos(&self) -> (i32, i32) {
        (self.row, self.col)
    }
}

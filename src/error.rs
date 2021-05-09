
#[derive(Debug)]
pub enum Error {
    Msg(String),
    EOF,
}

#[macro_export]
macro_rules! fmt_err {
    ($($arg:tt)*) => { Result::Err(Error::Msg(format!($($arg)*))) }
}


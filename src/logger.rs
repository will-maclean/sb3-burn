use csv::Writer;
use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsStr;
use std::{collections::HashMap, path::PathBuf};

// Logger class for logging training and evaluation data
pub trait Logger {

    // log a piece of data
    fn log(&mut self, data: HashMap<String, LogData>);

    // dump the entire training data
    fn dump(&self) -> Result<(), Box<dyn Error>>;

    // check whether logging is possible. Could check
    // e.g. whether directories exist, whether db connections
    // are available, if online, etc. if try_to_fix, then
    // the Logger will try to resolve the issue, e.g. by
    // creating the dir or fixing a db connection
    fn check_can_log(&self, try_to_fix: bool) -> Result<(), &str>;
}

#[derive(Debug, Clone)]
pub enum LogData {
    String(String),
    Float(f32),
    Int(i32),
}

pub struct CsvLogger {
    dump_path: PathBuf,
    to_stdout: bool,
    //TODO: some pretty printing using the step key will make logs nicer
    step_key: Option<String>,

    data: Vec<HashMap<String, LogData>>,
}

impl CsvLogger {
    pub fn new(dump_path: PathBuf, to_stdout: bool, step_key: Option<String>) -> Self {
        Self {
            dump_path,
            to_stdout,
            data: Vec::new(),
            step_key,
        }
    }
}

impl Logger for CsvLogger {
    fn log(&mut self, data: HashMap<String, LogData>) {
        if self.to_stdout {
            println!("{:?}", data);
        }

        self.data.push(data);
    }
    fn dump(&self) -> Result<(), Box<dyn Error>> {
        let mut wtr = Writer::from_path(self.dump_path.clone()).unwrap();

        // Determine the union of all keys
        let mut all_keys: HashSet<String> = HashSet::new();
        for record in &self.data {
            for key in record.keys() {
                all_keys.insert(key.clone());
            }
        }
        let headers: Vec<&String> = all_keys.iter().collect();

        // Write the header
        wtr.write_record(&headers)?;

        // Write the data
        for record in &self.data {
            let mut row = Vec::new();
            for key in &headers {
                match record.get(*key) {
                    Some(LogData::String(s)) => row.push(s.clone()),
                    Some(LogData::Float(f)) => row.push(f.to_string()),
                    Some(LogData::Int(f)) => row.push(f.to_string()),
                    None => row.push(String::new()), // Handle missing values
                }
            }
            wtr.write_record(&row)?;
        }

        wtr.flush()?;

        Ok(())
    }

    fn check_can_log(&self, try_to_fix: bool) -> Result<(), &str> {
        if self.dump_path.exists() {
            Err("logger dump file already exists")
        }
        else if self.dump_path.extension() != Some(OsStr::new("csv")) {
            Err("logger dump path should be a csv")
        } else if !self.dump_path.parent().unwrap().exists() {
            // the parent directory does not exist
            if try_to_fix {
                match std::fs::create_dir(self.dump_path.parent().unwrap()) {
                    Ok(_) => Ok(()),
                    Err(_) => Err("Couldn't create directory"),
                }
            } else {
                Err("logger dump path dir does not exist")
            }
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod test {
    #[allow(clippy::ignore)]
    use std::{env, fs::OpenOptions, path::PathBuf};

    use super::{CsvLogger, Logger};

    #[test]
    fn test_should_log(){
        let mut pth = env::current_dir().unwrap();
        pth.push("log.csv");
        let logger = CsvLogger::new(pth, false, None);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Ok(()));
    }

    #[test]
    fn test_shouldnt_log1(){
        let logger = CsvLogger::new(PathBuf::from("this/path/shouldnt/exist.csv"), false, None);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Err("logger dump path dir does not exist"));
    }

    #[test]
    fn test_shouldnt_log2(){
        let mut pth = env::current_dir().unwrap();
        pth.push("log.txt");
        let logger = CsvLogger::new(pth, false, None);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Err("logger dump path should be a csv"));
    }

    #[test]
    fn test_shouldnt_log3(){
        let mut pth = env::current_dir().unwrap();
        pth.push("__very_strange_name.csv");

        let _ = OpenOptions::new().create(true).write(true).open(pth.clone());

        let logger = CsvLogger::new(pth.clone(), false, None);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Err("logger dump file already exists"));

        // cleanup
        let _ = std::fs::remove_file(pth);
    }
}

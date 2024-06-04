use std::collections::HashSet;
use std::error::Error;
use std::{collections::HashMap, path::PathBuf};
use csv::Writer;

pub trait Logger {
    fn log(&mut self, data: HashMap<String, LogData>);
    fn dump(&self) -> Result<(), Box<dyn Error>>;
}

#[derive(Debug)]
pub enum LogData{
    String (String),
    Float (f32),
    Int (i32),
}


pub struct CsvLogger {
    dump_path: PathBuf,
    to_stdout: bool,
    step_key: Option<String>,

    data: Vec<HashMap<String, LogData>>
}

impl CsvLogger{
    fn new(dump_path: PathBuf, to_stdout: bool, step_key: Option<String>) -> Self {
        Self { 
            dump_path, 
            to_stdout, 
            data: Vec::new(), 
            step_key
        }
    }
}

impl Logger for CsvLogger {
    fn log(&mut self, data: HashMap<String, LogData>){
        if self.to_stdout {
            println!("{:?}", data);
        }

        self.data.push(data);
    }
    fn dump(&self) -> Result<(), Box<dyn Error>>{
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
}

mod test{
    //TODO
}
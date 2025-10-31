use csv::Writer;
use plotters::prelude::*;
use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsStr;
use std::{collections::HashMap, path::PathBuf};

use super::eval::EvalResult;

// Logger class for logging training and evaluation data
pub trait Logger {
    // log a piece of data
    fn log(&mut self, data: LogItem);

    // dump the entire training data
    fn dump(&self) -> Result<(), Box<dyn Error>>;

    // check whether logging is possible. Could check
    // e.g. whether directories exist, whether db connections
    // are available, if online, etc. if try_to_fix, then
    // the Logger will try to resolve the issue, e.g. by
    // creating the dir or fixing a db connection
    fn check_can_log(&self, try_to_fix: bool) -> Result<(), &str>;

    fn print_last(&self);
}

#[derive(Debug, Clone)]
pub enum LogData {
    String(String),
    Float(f32),
    Int(i32),
}

#[derive(Debug, Clone, Default)]
pub struct LogItem {
    items: HashMap<String, LogData>,
}

impl LogItem {
    pub fn push(mut self, k: String, v: LogData) -> Self {
        self.items.insert(k, v);

        self
    }

    pub fn print(&self) {
        for (k, v) in &self.items {
            println!("{}: {:?}", k, v);
        }
    }

    pub fn combine(&mut self, other: LogItem) {
        other.items.into_iter().for_each(|(k, v)| {
            self.items.insert(k, v);
        });
    }
}

impl From<EvalResult> for LogItem {
    fn from(value: EvalResult) -> Self {
        LogItem::default()
            .push(
                "eval_ep_mean_len".to_string(),
                LogData::Float(value.mean_len),
            )
            .push(
                "eval_ep_mean_rew".to_string(),
                LogData::Float(value.mean_reward),
            )
    }
}

pub struct CsvLogger {
    overwrite: bool,
    dump_path: PathBuf,
    to_stdout: bool,
    keys: Vec<String>,
    data: Vec<LogItem>,
}

impl CsvLogger {
    pub fn new(dump_path: PathBuf, to_stdout: bool, overwrite: bool) -> Self {
        Self {
            dump_path,
            to_stdout,
            data: Vec::new(),
            keys: Vec::new(),
            overwrite,
        }
    }
}

impl Logger for CsvLogger {
    fn log(&mut self, data: LogItem) {
        if self.to_stdout {
            println!("{:?}", data);
        }

        if data.items.keys().len() == 0 {
            return;
        }

        self.data.push(data.clone());

        for key in data.items.keys() {
            if !self.keys.contains(key) {
                self.keys.push(key.clone());
            }
        }
    }
    fn dump(&self) -> Result<(), Box<dyn Error>> {
        println!(
            "Dumping logs to {:?}. {} items to dump",
            self.dump_path,
            self.data.len()
        );

        let mut wtr = Writer::from_path(self.dump_path.clone()).unwrap();

        // Determine the union of all keys
        let mut all_keys: HashSet<String> = HashSet::new();
        for record in &self.data {
            for key in record.items.keys() {
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
                match record.items.get(*key) {
                    Some(LogData::String(s)) => row.push(s.clone()),
                    Some(LogData::Float(f)) => row.push(f.to_string()),
                    Some(LogData::Int(f)) => row.push(f.to_string()),
                    None => row.push(String::new()), // Handle missing values
                }
            }
            wtr.write_record(&row)?;
        }

        wtr.flush()?;

        let _ = create_plots(
            self.data.clone(),
            self.keys.clone(),
            self.dump_path.parent().unwrap().to_path_buf(),
        );

        Ok(())
    }

    fn check_can_log(&self, try_to_fix: bool) -> Result<(), &str> {
        if self.dump_path.exists() && !self.overwrite {
            Err("logger dump file already exists")
        } else if self.dump_path.extension() != Some(OsStr::new("csv")) {
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

    fn print_last(&self) {
        println!("Last Log:");
        if let Some(log) = self.data.last() {
            for (key, record) in &log.items {
                println!("\t{key}: {:#?}", record);
            }
        }
    }
}

pub fn create_plots(
    data: Vec<LogItem>,
    create: Vec<String>,
    dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut xmax = 10.0;

    for yvar in create {
        // bulid output file path
        let mut path = dir.clone();
        path.push(format!("{yvar}.png"));

        // find mins and maxes, and build the data vecs
        let mut ymin = f32::MAX;
        let mut ymax = f32::MIN;
        let mut plot_data = Vec::new();
        for (idx, point) in (&data).iter().enumerate() {
            let plot_y: f32;

            if let Some(y) = point.items.get(yvar.as_str()) {
                match y {
                    LogData::String(_) => todo!(),
                    LogData::Float(y) => {
                        plot_y = *y;

                        if *y < ymin {
                            ymin = *y;
                        }

                        if *y > ymax {
                            ymax = *y;
                        }
                    }
                    LogData::Int(y) => {
                        let y = (*y) as f32;

                        plot_y = y;

                        if y < ymin {
                            ymin = y;
                        }

                        if y > ymax {
                            ymax = y;
                        }
                    }
                }

                xmax = idx as f32;
                plot_data.push((idx as f32, plot_y));
            }
        }

        let title = yvar.to_string();

        let root_area = BitMapBackend::new(&path, (600, 400)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption(title, ("sans-serif", 40))
            .build_cartesian_2d(0.0..xmax, ymin.min(0.0)..ymax)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();

        ctx.draw_series(LineSeries::new(plot_data, &GREEN)).unwrap();
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use std::{env, fs::OpenOptions, path::PathBuf};

    use super::{CsvLogger, Logger};

    #[test]
    fn test_should_log() {
        let mut pth = env::current_dir().unwrap();
        pth.push("log.csv");
        let logger = CsvLogger::new(pth, false, true);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Ok(()));
    }

    #[test]
    fn test_shouldnt_log1() {
        let logger = CsvLogger::new(PathBuf::from("this/path/shouldnt/exist.csv"), false, true);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Err("logger dump path dir does not exist"));
    }

    #[test]
    fn test_shouldnt_log2() {
        let mut pth = env::current_dir().unwrap();
        pth.push("log.txt");
        let logger = CsvLogger::new(pth, false, true);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Err("logger dump path should be a csv"));
    }

    #[test]
    fn test_shouldnt_log3() {
        let mut pth = env::current_dir().unwrap();
        pth.push("__very_strange_name.csv");

        let _ = OpenOptions::new()
            .create(true)
            .write(true)
            .open(pth.clone());

        let logger = CsvLogger::new(pth.clone(), false, false);
        let can_check = logger.check_can_log(false);

        assert_eq!(can_check, Err("logger dump file already exists"));

        // cleanup
        let _ = std::fs::remove_file(pth);
    }
}

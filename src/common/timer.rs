use std::collections::HashMap;
use std::time::Instant;

use crate::common::logger::{LogData, LogItem};

/// Lightweight per-phase profiler with averaged timings.
/// - Use `time(name, || { ... })` to measure a closure.
/// - Or `record(name, secs)` to add a manual duration.
/// - Call `into_logitem(step, interval_steps, prefix)` to get averaged ms per phase.
/// - Call `reset()` after logging to start a new interval.
#[derive(Default, Debug, Clone)]
pub struct Profiler {
    enabled: bool,
    sums: HashMap<&'static str, f64>,     // seconds
    counts: HashMap<&'static str, usize>, // samples
}

impl Profiler {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, ..Default::default() }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn record(&mut self, name: &'static str, secs: f64) {
        if !self.enabled {
            return;
        }
        *self.sums.entry(name).or_insert(0.0) += secs;
        *self.counts.entry(name).or_insert(0) += 1;
    }

    pub fn time<T, F: FnOnce() -> T>(&mut self, name: &'static str, f: F) -> T {
        if !self.enabled {
            return f();
        }
        let t0 = Instant::now();
        let out = f();
        let dt = t0.elapsed().as_secs_f64();
        self.record(name, dt);
        out
    }

    /// Builds a LogItem with averaged ms timings per phase and optional prefix for keys.
    /// Example keys: `agent_avg_policy_ms` if prefix="agent_" and name="policy".
    pub fn into_logitem(
        &self,
        step: usize,
        interval_steps: usize,
        prefix: Option<&str>,
    ) -> Option<LogItem> {
        if !self.enabled {
            return None;
        }
        if self.counts.is_empty() {
            return None;
        }
        let mut item = LogItem::default()
            .push("global_step".to_string(), LogData::Int(step as i32))
            .push(
                "timing_interval_steps".to_string(),
                LogData::Int(interval_steps as i32),
            );

        let prefix = prefix.unwrap_or("");
        let mut avg_loop_ms: Option<f64> = None;
        for (name, sum) in &self.sums {
            let count = *self.counts.get(name).unwrap_or(&1) as f64;
            let avg_ms = (sum / count) * 1000.0;
            let key = format!("{}avg_{}_ms", prefix, name);
            item = item.push(key, LogData::Float(avg_ms as f32));

            if *name == "loop" {
                avg_loop_ms = Some(avg_ms);
            }
        }

        if let Some(loop_ms) = avg_loop_ms {
            if loop_ms > 0.0 {
                let sps = 1000.0 / loop_ms; // steps per second based on avg loop time
                let key = format!("{}steps_per_sec", prefix);
                item = item.push(key, LogData::Float(sps as f32));
            }
        }

        Some(item)
    }

    pub fn reset(&mut self) {
        self.sums.clear();
        self.counts.clear();
    }
}

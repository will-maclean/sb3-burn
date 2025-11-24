use std::{
    mem,
    sync::mpsc::{self, Receiver, Sender},
    thread::JoinHandle,
};

use crate::{
    common::{
        spaces::Space,
        vec_env::base_env::{VecEnv, VecEnvObservation},
    },
    env::base::{Env, Info, ResetOptions, RewardRange},
};

#[derive(Clone, Debug)]
struct SubProcEnvObservation<O> {
    pub obs: O,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub truncated_obs: Option<O>,
    pub info: Info<O>,
}

impl<O> From<Vec<SubProcEnvObservation<O>>> for VecEnvObservation<O> {
    fn from(mut value: Vec<SubProcEnvObservation<O>>) -> Self {
        let mut obs = Vec::with_capacity(value.len());
        let mut reward = Vec::with_capacity(value.len());
        let mut terminated = Vec::with_capacity(value.len());
        let mut truncated = Vec::with_capacity(value.len());
        let mut info = Vec::with_capacity(value.len());
        let mut truncated_obs = Vec::with_capacity(value.len());

        for _ in 0..value.len() {
            let o = value.pop().unwrap();

            obs.push(o.obs);
            reward.push(o.reward);
            terminated.push(o.terminated);
            truncated.push(o.truncated);
            info.push(o.info);
            truncated_obs.push(o.truncated_obs);
        }

        Self {
            obs,
            reward,
            terminated,
            truncated,
            info,
            truncated_obs,
        }
    }
}

#[derive(Debug)]
enum Command<A>
where
    A: Clone + Send + 'static,
{
    Reset(Option<u64>, Option<ResetOptions>),
    Step(A),
    Render,
    Close,
}

#[derive(Debug)]
enum EnvSignal<O>
where
    O: Clone + Send + 'static,
{
    StepResult(SubProcEnvObservation<O>),
    ResetResult(O),
    Error,
}

fn worker<O: Clone + Send + 'static, A: Clone + Send + 'static, F: Fn() -> Box<dyn Env<O, A>>>(
    env_fn: F,
    cmd_rx: Receiver<Command<A>>,
    signal_tx: Sender<EnvSignal<O>>,
    env_id: usize,
) {
    let mut env = env_fn();

    loop {
        match cmd_rx.recv() {
            Ok(cmd) => match cmd {
                Command::Reset(seed, options) => {
                    signal_tx
                        .send(EnvSignal::ResetResult(env.reset(seed, options)))
                        .expect(&format!(
                            "Subprov vec env {env_id}: failure communicating with main thread."
                        ));
                }
                Command::Step(a) => {
                    let raw_state = env.step(&a);
                    let mut result_state = SubProcEnvObservation {
                        obs: raw_state.obs,
                        reward: raw_state.reward,
                        terminated: raw_state.terminated,
                        truncated: raw_state.truncated,
                        info: raw_state.info,
                        truncated_obs: None,
                    };

                    if result_state.terminated || result_state.truncated {
                        // need to handle an env reset
                        if result_state.truncated {
                            // special handling if the env was truncated
                            result_state.truncated_obs = Some(result_state.obs.clone());
                        }

                        result_state.obs = env.reset(None, None);
                    }

                    signal_tx
                        .send(EnvSignal::StepResult(result_state))
                        .expect(&format!(
                            "Subprov vec env {env_id}: failure communicating with main thread."
                        ));
                }
                Command::Render => env.render(),
                Command::Close => {
                    env.close();
                    break;
                }
            },
            Err(_) => break,
        }
    }
}

pub struct SubProcVecEnv<O, A>
where
    O: Clone + Send + 'static,
    A: Clone + Send + 'static,
{
    n_envs: usize,
    observation_space: Box<dyn Space<O>>,
    action_space: Box<dyn Space<A>>,
    _renderable: bool,
    _reward_range: RewardRange,

    cmd_txs: Vec<Sender<Command<A>>>,
    signal_rxs: Vec<Receiver<EnvSignal<O>>>,
    handles: Vec<JoinHandle<()>>,
}

impl<O, A> SubProcVecEnv<O, A>
where
    O: Clone + Send + 'static,
    A: Clone + Send + 'static,
{
    pub fn new<F: Fn() -> Box<dyn Env<O, A>> + Clone + Send + 'static>(
        env_fn: F,
        n_envs: usize,
    ) -> Self {
        let env = env_fn();

        let mut cmd_txs = vec![];
        let mut signal_rxs = vec![];
        let mut handles = vec![];

        for i in 0..n_envs {
            let (cmd_tx, cmd_rx) = mpsc::channel::<Command<A>>();
            let (signal_tx, signal_rx) = mpsc::channel::<EnvSignal<O>>();

            let env_fn_clone = env_fn.clone();
            let handle = std::thread::spawn(move || {
                worker(env_fn_clone, cmd_rx, signal_tx, i);
            });

            cmd_txs.push(cmd_tx);
            signal_rxs.push(signal_rx);
            handles.push(handle);
        }

        Self {
            n_envs,
            observation_space: env.observation_space(),
            action_space: env.action_space(),
            _renderable: env.renderable(),
            _reward_range: env.reward_range(),
            cmd_txs,
            signal_rxs,
            handles,
        }
    }
}

impl<O, A> VecEnv<O, A> for SubProcVecEnv<O, A>
where
    O: Clone + Send + 'static,
    A: Clone + Send + 'static,
{
    fn step_async(&mut self, action: Vec<A>) {
        if action.len() != self.n_envs {
            panic!(
                "Wrong amount of actions! Got {}, expecting {}",
                action.len(),
                self.n_envs
            );
        }

        for (i, a) in action.into_iter().enumerate() {
            self.cmd_txs[i]
                .send(Command::Step(a))
                .expect(&format!("Failure communicating with subproc vec env {i}"));
        }
    }

    fn step_wait(&mut self) -> VecEnvObservation<O> {
        let mut vec_obs = vec![];

        for (i, sr) in self.signal_rxs.iter().enumerate() {
            loop {
                match sr
                    .recv()
                    .expect(&format!("Failure communicating with subproc vec env {i}"))
                {
                    EnvSignal::StepResult(sub_proc_env_observation) => {
                        vec_obs.push(sub_proc_env_observation);
                        break;
                    }
                    EnvSignal::ResetResult(_) => {}
                    EnvSignal::Error => panic!("Failure communicating with subproc vec env {}", i),
                }
            }
        }

        VecEnvObservation::from(vec_obs)
    }

    fn reset(&mut self, seed: Option<u64>, options: Option<ResetOptions>) -> Vec<O> {
        // This is blocking across all envs - can't be avoided
        self.cmd_txs.iter().enumerate().for_each(|(i, c)| {
            c.send(Command::Reset(seed, options.clone()))
                .expect(&format!("Failure communicating with subproc vec env {}", i))
        });

        let mut obs = Vec::with_capacity(self.n_envs);

        for (i, sr) in self.signal_rxs.iter().enumerate() {
            loop {
                match sr.recv() {
                    Ok(s) => match s {
                        EnvSignal::StepResult(_) => {}
                        EnvSignal::ResetResult(ob) => {
                            obs.push(ob);
                            break;
                        }
                        EnvSignal::Error => panic!("Error reported in subproc vec env {}", i),
                    },
                    Err(_) => panic!("Failure communicating with subproc vec env {}", i),
                }
            }
        }

        obs
    }

    fn action_space(&self) -> Box<dyn Space<A>> {
        dyn_clone::clone_box(&*self.action_space)
    }

    fn observation_space(&self) -> Box<dyn Space<O>> {
        dyn_clone::clone_box(&*self.observation_space)
    }

    fn reward_range(&self) -> RewardRange {
        self._reward_range.clone()
    }

    fn render(&self) {
        self.cmd_txs[0]
            .send(Command::Render)
            .expect("Failure communicating with subproc vec env 0")
    }

    fn renderable(&self) -> bool {
        self._renderable
    }

    fn close(&mut self) {
        self.cmd_txs.iter().enumerate().for_each(|(i, c)| {
            c.send(Command::Close)
                .expect(&format!("Failure communicating with subproc vec env {}", i))
        });

        let handles = mem::take(&mut self.handles);

        handles.into_iter().enumerate().for_each(|(i, h)| {
            h.join()
                .expect(&format!("Failure closing subproc vec env {}", i))
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::vec_env::{base_env::VecEnv, subproc_vec_env::SubProcVecEnv},
        env::classic_control::cartpole::CartpoleEnv,
    };

    #[test]
    fn test_sub_proc_vec_env() {
        let mut vec_env = SubProcVecEnv::new(|| return Box::new(CartpoleEnv::default()), 3);
        let mut action_space = vec_env.action_space();

        let reset_obs = vec_env.reset(None, None);

        assert_eq!(reset_obs.len(), 3);

        for _ in 0..10 {
            let act = (0..3).map(|_| action_space.sample()).collect();
            let obs = vec_env.step(act);
            assert_eq!(obs.len(), 3);
        }

        vec_env.close();
    }
}

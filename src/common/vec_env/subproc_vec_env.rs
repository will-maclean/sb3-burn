use std::sync::mpsc::SendError;

use tokio::{
    sync::mpsc::{self, Receiver, Sender},
    task::{JoinError, JoinHandle},
};

use crate::{
    common::{spaces::Space, vec_env::base_env::VecEnv},
    env::base::{Env, EnvObservation, ResetOptions, RewardRange},
};

#[derive(Debug)]
enum Command<A>
where
    A: Clone + Send + 'static,
{
    Reset,
    Step(A),
    Render,
    Close,
}

#[derive(Debug)]
enum EnvSignal<O>
where
    O: Clone + Send + 'static,
{
    StepResult(EnvObservation<O>),
    ResetResult(O),
}

async fn worker<
    O: Clone + Send + 'static,
    A: Clone + Send + 'static,
    F: Fn() -> Box<dyn Env<O, A>>,
>(
    env_fn: F,
    mut cmd_rx: Receiver<Command<A>>,
    signal_tx: Sender<EnvSignal<O>>,
) {
    let mut env = env_fn();

    loop {
        match cmd_rx.recv().await {
            Some(cmd) => match cmd {
                Command::Reset => {
                    signal_tx
                        .send(EnvSignal::ResetResult(env.reset(None, None)))
                        .await;
                }
                Command::Step(a) => {
                    signal_tx.send(EnvSignal::StepResult(env.step(&a))).await;
                }
                Command::Render => todo!(),
                Command::Close => todo!(),
            },
            None => break,
        }
    }
}

struct SubProcVecEnv<O, A>
where
    O: Clone + Send + 'static,
    A: Clone + Send + 'static,
{
    n_envs: usize,
    observation_space: Box<dyn Space<O>>,
    action_space: Box<dyn Space<A>>,

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

        for _ in 0..n_envs {
            let (cmd_tx, cmd_rx) = mpsc::channel::<Command<A>>(32);
            let (signal_tx, signal_rx) = mpsc::channel::<EnvSignal<O>>(32);

            let env_fn_clone = env_fn.clone();
            let handle = tokio::spawn(async move {
                worker(env_fn_clone, cmd_rx, signal_tx).await;
            });

            cmd_txs.push(cmd_tx);
            signal_rxs.push(signal_rx);
            handles.push(handle);
        }

        Self {
            n_envs,
            observation_space: env.observation_space(),
            action_space: env.action_space(),
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
    async fn step_async(&mut self, action: Vec<A>) {
        if action.len() != self.n_envs {
            panic!(
                "Wrong amount of actions! Got {}, expecting {}",
                action.len(),
                self.n_envs
            );
        }

        for (i, a) in action.into_iter().enumerate() {
            self.cmd_txs[i].send(Command::Step(a)).await;
        }
    }

    async fn step_wait(&mut self) -> Vec<EnvObservation<O>> {
        todo!()
    }

    async fn reset(&mut self, seed: Option<u64>, options: Option<ResetOptions>) -> Vec<O> {
        todo!()
    }

    fn action_space(&self) -> Box<dyn Space<A>> {
        todo!()
    }

    fn observation_space(&self) -> Box<dyn Space<O>> {
        todo!()
    }

    fn reward_range(&self) -> RewardRange {
        todo!()
    }

    fn render(&self) {
        todo!()
    }

    fn renderable(&self) -> bool {
        todo!()
    }

    fn close(&mut self) {
        todo!()
    }
}

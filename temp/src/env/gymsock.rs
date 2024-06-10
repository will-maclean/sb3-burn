use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use async_trait::async_trait;

use crate::spaces::{Action, ActionSpace, Obs, ObsSpace};
use super::base::{Env, EnvObservation, EnvError};

#[derive(Serialize, Deserialize, Debug)]
pub struct RunningParams {
    pub env_name: String,
    pub env_type: String,
    pub render: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct EnvObservationData {
    obs: Vec<f32>,
    reward: f32,
    done: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct EnvInfo {
    observation_space: Vec<i32>,
    action_space: i32,
    reward_range: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ServerMessage {
    info: String,
    data: HashMap<String, String>,
}

pub struct GymSockEnv {
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
    writer: tokio::net::tcp::OwnedWriteHalf,
    action_space: ActionSpace,
    observation_space: ObsSpace,
}

impl GymSockEnv {
    pub async fn new(addr: &str, params: RunningParams) -> Result<Self, EnvError> {
        match GymSockEnv::load_env(addr, params).await {
            Ok(e) => Ok(e),
            Err(e) => Err(EnvError::InitError(e.to_string())),
        }
    }

    pub async fn load_env(addr: &str, params: RunningParams) -> Result<Self, Box<dyn Error>> {
        let stream = TcpStream::connect(addr).await?;
        let (read_half, write_half) = stream.into_split();
        let mut reader = BufReader::new(read_half);

        // Receive initial message from server
        let mut buffer = String::new();
        reader.read_line(&mut buffer).await?;
        let received_message: ServerMessage = serde_json::from_str(&buffer)?;

        let mut env: GymSockEnv = Self {
            reader,
            writer: write_half,
            action_space: ActionSpace::Discrete { size: 0 },
            observation_space: ObsSpace::Discrete { size: 0 },
        };

        // Send running parameters to the server
        let message = json!({
            "env_name": params.env_name,
            "env_type": params.env_type,
            "render": params.render,
        });

        env.writer.write_all((message.to_string() + "\n").as_bytes()).await?;

        // Receive environment info from server
        let mut buffer = String::new();
        env.reader.read_line(&mut buffer).await?;
        let mut env_info: EnvInfo = serde_json::from_str(&buffer)?;

        // Convert the string representation of infinity to f32
        let reward_range = (
            if env_info.reward_range[0] == "-inf" {
                std::f32::NEG_INFINITY
            } else {
                env_info.reward_range[0].parse::<f32>()?
            },
            if env_info.reward_range[1] == "inf" {
                std::f32::INFINITY
            } else {
                env_info.reward_range[1].parse::<f32>()?
            }
        );

        env_info.reward_range = vec![reward_range.0.to_string(), reward_range.1.to_string()]; // Update reward_range to use actual float values

        println!("Environment Info: {:?}\n", env_info);

        env.action_space = ActionSpace::Discrete { size: env_info.action_space as usize };
        env.observation_space = ObsSpace::Continuous {
            lows: vec![f32::NEG_INFINITY; env_info.observation_space.len()],
            highs: vec![f32::INFINITY; env_info.observation_space.len()],
        };

        // env.reward_space ?

        Ok(env)
    }

    async fn send_action(&mut self, action: &Action) -> Result<EnvObservation, Box<dyn Error>> {
        let action_message = json!({
            "action": match action {
                Action::Discrete { idx, .. } => idx,
                _ => return Err("Unsupported action type".into()),
            }
        });

        self.writer.write_all((action_message.to_string() + "\n").as_bytes()).await?;

        let mut buffer = String::new();
        self.reader.read_line(&mut buffer).await?;
        let response: EnvObservationData = serde_json::from_str(&buffer)?;

        println!("data {:?}", response);

        Ok(EnvObservation {
            obs: Obs::Continuous { space: self.observation_space.clone(), data: response.obs },
            reward: response.reward,
            done: response.done,
        })
    }

    async fn send_reset(&mut self) -> Result<Obs, Box<dyn Error>> {
        let reset_message = json!({
            "action": "reset"
        });

        self.writer.write_all((reset_message.to_string() + "\n").as_bytes()).await?;

        let mut buffer = String::new();
        self.reader.read_line(&mut buffer).await?;
        let response: EnvObservationData = serde_json::from_str(&buffer)?;

        Ok(Obs::Continuous { space: self.observation_space.clone(), data: response.obs })
    }
}

#[async_trait]
impl Env for GymSockEnv {
    async fn step(&mut self, action: &Action) -> Result<EnvObservation, EnvError> {
        match self.send_action(action).await {
            Ok(d) => Ok(d),
            Err(e) => Err(EnvError::FatalEnvError(e.to_string())),
        }  
    }

    async fn reset(&mut self) -> Result<Obs, EnvError> {
        match self.send_reset().await {
            Ok(d) => Ok(d),
            Err(e) => Err(EnvError::FatalEnvError(e.to_string())),
        }
    }

    fn action_space(&self) -> ActionSpace {
        self.action_space.clone()
    }

    fn observation_space(&self) -> ObsSpace {
        self.observation_space.clone()
    }

    fn render(&self) {}

    fn renderable(&self) -> bool {
        false
    }
}

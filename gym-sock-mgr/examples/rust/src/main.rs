use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

#[derive(Serialize, Deserialize, Debug)]
struct RunningParams {
    env_name: String,
    env_type: String,
    render: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct EnvInfo {
    observation_space: Vec<i32>,
    action_space: i32,
    reward_range: Vec<String>,
    initial_observation: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ServerMessage {
    info: String,
    data: std::collections::HashMap<String, String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let addr = "127.0.0.1:65432";
    let mut stream = TcpStream::connect(addr).await?;
    let (read_half, mut write_half) = stream.split();
    let mut reader = BufReader::new(read_half);

    // Receive initial message from server
    let mut buffer = String::new();
    reader.read_line(&mut buffer).await?;
    let received_message: ServerMessage = serde_json::from_str(&buffer)?;
    println!("Received: {:?}\n", received_message);

    // Send running parameters to the server
    let params = RunningParams {
        env_name: "CartPole-v1".to_string(),
        env_type: "simple".to_string(),
        render: false,
    };

    let message = json!({
        "env_name": params.env_name,
        "env_type": params.env_type,
        "render": params.render,
    });

    write_half.write_all((message.to_string() + "\n").as_bytes()).await?;

    // Receive environment info from server
    buffer.clear();
    reader.read_line(&mut buffer).await?;
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

    // Example interaction with the server (sending actions and receiving state data)
    for i in 1..=5 {
        println!("\nEpisode {}", i);
        loop {
            // Send an action to the server
            let action_message = json!({
                "action": 1
            });

            write_half.write_all((action_message.to_string() + "\n").as_bytes()).await?;

            // Receive the response from the server
            buffer.clear();
            reader.read_line(&mut buffer).await?;
            let response: serde_json::Value = serde_json::from_str(&buffer)?;
            println!("Response: {:?}", response);

            // Break if done
            if response["done"].as_bool().unwrap_or(false) {
                break;
            }
        }
    }
    

    // Send quit action to server to end session
    let action_message = json!({
            "action": "quit"
    });

    write_half.write_all((action_message.to_string() + "\n").as_bytes()).await?;

    // Disconnect from socket gracefully
    stream.shutdown().await?;
    println!("Disconnected from the server.");

    Ok(())
}

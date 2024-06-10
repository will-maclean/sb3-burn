from socket import socket, AF_INET, SOCK_STREAM
import gym
from json import dumps, loads

from gym_sock_mgr.wrappers.builders import *


class GymSocketMgr:

    def __init__(self, host='127.0.0.1', port=65432) -> 'GymSocketMgr':
        self.host = host
        self.port = port
        self.socket = self._create_socket()
        self.env = None
        self.running_params = None

    def _create_socket(self):
        server_socket = socket(AF_INET, SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        print(f'Server listening on {self.host}:{self.port}')
        server_socket.listen()
        conn, addr = server_socket.accept()
        print(f'Accepted connection from: {addr}')
        return conn
    
    def _send_message(self, message):
        self.socket.sendall((dumps(message) + '\n').encode())

    def _receive_message(self):
        data = b""
        while True:
            part = self.socket.recv(1024)
            data += part
            if b'\n' in part:
                break
        return loads(data.decode().strip())
    
    def _get_running_params(self):
        required_params = {
            "env_name": "The gym env string such as 'CartPole-v1'",
            "env_type": "Should be either 'simple' for an env with no extra wrappers, or 'full' for a wrapped env",
            "render": "A boolean which specifies if the env should be rendered to the screen"
            # Add others as required
        }

        required_keys = sorted(list(required_params.keys()))

        self._send_message({
            "info": "Please return the object stored at 'data' with the desired values for the env",
            "data": required_params
        })

        while self.running_params is None:

            res = self._receive_message()

            if sorted(list(res.keys())) != required_keys:
                self._send_message({
                        "Error": f"'data' object missing required key/s",
                        "data": required_params
                    })
            else:
                self.running_params = res
                print(f"\nLoaded running parameters from client: \n{self.running_params}")
    
    def _load_env(self):
        env_builder = make_full_env if self.running_params["env_type"] == "full" else make_simple_env
        self.env = env_builder(gym.make(self.running_params["env_name"]))

        _ = self.env.reset() # Confirm env can be reset
        print(f"\nLoaded {self.running_params['env_type']} gym environment {self.running_params['env_name']}")

        # Send the observation, action, and reward space information to the client
        env_info = {
            'observation_space': list(self.env.observation_space.shape),
            'action_space': self.env.action_space.n,
            'reward_range': list(self.env.reward_range)
        }

        # Look for infinity values and replace with string for json compatibility
        for k, v in env_info.items():
            if isinstance(v, list):
                for i, el in enumerate(v):
                    if el == -np.inf:
                        v[i] = '-inf'
                    elif el == np.inf:
                        v[i] = 'inf'
                if k == "reward_range":
                    env_info[k] = [str(el) for el in v] 


        print(f"Sending env info: \n{env_info}")
        self._send_message(env_info)

    def safely_run(self):
        try:
            # Get params for running and load env
            self._get_running_params()
            self._load_env()

            render = self.running_params["render"]

            while True:
                # Receive action from client
                if not (action := self._receive_message().get("action")):
                    self._send_message({ "Error": "action expected in form {\"action\": value}" })
                    continue

                if action == "quit":
                    return
                elif action == "reset":
                    observation = self.env.reset()
                    reward, done = 0.0, False   # dummy values
                else:
                    # Step through the environment with the received action
                    observation, reward, done, _ = self.env.step(action)

                # Render the env if requested by client
                if render:
                    self.env.render()

                # Send the resulting state, reward, and done state back to the client
                response = {
                    'obs': observation.tolist(),
                    'reward': reward,
                    'done': done
                }

                self._send_message(response)

        except Exception as e:
            print(e.with_traceback())
        finally:
            self.socket.close()
            print("Socket closed successfully")
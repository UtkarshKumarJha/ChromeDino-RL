# 🦖 Chrome Dino AI - Reinforcement Learning with Gymnasium

This project implements a custom Gymnasium environment to train an AI agent to play the Chrome Dino game (inspired by the offline Google Chrome dinosaur runner game). The environment is built using PyGame, and the agent is trained using reinforcement learning algorithms such as PPO from Stable-Baselines3.

## 🚀 Features

- Custom PyGame-based Chrome Dino game environment.
- Gymnasium-compatible interface (`reset`, `step`, `render`, etc.).
- Multiple obstacles: small cactus, large cactus, and bird.
- Discrete action space: `0` (no action), `1` (jump), `2` (duck).
- Clean observation space with obstacle-relative distance and state flags.
- Easy integration with RL libraries like Stable-Baselines3.

---

## 🧠 Tech Stack

- **Python 3.8+**
- **Gymnasium** – for creating the RL environment.
- **PyGame** – for graphics and game logic.
- **Stable-Baselines3** – for training agents with PPO/DQN/A2C etc.
- **NumPy** – for efficient array computations.

---

## 📁 Project Structure

```bash
Chrome-Dino-Runner/
├── assets/                     # Game images (cactus, dino, bird, etc.)
│   ├── Cactus/
│   ├── Bird/
│   └── Other/
├── dino_env.py                # Gymnasium-compatible custom environment and game logic
├── train_agent.py             # RL agent training script (PPO, etc.)
├── test_agent.py              # Run the trained agent
└── README.md                  # Project documentation

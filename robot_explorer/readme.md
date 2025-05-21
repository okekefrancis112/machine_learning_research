# Multi-Agent Exploration Simulation

This project is a Python-based simulation of multiple autonomous bots exploring an unknown 2D map. The bots use the A* pathfinding algorithm to navigate and map unmapped areas while avoiding walls. The simulation visualizes the exploration process using Matplotlib, showing bots' positions, planned routes, and mapped areas in real-time.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Media](#media)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Multi-Agent System**: Simulates up to 15 bots exploring a 50x50 grid map.
- **A* Pathfinding**: Bots use the A* algorithm to find optimal paths to unexplored areas.
- **Real-Time Visualization**: Displays bot positions (red squares), planned routes (red lines), destinations (yellow dots), and the exploration map (grayscale) using Matplotlib.
- **Dynamic Exploration**: Bots mark areas as mapped, planned, or walls, updating the map as they move.
- **Customizable Parameters**: Adjustable map size, number of bots, and iteration limits.

## Installation

### Prerequisites
- Python 3.7–3.11 (Python 3.13 is not supported by TensorFlow as of May 2025).
- A compatible operating system (Windows, macOS, or Linux).
- `pip` or `conda` for package management.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/multi-agent-exploration.git
   cd multi-agent-exploration
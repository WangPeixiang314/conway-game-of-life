# Conway's Game of Life - GPU/CPU High Performance Version

A high-performance cellular automaton simulator based on Conway's Game of Life rules, featuring GPU acceleration and interactive controls. This project uses Pygame for the graphical interface, Numba for CPU optimization, and optional CUDA for GPU acceleration.

## 🎮 Features

### Core Features
- **Classic Game of Life Rules**: Strictly follows Conway's four fundamental rules
- **GPU Acceleration Support**: Utilizes CUDA for parallel GPU computing, significantly boosting performance
- **CPU Optimization**: Uses Numba JIT compiler for optimized CPU performance
- **Real-time Interaction**: Supports mouse drawing and erasing operations

### Visualization Features
- **High-definition Rendering**: 800×440 grid with 2-pixel cell size
- **Smooth Animation**: Adjustable frame rate (15-300 FPS)
- **Modern Interface**: Dark theme with real-time status display
- **Real-time Statistics**: Displays current generation, live cell count, compute time, FPS, and more

### Interactive Functions
- **Brush Tool**: Adjustable circular brush (1-30 pixels)
- **Cluster Generation**: Random initial state generation based on Gaussian distribution
- **Pause/Resume**: Pause and resume simulation at any time
- **Clear Function**: One-click clearing of all cells
- **Reset Function**: Regenerate random initial state
- **Grid Display**: Toggle grid lines on/off

### Performance Features
- **Smart Device Switching**: Dynamically switch between CPU/GPU computation modes at runtime
- **Memory Optimization**: Uses NumPy arrays and Numba JIT optimization
- **Parallel Computing**: Supports multi-core CPU and GPU parallel processing

## 🚀 Quick Start

### Requirements
- Python 3.7+
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Program
```bash
python main.py
```

## 🎯 Game Rules

Conway's Game of Life four basic rules:

1. **Survival**: Any live cell with 2 or 3 live neighbors survives to the next generation
2. **Death**: Any live cell with fewer than 2 or more than 3 live neighbors dies
3. **Reproduction**: Any dead cell with exactly 3 live neighbors becomes a live cell
4. **Stasis**: All other cells maintain their current state

*Note: Uses periodic (wrap-around) boundary conditions*

## 🎮 Controls

### Mouse Operations
- **Left Drag**: Draw live cells (based on current brush size)
- **Save Mode ON**: Left click only adds live cells, no erasing
- **Save Mode OFF**: Left click toggles cell state (alive ↔ dead)

### Interface Buttons
- **Pause/Resume**: Control simulation running state
- **Reset**: Regenerate random initial state
- **Toggle Grid**: Show/hide grid lines
- **Clear**: Remove all live cells
- **Save Mode**: Toggle brush mode (add/toggle)
- **Decrease Brush**: Decrease brush radius (minimum 1)
- **Increase Brush**: Increase brush radius (maximum 30)
- **FPS-**: Decrease simulation frame rate (minimum 15 FPS)
- **FPS+**: Increase simulation frame rate (maximum 300 FPS)
- **Switch to CPU/GPU**: Toggle between CPU and GPU computation modes

### Keyboard Shortcuts
The program mainly relies on mouse operations, no preset keyboard shortcuts.

## 🔧 Technical Architecture

### Core Technology Stack
- **Pygame 2.5.2+**: Graphics interface and rendering
- **NumPy 1.24.3+**: Numerical computation and array operations
- **Numba 0.58.0+**: JIT compilation optimization
- **CUDA** (optional): GPU parallel computing

### Performance Characteristics
- **CPU Mode**: Uses Numba JIT and parallel computing, significantly faster than pure Python
- **GPU Mode**: CUDA kernel parallel processing, suitable for large-scale computation
- **Memory Efficiency**: Uses uint8 data type, optimized memory usage

### Configuration Parameters
- **Grid Size**: 800×440 (352,000 cells)
- **Initial Density**: 41%
- **Cluster Centers**: 4 Gaussian distribution centers
- **Cluster Radius**: 345 pixels
- **Default FPS**: 60 FPS

## 🎨 Interface Guide

### Main Interface Layout
- **Upper Grid Area**: Displays cell states and grid
- **Lower Control Panel**: Contains all control buttons and status information

### Status Information
- **Device**: Current computation device (CPU/GPU)
- **Compute Time**: Time per iteration (milliseconds)
- **FPS**: Current running frame rate
- **Generation**: Number of simulation generations completed
- **Live Cells**: Current live cell count and percentage
- **Status**: Running/paused state
- **Save Mode**: Brush mode status
- **Brush Size**: Current brush radius

## ⚠️ Notes

1. **GPU Support**: Requires NVIDIA GPU with CUDA support and drivers installed
2. **Font Display**: Program attempts to load Chinese fonts, falls back to system default if failed
3. **Performance Difference**: GPU mode performs better on supported devices, but has initialization overhead on first run
4. **Boundary Handling**: Uses periodic boundary conditions, cells can exit one side and enter from the opposite side

## 📄 License

This project uses the MIT License, see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Issues and Pull Requests to improve the project are welcome!

---

*Game of Life, evolving in code*
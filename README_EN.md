# Conway's Game of Life - GPU/CPU High Performance Version

A high-performance cellular automaton simulator based on Conway's Game of Life rules, featuring GPU acceleration and interactive controls.

## 🎮 Features

### Core Features
- **Classic Game of Life Rules**: Strictly follows Conway's four fundamental rules
- **GPU Acceleration Support**: Utilizes CUDA for parallel GPU computing, significantly boosting performance
- **CPU Optimization**: Uses Numba JIT compiler for optimized CPU performance
- **Real-time Interaction**: Supports mouse drawing, erasing, dragging and other operations

### Visualization Features
- **High-definition Rendering**: Supports 800x440 resolution grid with 2-pixel cell size
- **Smooth Animation**: Up to 300 FPS with adjustable frame rate
- **Modern UI**: Contemporary interface design with Chinese language support
- **Real-time Statistics**: Displays current generation, live cell count, FPS and more

### Interactive Functions
- **Brush Tool**: Adjustable brush size for drawing and erasing
- **Preset Patterns**: Built-in classic patterns (gliders, spaceships, etc.)
- **Random Generation**: Random initial states with adjustable density and clustering
- **Pause/Resume**: Pause and resume simulation at any time
- **Step Execution**: Supports single-step debugging mode
- **Clear Function**: One-click clearing of all cells

### Performance Optimization
- **Smart Device Selection**: Automatically detects and uses GPU (if available)
- **Memory Optimization**: Efficient data structures and memory management
- **Parallel Computing**: Full utilization of multi-core CPU and GPU parallel capabilities

## 🚀 Quick Start

### Requirements
- Python 3.11+
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Program
```bash
python main.py
```

### Controls
- **Left Mouse Button**: Draw live cells
- **Right Mouse Button**: Erase cells
- **Mouse Wheel**: Adjust brush size
- **Space**: Pause/Resume
- **R**: Regenerate random state
- **C**: Clear screen
- **+/-**: Adjust frame rate

## 🎯 Game Rules

Conway's Game of Life four basic rules:

1. **Survival**: Any live cell with 2 or 3 live neighbors survives to the next generation
2. **Death**: Any live cell with fewer than 2 or more than 3 live neighbors dies
3. **Reproduction**: Any dead cell with exactly 3 live neighbors becomes a live cell
4. **Stasis**: All other cells maintain their current state

## 🔧 Technical Architecture

### Core Technology Stack
- **Pygame**: Graphics interface and rendering engine
- **NumPy**: Efficient numerical computation
- **Numba**: JIT compiler for accelerating Python code
- **CUDA**: GPU parallel computing (optional)

### Performance Metrics
- **CPU Mode**: Multi-core parallel support, 50-100x faster than pure Python
- **GPU Mode**: On CUDA-supported devices, can handle larger grid sizes
- **Memory Usage**: Optimized memory layout, supports long-running sessions

## 🎨 Interface Guide

### Main Interface Areas
- **Grid Area**: Displays Game of Life cell states
- **Control Panel**: Contains all interactive buttons and status displays
- **Statistics**: Real-time display of generation, live cell count, FPS, etc.

### Button Functions
- **▶️/⏸️**: Start/Pause simulation
- **⏭️**: Single step execution
- **🔄**: Regenerate
- **🗑️**: Clear screen
- **💾**: Save current state
- **📁**: Load preset patterns

## 🤝 Contributing

Issues and Pull Requests are welcome! Before contributing code, please ensure:

1. Code follows PEP 8 standards
2. Add appropriate comments and documentation
3. Test compatibility across different configurations
4. Update relevant documentation

## 📄 License

This project uses the MIT License, see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- John Conway: Creator of the Game of Life
- Pygame Community: Provides excellent game development framework
- Numba Team: Making Python numerical computing faster
- All contributors and users

## 📞 Contact

For questions or suggestions, please contact us via GitHub Issues.

---

*Let life bloom in code!*
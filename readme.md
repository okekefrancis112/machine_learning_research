# Computer Vision and Robotics Implementations

This repository contains five Python implementations for various computer vision and robotics tasks including image classification, noise removal, histogram analysis, edge detection, and multi-robot exploration.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Image Classification](#image-classification)
- [Image Noise Remover](#image-noise-remover)
- [Histogram Analysis](#histogram-analysis)
- [Edge Detection](#edge-detection)
- [Multi-Robot Exploration](#multi-robot-exploration)
- [Usage Examples](#usage-examples)
- [File Structure](#file-structure)

## Prerequisites

- Python 3.7+
- OpenCV (cv2)
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Heapq (built-in Python module)

## Installation

```bash
pip install opencv-python tensorflow pandas numpy matplotlib
```

## Image Classification

### `image_classification.py`

A convolutional neural network implementation for traffic sign classification using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

#### Features

- **Dataset**: GTSRB Training dataset with 43 traffic sign classes
- **Architecture**: Deep CNN with batch normalization and dropout layers
- **Preprocessing**: Image resizing to 64x64 pixels and ROI extraction
- **Data Split**: 60% training, 20% validation, 20% testing

#### Model Architecture

```
Conv2D(16) → BatchNorm → Conv2D(16) → BatchNorm → MaxPool → Dropout(0.2)
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.2)
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.2)
Flatten → Dense(2048) → Dropout(0.2) → Dense(1024) → Dropout(0.2) → Dense(128) → Dropout(0.2) → Dense(43)
```

#### Key Parameters

- **Input Shape**: (64, 64, 3)
- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Categorical crossentropy
- **Epochs**: 10
- **Batch Size**: 16
- **Classes**: 43 traffic sign categories

#### Usage

1. Ensure GTSRB dataset is in `GTSRB/Training/` directory
2. Each class should have its own subdirectory with images and CSV metadata
3. Run the script to train and evaluate the model

```python
python image_classification.py
```

## Image Noise Remover

### `image_noise_remover.py`

Implements various noise reduction and image processing techniques including filtering, template matching, and histogram analysis.

#### Features

- **Histogram Analysis**: Generates histograms for input images
- **Mean Filtering**: 5x5 and 81x81 kernel sizes
- **Median Filtering**: Custom implementation with 5x5 and 81x81 windows
- **Template Matching**: Two methods (TM_CCORR and TM_CCORR_NORMED)

#### Filtering Methods

1. **Mean Filter (5x5)**
   - Fast smoothing for light noise
   - Kernel: 5x5 uniform weights (1/25)

2. **Mean Filter (81x81)**
   - Heavy smoothing for significant noise
   - Kernel: 81x81 uniform weights (1/6561)

3. **Median Filter**
   - Preserves edges while removing salt-and-pepper noise
   - Custom implementation with padding

#### Template Matching

- **Method 1**: `TM_CCORR` - Basic correlation
- **Method 2**: `TM_CCORR_NORMED` - Normalized correlation

#### Required Images

- `forest.jpg` - Primary test image
- `lane.png` - Secondary test image
- `beach.png` - Template image
- `pic.jpeg` - Target image for template matching

#### Output Files

- `Mean_5x5_#1.png`, `Mean_5x5_#2.png`
- `Mean_81x81_#1.png`, `Mean_81x81_#2.png`
- `new_median_filtered1.png`, `new_median_filtered2.png`

## Histogram Analysis

### `histogram.py`

Comprehensive histogram analysis and adaptive histogram equalization implementation.

#### Functions

##### `computeNormGrayHistogram(image)`
- Converts image to grayscale
- Creates 32-bin normalized histogram
- Displays bar chart with color values vs pixel count

##### `computeNormRGBHistogram(image)`
- Analyzes RGB channels separately
- Creates overlaid histograms for red, green, and blue channels
- 32-bin normalized histograms with density calculation

##### `AHE(img, winSize)`
- **Adaptive Histogram Equalization** implementation
- Uses contextual regions for local contrast enhancement
- Parameters:
  - `img`: Input grayscale image
  - `winSize`: Window size for local analysis (recommended: odd numbers like 129)

#### Features

- Image flipping demonstration
- Channel manipulation (red channel doubling)
- Comparison between regular histogram equalization and AHE
- Border replication for edge handling

#### Usage Example

```python
# Load and process image
image = cv2.imread('images/pic.jpeg')
computeNormGrayHistogram(image)
computeNormRGBHistogram(image)

# Apply AHE
beach = cv2.imread('images/beach.png', 0)
ahe_result = AHE(beach, 129)
```

## Edge Detection

### `edge_detection.py`

Complete Canny edge detection implementation from scratch with custom kernels and non-maximum suppression.

#### Processing Pipeline

1. **Smoothing**
   - Custom 5x5 Gaussian-like kernel
   - Reduces noise before edge detection

2. **Gradient Calculation**
   - Sobel operators for X and Y directions
   - Magnitude: `G = √(Gx² + Gy²)`
   - Direction: `θ = arctan2(Gy, Gx)`

3. **Non-Maximum Suppression (NMS)**
   - Thins edges to single-pixel width
   - Checks gradient direction (0°, 45°, 90°, 135°)
   - Suppresses non-maximum pixels

4. **Thresholding**
   - High threshold: 14.5% of maximum gradient
   - Binary edge map output

#### Algorithm Details

##### Sobel Kernels
```
Kx = [[-1, 0, 1],     Ky = [[-1, -2, -1],
      [-2, 0, 2],           [ 0,  0,  0],
      [-1, 0, 1]]           [ 1,  2,  1]]
```

##### Smoothing Kernel
```
K = [[2, 4, 5, 6, 2],
     [4, 9,12, 9, 4],
     [5,12,15,12, 5],
     [4, 9,12, 9, 4],
     [2, 4, 5, 4, 2]] / 159
```

#### Input Requirements

- Input image: `images/lane.png` (grayscale)
- Image should have clear edges for optimal results

#### Output

- Displays intermediate results at each processing stage
- Final binary edge map highlighting detected edges

## Multi-Robot Exploration

### `multi_robot_exploration.py`

A sophisticated multi-robot exploration simulation that coordinates multiple robots to map an unknown environment using A* pathfinding and intelligent task allocation.

#### Features

- **Multi-Agent Coordination**: 5 robots working collaboratively
- **Intelligent Pathfinding**: A* algorithm for optimal route planning
- **Real-time Visualization**: Live map updates showing robot positions and paths
- **Adaptive Exploration**: Dynamic destination selection based on unexplored areas
- **Collision Avoidance**: Wall detection and boundary handling

#### Algorithm Components

##### Core Parameters
```python
height = 50          # Map height
width = 50           # Map width
num_bots = 5         # Number of exploration robots
max_itr = 2500       # Maximum simulation iterations
```

##### Map Values
- `wall = 1`: Obstacle/wall cells
- `mapped = 0.4`: Successfully explored areas
- `planned = 0.2`: Planned robot paths
- `unmapped = 0`: Unknown/unexplored areas

#### Key Functions

##### `get_unexplored_areas(explore_map, unmapped_value)`
- Identifies all unmapped locations in the environment
- Returns Nx2 matrix of unexplored coordinates
- Filters out walls, mapped areas, and planned paths

##### `get_new_destination(current_position, unexplored_areas)`
- Selects closest unexplored area as next destination
- Uses Euclidean distance: `√((x₁-x₂)² + (y₁-y₂)²)`
- Implements greedy nearest-neighbor strategy

##### `update_explore_map(dest, route, explore_map, planned, unmapped)`
- Marks destination and route as planned
- Only updates previously unmapped areas
- Prevents overwriting walls or mapped regions

##### `update_position(curPos, route, dest, explore_map, mapped)`
- Moves robot one step along calculated route
- Marks new position as mapped
- Updates destination status and route queue
- Handles destination arrival logic

##### `a_star(array, start, goal)`
- Implements A* pathfinding algorithm
- Uses Manhattan distance heuristic
- Handles obstacle avoidance and boundary checking
- Returns optimal path as coordinate sequence

#### Simulation Flow

1. **Initialization**
   - Create 50x50 map with predefined walls
   - Position all robots at center coordinates
   - Initialize empty destinations and routes

2. **Exploration Loop** (max 2500 iterations)
   - For each robot:
     - Check if destination exists
     - If no destination: find nearest unexplored area
     - Calculate optimal path using A*
     - Move one step toward destination
     - Update exploration map

3. **Termination**
   - Stops when all areas are explored or max iterations reached
   - Reports total iterations required

#### Blueprint Environment

The simulation includes a predefined environment with obstacles:
```
Wall configuration:
- Vertical walls: columns 35 and 44, rows 9-19
- Horizontal connectors and barriers
- Padded boundaries around entire map
```

#### Visualization Features

- **Real-time Display**: Updates every 0.25 seconds
- **Robot Representation**: Red squares with blue borders
- **Path Visualization**: Red lines showing planned routes
- **Destination Markers**: Yellow dots indicating targets
- **Grayscale Map**: Different intensities for different area types

#### Performance Metrics

- **Completion Detection**: Monitors mapped vs total explorable area
- **Iteration Counting**: Tracks simulation efficiency
- **Coverage Analysis**: Calculates exploration percentage

#### Advanced Features

##### Pathfinding Optimization
- **Heuristic Function**: Euclidean distance estimation
- **Open/Closed Sets**: Efficient node management
- **Path Reconstruction**: Backtracking from goal to start
- **Obstacle Handling**: Dynamic wall detection

##### Multi-Agent Coordination
- **Distributed Planning**: Each robot plans independently
- **Shared Map**: Common exploration state
- **Non-blocking Updates**: Concurrent robot operations
- **Resource Allocation**: Automatic task distribution

#### Usage

```python
python multi_robot_exploration.py
```

The simulation will:
1. Display initial map with "Press 'q' to begin" prompt
2. Start exploration animation
3. Show real-time robot movements and mapping progress
4. Report completion statistics

#### Customization Options

```python
# Adjust robot count
num_bots = 3  # Reduce for simpler scenarios

# Modify environment size
height = 30
width = 30

# Change simulation parameters
max_itr = 1000  # Reduce for faster completion
```

#### Algorithm Complexity

- **Time Complexity**: O(n × m × log(nm)) per A* call
- **Space Complexity**: O(nm) for map storage
- **Scalability**: Efficient for maps up to 100x100

#### Applications

- **Robotic Mapping**: Real-world robot exploration
- **Search and Rescue**: Emergency response scenarios
- **Environmental Monitoring**: Autonomous survey missions
- **Game AI**: Strategy game pathfinding
- **Research**: Multi-agent systems study

## Usage Examples

### Basic Usage

```bash
# Run individual scripts
python image_classification.py
python image_noise_remover.py
python histogram.py
python edge_detection.py
```

### Customization

#### Image Classification
- Modify `epochs` and `batch_size` for different training configurations
- Adjust model architecture by changing layer parameters
- Change `data_dir` path for different datasets

#### Noise Removal
- Experiment with different kernel sizes
- Try various template matching methods
- Adjust median filter window sizes

#### Histogram Analysis
- Change bin count in histogram functions
- Modify AHE window size for different enhancement levels
- Experiment with different color channel manipulations

#### Edge Detection
- Adjust threshold percentage (currently 14.5%)
- Modify smoothing kernel for different noise levels
- Experiment with different gradient operators

## File Structure

```
project/
├── image_classification.py
├── image_noise_remover.py
├── histogram.py
├── edge_detection.py
├── GTSRB/
│   └── Training/
│       ├── 00000/
│       ├── 00001/
│       └── ...
├── images/
│   ├── forest.jpg
│   ├── lane.png
│   ├── beach.png
│   ├── pic.jpeg
│   └── ...
└── README.md
```

## Notes and Considerations

- **Memory Usage**: The CNN model requires significant memory for training
- **Processing Time**: Median filtering with large kernels (81x81) is computationally intensive
- **Image Quality**: Results depend heavily on input image quality and appropriate parameter tuning
- **Dataset Size**: Traffic sign classification uses only 100 samples per class for demonstration

## Troubleshooting

1. **Import Errors**: Ensure all required packages are installed
2. **File Not Found**: Check image paths and ensure all required images are present
3. **Memory Issues**: Reduce batch size or image resolution for classification
4. **Slow Performance**: Consider using OpenCV's built-in functions for production use

## Future Improvements

- Add data augmentation for better classification performance
- Implement GPU acceleration for faster training
- Add more sophisticated edge detection algorithms
- Include additional noise reduction methods
- Implement real-time processing capabilities
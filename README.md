# RGB-D Pose Estimation with OpenMP - Quick Start Guide

## Prerequisites

1. **Visual Studio 2019/2022** with C++ development tools
2. **CMake** (>= 3.15)
3. **Ninja** build system
4. **Python 3.8+**
5. **OpenCV 4.x** and **Eigen3** (via vcpkg)

### Install C++ Dependencies

```powershell
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install libraries
.\vcpkg install opencv[core,imgproc,imgcodecs]:x64-windows
.\vcpkg install eigen3:x64-windows
```

### Setup Python Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate.bat

# Install Python packages
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Automated (Recommended)

Run the complete pipeline with one command:

```powershell
.\run_berkeley_analysis.bat
```

This will:

1. Build the C++ project
2. Convert Berkeley dataset (7 frames)
3. Run 140 benchmarks (1, 2, 4, 8 threads × 5 runs each)
4. Generate all plots

**Duration**: 10-15 minutes

---

### Option 2: Manual Steps

#### Step 1: Build

```powershell
mkdir build
cd build
cmake -G Ninja ..
ninja
cd ..
```

#### Step 2: Convert Dataset

```powershell
.venv\Scripts\activate.bat
python scripts\convert_berkeley_data.py --input "data\004_sugar_box_berkeley_rgbd\004_sugar_box" --output "data\berkeley_diverse"
```

#### Step 3: Run Benchmarks

```powershell
python scripts\generate_per_frame_analysis.py
```

---

## Output

Results are saved in `output\berkeley_frames\{frame_name}\`:

- `data/benchmark_results.json` - Performance metrics
- `images/histogram_equalization.png` - Image preprocessing
- `images/performance_analysis.png` - Speedup charts
- `images/convergence_comparison.png` - Algorithm convergence

---

## Troubleshooting

**CMake can't find OpenCV:**

```powershell
set OpenCV_DIR=C:\path\to\opencv\build
```

**Python packages not found:**

```powershell
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

@echo off
REM ============================================================================
REM RGB-D Pose Estimation - Berkeley Dataset Complete Analysis Workflow
REM ============================================================================
REM This script runs the complete pipeline for Berkeley RGB-D dataset analysis:
REM 1. Build the C++ project with OpenMP
REM 2. Convert Berkeley H5 data to PNG/OBJ format
REM 3. Run comprehensive benchmarks on all frames
REM 4. Generate performance analysis plots
REM ============================================================================

echo.
echo ============================================================================
echo   RGB-D Pose Estimation - Berkeley Dataset Analysis
echo ============================================================================
echo.

REM Check if Python virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then run: .venv\Scripts\activate.bat
    echo Then run: pip install matplotlib h5py numpy opencv-python pandas
    pause
    exit /b 1
)

REM Activate Python virtual environment
echo [1/4] Activating Python virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Step 1: Build the C++ project
echo ============================================================================
echo [2/4] Building C++ Project with OpenMP...
echo ============================================================================
if not exist "build" mkdir build
cd build
cmake -G Ninja ..
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    cd ..
    pause
    exit /b 1
)

ninja
if errorlevel 1 (
    echo [ERROR] Build failed
    cd ..
    pause
    exit /b 1
)
cd ..
echo [OK] Build completed successfully
echo.

REM Step 2: Convert Berkeley dataset
echo ============================================================================
echo [3/4] Converting Berkeley RGB-D Dataset (H5 to PNG/OBJ)...
echo ============================================================================
echo This will process 7 diverse frames (4 side views + 3 top-down views)
echo.
python scripts\convert_berkeley_data.py --input "data\004_sugar_box_berkeley_rgbd\004_sugar_box" --output "data\berkeley_diverse"
if errorlevel 1 (
    echo [ERROR] Data conversion failed
    pause
    exit /b 1
)
echo [OK] Dataset conversion completed
echo.

REM Step 3: Run comprehensive benchmarks
echo ============================================================================
echo [4/4] Running Comprehensive Benchmarks...
echo ============================================================================
echo This will run 140 benchmarks (7 frames x 4 thread counts x 5 runs each)
echo Estimated time: 10-15 minutes
echo.
echo Press Ctrl+C to cancel, or
pause

python scripts\generate_per_frame_analysis.py
if errorlevel 1 (
    echo [ERROR] Benchmark analysis failed
    pause
    exit /b 1
)
echo [OK] Benchmarks and plots generated successfully
echo.

REM Summary
echo ============================================================================
echo   ANALYSIS COMPLETE!
echo ============================================================================
echo.
echo Output Location: output\berkeley_frames\
echo.
echo Generated for each frame (7 frames total):
echo   - benchmark_results.json       : Performance metrics
echo   - convergence_*.csv            : SA convergence data (20 files per frame)
echo   - histogram_equalization.png   : Image preprocessing visualization
echo   - performance_analysis.png     : Speedup and efficiency charts
echo   - convergence_comparison.png   : Algorithm convergence across threads
echo.
echo To view results:
echo   explorer output\berkeley_frames
echo.
echo ============================================================================
pause

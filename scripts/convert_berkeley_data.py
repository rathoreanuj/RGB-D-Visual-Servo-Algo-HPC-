"""
Convert Berkeley RGB-D Dataset to Format Compatible with RGB-D Pose Estimation

This script:
1. Loads depth data from HDF5 files
2. Generates RGB images from depth (colormap)
3. Creates edge-based wireframe model using Canny edge detection
4. Saves in format compatible with the C++ pose estimation algorithm
"""

import h5py
import numpy as np
import cv2
import json
import argparse
from pathlib import Path

def load_depth_from_h5(h5_path):
    """Load depth map from Berkeley RGB-D H5 file"""
    with h5py.File(h5_path, 'r') as f:
        depth = np.array(f['depth'])
    return depth

def depth_to_rgb(depth, min_depth=0, max_depth=None):
    """Convert depth map to colorized RGB image using COLORMAP_JET"""
    if max_depth is None:
        max_depth = depth[depth > 0].max() if np.any(depth > 0) else 1
    
    # Normalize to 0-255
    depth_normalized = np.zeros_like(depth, dtype=np.uint8)
    valid_mask = depth > min_depth
    if np.any(valid_mask):
        depth_normalized[valid_mask] = cv2.normalize(
            depth[valid_mask], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        ).flatten()
    
    # Apply colormap
    rgb = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # Set invalid regions to black
    rgb[~valid_mask] = [0, 0, 0]
    
    return rgb

def extract_edges_canny(image, low_threshold=50, high_threshold=150):
    """Extract edges using Canny edge detector"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges

def create_wireframe_from_edges(edges, depth, camera_K, subsample=5):
    """
    Create 3D wireframe model from edge pixels using depth information
    
    Args:
        edges: Binary edge map (H x W)
        depth: Depth map in mm (H x W)
        camera_K: Camera intrinsic matrix (3x3)
        subsample: Subsample edges to reduce point count
    
    Returns:
        vertices: List of 3D points
        edges: List of edge connections (simplified as sequential connections)
    """
    # Find edge pixels
    edge_coords = np.argwhere(edges > 0)
    
    # Subsample to reduce point count
    if subsample > 1:
        edge_coords = edge_coords[::subsample]
    
    vertices = []
    fx, fy = camera_K[0, 0], camera_K[1, 1]
    cx, cy = camera_K[0, 2], camera_K[1, 2]
    
    # Convert 2D edge pixels to 3D using depth
    for y, x in edge_coords:
        z = depth[y, x]
        if z > 0:  # Valid depth
            # Backproject to 3D (convert mm to meters)
            X = (x - cx) * z / fx / 1000.0
            Y = (y - cy) * z / fy / 1000.0
            Z = z / 1000.0
            vertices.append([X, Y, Z])
    
    # Create edges as sequential connections (simplified wireframe)
    # In real scenario, you'd use more sophisticated edge connectivity
    edge_list = []
    for i in range(len(vertices) - 1):
        edge_list.append([i, i + 1])
    
    # Close the loop
    if len(vertices) > 0:
        edge_list.append([len(vertices) - 1, 0])
    
    return vertices, edge_list

def save_as_obj(vertices, edges, output_path):
    """Save wireframe model as OBJ file"""
    with open(output_path, 'w') as f:
        f.write("# Wireframe model from Berkeley RGB-D dataset\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Edges: {len(edges)}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Write edges as lines (OBJ line element)
        for e in edges:
            # OBJ indices are 1-based
            f.write(f"l {e[0]+1} {e[1]+1}\n")

def process_berkeley_frame(h5_path, output_dir, camera_K, 
                           edge_low=50, edge_high=150, subsample=5):
    """Process a single Berkeley RGB-D frame"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_name = Path(h5_path).stem
    
    # Load depth
    depth = load_depth_from_h5(h5_path)
    
    # Generate RGB from depth
    rgb = depth_to_rgb(depth)
    
    # Save RGB
    rgb_path = output_dir / f"{frame_name}_rgb.png"
    cv2.imwrite(str(rgb_path), rgb)
    
    # Save depth as 16-bit PNG
    depth_path = output_dir / f"{frame_name}_depth.png"
    cv2.imwrite(str(depth_path), depth.astype(np.uint16))
    
    # Extract edges
    edges = extract_edges_canny(rgb, edge_low, edge_high)
    
    # Save edge visualization
    edge_vis_path = output_dir / f"{frame_name}_edges.png"
    cv2.imwrite(str(edge_vis_path), edges)
    
    # Create wireframe from edges
    vertices, edge_list = create_wireframe_from_edges(edges, depth, camera_K, subsample)
    
    # Save wireframe as OBJ
    obj_path = output_dir / f"{frame_name}_wireframe.obj"
    save_as_obj(vertices, edge_list, obj_path)
    
    # Save metadata
    metadata = {
        'frame_name': frame_name,
        'rgb_image': str(rgb_path.name),
        'depth_map': str(depth_path.name),
        'wireframe': str(obj_path.name),
        'num_vertices': len(vertices),
        'num_edges': len(edge_list),
        'depth_stats': {
            'min': int(depth[depth > 0].min()) if np.any(depth > 0) else 0,
            'max': int(depth.max()),
            'mean': float(depth[depth > 0].mean()) if np.any(depth > 0) else 0,
        },
        'edge_pixels': int(edges.sum() / 255)
    }
    
    metadata_path = output_dir / f"{frame_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processed {frame_name}:")
    print(f"  - RGB: {rgb_path.name}")
    print(f"  - Depth: {depth_path.name}")
    print(f"  - Wireframe: {obj_path.name} ({len(vertices)} vertices, {len(edge_list)} edges)")
    print(f"  - Edge pixels: {metadata['edge_pixels']}")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Convert Berkeley RGB-D data to pose estimation format')
    parser.add_argument('--input', type=str, 
                       default='004_sugar_box_berkeley_rgbd/004_sugar_box',
                       help='Input directory with H5 files')
    parser.add_argument('--output', type=str, 
                       default='data/berkeley_real',
                       help='Output directory')
    parser.add_argument('--frames', type=int, nargs='+',
                       default=[0, 50, 100, 150, 200],
                       help='Frame indices to process (e.g., 0 50 100)')
    parser.add_argument('--edge-low', type=int, default=50,
                       help='Canny lower threshold')
    parser.add_argument('--edge-high', type=int, default=150,
                       help='Canny upper threshold')
    parser.add_argument('--subsample', type=int, default=5,
                       help='Subsample factor for edge points')
    
    args = parser.parse_args()
    
    # Camera intrinsics from calibration.h5
    # Using NP5_depth_K (adjust based on your camera)
    camera_K = np.array([
        [574.06408607, 0.0, 314.66898338],
        [0.0, 574.05978096, 220.79854667],
        [0.0, 0.0, 1.0]
    ])
    
    input_dir = Path(args.input)
    
    # Find both NP1 (side views) and NP2 (top-down views)
    np1_files = sorted(input_dir.glob('NP1_*.h5'))
    np2_files = sorted(input_dir.glob('NP2_*.h5'))
    
    print(f"Found {len(np1_files)} NP1 files (side views)")
    print(f"Found {len(np2_files)} NP2 files (top-down views)")
    
    # Select diverse frames: mix of side views (NP1) and top-down views (NP2)
    diverse_frames = [
        ('NP1', 0),    # Side view - 0°
        ('NP1', 90),   # Side view - 90°
        ('NP1', 180),  # Side view - 180°
        ('NP1', 270),  # Side view - 270°
        ('NP2', 0),    # Top-down view - 0°
        ('NP2', 90),   # Top-down view - 90°
        ('NP2', 180),  # Top-down view - 180°
    ]
    
    # Build list of H5 files to process
    h5_files = []
    for series, angle in diverse_frames:
        pattern = f'{series}_{angle}.h5'
        matched_files = list(input_dir.glob(pattern))
        if matched_files:
            h5_files.append(matched_files[0])
            print(f"  ✓ {series}_{angle} - {'Top-down' if series == 'NP2' else 'Side'} view")
        else:
            print(f"  ✗ {pattern} not found")
    
    if not h5_files:
        print(f"No H5 files found in {input_dir}")
        return
    
    print(f"\nProcessing {len(h5_files)} diverse frames\n")
    
    all_metadata = []
    
    for h5_path in h5_files:
        metadata = process_berkeley_frame(
            h5_path, args.output, camera_K,
            args.edge_low, args.edge_high, args.subsample
        )
        all_metadata.append(metadata)
        print()
    
    # Save summary
    summary_path = Path(args.output) / 'dataset_summary.json'
    summary = {
        'dataset': 'Berkeley RGB-D (004_sugar_box)',
        'num_frames': len(all_metadata),
        'frames': all_metadata,
        'camera_intrinsics': camera_K.tolist(),
        'processing_params': {
            'edge_detection': {
                'low_threshold': args.edge_low,
                'high_threshold': args.edge_high
            },
            'subsample': args.subsample
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Dataset conversion complete!")
    print(f"Output directory: {args.output}")
    print(f"Processed {len(all_metadata)} frames")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

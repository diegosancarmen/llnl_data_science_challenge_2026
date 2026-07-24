import argparse
import napari
import tifffile
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Step 2: Load CT and Centerlines")
    parser.add_argument('--scan', type=str, required=True, help='Path to the 3D TIFF scan')
    parser.add_argument('--centerlines', type=str, required=True, help='Path to napari_centerlines.csv')
    
    args = parser.parse_args()

    print(f"Loading TIFF from {args.scan} (memory-mapped)...")
    volume = tifffile.memmap(args.scan)

    print("Downsampling volume 2x for rendering...")
    volume_2x = volume[::2, ::2, ::2]

    print(f"Loading centerlines from {args.centerlines}...")
    table = pd.read_csv(args.centerlines)

    print("Initializing Napari viewer...")
    viewer = napari.Viewer(ndisplay=3)

    # 1. Load the CT image (scaled 2x to match the 1x voxel world space)
    viewer.add_image(
        volume_2x, 
        name='CT Scan (2x Downsampled)', 
        colormap='gray', 
        rendering='mip', 
        depiction='volume',
        scale=(2, 2, 2) 
    )

    # 2. Extract origin and direction vectors directly from the CSV
    print("Overlaying strut centerlines...")
    origin = table[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy()
    direction = table[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy()
    vectors = np.stack((origin, direction), axis=1)  # Format required by Napari: (N, 2, 3)

    # 3. Add the vectors layer, color-coded and packed with metadata
    viewer.add_vectors(
        vectors,
        name="Strut Centerlines",
        edge_color=table["napari_color_hex"].tolist(),
        edge_width=2, 
        properties=table, # Attaches all inventory metadata to the vectors
    )

    print("Launching window...")
    napari.run()

if __name__ == "__main__":
    main()
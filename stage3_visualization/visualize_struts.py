import argparse
import napari
import tifffile

def main():
    parser = argparse.ArgumentParser(description="Step 1: Load CT Scan")
    parser.add_argument('--scan', type=str, required=True, help='Path to the 3D TIFF scan')
    parser.add_argument('--registration', type=str, required=True, help='Path to the JSON')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV')
    
    args = parser.parse_args()

    print(f"Loading TIFF from {args.scan} (memory-mapped)...")
    volume = tifffile.memmap(args.scan)

    # Downsample by 2 in all dimensions to save RAM
    print("Downsampling volume 2x for rendering...")
    volume_2x = volume[::2, ::2, ::2]

    print("Initializing Napari viewer...")
    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(
        volume_2x, 
        name='CT Scan (2x Downsampled)', 
        colormap='gray', 
        rendering='mip', 
        depiction='volume',
        scale=(2, 2, 2) # Tells Napari these voxels are twice as large, keeping coordinates aligned
    )

    print("Launching window...")
    napari.run()

if __name__ == "__main__":
    main()
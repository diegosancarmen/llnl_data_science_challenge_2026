from fastmcp import FastMCP
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")   # render without needing a display window
import matplotlib.pyplot as plt
from skeletonization import skeletonize_mask

# Small program to demonstrate how to use FastMCP to create a server that exposes functions for CT segmentation and visualization.

# Initialize the MCP server
mcp = FastMCP("CT Segmentation") # creates server object and names it

@mcp.tool() # decorator to register the function as a tool in the MCP server
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    if not os.path.exists(input_filepath):
        return f"Error: Input file '{input_filepath}' does not exist."
    data = np.load(input_filepath)
    mask = (data >= threshold).astype(np.uint8)  # Create binary mask
    np.save(output_filepath, mask)
    return f"Segmentation completed. Mask saved to '{output_filepath}'."

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy file and saves a visualization of a specific slice to an image file.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    # data is a 256 x 256 x 256 3D array and we want to visualize a slice along the specified axis
    if not os.path.exists(input_filepath):
        return f"Error: Input file '{input_filepath}' does not exist."
    data = np.load(input_filepath)
    if axis not in (0, 1, 2):
        return f"Error: Invalid axis '{axis}'. Must be 0, 1, or 2, got {axis}."
    if not(0 <= slice_index < data.shape[axis]):
        return f"Error: Slice index '{slice_index}' is out of bounds for axis {axis} with size {data.shape[axis]}."
    slice_2d = np.take(data, slice_index, axis=axis)
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d, cmap='gray')
    plt.colorbar(label="density")
    plt.title(f"Slice {slice_index} along axis {axis}")
    plt.savefig(output_filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return f"Slice visualization saved to '{output_filepath}'."
    


@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    if not os.path.exists(input_filepath):
        return f"Error: File not found at {input_filepath}"
    try:
        skeleton = skeletonize_mask(file_path=input_filepath, output_path=output_filepath)
        if skeleton is None:
            return f"Error: skeletonization failed for {input_filepath}"
        return (f"Saved skeleton to {output_filepath}. "
                f"Skeleton voxels: {int(np.count_nonzero(skeleton))}")
    except Exception as e:
        return f"Error during skeletonization: {e}"

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
from fastmcp import FastMCP
import numpy as np
import matplotlib.pyplot as plt
import skeletonization
mcp = FastMCP("CT Segmentation")

@mcp.tool()
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
    try:
        arr = np.load(input_filepath)
        result = np.where(arr > threshold, arr, 0)
        
        np.save(output_filepath, result)
        return f"success: filename: {output_filepath}"
    except Exception as e:
        return f"segmentation failed: {input_filepath}"
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
    try:
        arr = np.load(input_filepath)
        if axis == 0:
            arr2d = arr[slice_index, :, :]

        elif axis == 1:
            arr2d = arr[:,slice_index, :]
        elif axis == 2:
            arr2d = arr[:, :, slice_index]
        else:
            raise Exception
        
        plt.figure(figsize=(6, 6))
        plt.imshow(arr2d, cmap='gray')
        plt.axis('off')
        plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    
    except:
        return f"Failed to visualize slice: {input_filepath}"

    return f"Success output file: {input_filepath}"

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
    
    try:
        skeletonization.skeletonize_mask(input_filepath, output_filepath)

    except Exception as e:
        return f"failed to skeletonize {input_filepath}"

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()

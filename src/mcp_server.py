from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from skeletonization import skeletonize_mask

from fastmcp import FastMCP

# Initialize the MCP server
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
        input_path = Path(input_filepath)
        output_path = Path(output_filepath)

        data = np.load(input_path, allow_pickle=False)
        if data.ndim != 3:
            raise ValueError(
                f"expected a 3D CT dataset, but received an array with shape {data.shape}"
            )

        # Use uint8 so the saved mask contains only the values 0 and 1.
        segmented = (data >= threshold).astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, segmented)
        return f"Segmentation completed successfully. Saved to {output_path}"
    except (OSError, TypeError, ValueError) as error:
        return f"Error segmenting CT dataset: {error}"

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
        input_path = Path(input_filepath)
        output_path = Path(output_filepath)

        data = np.load(input_path, allow_pickle=False)
        if data.ndim != 3:
            raise ValueError(
                f"expected a 3D CT dataset, but received an array with shape {data.shape}"
            )
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, but received {axis}")
        if not 0 <= slice_index < data.shape[axis]:
            raise ValueError(
                f"slice_index must be between 0 and {data.shape[axis] - 1} "
                f"for axis {axis}, but received {slice_index}"
            )

        # Select one 2D plane along the requested axis.
        image = np.take(data, slice_index, axis=axis)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        try:
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
        finally:
            plt.close(fig)

        return f"Slice visualization completed successfully. Saved to {output_path}"
    except (OSError, TypeError, ValueError) as error:
        return f"Error visualizing CT slice: {error}"

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
        skeletonize_mask(
            file_path=input_filepath, 
            output_path=output_filepath
        )

        return f"Skeletonization completed successfully. Saved to {output_filepath}"
    except (OSError, TypeError, ValueError) as error:
        return f"Error skeletonizing mask: {error}"

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()

import argparse
import napari
import tifffile
import pandas as pd
import numpy as np
from magicgui.widgets import Container, Label, ComboBox, TextEdit

def main():
    parser = argparse.ArgumentParser(description="Step 3: Interactive UI")
    parser.add_argument('--scan', type=str, required=True, help='Path to the 3D TIFF scan')
    parser.add_argument('--centerlines', type=str, required=True, help='Path to napari_centerlines.csv')
    
    args = parser.parse_args()

    print(f"Loading TIFF from {args.scan} (memory-mapped)...")
    volume = tifffile.memmap(args.scan)
    volume_2x = volume[::2, ::2, ::2]

    print(f"Loading centerlines from {args.centerlines}...")
    table = pd.read_csv(args.centerlines)

    # Fallback in case the ID column is named slightly differently in your CSV
    id_col = 'inventory_strut_ID' if 'inventory_strut_ID' in table.columns else table.columns[0]

    print("Initializing Napari viewer...")
    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(
        volume_2x, 
        name='CT Scan (2x)', 
        colormap='gray', 
        rendering='mip', 
        depiction='volume',
        scale=(2, 2, 2) 
    )

    origin = table[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy()
    direction = table[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy()
    vectors = np.stack((origin, direction), axis=1)

    vectors_layer = viewer.add_vectors(
        vectors,
        name="Strut Centerlines",
        edge_color=table["napari_color_hex"].tolist(),
        edge_width=2, 
        properties=table, 
    )

    # ==========================================
    # 4. BUILD THE INTERACTIVE UI
    # ==========================================
    
    id_col = 'strut_id'
    strut_choices = table[id_col].tolist()
    dropdown = ComboBox(choices=strut_choices, label="Select Strut:")
    metadata_display = TextEdit(label="Metadata:", value="Select a strut...")
    
    def update_ui(strut_id):
        # Find the row index for this strut
        idx = table.index[table[id_col] == strut_id].tolist()[0]
        row = table.iloc[idx]
        
        # Format the specific columns we now know exist!
        info = (
            f"--- STRUT {row['strut_id']} ---\n"
            f"Classification: {row.get('inventory_classification', 'N/A')}\n"
            f"Material Occupancy: {row.get('inventory_ct_material_occupancy', 0):.2f}\n"
            f"Length (µm): {row.get('length_um_from_centerline', 0):.2f}\n"
            f"Missing Vol (mm³): {row.get('inventory_estimated_missing_volume_mm3', 0):.6f}"
        )
        metadata_display.value = info
        
        # Highlight the selected vector natively
        vectors_layer.selected_data = {idx}
        
        # --- CAMERA FLY-TO LOGIC ---
        # Move the camera center to the exact middle of the selected strut
        viewer.camera.center = (row['center_z_vox'], row['center_y_vox'], row['center_x_vox'])
        # Zoom in (default is usually ~0.5 to 1. Higher numbers = closer)
        viewer.camera.zoom = 5

    dropdown.changed.connect(update_ui)

    # 5. Two-Way Sync: Shift + Click 3D Vector -> UI
    @vectors_layer.mouse_drag_callbacks.append
    def on_3d_click(layer, event):
        if 'Shift' not in event.modifiers:
            return
            
        clicked_index = layer.get_value(
            event.position, 
            world=True, 
            view_direction=event.view_direction, 
            dims_displayed=event.dims_displayed
        )
        
        if clicked_index is not None and isinstance(clicked_index, int):
            clicked_strut_id = table.iloc[clicked_index][id_col]
            dropdown.value = clicked_strut_id # Triggers update_ui and camera fly-to!

    ui_container = Container(widgets=[dropdown, metadata_display])
    viewer.window.add_dock_widget(ui_container, name="Inspector", area="right")

    print("Launching window...")
    napari.run()

if __name__ == "__main__":
    main()
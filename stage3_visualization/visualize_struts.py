import argparse
import napari
import tifffile
import pandas as pd
import numpy as np
from magicgui.widgets import Container, Label, ComboBox, TextEdit, RadioButtons

def main():
    parser = argparse.ArgumentParser(description="Interactive Strut Visualizer")
    parser.add_argument('--scan', type=str, required=True, help='Path to the 3D TIFF scan')
    parser.add_argument('--centerlines', type=str, required=True, help='Path to napari_centerlines.csv')
    
    args = parser.parse_args()

    print(f"Loading TIFF from {args.scan} (memory-mapped)...")
    volume = tifffile.memmap(args.scan)
    
    print("Downsampling volume 2x for full-lattice rendering...")
    volume_2x = volume[::2, ::2, ::2]

    print(f"Loading centerlines from {args.centerlines}...")
    table = pd.read_csv(args.centerlines)
    id_col = 'strut_id'

    print("Initializing Napari viewer...")
    viewer = napari.Viewer(ndisplay=3)

    # ==========================================
    # 1. SETUP FULL LATTICE LAYERS (2x)
    # ==========================================
    full_ct_layer = viewer.add_image(
        volume_2x, 
        name='Full CT Scan (2x)', 
        colormap='gray', 
        rendering='mip', 
        depiction='volume',
        scale=(2, 2, 2) 
    )

    origin = table[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy()
    direction = table[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy()
    vectors = np.stack((origin, direction), axis=1)

    full_vectors_layer = viewer.add_vectors(
        vectors,
        name="All Centerlines",
        edge_color=table["napari_color_hex"].tolist(),
        edge_width=2, 
        properties=table, 
    )

    # ==========================================
    # 2. SETUP ISOLATED LAYERS (1x, Initially Empty & Hidden)
    # ==========================================
    isolated_ct_layer = viewer.add_image(
        np.zeros((1, 1, 1)), 
        name='Isolated 3D CT (1x)', 
        colormap='gray', 
        rendering='mip', 
        depiction='volume',
        visible=False
    )
    
    isolated_slice_layer = viewer.add_image(
        np.zeros((1, 1, 1)), 
        name='Isolated 2D Slice', 
        colormap='gray',
        visible=False
    )
    
    isolated_vector_layer = viewer.add_vectors(
        np.zeros((1, 2, 3)), 
        name="Isolated Centerline",
        edge_width=10, 
        visible=False
    )

    # ==========================================
    # 3. BUILD THE INTERACTIVE UI
    # ==========================================
    strut_choices = table[id_col].tolist()
    dropdown = ComboBox(choices=strut_choices, label="Select Strut:")
    
    view_mode = RadioButtons(
        choices=["Full Lattice", "Isolated 3D", "Isolated 2D Slice"],
        value="Full Lattice",
        label="View Mode:"
    )
    
    metadata_display = TextEdit(label="Metadata:", value="Select a strut...")
    
    def update_ui(*args):
        strut_id = dropdown.value
        mode = view_mode.value
        
        idx = table.index[table[id_col] == strut_id].tolist()[0]
        row = table.iloc[idx]
        
        # --- Update Metadata Text ---
        info = (
            f"--- STRUT {row['strut_id']} ---\n"
            f"Classification: {row.get('inventory_classification', 'N/A')}\n"
            f"Material Occupancy: {row.get('inventory_ct_material_occupancy', 0):.2f}\n"
            f"Length (µm): {row.get('length_um_from_centerline', 0):.2f}\n"
            f"Missing Vol (mm³): {row.get('inventory_estimated_missing_volume_mm3', 0):.6f}"
        )
        metadata_display.value = info
        
        # --- Extract 1x High-Res Data from Disk ---
        z_min, z_max = int(row['bbox_z_min_inclusive']), int(row['bbox_z_max_exclusive'])
        y_min, y_max = int(row['bbox_y_min_inclusive']), int(row['bbox_y_max_exclusive'])
        x_min, x_max = int(row['bbox_x_min_inclusive']), int(row['bbox_x_max_exclusive'])
        center_z = int(row['center_z_vox'])
        
        # Update 3D Crop (shift it into place using translate)
        isolated_ct_layer.data = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        isolated_ct_layer.translate = (z_min, y_min, x_min)
        
        # Update 2D Slice (extract a 1-voxel thick Z-slice)
        isolated_slice_layer.data = volume[center_z:center_z+1, y_min:y_max, x_min:x_max]
        isolated_slice_layer.translate = (center_z, y_min, x_min)
        
        # Update Isolated Vector
        v_orig = [row['start_z_vox'], row['start_y_vox'], row['start_x_vox']]
        v_dir = [row['direction_z_vox'], row['direction_y_vox'], row['direction_x_vox']]
        isolated_vector_layer.data = np.array([[v_orig, v_dir]])
        isolated_vector_layer.edge_color = [row['napari_color_hex']]
        
        # --- Handle Visibility Toggles ---
        if mode == "Full Lattice":
            full_ct_layer.visible = True
            full_vectors_layer.visible = True
            isolated_ct_layer.visible = False
            isolated_slice_layer.visible = False
            isolated_vector_layer.visible = False
            viewer.camera.zoom = 5
        elif mode == "Isolated 3D":
            full_ct_layer.visible = False
            full_vectors_layer.visible = False
            isolated_ct_layer.visible = True
            isolated_slice_layer.visible = False
            isolated_vector_layer.visible = True
            viewer.camera.zoom = 15
        elif mode == "Isolated 2D Slice":
            full_ct_layer.visible = False
            full_vectors_layer.visible = False
            isolated_ct_layer.visible = False
            isolated_slice_layer.visible = True
            isolated_vector_layer.visible = True
            viewer.camera.zoom = 15
            
        # Highlight in full view just in case we swap back
        full_vectors_layer.selected_data = {idx}
        
        # --- Fly Camera to Center ---
        viewer.camera.center = (row['center_z_vox'], row['center_y_vox'], row['center_x_vox'])

    # Connect both widgets to trigger the same update function
    dropdown.changed.connect(update_ui)
    view_mode.changed.connect(update_ui)

    # 4. Two-Way Sync: Shift + Click 3D Vector -> UI
    @full_vectors_layer.mouse_drag_callbacks.append
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
            dropdown.value = clicked_strut_id 

    ui_container = Container(widgets=[dropdown, view_mode, metadata_display])
    viewer.window.add_dock_widget(ui_container, name="Inspector", area="right")

    print("Launching window...")
    napari.run()

if __name__ == "__main__":
    main()
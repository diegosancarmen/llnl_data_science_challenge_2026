import argparse
import napari
import tifffile
import pandas as pd
import numpy as np
from magicgui.widgets import Container, ComboBox, TextEdit

def main():
    parser = argparse.ArgumentParser(description="Dual-Window Strut Visualizer")
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

    # ==========================================
    # 0. DISTINCT COLOR MAPPING (Fixed)
    # ==========================================
    def get_color(cls_val):
        if pd.isna(cls_val): return 'magenta'
        val = str(cls_val).lower()
        if 'missing' in val: return 'red'
        if 'partial' in val: return 'orange'
        if 'nominal' in val: return 'cyan'
        return 'yellow' # Fallback
        
    # Inject the exact categorical color into the dataframe as a new column
    table['custom_display_color'] = table['inventory_classification'].apply(get_color)

    print("Initializing Dual Napari Viewers...")
    viewer_full = napari.Viewer(title="Full Lattice (Macro)", ndisplay=3)
    viewer_iso = napari.Viewer(title="Isolated Strut (Micro)", ndisplay=3)

    # ==========================================
    # 1. SETUP FULL LATTICE WINDOW
    # ==========================================
    viewer_full.add_image(
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

    full_vectors_layer = viewer_full.add_vectors(
        vectors,
        name="All Centerlines",
        properties=table, 
        edge_color='custom_display_color', # Tell Napari to use our injected column!
        edge_width=2, 
    )

    # ==========================================
    # 2. SETUP ISOLATED WINDOW
    # ==========================================
    isolated_ct_layer = viewer_iso.add_image(
        np.zeros((1, 1, 1)), 
        name='Isolated 3D CT', 
        colormap='gray', 
        rendering='mip', 
        depiction='volume'
    )
    
    isolated_vector_layer = viewer_iso.add_vectors(
        np.zeros((1, 2, 3)), 
        name="Isolated Centerline",
        edge_width=4
    )

    # ==========================================
    # 3. BUILD THE INTERACTIVE UI
    # ==========================================
    strut_choices = table[id_col].tolist()
    dropdown = ComboBox(choices=strut_choices, label="Select Strut:")
    metadata_display = TextEdit(label="Metadata:", value="Select a strut...")
    
    def update_ui(*args):
        strut_id = dropdown.value
        
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
        
        # --- Update ISOLATED Window ---
        z_min, z_max = int(row['bbox_z_min_inclusive']), int(row['bbox_z_max_exclusive'])
        y_min, y_max = int(row['bbox_y_min_inclusive']), int(row['bbox_y_max_exclusive'])
        x_min, x_max = int(row['bbox_x_min_inclusive']), int(row['bbox_x_max_exclusive'])
        
        # Extract the crop and calculate its true min/max values
        crop_data = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        isolated_ct_layer.data = crop_data
        isolated_ct_layer.translate = (z_min, y_min, x_min)
        
        # FIX: Force Napari to recalculate contrast limits for this specific crop
        c_min, c_max = float(np.min(crop_data)), float(np.max(crop_data))
        if c_min < c_max:
            isolated_ct_layer.contrast_limits = (c_min, c_max)
        
        v_orig = [row['start_z_vox'], row['start_y_vox'], row['start_x_vox']]
        v_dir = [row['direction_z_vox'], row['direction_y_vox'], row['direction_x_vox']]
        isolated_vector_layer.data = np.array([[v_orig, v_dir]])
        isolated_vector_layer.edge_color = row['custom_display_color']
        
        # --- Update Cameras ---
        center_pt = (row['center_z_vox'], row['center_y_vox'], row['center_x_vox'])
        
        viewer_full.camera.center = center_pt
        viewer_iso.camera.center = center_pt
        viewer_iso.camera.zoom = 10
        
        full_vectors_layer.selected_data = {idx}

    dropdown.changed.connect(update_ui)

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

    ui_container = Container(widgets=[dropdown, metadata_display])
    viewer_full.window.add_dock_widget(ui_container, name="Inspector", area="right")

    print("Launching windows...")
    napari.run()

if __name__ == "__main__":
    main()
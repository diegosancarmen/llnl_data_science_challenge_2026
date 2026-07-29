"""Focused tests for renderer-independent PyVista scene helpers."""

import numpy as np

from part2.napari_visualizer.scene_data import (
    TRANSPARENT_RGBA,
    classification_color,
    image_grid,
    make_line_mesh,
    parse_strut_ids,
    placeholder_line_mesh,
    unit_cell_context_color,
)


def test_parse_strut_ids_preserves_first_occurrence_order():
    assert parse_strut_ids("12, 7 12 8") == ["12", "7", "8"]


def test_parse_strut_ids_handles_empty_input_for_web_selection():
    assert parse_strut_ids("") == []
    assert parse_strut_ids("   ,  ") == []


def test_classification_colors_match_existing_visualizer_contract():
    assert classification_color("Missing_Intentional") == "red"
    assert classification_color("Partial") == "orange"
    assert classification_color("Nominal") == "cyan"
    assert unit_cell_context_color("Nominal") == "blue"
    assert unit_cell_context_color("Missing") == "red"
    assert TRANSPARENT_RGBA == (0, 0, 0, 0)


def test_line_mesh_has_one_line_cell_per_strut_and_retains_colors():
    mesh = make_line_mesh(
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=float),
        ["red", "blue"],
    )
    assert mesh.n_cells == 2
    assert mesh.n_points == 4
    assert mesh.cell_data["rgba"].shape == (2, 4)
    assert mesh.points[0].tolist() == [3.0, 2.0, 1.0]


def test_placeholder_line_mesh_is_non_empty_and_can_carry_rgba_colors():
    mesh = placeholder_line_mesh("gray")
    assert mesh.n_cells == 1
    assert mesh.n_points == 2
    assert np.isfinite(mesh.points).all()
    assert mesh.cell_data["rgba"].shape == (1, 4)


def test_image_grid_preserves_zyx_origin_and_xyz_dimensions():
    grid = image_grid(np.arange(24, dtype=np.uint16).reshape(2, 3, 4), 1, np.array([10, 20, 30]))
    assert grid.dimensions == (4, 3, 2)
    assert grid.origin == (30.0, 20.0, 10.0)
    assert grid.point_data["ct_intensity"].size == 24

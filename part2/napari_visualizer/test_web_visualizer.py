"""Selection-state regression tests for the browser visualizer."""

from unittest import TestCase
from unittest.mock import Mock

from part2.napari_visualizer.web_visualizer import WebStrutVisualizer


class _State:
    def __init__(self):
        self.flush_calls = 0
        self.context_entries = 0

    def __enter__(self):
        self.context_entries += 1
        return self

    def __exit__(self, *_args):
        return False

    def flush(self):
        self.flush_calls += 1


class BrokerSelectionStateTests(TestCase):
    def test_broker_selection_flushes_the_id_controls(self):
        viewer = object.__new__(WebStrutVisualizer)
        viewer.strut_index = {"10": 0, "11": 1}
        viewer.state = _State()
        viewer._selection_state_update = False
        viewer._set_selection_status = Mock()
        viewer._update_selection = Mock()

        viewer._apply_ids(["10", "11"], publish=False, active_strut_id="11")

        self.assertEqual(viewer.selected_ids, ["10", "11"])
        self.assertEqual(viewer.state.strut_ids_input, "10, 11")
        self.assertEqual(
            viewer.state.selected_strut_options,
            [{"title": "10", "value": "10"}, {"title": "11", "value": "11"}],
        )
        self.assertEqual(viewer.state.selected_strut_id, "11")
        self.assertEqual(viewer.state.context_entries, 1)
        self.assertEqual(viewer.state.flush_calls, 1)
        viewer._update_selection.assert_called_once_with("11", publish=False)

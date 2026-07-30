"""Chat-parser and selection-broker regression tests."""

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import HTTPException

from part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi import (
    _chat_format_metric,
    _chat_response,
    current_state,
    publish_active_strut,
    publish_selection,
    ReviewDecisionStore,
    _chat_filter,
    store,
)


class ChatListingTests(unittest.TestCase):
    def setUp(self):
        self._state = (
            current_state.active_strut_id,
            list(current_state.active_strut_ids),
        )

    def tearDown(self):
        current_state.active_strut_id, current_state.active_strut_ids = self._state

    def test_dashboard_summary_is_complete_and_geometry_free(self):
        payload = store.dashboard_records

        self.assertEqual(len(payload), len(store.defect_by_strut))
        self.assertIn("missing_score", payload[0])
        self.assertIn("needs_review", payload[0])
        self.assertNotIn("start_x_vox", payload[0])

    def test_indexed_compact_records_are_fast(self):
        keys = store.defect_by_strut["_strut_key"].iloc[:1000]
        started = perf_counter()
        for key in keys:
            store._compact_strut_record(key)

        self.assertLess(perf_counter() - started, 2.0)

    def test_strut_records_include_full_defect_summary(self):
        record = store._compact_strut_record(str(store.centerlines.iloc[0]["_strut_key"]))

        self.assertIn("defect_summary", record)
        self.assertIn("missing_score", record["defect_summary"])
        self.assertIn("needs_review", record["defect_summary"])

    def test_active_strut_supports_station_deviation_question(self):
        strut_id = int(store.defect_by_strut.iloc[0]["strut_id"])
        response = _chat_response("Where is the largest deviation?", active_strut_id=strut_id)

        self.assertEqual(response["strut_id"], strut_id)
        self.assertIn("position fraction", response["reply"])

    def test_chat_measurements_use_two_decimal_places(self):
        self.assertEqual(_chat_format_metric(3225.5712106704714, "length"), "3,225.57 µm")
        self.assertEqual(_chat_format_metric(0.6043185424804688, "occupancy"), "60.43 %")

    def test_single_strut_summary_is_labeled_and_rounded(self):
        strut_id = int(store.defect_by_strut.iloc[0]["strut_id"])
        response = _chat_response(f"Inspect strut {strut_id}")

        self.assertEqual(response["result_type"], "strut_summary")
        self.assertTrue(response["reply"].startswith(f"### Strut {strut_id}"))
        self.assertIn("**Length:**", response["reply"])
        self.assertIn("**Median diameter:**", response["reply"])
        self.assertNotIn("**secondary defects:** nan", response["reply"].casefold())

    def test_explicit_strut_list_selects_each_valid_id(self):
        strut_ids = store.defect_by_strut.iloc[:4]["strut_id"].astype(int).tolist()
        response = _chat_response("Select struts " + ", ".join(map(str, strut_ids)))

        self.assertEqual(response["select_strut_ids"], strut_ids)

    def test_selection_can_activate_a_nonfirst_selected_strut(self):
        strut_ids = store.defect_by_strut.iloc[:2]["strut_id"].astype(int).tolist()
        result = asyncio.run(publish_selection(strut_ids, active_strut_id=strut_ids[1]))

        self.assertEqual(result["strut_ids"], strut_ids)
        self.assertEqual(result["active_strut_id"], strut_ids[1])
        self.assertEqual(current_state.active_strut_id, strut_ids[1])

    def test_active_strut_change_preserves_selected_struts(self):
        strut_ids = store.defect_by_strut.iloc[:2]["strut_id"].astype(int).tolist()
        asyncio.run(publish_selection(strut_ids))
        result = asyncio.run(publish_active_strut(strut_ids[1]))

        self.assertEqual(result["strut_ids"], strut_ids)
        self.assertEqual(current_state.active_strut_id, strut_ids[1])

    def test_active_strut_must_belong_to_selected_struts(self):
        strut_ids = store.defect_by_strut.iloc[:2]["strut_id"].astype(int).tolist()
        asyncio.run(publish_selection(strut_ids))
        outsider = int(store.defect_by_strut.iloc[2]["strut_id"])

        with self.assertRaises(HTTPException):
            asyncio.run(publish_active_strut(outsider))

    def test_review_decision_store_persists_latest_decision(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            decisions = ReviewDecisionStore(path)
            decisions.set(7, "needs_review")
            decisions.set(7, "nominal")

            restored = ReviewDecisionStore(path)
            self.assertEqual(restored.items()[0]["strut_id"], 7)
            self.assertEqual(restored.items()[0]["decision"], "nominal")

    def test_dashboard_records_include_geometry_length(self):
        self.assertIn("length_um", store.dashboard_records[0])

    def test_chat_filter_counts_primary_defect_rows(self):
        inflated = _chat_filter(primary_defect="Inflated")
        expected = store.defect_by_strut[store.defect_by_strut["primary_defect"] == "Inflated"]
        self.assertEqual(len(inflated), len(expected))

    def test_bent_listing_selects_all_matching_struts(self):
        response = _chat_response("Which struts are classified as bent?")
        expected = store.defect_by_strut[
            store.defect_by_strut["primary_defect"].astype(str).str.casefold() == "bent"
        ]["strut_id"].astype(int).tolist()

        self.assertEqual(response["field"], "primary_defect")
        self.assertEqual(response["value"], "Bent")
        self.assertEqual(response["total"], len(expected))
        self.assertEqual(response["strut_ids"], expected)
        self.assertEqual(response["select_strut_ids"], expected)
        self.assertEqual(response["references"], ["/struts?primary_defect=Bent"])

    def test_all_struts_defect_request_selects_every_matching_strut(self):
        response = _chat_response("I want to know all the struts that are bent")
        expected = store.defect_by_strut[
            store.defect_by_strut["primary_defect"].astype(str).str.casefold() == "bent"
        ]["strut_id"].astype(int).tolist()

        self.assertEqual(response["select_strut_ids"], expected)
        self.assertIn(f"Selected all {len(expected)}", response["reply"])

    def test_count_question_does_not_change_selection(self):
        response = _chat_response("How many struts are bent?")

        self.assertNotIn("select_strut_ids", response)

    def test_explicit_classification_uses_stage2_field(self):
        response = _chat_response("List struts with Stage 2 classification Missing_Intentional")

        self.assertEqual(response["field"], "stage2_classification")
        self.assertEqual(response["value"], "Missing_Intentional")
        self.assertEqual(response["total"], 87)
        self.assertEqual(response["references"], ["/struts?classification=Missing_Intentional"])

    def test_bare_nominal_prefers_classification(self):
        response = _chat_response("Show Nominal struts")

        self.assertEqual(response["field"], "stage2_classification")
        self.assertEqual(response["value"], "Nominal")

    def test_missing_subtypes_accept_human_readable_spacing(self):
        for prompt in (
            "Which struts are Missing Intentional?",
            "Which struts are missing-intentional?",
            "Which struts are MISSING_INTENTIONAL?",
        ):
            response = _chat_response(prompt)
            self.assertEqual(response["field"], "stage2_classification")
            self.assertEqual(response["value"], "Missing_Intentional")
            self.assertEqual(response["total"], 87)

    def test_bare_missing_requests_a_subtype(self):
        response = _chat_response("Which struts are missing?")

        self.assertNotIn("strut_ids", response)
        self.assertEqual(
            response["missing_subtypes"],
            {"Missing_Intentional": 87, "Missing_Unintentional": 331},
        )
        self.assertIn("Please specify", response["reply"])

    def test_chat_explains_known_data_terms(self):
        response = _chat_response("What does material occupancy mean?")

        self.assertEqual(response["result_type"], "glossary")
        self.assertIn("sampled fraction", response["reply"])

    def test_chat_compares_current_strut_with_nominal_baseline(self):
        strut_id = int(store.defect_by_strut.iloc[0]["strut_id"])
        response = _chat_response("Compare this strut with all nominal struts", active_strut_id=strut_id)

        self.assertEqual(response["result_type"], "comparison")
        self.assertEqual(response["strut_ids"], [strut_id])
        self.assertIn("nominal_median", next(iter(response["comparison"].values())))

    def test_chat_ranks_and_selects_requested_struts(self):
        response = _chat_response("Show the top 3 struts by maximum deviation")

        self.assertEqual(response["result_type"], "ranking")
        self.assertEqual(len(response["strut_ids"]), 3)
        self.assertEqual(response["select_strut_ids"], response["strut_ids"])

    def test_chat_summarizes_multiple_explicit_struts(self):
        strut_ids = store.defect_by_strut.iloc[:2]["strut_id"].astype(int).tolist()
        response = _chat_response("Inspect struts " + ", ".join(map(str, strut_ids)))

        self.assertEqual(response["result_type"], "multi_strut_summary")
        self.assertEqual(response["strut_ids"], strut_ids)
        self.assertEqual(response["total"], len(strut_ids))

    def test_chat_reports_unknown_explicit_strut(self):
        response = _chat_response("Inspect strut 999999999")

        self.assertEqual(response["result_type"], "unknown_strut")
        self.assertIn("loaded dataset", response["reply"])


if __name__ == "__main__":
    unittest.main()

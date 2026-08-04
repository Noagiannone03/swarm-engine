from __future__ import annotations

import unittest

from fabi_context_placement_oracle import solve


class PlacementOracleTest(unittest.TestCase):
    def test_prefers_complete_weighted_long_route_over_short_full_model(self) -> None:
        scenario = {
            "num_layers": 4,
            "context_classes": [10, 20],
            "demands": [
                {"context_tokens": 10, "target_slots": 1, "weight": 1},
                {"context_tokens": 20, "target_slots": 1, "weight": 10},
            ],
            "workers": [
                {
                    "worker_id": "adaptive",
                    "options": [
                        {
                            "option_id": "short-full",
                            "start_layer": 0,
                            "end_layer": 4,
                            "context_tokens": 10,
                            "max_sessions": 1,
                        },
                        {
                            "option_id": "long-head",
                            "start_layer": 0,
                            "end_layer": 2,
                            "context_tokens": 20,
                            "max_sessions": 1,
                        },
                    ],
                },
                {
                    "worker_id": "tail",
                    "options": [
                        {
                            "option_id": "long-tail",
                            "start_layer": 2,
                            "end_layer": 4,
                            "context_tokens": 20,
                            "max_sessions": 1,
                        }
                    ],
                },
            ],
        }

        result = solve(scenario)

        self.assertEqual(result["served_slots_by_context"], {"10": 0, "20": 1})
        self.assertEqual(
            {(item["worker_id"], item["option_id"]) for item in result["placements"]},
            {("adaptive", "long-head"), ("tail", "long-tail")},
        )

    def test_slots_are_shared_across_context_classes(self) -> None:
        scenario = {
            "num_layers": 2,
            "context_classes": [10, 20],
            "demands": [
                {"context_tokens": 10, "target_slots": 2, "weight": 1},
                {"context_tokens": 20, "target_slots": 2, "weight": 1},
            ],
            "workers": [
                {
                    "worker_id": "full",
                    "options": [
                        {
                            "option_id": "long-full",
                            "start_layer": 0,
                            "end_layer": 2,
                            "context_tokens": 20,
                            "max_sessions": 2,
                        }
                    ],
                }
            ],
        }

        result = solve(scenario)

        self.assertEqual(sum(result["served_slots_by_context"].values()), 2)


if __name__ == "__main__":
    unittest.main()

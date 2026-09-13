from scripts.hpc.prepare_recurrent_training_probe import recurrent_exclusions


def _window(component, length, index):
    landpoint_id = f"development-{component}-{index}"
    start_row = 40 * index
    return {
        "artifact_id": index,
        "component": component,
        "window_index": 0,
        "landpoint_id": landpoint_id,
        "year": 2000,
        "start_row": start_row,
        "length": length,
        "teacher_day_keys": [
            (landpoint_id, 2000, row) for row in range(start_row, start_row + length)
        ],
    }


def test_recurrent_exclusions_keep_expanded_train_rows_eligible_but_exclude_monitor_and_development_windows():
    development = {
        "windows": [
            *[_window("seven_day", 7, index) for index in range(8)],
            *[_window("thirty_day", 30, index + 8) for index in range(8)],
        ]
    }
    catalogue = {
        "one_day_entries": [
            {
                "components": {
                    "one_day": [
                        {"landpoint_id": "expanded-train", "year": 2001, "start_rows": [44]}
                    ]
                }
            }
        ],
        "monitor_rows": [
            {"landpoint_id": "monitor", "year": 2002, "start_row": 55}
        ],
    }

    monitor_days, development_days, excluded = recurrent_exclusions(catalogue, development)

    assert monitor_days == {("monitor", 2002, 55)}
    assert len(development_days) == 296
    assert ("development-seven_day-0", 2000, 0) in excluded
    assert ("development-thirty_day-15", 2000, 629) in excluded
    assert ("expanded-train", 2001, 44) not in excluded

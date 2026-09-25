"""shift_hours in every accepted form, and clear errors for the rest."""

from __future__ import annotations

import pytest

import production_anomaly as pa
from production_anomaly import Schedule, Shift, parse_shift_hours


@pytest.mark.parametrize(
    "spec, windows",
    [
        ((6, 22), ["06:00-22:00"]),
        ([(6, 14), (14, 22)], ["06:00-14:00", "14:00-22:00"]),
        ("06:00-14:00,14:00-22:00", ["06:00-14:00", "14:00-22:00"]),
        ("6-14; 14-22.5", ["06:00-14:00", "14:00-22:30"]),
        (("06:00", "22:00"), ["06:00-22:00"]),
        (("6-14", "14-22"), ["06:00-14:00", "14:00-22:00"]),
        (8, ["06:00-14:00", "14:00-22:00", "22:00-06:00"]),
        ("12", ["06:00-18:00", "18:00-06:00"]),
        ((0, 24), ["00:00-00:00"]),
    ],
)
def test_accepted_forms(spec, windows):
    schedule = parse_shift_hours(spec)
    assert isinstance(schedule, Schedule) and schedule.given
    assert [s.window for s in schedule.shifts] == windows


def test_named_shifts_and_midnight_wrap():
    schedule = parse_shift_hours({"A": (6, 14), "B": (14, 22), "C": (22, 6)})
    assert [s.name for s in schedule.shifts] == ["A", "B", "C"]
    assert schedule.covers_whole_day
    night = schedule.shifts[2]
    assert isinstance(night, Shift) and night.minutes == 480
    assert parse_shift_hours("A=6-14,B=14-22").shifts[1].name == "B"


def test_default_schedule_counts_every_hour():
    schedule = parse_shift_hours(None)
    assert not schedule.given and schedule.covers_whole_day
    assert "every hour counts as scheduled" in schedule.describe()


def test_lookup_maps_each_minute_to_its_shift():
    table = parse_shift_hours([(6, 14), (22, 2)]).lookup()
    assert table[6 * 60] == 0 and table[14 * 60 - 1] == 0 and table[14 * 60] == -1
    assert table[23 * 60] == 1 and table[60] == 1 and table[2 * 60] == -1


@pytest.mark.parametrize(
    "spec, message",
    [
        ([(6, 14), (13, 22)], "overlap"),
        (7, "divide the day evenly"),
        ("x-y", "not a time of day"),
        ((6, 25), "outside 0-24"),
        ([], "empty"),
        ({}, "empty dict"),
        (True, "must be a"),
        ([(6, 14, 3)], "pair"),
        ("06:75-14:00", "more than 59 minutes"),
        ([(6, 6), (8, 9)], "no length"),
    ],
)
def test_bad_forms_raise_value_error(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_shift_hours(spec)


def test_bad_shift_hours_raise_at_the_entry_point(steady_day):
    with pytest.raises(ValueError, match="overlap"):
        pa.analyze(steady_day, shift_hours=[(6, 14), (12, 20)])


def test_by_shift_has_one_row_per_shift_worked(steady_day):
    report = pa.analyze(steady_day, shift_hours=[(6, 14), (14, 22)])
    assert list(report.by_shift["shift"]) == ["06:00-14:00", "14:00-22:00"]
    assert list(report.by_shift["scheduled_min"]) == [480, 480]
    assert list(report.by_shift["rate_per_hour"]) == [3600, 3600]
    assert report.unscheduled_minutes == 480

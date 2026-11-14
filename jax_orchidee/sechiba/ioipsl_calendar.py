"""Pure scalar subset of the identified IOIPSL calendar implementation.

Provenance: IOIPSL/src/calendar.f90, SHA256
4143df773973bfa0d7b9a8fa1aea36475e23f210b7c5bb6daec95ac02e05acc1.
The numbered source is retained in outputs/reviews/main-physics-20261113/
clock-authority-cold-inputs/clock-source-evidence.txt and the 20261114 native
clock package. Only canonical Gregorian and noleap calendars are admitted.
Gregorian day numbers are Lilian, NOT astronomical Julian dates. A noleap
January 1 has day number year*365. This module performs no IO or scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _idiv(numerator: int, denominator: int) -> int:
    """Fortran integer division, truncating toward zero (calendar136-309)."""
    quotient = abs(numerator) // abs(denominator)
    return -quotient if (numerator < 0) != (denominator < 0) else quotient


def fortran_nint(value: float) -> int:
    """Fortran NINT, ties away from zero (calendar313-402; intersurf1933)."""
    integer = int(value)
    fraction = value - integer
    if fraction >= 0.5:
        return integer + 1
    if fraction <= -0.5:
        return integer - 1
    return integer


@dataclass(frozen=True)
class IOIPSLCalendar:
    """Immutable calendar and optional independent IOIPSL startdate lock.

    The split lock models calendar.f90::ioconf_startdate_internal754-783.
    It is explicit: a non-Gregorian owner date0 must not implicitly establish
    the separately initialized itau2ymds startdate. Gregorian owner decoding
    uses its scalar in_julian and does not require this lock.
    """

    calendar: str
    startdate_day: int | None = None
    startdate_second: float = 0.0

    def __post_init__(self):
        name = self.calendar.strip()
        if name not in {"gregorian", "noleap"}:
            raise ValueError(f"unsupported IOIPSL calendar {name!r}; measured scope is gregorian/noleap")
        if self.startdate_day is not None and int(self.startdate_day) != self.startdate_day:
            raise ValueError("startdate_day must be an integer")
        if not math.isfinite(self.startdate_second):
            raise ValueError("startdate_second must be finite")
        if self.startdate_day is None and self.startdate_second != 0:
            raise ValueError("an unlocked calendar cannot carry startdate_second")
        object.__setattr__(self, "calendar", name)

    @property
    def one_year_days(self) -> float:
        """ioget_calendar constants, calendar.f90::ioconf_calendar636-698."""
        return 365.2425 if self.calendar == "gregorian" else 365.0

    @property
    def one_day_seconds(self) -> float:
        """IOIPSL's second-valued day constant, calendar.f90::calendar module."""
        return 86400.0

    def month_length(self, year: int, month: int) -> int:
        """ioget_mon_len, calendar.f90:858-909; Gregorian leap rule."""
        if not 1 <= month <= 12:
            raise ValueError("month must be in 1..12")
        leap = self.calendar == "gregorian" and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]

    def _ymds_day(self, year: int, month: int, day: int) -> int:
        """ymds2ju_internal136-197; intentionally keeps source date arithmetic."""
        if not 1 <= month <= 12:
            raise ValueError("month must be in 1..12")
        if self.calendar == "noleap":
            return year * 365 + sum(self.month_length(year, m) for m in range(1, month)) + day - 1
        q = _idiv(month - 14, 12)
        jd = (_idiv(1461 * (year + 4800 + q), 4)
              + _idiv(367 * (month - 2 - 12 * q), 12)
              - _idiv(3 * _idiv(year + 4900 + q, 100), 4) + day - 32075)
        return jd - 2299160

    def ymds2ju(self, year: int, month: int, day: int, second: float = 0.0) -> float:
        """ymds2ju112-132 and internal136-197; seconds remain REAL."""
        return float(self._ymds_day(year, month, day)) + float(second) / self.one_day_seconds

    def _decode_split(self, julian_day: int, julian_second: float) -> tuple[int, int, int, float]:
        """ju2ymds_internal222-309, including its exact normalization order."""
        jd, sec = int(julian_day), float(julian_second)
        eps_day = math.ulp(self.one_day_seconds)
        if sec > self.one_day_seconds - eps_day:
            add_day = int(sec / self.one_day_seconds)
            sec = sec - add_day * self.one_day_seconds
            jd = jd + add_day
        if sec < -eps_day:
            sec = sec + self.one_day_seconds
            jd = jd - 1
        if self.calendar == "gregorian":
            jd = jd + 2299160
            l = jd + 68569
            n = _idiv(4 * l, 146097)
            l = l - _idiv(146097 * n + 3, 4)
            i = _idiv(4000 * (l + 1), 1461001)
            l = l - _idiv(1461 * i, 4) + 31
            j = _idiv(80 * l, 2447)
            day = l - _idiv(2447 * j, 80)
            l = _idiv(j, 11)
            month = j + 2 - 12 * l
            year = 100 * (n - 49) + i + l
        else:
            year = _idiv(jd, 365)
            remaining = jd - year * 365
            month, elapsed = 1, 0
            while elapsed + self.month_length(year, month) <= remaining:
                elapsed += self.month_length(year, month)
                month += 1
            day = remaining - elapsed + 1
        return year, month, day, sec

    def ju2ymds(self, julian: float) -> tuple[int, int, int, float]:
        """ju2ymds201-218: INT scalar split, not a noon/.5 transformation."""
        day = int(julian)
        return self._decode_split(day, (float(julian) - day) * self.one_day_seconds)

    def itau2date(self, istp: int, date0: float, dt: float) -> float:
        """itau2date406-426; preserve multiplication/division evaluation."""
        return float(istp) * float(dt) / self.one_day_seconds + float(date0)

    def itau2ymds(self, istp: int, dt: float) -> tuple[int, int, int, float]:
        """itau2ymds430-466 uses the independent split startdate lock."""
        if self.startdate_day is None:
            raise ValueError("itau2ymds requires an explicitly locked IOIPSL startdate")
        return self._decode_split(self.startdate_day, self.startdate_second + float(istp) * float(dt))

    def year_steps(self, dt: float, date: float) -> int:
        """tlen2itau313-402, exactly the '1Y' branch358-364 (not average year)."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        year, month, day, second = self.ju2ymds(date)
        date_new = self.ymds2ju(year + 1, month, day, second)
        dd = date_new - date
        # The odd fractional-day expression is the source contract verbatim.
        ss = int(dd) * self.one_day_seconds + dd - int(dd)
        return fortran_nint(ss / dt)

    @classmethod
    def from_ymds(cls, calendar: str, year: int, month: int, day: int, second: float = 0.0):
        """ioconf_startdate_ymds736-750 retains split seconds without scalar loss."""
        provider = cls(calendar)
        return cls(calendar, provider._ymds_day(year, month, day), float(second))

    @classmethod
    def from_scalar_startdate(cls, calendar: str, date: float):
        """ioconf_startdate_simple718-732 retains the scalar split's rounding."""
        if not math.isfinite(date):
            raise ValueError("startdate must be finite")
        provider = cls(calendar)
        day = int(date)
        return cls(calendar, day, (float(date) - day) * provider.one_day_seconds)

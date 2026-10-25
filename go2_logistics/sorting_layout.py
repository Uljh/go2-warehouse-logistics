"""Pure data contract for camera-decoded parcel IDs and numbered shelf poses."""
from dataclasses import dataclass
import re
from .interfaces import Pose2D

SHELF_DECK_SIZE = (.30, .40, .03)


@dataclass(frozen=True)
class ParcelCode:
    shelf_id: int
    serial: str

    @classmethod
    def parse(cls, text: str) -> "ParcelCode":
        match = re.fullmatch(r"([1-9][0-9]*)-([0-9]+)", text)
        if match is None or not 1 <= int(match[1]) <= 30:
            raise ValueError(f"Expected warehouse code 1..30-numeric_serial, got {text!r}")
        return cls(int(match[1]), match[2])

    def __str__(self):
        return f"{self.shelf_id}-{self.serial}"


@dataclass(frozen=True)
class Shelf:
    number: int
    row: int
    column: int
    surface: tuple[float, float, float]
    dock: Pose2D


def numbered_shelves() -> tuple[Shelf, ...]:
    """Three east-west rows; each shelf's open picking face points west.

    Positions are a known warehouse map, not perceived parcel poses. Shelf
    top height matches the measured feasible placement height of the baseline.
    """
    shelves = []
    for number in range(1, 31):
        row, column = divmod(number - 1, 10)
        x, y = -5.4 + column * 1.2, 3.0 + row * 3.0
        shelves.append(Shelf(number, row, column, (x, y, .42), Pose2D(x - .50, y)))
    return tuple(shelves)


def shelf_for_code(text: str) -> Shelf:
    return numbered_shelves()[ParcelCode.parse(text).shelf_id - 1]

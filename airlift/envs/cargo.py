from __future__ import annotations

# The following imports from the airport module for type checking only (and avoids circular imports)
# See https://www.stefaanlippens.net/circular-imports-type-hints-python.html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from airlift.envs import Airport

CargoID = int


class Cargo:
    """Defines the properties of each single piece of Cargo"""
    def __init__(self, id, source_airport, end_airport, weight=1, soft_deadline=2 ** 32, hard_deadline=2 ** 32):
        self.weight: int = weight
        self.id: CargoID = id
        self.soft_deadline = soft_deadline
        self.hard_deadline = hard_deadline
        self.source_airport: Airport = source_airport
        self.end_airport: Airport = end_airport
        self.delivery_time = None
        self.reward = 0
        self.missed_softdeadline = False
        self.missed_hardeadline = False

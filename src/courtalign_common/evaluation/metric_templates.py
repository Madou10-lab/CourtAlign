from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricTemplate:
    name: str
    sport: str
    width_m: float
    length_m: float
    source_note: str
    zone_polygons_tl_tr_bl_br: dict[int, list[tuple[float, float]]]

    @property
    def fullcourt_corners_tl_tr_bl_br(self) -> np.ndarray:
        return np.asarray(
            [
                (0.0, 0.0),
                (self.width_m, 0.0),
                (0.0, self.length_m),
                (self.width_m, self.length_m),
            ],
            dtype=np.float64,
        )


def tennis_fullcourt_template(width_m: float, length_m: float, source_note: str) -> MetricTemplate:
    return MetricTemplate(
        name="tennis_fullcourt_metric",
        sport="tennis",
        width_m=float(width_m),
        length_m=float(length_m),
        source_note=source_note,
        zone_polygons_tl_tr_bl_br={
            1: [
                (0.0, 0.0),
                (float(width_m), 0.0),
                (0.0, float(length_m)),
                (float(width_m), float(length_m)),
            ]
        },
    )


def badminton_doubles_zone_template(
    width_m: float,
    length_m: float,
    singles_width_m: float,
    doubles_long_service_from_back_m: float,
    short_service_from_net_m: float,
    source_note: str,
) -> MetricTemplate:
    width_m = float(width_m)
    length_m = float(length_m)
    singles_width_m = float(singles_width_m)
    long_service = float(doubles_long_service_from_back_m)
    short_service = float(short_service_from_net_m)

    alley = (width_m - singles_width_m) / 2.0
    x0 = 0.0
    x1 = alley
    x2 = width_m / 2.0
    x3 = width_m - alley
    x4 = width_m

    net = length_m / 2.0
    y0 = 0.0
    y1 = long_service
    y2 = net - short_service
    y3 = net + short_service
    y4 = length_m - long_service
    y5 = length_m

    zones = {
        1: [(x0, y3), (x1, y3), (x0, y4), (x1, y4)],  # fl_doubles
        2: [(x1, y3), (x2, y3), (x1, y4), (x2, y4)],  # fl_service
        3: [(x1, y4), (x2, y4), (x1, y5), (x2, y5)],  # fs_singles
        4: [(x2, y4), (x3, y4), (x2, y5), (x3, y5)],  # fr_singles
        5: [(x2, y3), (x3, y3), (x2, y4), (x3, y4)],  # fr_service
        6: [(x3, y3), (x4, y3), (x3, y4), (x4, y4)],  # fr_doubles
        7: [(x0, y2), (x4, y2), (x0, y3), (x4, y3)],  # central_zone
        8: [(x0, y1), (x1, y1), (x0, y2), (x1, y2)],  # bl_doubles
        9: [(x1, y1), (x2, y1), (x1, y2), (x2, y2)],  # bl_service
        10: [(x1, y0), (x2, y0), (x1, y1), (x2, y1)],  # bl_singles
        11: [(x2, y1), (x3, y1), (x2, y2), (x3, y2)],  # br_service
        12: [(x2, y0), (x3, y0), (x2, y1), (x3, y1)],  # br_singles
        13: [(x3, y1), (x4, y1), (x3, y2), (x4, y2)],  # br_doubles
    }

    return MetricTemplate(
        name="badminton_doubles_13_zone_metric",
        sport="badminton",
        width_m=width_m,
        length_m=length_m,
        source_note=source_note,
        zone_polygons_tl_tr_bl_br=zones,
    )


def template_from_config(dataset_id: str, config: dict) -> MetricTemplate:
    if dataset_id == "tennis_fullcourt":
        tennis = config["tennis_fullcourt"]
        return tennis_fullcourt_template(
            width_m=tennis["width_m"],
            length_m=tennis["length_m"],
            source_note=tennis.get("source_note", ""),
        )
    if dataset_id == "badminton_zones":
        badminton = config["badminton_doubles"]
        return badminton_doubles_zone_template(
            width_m=badminton["width_m"],
            length_m=badminton["length_m"],
            singles_width_m=badminton["singles_width_m"],
            doubles_long_service_from_back_m=badminton["doubles_long_service_from_back_m"],
            short_service_from_net_m=badminton["short_service_from_net_m"],
            source_note=badminton.get("source_note", ""),
        )
    raise ValueError(f"No metric template configured for dataset_id={dataset_id!r}")

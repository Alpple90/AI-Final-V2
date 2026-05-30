# travel_time.py
import math
from config import (
    SPEED_LIMIT_KPH,
    INTERSECTION_DELAY_SEC,
    CAPACITY_FLOW,
    FREE_FLOW_THRESHOLD,
    FLOW_TO_SPEED_A,
    FLOW_TO_SPEED_B,
)

_INTERSECTION_DELAY_MIN = INTERSECTION_DELAY_SEC / 60


def flow_to_speed(flow_per_hour):
    if flow_per_hour <= FREE_FLOW_THRESHOLD:
        return SPEED_LIMIT_KPH

    discriminant = FLOW_TO_SPEED_B**2 - 4 * FLOW_TO_SPEED_A * (-flow_per_hour)
    if discriminant < 0:
        return 32

    sqrt_disc = math.sqrt(discriminant)
    speeds = [s for s in [
        (-FLOW_TO_SPEED_B + sqrt_disc) / (2 * FLOW_TO_SPEED_A),
        (-FLOW_TO_SPEED_B - sqrt_disc) / (2 * FLOW_TO_SPEED_A),
    ] if s > 0]

    if not speeds:
        return 32

    if flow_per_hour <= CAPACITY_FLOW:
        return min(max(speeds), SPEED_LIMIT_KPH)
    return min(speeds)


def calculate_travel_time(distance_km, flow_per_15min):
    speed = min(flow_to_speed(flow_per_15min * 4), SPEED_LIMIT_KPH)
    if speed <= 0.1:
        return 999
    return round((distance_km / speed) * 60 + _INTERSECTION_DELAY_MIN, 2)

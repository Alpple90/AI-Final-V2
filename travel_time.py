# travel_time.py - convert traffic flow to travel time
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


# convert hourly flow to speed in km/h using the quadratic flow-speed model
def flowToSpeed(flowPerHour):
    # free flow below threshold, otherwise use quadratic model
    if flowPerHour <= FREE_FLOW_THRESHOLD:
        return SPEED_LIMIT_KPH

    discriminant = FLOW_TO_SPEED_B**2 - 4 * FLOW_TO_SPEED_A * (-flowPerHour)
    if discriminant < 0:
        return 32

    sqrtDisc = math.sqrt(discriminant)
    speeds = [s for s in [
        (-FLOW_TO_SPEED_B + sqrtDisc) / (2 * FLOW_TO_SPEED_A),
        (-FLOW_TO_SPEED_B - sqrtDisc) / (2 * FLOW_TO_SPEED_A),
    ] if s > 0]

    if not speeds:
        return 32

    if flowPerHour <= CAPACITY_FLOW:
        return min(max(speeds), SPEED_LIMIT_KPH)
    return min(speeds)


# convert a road segment distance and 15-min flow count into a travel time in minutes
def calcTravelTime(distanceKm, flowPer15min):
    speed = min(flowToSpeed(flowPer15min * 4), SPEED_LIMIT_KPH)
    if speed <= 0.1:
        return 999
    return round((distanceKm / speed) * 60 + _INTERSECTION_DELAY_MIN, 2)

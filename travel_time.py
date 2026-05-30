# travel_time.py
"""
Traffic Flow to Travel Time Conversion
Based on Assignment 2B PDF v1.0

Formula: flow = -1.4648375 * speed^2 + 93.75 * speed
"""

import math
from config import (
    SPEED_LIMIT_KPH,
    INTERSECTION_DELAY_SEC,
    CAPACITY_FLOW,
    FREE_FLOW_THRESHOLD,
    FLOW_TO_SPEED_A,
    FLOW_TO_SPEED_B
)


class TravelTimeCalculator:
    """
    Converts predicted traffic flow to travel time using the
    fundamental diagram from the assignment PDF.
    """
    
    def __init__(self):
        self.speed_limit = SPEED_LIMIT_KPH
        self.a = FLOW_TO_SPEED_A
        self.b = FLOW_TO_SPEED_B
        self.capacity_flow = CAPACITY_FLOW
        self.free_flow_threshold = FREE_FLOW_THRESHOLD
        self.intersection_delay_min = INTERSECTION_DELAY_SEC / 60

    def flow_to_speed(self, flow_per_hour):
        """
        Convert traffic flow to expected speed.
        
        Args:
            flow_per_hour: Traffic flow in vehicles per hour
            
        Returns:
            Speed in km/h
        """
        # Free flow condition (≤ 351 vehicles/hour)
        if flow_per_hour <= self.free_flow_threshold:
            return self.speed_limit

        # Solve quadratic: a*s² + b*s - flow = 0
        discriminant = self.b**2 - 4 * self.a * (-flow_per_hour)
        
        if discriminant < 0:
            return 32  # Default congested speed

        sqrt_disc = math.sqrt(discriminant)
        speed1 = (-self.b + sqrt_disc) / (2 * self.a)
        speed2 = (-self.b - sqrt_disc) / (2 * self.a)
        
        # Take positive speeds only
        speeds = [s for s in [speed1, speed2] if s > 0]

        if not speeds:
            return 32

        # Green line (under capacity): higher speed
        # Red line (over capacity): lower speed
        if flow_per_hour <= self.capacity_flow:
            return min(max(speeds), self.speed_limit)
        else:
            return min(speeds)

    def calculate_travel_time(self, distance_km, flow_per_15min):
        """
        Calculate travel time in minutes for a road segment.
        
        Args:
            distance_km: Distance between SCATS sites in km
            flow_per_15min: Predicted traffic flow (vehicles per 15 minutes)
            
        Returns:
            Travel time in minutes
        """
        hourly_flow = flow_per_15min * 4
        speed = self.flow_to_speed(hourly_flow)
        speed = min(speed, self.speed_limit)

        if speed <= 0.1:
            return 999  # Very high cost for gridlock

        # Travel time = (distance / speed) * 60 + intersection delay
        travel_time = (distance_km / speed) * 60 + self.intersection_delay_min
        return round(travel_time, 2)
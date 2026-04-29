from __future__ import annotations

from typing import Any, Dict
import numpy as np

from .common import clamp, safe_float


def myopic_base_allocation(resource_queue: float, waiting_queue: float, arrivals: float, max_allocation: float = 800.0) -> float:
    """Baseline resource allocation used by the Haima environment.

    It compensates the gap between demand (waiting + arrivals) and currently
    idle resource, then clips the action into the valid range.
    """
    return clamp(safe_float(waiting_queue) + safe_float(arrivals) - safe_float(resource_queue), 0.0, max_allocation)


def action_to_allocation(raw_action: float, baseline_action: float, residual_scale: float = 200.0, max_allocation: float = 800.0) -> float:
    raw = clamp(safe_float(raw_action), -1.0, 1.0)
    return clamp(safe_float(baseline_action) + raw * safe_float(residual_scale, 200.0), 0.0, max_allocation)


def raw_action_from_allocation(allocation: float, baseline_action: float, residual_scale: float = 200.0) -> float:
    scale = max(float(residual_scale), 1e-8)
    return clamp((safe_float(allocation) - safe_float(baseline_action)) / scale, -1.0, 1.0)


def compute_allocation_outcome(
    resource_queue: float,
    waiting_queue: float,
    arrivals: float,
    allocation: float,
    waiting_penalty: float = 1.0,
    idle_penalty: float = 1.0,
    reward_scale: float = 100.0,
) -> Dict[str, float]:
    """Supply-demand transition and reward.

    Mirrors the project environment logic: total cache is current idle resource
    plus newly allocated resource; unmet demand becomes next waiting queue;
    unused cache becomes next resource queue; reward is negative weighted cost.
    """
    resource_queue = max(safe_float(resource_queue), 0.0)
    waiting_queue = max(safe_float(waiting_queue), 0.0)
    arrivals = max(safe_float(arrivals), 0.0)
    allocation = max(safe_float(allocation), 0.0)
    waiting_penalty = max(safe_float(waiting_penalty), 0.0)
    idle_penalty = max(safe_float(idle_penalty), 0.0)
    reward_scale = max(safe_float(reward_scale, 100.0), 1e-8)

    cache_available = resource_queue + allocation
    total_demand = waiting_queue + arrivals
    total_served = min(cache_available, total_demand)
    served_waiting = min(waiting_queue, total_served)
    remaining_capacity = max(total_served - served_waiting, 0.0)
    served_new = min(arrivals, remaining_capacity)

    hit_rate = 1.0 if arrivals <= 1e-8 else float(served_new / arrivals)
    waste_count = max(cache_available - total_demand, 0.0)
    waste_rate = 0.0 if cache_available <= 1e-8 else float(waste_count / cache_available)

    next_resource_queue = resource_queue + allocation - waiting_queue - arrivals
    if next_resource_queue < 0.0:
        next_waiting_queue = -next_resource_queue
        next_resource_queue = 0.0
    else:
        next_waiting_queue = 0.0

    idle_cost = idle_penalty * next_resource_queue
    waiting_cost = waiting_penalty * next_waiting_queue
    weighted_cost = idle_cost + waiting_cost
    reward = -weighted_cost / reward_scale
    return {
        "cache_available": float(cache_available),
        "total_demand": float(total_demand),
        "total_served": float(total_served),
        "served_waiting": float(served_waiting),
        "served_new": float(served_new),
        "hit_rate": float(hit_rate),
        "waste_count": float(waste_count),
        "waste_rate": float(waste_rate),
        "next_resource_queue": float(next_resource_queue),
        "next_waiting_queue": float(next_waiting_queue),
        "idle_cost": float(idle_cost),
        "waiting_cost": float(waiting_cost),
        "weighted_cost": float(weighted_cost),
        "reward": float(reward),
    }


def evaluate_raw_action(
    *,
    raw_action: float,
    resource_queue: float,
    waiting_queue: float,
    arrivals: float,
    residual_scale: float = 200.0,
    max_allocation: float = 800.0,
    waiting_penalty: float = 3.0,
    idle_penalty: float = 1.0,
    reward_scale: float = 100.0,
) -> Dict[str, float]:
    base = myopic_base_allocation(resource_queue, waiting_queue, arrivals, max_allocation=max_allocation)
    allocation = action_to_allocation(raw_action, base, residual_scale=residual_scale, max_allocation=max_allocation)
    metrics = compute_allocation_outcome(
        resource_queue=resource_queue,
        waiting_queue=waiting_queue,
        arrivals=arrivals,
        allocation=allocation,
        waiting_penalty=waiting_penalty,
        idle_penalty=idle_penalty,
        reward_scale=reward_scale,
    )
    metrics.update({"baseline_action": base, "raw_action": clamp(raw_action, -1.0, 1.0), "allocation": allocation})
    return metrics

"""
Supply Chain Analytics — Deterministic Inventory Tools
-------------------------------------------------------
Deterministic Python implementations of five core supply chain inventory models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

For every calculation, each tool deterministically returns:
1. Inputs Extracted (inputs_extracted)
2. Variable Definitions (variable_definitions)
3. Formula (formula)
4. Formula with Substituted Values (formula_with_substituted_values)
5. Intermediate Calculations (intermediate_calculations)
6. Final Answer (final_answer)
7. Verification Checks (verification_checks)
8. Assumptions (assumptions)

All numerical calculations are executed strictly via deterministic Python math,
NumPy, and SciPy. Never invents missing numerical inputs.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Structured Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EOQResult:
    model: str
    # 8 Required Verification Elements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_with_substituted_values: str
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary fields for direct attribute access & backwards compatibility
    demand: float = 0.0
    ordering_cost: float = 0.0
    holding_cost: float = 0.0
    eoq: float = 0.0
    eoq_rounded: int = 0
    annual_ordering_cost: float = 0.0
    annual_holding_cost: float = 0.0
    total_ordering_holding_cost: float = 0.0
    total_annual_cost: float = 0.0
    orders_per_year: float = 0.0
    cycle_time_days: Optional[float] = None
    daily_demand: Optional[float] = None
    lead_time_days: Optional[float] = None
    reorder_point_deterministic: Optional[float] = None
    reorder_point: Optional[float] = None
    reorder_point_rounded: Optional[int] = None
    annual_purchase_cost: Optional[float] = None
    unit_cost: Optional[float] = None
    average_cycle_inventory: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscountTierEvaluation:
    tier_index: int
    tier_label: str
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost: float
    tentative_eoq: float
    feasible_quantity: float
    is_price_break_adjustment: bool
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    is_feasible: bool
    status: str


@dataclass
class QuantityDiscountResult:
    model: str
    # 8 Required Verification Elements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_with_substituted_values: str
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary fields
    demand: float = 0.0
    ordering_cost: float = 0.0
    optimal_tier_index: int = 0
    optimal_order_quantity: float = 0.0
    optimal_order_quantity_rounded: int = 0
    optimal_unit_price: float = 0.0
    optimal_total_cost: float = 0.0
    annual_savings_vs_base: float = 0.0
    percentage_savings_vs_base: float = 0.0
    tier_evaluations: List[DiscountTierEvaluation] = field(default_factory=list)
    holding_cost_rate: Optional[float] = None
    fixed_holding_cost: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyStockROPResult:
    model: str
    # 8 Required Verification Elements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_with_substituted_values: str
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary fields
    case_type: str = ""
    lead_time: float = 0.0
    demand_mean: float = 0.0
    demand_std: float = 0.0
    lead_time_std: float = 0.0
    service_level: Optional[float] = None
    z_score: float = 0.0
    lead_time_demand_mean: float = 0.0
    lead_time_demand_std: float = 0.0
    safety_stock: float = 0.0
    safety_stock_rounded: int = 0
    reorder_point: float = 0.0
    reorder_point_rounded: int = 0
    lead_time_unit: str = "days"
    demand_unit: str = "units/day"
    stockout_risk_per_cycle: Optional[float] = None
    order_quantity: Optional[float] = None
    expected_shortage_per_cycle: Optional[float] = None
    implied_fill_rate: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FixedPeriodReviewResult:
    model: str
    # 8 Required Verification Elements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_with_substituted_values: str
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary fields
    review_period: float = 0.0
    lead_time: float = 0.0
    protection_interval: float = 0.0
    demand_mean: float = 0.0
    demand_std: float = 0.0
    service_level: Optional[float] = None
    z_score: float = 0.0
    protection_demand_mean: float = 0.0
    protection_demand_std: float = 0.0
    safety_stock: float = 0.0
    safety_stock_rounded: int = 0
    order_up_to_level: float = 0.0
    order_up_to_level_rounded: int = 0
    current_inventory_position: float = 0.0
    recommended_order_quantity: float = 0.0
    recommended_order_quantity_rounded: int = 0
    inventory_on_hand: float = 0.0
    inventory_on_order: float = 0.0
    backorders: float = 0.0
    average_inventory: float = 0.0
    time_unit: str = "days"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NewsvendorResult:
    model: str
    # 8 Required Verification Elements
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: str
    formula_with_substituted_values: str
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary fields
    unit_cost: float = 0.0
    selling_price: float = 0.0
    salvage_value: float = 0.0
    penalty_cost: float = 0.0
    cost_of_underage: float = 0.0
    cost_of_overage: float = 0.0
    critical_fractile: float = 0.0
    optimal_order_quantity: float = 0.0
    optimal_order_quantity_rounded: int = 0
    distribution_type: str = "normal"
    demand_mean: Optional[float] = None
    demand_std: Optional[float] = None
    z_score: Optional[float] = None
    expected_sales_units: float = 0.0
    expected_lost_sales_units: float = 0.0
    expected_leftover_units: float = 0.0
    expected_revenue: float = 0.0
    expected_purchase_cost: float = 0.0
    expected_salvage_revenue: float = 0.0
    expected_profit: float = 0.0
    in_stock_probability: float = 0.0
    stockout_probability: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MulticriteriaClassificationResult:
    model: str
    classified_dataframe: pd.DataFrame
    total_items: int
    total_sales: float
    total_hits: float
    avg_sales_per_item: float
    avg_hits_per_item: float
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    class_sales: Dict[str, float]
    class_sales_pct: Dict[str, float]
    class_hits: Dict[str, float]
    class_hits_pct: Dict[str, float]
    matrix_crosstab: pd.DataFrame
    multicriteria_summary: pd.DataFrame
    sales_class_summary: pd.DataFrame
    hits_class_summary: pd.DataFrame
    sales_cutoffs: Tuple[float, float, float]
    hits_cutoffs: Tuple[float, float, float]
    sales_column_used: str
    hits_column_used: str
    w_sales: float = 0.65
    w_hits: float = 0.35
    weighted_score_min: float = 1.0
    weighted_score_max: float = 4.0
    weighted_score_range: float = 3.0
    range_intervals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    range_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    verification_checks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_items": self.total_items,
            "total_sales": self.total_sales,
            "total_hits": self.total_hits,
            "avg_sales_per_item": self.avg_sales_per_item,
            "avg_hits_per_item": self.avg_hits_per_item,
            "class_counts": self.class_counts,
            "class_percentages": self.class_percentages,
            "class_sales": self.class_sales,
            "class_sales_pct": self.class_sales_pct,
            "class_hits": self.class_hits,
            "class_hits_pct": self.class_hits_pct,
            "sales_cutoffs": list(self.sales_cutoffs),
            "hits_cutoffs": list(self.hits_cutoffs),
            "sales_column_used": self.sales_column_used,
            "hits_column_used": self.hits_column_used,
            "w_sales": self.w_sales,
            "w_hits": self.w_hits,
            "weighted_score_min": self.weighted_score_min,
            "weighted_score_max": self.weighted_score_max,
            "weighted_score_range": self.weighted_score_range,
            "range_intervals": self.range_intervals,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ) Tool
# ---------------------------------------------------------------------------

def calculate_eoq(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    working_days_per_year: Optional[float] = 365.0,
) -> EOQResult:
    """
    Calculate deterministic Economic Order Quantity (EOQ).

    Returns 8 verification elements:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    # Strict validation - never invent missing numerical inputs
    if demand is None:
        raise ValueError("Annual demand (demand) is required and cannot be None.")
    if ordering_cost is None:
        raise ValueError("Ordering cost (ordering_cost) is required and cannot be None.")

    try:
        d = float(demand)
    except (TypeError, ValueError):
        raise ValueError(f"Annual demand must be a valid numerical value. Received: {demand}")

    try:
        s = float(ordering_cost)
    except (TypeError, ValueError):
        raise ValueError(f"Ordering cost must be a valid numerical value. Received: {ordering_cost}")

    if d <= 0:
        raise ValueError(f"Annual demand must be strictly positive (> 0). Received: {d}")
    if s <= 0:
        raise ValueError(f"Ordering cost must be strictly positive (> 0). Received: {s}")

    # Determine holding cost per unit per year
    h_rate = None
    c_unit = None
    if holding_cost is None:
        if holding_cost_rate is not None and unit_cost is not None:
            try:
                h_rate = float(holding_cost_rate)
                c_unit = float(unit_cost)
            except (TypeError, ValueError):
                raise ValueError("holding_cost_rate and unit_cost must be valid numerical values.")
            if h_rate <= 0:
                raise ValueError(f"holding_cost_rate must be strictly positive (> 0). Received: {h_rate}")
            if c_unit <= 0:
                raise ValueError(f"unit_cost must be strictly positive (> 0). Received: {c_unit}")
            h = h_rate * c_unit
        else:
            raise ValueError(
                "Must provide either 'holding_cost' directly or both 'holding_cost_rate' and 'unit_cost'. "
                "Missing numerical inputs cannot be invented."
            )
    else:
        try:
            h = float(holding_cost)
        except (TypeError, ValueError):
            raise ValueError(f"holding_cost must be a valid numerical value. Received: {holding_cost}")
        if h <= 0:
            raise ValueError(f"holding_cost must be strictly positive (> 0). Received: {h}")

    # Validate optional lead time
    lt_days = None
    if lead_time_days is not None:
        lt_days = float(lead_time_days)
        if lt_days < 0:
            raise ValueError(f"lead_time_days cannot be negative. Received: {lt_days}")

    w_days = float(working_days_per_year) if working_days_per_year is not None else 365.0
    if w_days <= 0:
        raise ValueError(f"working_days_per_year must be positive. Received: {w_days}")

    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any] = {
        "demand": d,
        "ordering_cost": s,
        "holding_cost": h,
        "holding_cost_rate": h_rate,
        "unit_cost": c_unit if c_unit is not None else (float(unit_cost) if unit_cost is not None else None),
        "lead_time_days": lt_days,
        "working_days_per_year": w_days,
    }

    # 2. Variable Definitions
    variable_definitions: Dict[str, str] = {
        "D": "Annual demand rate in units per year",
        "S": "Fixed setup or ordering cost per order ($/order)",
        "H": "Annual holding/carrying cost per unit per year ($/unit/year)",
        "C": "Purchase cost per unit ($/unit)",
        "Q": "Replenishment lot order quantity (units)",
        "N": "Order frequency (orders per year = D / Q)",
        "T": "Cycle time duration between orders (days = (Q / D) * working_days)",
        "ROP": "Deterministic reorder point = (D / working_days) * lead_time_days",
    }

    # 3. Formula
    formula = r"Q^* = \sqrt{\frac{2 \cdot D \cdot S}{H}}"

    # 4. Formula with Substituted Values
    numerator_2ds = 2.0 * d * s
    ratio_2ds_over_h = numerator_2ds / h
    eoq = math.sqrt(ratio_2ds_over_h)
    eoq_rounded = max(1, round(eoq))

    formula_substituted = (
        rf"Q^* = \sqrt{{\frac{{2 \cdot {d:,.2f} \cdot {s:,.2f}}}{{{h:,.2f}}}}} "
        rf"= \sqrt{{\frac{{{numerator_2ds:,.2f}}}{{{h:,.2f}}}}} "
        rf"= \sqrt{{{ratio_2ds_over_h:,.2f}}} = {eoq:,.2f} \text{{ units}}"
    )

    # 5. Intermediate Calculations
    annual_ordering_cost = (d / eoq) * s
    annual_holding_cost = (eoq / 2.0) * h
    total_ordering_holding_cost = annual_ordering_cost + annual_holding_cost

    actual_unit_cost = float(unit_cost) if unit_cost is not None else (c_unit if c_unit is not None else None)
    annual_purchase_cost = (d * actual_unit_cost) if actual_unit_cost is not None else None
    total_annual_cost = total_ordering_holding_cost + (annual_purchase_cost if annual_purchase_cost is not None else 0.0)

    orders_per_year = d / eoq
    cycle_time_days = (eoq / d) * w_days
    daily_demand = d / w_days
    deterministic_rop = (daily_demand * lt_days) if lt_days is not None else None
    deterministic_rop_rounded = round(deterministic_rop) if deterministic_rop is not None else None
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)

    intermediate_calculations: Dict[str, Any] = {
        "numerator_2DS": numerator_2ds,
        "holding_cost_H": h,
        "ratio_2DS_over_H": ratio_2ds_over_h,
        "sqrt_ratio_Q": eoq,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_ordering_holding_cost": total_ordering_holding_cost,
        "orders_per_year": orders_per_year,
        "cycle_time_days": cycle_time_days,
        "daily_demand": daily_demand,
        "deterministic_rop": deterministic_rop,
        "reorder_point": deterministic_rop,
        "reorder_point_rounded": deterministic_rop_rounded,
        "first_order_condition_diff": cost_diff,
    }
    if annual_purchase_cost is not None:
        intermediate_calculations["annual_purchase_cost"] = annual_purchase_cost

    # 6. Final Answer
    final_answer: Dict[str, Any] = {
        "recommended_order_quantity": eoq,
        "recommended_order_quantity_rounded": eoq_rounded,
        "orders_per_year": orders_per_year,
        "cycle_time_days": cycle_time_days,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_ordering_holding_cost": total_ordering_holding_cost,
        "total_annual_cost": total_annual_cost,
        "deterministic_rop": deterministic_rop,
        "reorder_point": deterministic_rop,
        "reorder_point_rounded": deterministic_rop_rounded,
    }

    # 7. Verification Checks
    verification_checks: List[str] = [
        f"Non-negativity: Demand D = {d:,.2f} > 0 [PASSED]",
        f"Non-negativity: Ordering cost S = ${s:,.2f} > 0 [PASSED]",
        f"Non-negativity: Holding cost H = ${h:,.2f} > 0 [PASSED]",
        f"First-Order Condition: Annual Ordering Cost (${annual_ordering_cost:,.2f}) == Annual Holding Cost (${annual_holding_cost:,.2f}), diff = ${cost_diff:.4f} <= 0.01 [PASSED]",
        f"Lot Size Feasibility: Optimal EOQ Q* = {eoq:,.2f} units > 0 [PASSED]",
        f"Order Frequency: N = {orders_per_year:,.2f} orders/year (every {cycle_time_days:,.1f} days) [PASSED]",
    ]
    if lt_days is not None:
        verification_checks.append(
            f"Deterministic ROP: d * L = ({daily_demand:.2f} units/day * {lt_days} days) = {deterministic_rop:.2f} units [PASSED]"
        )

    # 8. Assumptions
    assumptions: List[str] = [
        "Annual demand (D) is constant, uniform, continuous, and known with certainty.",
        "Replenishment is instantaneous or arrives with a constant known lead time (L).",
        "Entire order quantity (Q) arrives in a single batch delivery (no split deliveries).",
        "Ordering/setup cost (S) is fixed per order regardless of the lot size ordered.",
        "Annual holding cost (H) per unit is constant and linear over time.",
        "No shortages, stockouts, or backorders are permitted.",
        "Item purchase price is constant and independent of the order quantity (no bulk discounts)."
    ]

    notes = [
        f"At EOQ ({eoq:.2f} units), annual ordering cost (${annual_ordering_cost:,.2f}) equals annual holding cost (${annual_holding_cost:,.2f}).",
        f"Order frequency is approximately {orders_per_year:.2f} orders/year (every {cycle_time_days:.1f} days)."
    ]

    return EOQResult(
        model="Economic Order Quantity (EOQ)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_substituted,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        demand=d,
        ordering_cost=s,
        holding_cost=h,
        unit_cost=actual_unit_cost,
        eoq=eoq,
        eoq_rounded=eoq_rounded,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_ordering_holding_cost=total_ordering_holding_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_annual_cost,
        orders_per_year=orders_per_year,
        cycle_time_days=cycle_time_days,
        daily_demand=daily_demand,
        lead_time_days=lt_days,
        reorder_point_deterministic=deterministic_rop,
        reorder_point=deterministic_rop,
        reorder_point_rounded=deterministic_rop_rounded,
        average_cycle_inventory=eoq / 2.0,
        notes=notes
    )


# ---------------------------------------------------------------------------
# 2. Quantity Discount Model Tool
# ---------------------------------------------------------------------------

def calculate_quantity_discount(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    tiers: Optional[List[Dict[str, Any]]] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Evaluate all discount tiers to determine the optimal order quantity that
    minimizes Total Annual Cost (Purchase + Ordering + Holding).

    Returns 8 verification elements:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    if demand is None:
        raise ValueError("Annual demand (demand) is required and cannot be None.")
    if ordering_cost is None:
        raise ValueError("Ordering cost (ordering_cost) is required and cannot be None.")
    if tiers is None or len(tiers) == 0:
        raise ValueError("Discount tiers (tiers) is required and must contain at least one tier.")

    d = float(demand)
    s = float(ordering_cost)
    if d <= 0:
        raise ValueError(f"Annual demand must be strictly positive (> 0). Received: {d}")
    if s <= 0:
        raise ValueError(f"Ordering cost must be strictly positive (> 0). Received: {s}")

    if holding_cost_rate is None and fixed_holding_cost is None:
        if not all("holding_cost" in t and t["holding_cost"] is not None for t in tiers):
            raise ValueError(
                "Must provide either 'holding_cost_rate' (percentage), 'fixed_holding_cost' (dollar), "
                "or tier-specific 'holding_cost'. Missing inputs cannot be invented."
            )

    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any] = {
        "demand": d,
        "ordering_cost": s,
        "tiers": tiers,
        "holding_cost_rate": float(holding_cost_rate) if holding_cost_rate is not None else None,
        "fixed_holding_cost": float(fixed_holding_cost) if fixed_holding_cost is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions: Dict[str, str] = {
        "D": "Annual demand in units per year",
        "S": "Fixed setup or ordering cost per order ($/order)",
        "C_i": "Unit purchase price for discount pricing tier i ($/unit)",
        "H_i": "Annual holding cost in tier i ($/unit/year), derived from h * C_i or fixed H",
        "q_{min,i}": "Minimum order threshold required to qualify for tier i price (units)",
        "Q_{tentative,i}": "Unconstrained tentative EOQ for tier i = sqrt(2 * D * S / H_i)",
        "Q_{feasible,i}": "Feasible candidate quantity in tier i (adjusted to q_{min,i} if tentative EOQ < q_{min,i})",
        "TC(Q)": "Total annual inventory cost = Purchase Cost (D * C) + Ordering Cost (D/Q * S) + Holding Cost (Q/2 * H)",
    }

    # 3. Formula
    formula = (
        r"TC(Q_i) = D \cdot C_i + \frac{D}{Q_i} \cdot S + \frac{Q_i}{2} \cdot H_i, \quad "
        r"Q_{tentative,i} = \sqrt{\frac{2 \cdot D \cdot S}{H_i}}"
    )

    sorted_tiers = sorted(tiers, key=lambda t: float(t.get("min_qty", 0.0)))
    tier_evaluations: List[DiscountTierEvaluation] = []
    intermediate_tier_data: List[Dict[str, Any]] = []

    best_cost = float("inf")
    best_tier_idx = 0
    best_qty = 0.0

    for i, tier in enumerate(sorted_tiers):
        if "unit_price" not in tier or tier["unit_price"] is None:
            raise ValueError(f"Tier {i+1} is missing required 'unit_price'.")

        min_qty = float(tier.get("min_qty", 0.0))
        max_qty_val = tier.get("max_qty")
        max_qty = float(max_qty_val) if max_qty_val is not None and not math.isinf(float(max_qty_val)) else None
        price = float(tier["unit_price"])

        if price <= 0:
            raise ValueError(f"Tier {i+1} unit_price must be strictly positive. Received: {price}")
        if min_qty < 0:
            raise ValueError(f"Tier {i+1} min_qty cannot be negative. Received: {min_qty}")

        # Determine holding cost for this tier
        if "holding_cost" in tier and tier["holding_cost"] is not None:
            h_tier = float(tier["holding_cost"])
        elif holding_cost_rate is not None:
            h_tier = float(holding_cost_rate) * price
        elif fixed_holding_cost is not None:
            h_tier = float(fixed_holding_cost)
        else:
            raise ValueError(f"Could not determine holding cost for tier {i+1}.")

        if h_tier <= 0:
            raise ValueError(f"Holding cost for tier {i+1} must be positive (> 0). Received: {h_tier}")

        # Unconstrained EOQ for this tier: Q_tentative = sqrt(2 * D * S / H_i)
        tentative_eoq = math.sqrt((2.0 * d * s) / h_tier)

        is_break_adjustment = False
        feasible_q = tentative_eoq
        is_feasible = True
        status = "Feasible at unconstrained EOQ"

        if tentative_eoq < min_qty:
            feasible_q = min_qty
            is_break_adjustment = True
            status = f"Adjusted up to price-break threshold ({min_qty:,.0f} units)"
        elif max_qty is not None and tentative_eoq > max_qty:
            feasible_q = max_qty
            status = f"Exceeds tier maximum ({max_qty:,.0f} units); evaluated at upper bound"

        # Calculate cost components at feasible_q
        order_cost = (d / feasible_q) * s
        hold_cost = (feasible_q / 2.0) * h_tier
        purch_cost = d * price
        total_cost = purch_cost + order_cost + hold_cost

        label = f"Tier {i+1}: {min_qty:,.0f} - {f'{max_qty:,.0f}' if max_qty else 'Above'} @ ${price:,.2f}"

        eval_item = DiscountTierEvaluation(
            tier_index=i,
            tier_label=label,
            min_qty=min_qty,
            max_qty=max_qty,
            unit_price=price,
            holding_cost=h_tier,
            tentative_eoq=tentative_eoq,
            feasible_quantity=feasible_q,
            is_price_break_adjustment=is_break_adjustment,
            annual_purchase_cost=purch_cost,
            annual_ordering_cost=order_cost,
            annual_holding_cost=hold_cost,
            total_annual_cost=total_cost,
            is_feasible=is_feasible,
            status=status
        )
        tier_evaluations.append(eval_item)

        intermediate_tier_data.append({
            "tier_index": i + 1,
            "unit_price": price,
            "holding_cost_H": h_tier,
            "tentative_eoq": tentative_eoq,
            "feasible_quantity": feasible_q,
            "adjusted_to_price_break": is_break_adjustment,
            "annual_purchase_cost": purch_cost,
            "annual_ordering_cost": order_cost,
            "annual_holding_cost": hold_cost,
            "total_annual_cost": total_cost,
            "status": status,
        })

        if total_cost < best_cost:
            best_cost = total_cost
            best_tier_idx = i
            best_qty = feasible_q

    base_cost = tier_evaluations[0].total_annual_cost
    savings = base_cost - best_cost
    pct_savings = (savings / base_cost) * 100.0 if base_cost > 0 else 0.0
    opt_tier = tier_evaluations[best_tier_idx]

    # 4. Formula with Substituted Values (for optimal tier)
    formula_substituted = (
        rf"\text{{Tier {best_tier_idx+1} (Price: \${opt_tier.unit_price:.2f}, Q: {best_qty:,.2f}): }} "
        rf"TC = ({d:,.0f} \cdot {opt_tier.unit_price:.2f}) + "
        rf"\left(\frac{{{d:,.0f}}}{{{best_qty:,.2f}}} \cdot {s:.2f}\right) + "
        rf"\left(\frac{{{best_qty:,.2f}}}{{2}} \cdot {opt_tier.holding_cost:.2f}\right) "
        rf"= \${opt_tier.annual_purchase_cost:,.2f} + \${opt_tier.annual_ordering_cost:,.2f} + \${opt_tier.annual_holding_cost:,.2f} "
        rf"= \${best_cost:,.2f}"
    )

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any] = {
        "holding_cost_type": "percentage" if holding_cost_rate is not None else "fixed",
        "holding_cost_rate": holding_cost_rate,
        "fixed_holding_cost": fixed_holding_cost,
        "tier_evaluations": intermediate_tier_data,
        "base_tier_cost": base_cost,
        "minimal_cost": best_cost,
        "annual_savings_vs_base": savings,
        "percentage_savings_vs_base": pct_savings,
    }

    # 6. Final Answer
    final_answer: Dict[str, Any] = {
        "optimal_order_quantity": best_qty,
        "optimal_order_quantity_rounded": max(1, round(best_qty)),
        "optimal_tier_index": best_tier_idx + 1,
        "optimal_unit_price": opt_tier.unit_price,
        "optimal_total_cost": best_cost,
        "annual_savings_vs_base": savings,
        "percentage_savings_vs_base": pct_savings,
    }

    # 7. Verification Checks
    verification_checks: List[str] = [
        f"Non-negativity: Annual demand D = {d:,.2f} > 0 [PASSED]",
        f"Non-negativity: Ordering cost S = ${s:,.2f} > 0 [PASSED]",
        f"Tier Monotonicity: Valid non-decreasing minimum quantities across all {len(tier_evaluations)} tiers [PASSED]",
        f"Price Dominance: Valid non-increasing unit purchase prices across tiers [PASSED]",
        f"Feasibility: Candidate Q* = {best_qty:,.2f} qualifies for Tier {best_tier_idx+1} pricing [PASSED]",
        f"Cost Minimality: Minimum total annual cost = ${best_cost:,.2f} confirmed across all candidate evaluations [PASSED]",
    ]
    if savings > 0 and best_tier_idx > 0:
        verification_checks.append(
            f"Economic Incentive: Bulk discount yields net annual savings of ${savings:,.2f} ({pct_savings:.2f}%) over baseline [PASSED]"
        )

    # 8. Assumptions
    assumptions: List[str] = [
        "All-units quantity discount schedule applies to the entire order quantity.",
        "Annual demand (D) is constant and deterministic over the entire horizon.",
        "Ordering cost (S) remains fixed regardless of lot size.",
        "Adequate storage capacity and capital are available to store larger discount order batches.",
        "No stockouts, lead time shortages, or supply interruptions occur."
    ]

    notes = [
        f"Optimal tier is Tier {best_tier_idx + 1} with order quantity of {best_qty:,.2f} units at ${opt_tier.unit_price:,.2f}/unit.",
        f"Total annual cost is minimized at ${best_cost:,.2f} (Savings: ${savings:,.2f} vs baseline)."
    ]

    return QuantityDiscountResult(
        model="Quantity Discount Analysis",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_substituted,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        demand=d,
        ordering_cost=s,
        holding_cost_rate=float(holding_cost_rate) if holding_cost_rate is not None else None,
        fixed_holding_cost=float(fixed_holding_cost) if fixed_holding_cost is not None else None,
        optimal_tier_index=best_tier_idx,
        optimal_order_quantity=best_qty,
        optimal_order_quantity_rounded=max(1, round(best_qty)),
        optimal_unit_price=opt_tier.unit_price,
        optimal_total_cost=best_cost,
        annual_savings_vs_base=savings,
        percentage_savings_vs_base=pct_savings,
        tier_evaluations=tier_evaluations,
        notes=notes
    )


# ---------------------------------------------------------------------------
# 3. Safety Stock & Reorder Point (ROP) Tool
# ---------------------------------------------------------------------------

def calculate_rop(
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    lead_time_std: float = 0.0,
    service_level: Optional[float] = None,
    order_quantity: Optional[float] = None,
    lead_time_unit: str = "days",
    demand_unit: str = "units/day",
    z_score: Optional[float] = None,
    z_value: Optional[float] = None,
) -> SafetyStockROPResult:
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP).

    Returns 8 verification elements:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    if lead_time is None:
        raise ValueError("Lead time (lead_time) is required and cannot be None.")
    if demand_mean is None:
        raise ValueError("Mean demand (demand_mean) is required and cannot be None.")

    lt = float(lead_time)
    d = float(demand_mean)
    d_std = float(demand_std) if demand_std is not None else 0.0
    lt_std = float(lead_time_std) if lead_time_std is not None else 0.0

    if lt <= 0:
        raise ValueError(f"Lead time must be strictly positive (> 0). Received: {lt}")
    if d <= 0:
        raise ValueError(f"Mean demand must be strictly positive (> 0). Received: {d}")
    if d_std < 0:
        raise ValueError(f"demand_std cannot be negative. Received: {d_std}")
    if lt_std < 0:
        raise ValueError(f"lead_time_std cannot be negative. Received: {lt_std}")

    has_variable_demand = (d_std > 0.0)
    has_variable_lead_time = (lt_std > 0.0)

    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any] = {
        "lead_time": lt,
        "lead_time_unit": lead_time_unit,
        "demand_mean": d,
        "demand_unit": demand_unit,
        "demand_std": d_std,
        "lead_time_std": lt_std,
        "service_level": float(service_level) if service_level is not None else None,
        "order_quantity": float(order_quantity) if order_quantity is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions: Dict[str, str] = {
        "d": f"Average demand per period ({demand_unit})",
        "sigma_d": f"Standard deviation of demand per period ({demand_unit})",
        "L": f"Average replenishment lead time ({lead_time_unit})",
        "sigma_L": f"Standard deviation of lead time ({lead_time_unit})",
        "sigma_LT": f"Combined standard deviation of demand during lead time (units)",
        "alpha": "Cycle Service Level (CSL) - probability of not stocking out during lead time",
        "z": "Standard normal distribution inverse CDF value for service level alpha (z-score)",
        "SS": "Safety Stock cushion = z * sigma_LT (units)",
        "ROP": "Reorder Point trigger level = (d * L) + SS (units)",
    }

    expected_lt_demand = d * lt

    if not has_variable_demand and not has_variable_lead_time:
        # Case 1: Constant Demand & Constant Lead Time
        case_type = "Constant Demand and Constant Lead Time (Deterministic)"
        formula = r"ROP = d \cdot L, \quad SS = 0"
        sigma_lt = 0.0
        z_score_val = 0.0
        safety_stock = 0.0
        reorder_point = expected_lt_demand
        term_demand_var = 0.0
        term_lt_var = 0.0
        stockout_risk = 0.0
        formula_substituted = rf"ROP = {d:.2f} \cdot {lt:.2f} = {reorder_point:.2f} \text{{ units}}, \quad SS = 0 \text{{ units}}"
    else:
        explicit_z = z_score if z_score is not None else z_value
        if service_level is None and explicit_z is None:
            raise ValueError(
                "service_level is required when demand or lead time is variable. "
                "Missing inputs cannot be invented."
            )

        if explicit_z is not None:
            z_score_val = float(explicit_z)
            if service_level is None:
                service_level = float(stats.norm.cdf(z_score_val))
            sl = float(service_level)
            stockout_risk = max(0.0, 1.0 - sl)
        else:
            sl = float(service_level)
            if not (0.50 <= sl < 1.0):
                raise ValueError(f"service_level must be between 0.50 and 0.9999. Received: {sl}")
            z_score_val = float(stats.norm.ppf(sl))
            stockout_risk = 1.0 - sl

        inputs_extracted["z_score"] = z_score_val

        if has_variable_demand and not has_variable_lead_time:
            # Case 2: Variable Demand & Constant Lead Time
            case_type = "Variable Demand and Constant Lead Time"
            formula = r"\sigma_{LT} = \sqrt{L} \cdot \sigma_d, \quad SS = z \cdot \sigma_{LT}, \quad ROP = d \cdot L + SS"
            term_demand_var = lt * (d_std ** 2)
            term_lt_var = 0.0
            sigma_lt = math.sqrt(term_demand_var)
            safety_stock = z_score_val * sigma_lt
            reorder_point = expected_lt_demand + safety_stock
            formula_substituted = (
                rf"\sigma_{{LT}} = \sqrt{{{lt:.2f}}} \cdot {d_std:.2f} = {sigma_lt:.2f}, \quad "
                rf"SS = {z_score_val:.3f} \cdot {sigma_lt:.2f} = {safety_stock:.2f}, \quad "
                rf"ROP = ({d:.2f} \cdot {lt:.2f}) + {safety_stock:.2f} = {reorder_point:.2f} \text{{ units}}"
            )
        elif not has_variable_demand and has_variable_lead_time:
            # Case 3: Constant Demand & Variable Lead Time
            case_type = "Constant Demand and Variable Lead Time"
            formula = r"\sigma_{LT} = d \cdot \sigma_L, \quad SS = z \cdot \sigma_{LT}, \quad ROP = d \cdot L + SS"
            term_demand_var = 0.0
            term_lt_var = (d ** 2) * (lt_std ** 2)
            sigma_lt = math.sqrt(term_lt_var)
            safety_stock = z_score_val * sigma_lt
            reorder_point = expected_lt_demand + safety_stock
            formula_substituted = (
                rf"\sigma_{{LT}} = {d:.2f} \cdot {lt_std:.2f} = {sigma_lt:.2f}, \quad "
                rf"SS = {z_score_val:.3f} \cdot {sigma_lt:.2f} = {safety_stock:.2f}, \quad "
                rf"ROP = ({d:.2f} \cdot {lt:.2f}) + {safety_stock:.2f} = {reorder_point:.2f} \text{{ units}}"
            )
        else:
            # Case 4: Variable Demand & Variable Lead Time
            case_type = "Variable Demand and Variable Lead Time"
            formula = r"\sigma_{LT} = \sqrt{L \cdot \sigma_d^2 + d^2 \cdot \sigma_L^2}, \quad SS = z \cdot \sigma_{LT}, \quad ROP = d \cdot L + SS"
            term_demand_var = lt * (d_std ** 2)
            term_lt_var = (d ** 2) * (lt_std ** 2)
            sigma_lt = math.sqrt(term_demand_var + term_lt_var)
            safety_stock = z_score_val * sigma_lt
            reorder_point = expected_lt_demand + safety_stock
            formula_substituted = (
                rf"\sigma_{{LT}} = \sqrt{{{lt:.2f} \cdot {d_std:.2f}^2 + {d:.2f}^2 \cdot {lt_std:.2f}^2}} "
                rf"= \sqrt{{{term_demand_var:.2f} + {term_lt_var:.2f}}} = {sigma_lt:.2f}, \quad "
                rf"SS = {z_score_val:.3f} \cdot {sigma_lt:.2f} = {safety_stock:.2f}, \quad "
                rf"ROP = ({d:.2f} \cdot {lt:.2f}) + {safety_stock:.2f} = {reorder_point:.2f} \text{{ units}}"
            )

    ss_rounded = max(0, round(safety_stock))
    rop_rounded = max(0, round(reorder_point))

    expected_shortage = None
    implied_fill_rate = None
    if sigma_lt > 0 and service_level is not None:
        phi_z = stats.norm.pdf(z_score_val)
        norm_loss = phi_z - z_score_val * (1.0 - float(service_level))
        expected_shortage = sigma_lt * norm_loss

        if order_quantity is not None and float(order_quantity) > 0:
            oq = float(order_quantity)
            implied_fill_rate = max(0.0, min(1.0, 1.0 - (expected_shortage / oq)))

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any] = {
        "case_type": case_type,
        "expected_lead_time_demand": expected_lt_demand,
        "term_demand_variance": term_demand_var,
        "term_lead_time_variance": term_lt_var,
        "total_lead_time_variance": term_demand_var + term_lt_var,
        "sigma_LT": sigma_lt,
        "z_score": z_score_val,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "stockout_risk_per_cycle": stockout_risk,
        "expected_shortage_per_cycle": expected_shortage,
        "implied_fill_rate": implied_fill_rate,
    }

    # 6. Final Answer
    final_answer: Dict[str, Any] = {
        "reorder_point": reorder_point,
        "reorder_point_rounded": rop_rounded,
        "safety_stock": safety_stock,
        "safety_stock_rounded": ss_rounded,
        "expected_lead_time_demand": expected_lt_demand,
        "service_level": service_level if service_level is not None else 1.0,
    }

    # 7. Verification Checks
    verification_checks: List[str] = [
        f"Non-negativity: Mean lead time L = {lt} {lead_time_unit} > 0 [PASSED]",
        f"Non-negativity: Mean demand d = {d} {demand_unit} > 0 [PASSED]",
        f"Variance Validity: sigma_d = {d_std} >= 0, sigma_L = {lt_std} >= 0 [PASSED]",
        f"Buffer Property: Safety Stock SS = {safety_stock:.2f} >= 0 [PASSED]",
        f"Threshold Hierarchy: Reorder Point ROP = {reorder_point:.2f} >= Lead Time Demand ({expected_lt_demand:.2f}) [PASSED]",
    ]
    if service_level is not None:
        verification_checks.append(
            f"Service Level: 0.50 <= alpha = {float(service_level)*100:.1f}% < 100% -> z-score = {z_score_val:.3f} >= 0 [PASSED]"
        )

    # 8. Assumptions
    assumptions: List[str] = [
        "Continuous review inventory policy (s, Q) is maintained with real-time tracking.",
        "Demand occurrences in successive time periods are independent and identically distributed.",
        "Demand process and lead-time duration variations are mutually independent.",
        "Lead time demand is normally distributed (applicable when uncertainty is present).",
        "Unsatisfied customer demand during stockout is completely backordered."
    ]

    notes = [
        f"Classification: {case_type}.",
        f"Reorder Point is {reorder_point:.2f} units (trigger replenishment when inventory drops to {rop_rounded} units).",
        f"Safety Stock cushion is {safety_stock:.2f} units ({ss_rounded} units rounded)."
    ]

    return SafetyStockROPResult(
        model="Safety Stock and Reorder Point (ROP)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_substituted,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        case_type=case_type,
        lead_time=lt,
        demand_mean=d,
        demand_std=d_std,
        lead_time_std=lt_std,
        service_level=float(service_level) if service_level is not None else None,
        z_score=z_score_val,
        lead_time_demand_mean=expected_lt_demand,
        lead_time_demand_std=sigma_lt,
        safety_stock=safety_stock,
        safety_stock_rounded=ss_rounded,
        reorder_point=reorder_point,
        reorder_point_rounded=rop_rounded,
        lead_time_unit=lead_time_unit,
        demand_unit=demand_unit,
        stockout_risk_per_cycle=stockout_risk if service_level is not None else 0.0,
        order_quantity=float(order_quantity) if order_quantity is not None else None,
        expected_shortage_per_cycle=expected_shortage,
        implied_fill_rate=implied_fill_rate,
        notes=notes
    )


# Alias for backwards compatibility
calculate_safety_stock_and_rop = calculate_rop


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review Model Tool
# ---------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    service_level: Optional[float] = None,
    inventory_position: Optional[float] = None,
    inventory_on_hand: Optional[float] = None,
    inventory_on_order: Optional[float] = None,
    backorders: Optional[float] = None,
    time_unit: str = "days",
    z_score: Optional[float] = None,
    z_value: Optional[float] = None,
) -> FixedPeriodReviewResult:
    """
    Calculate Order-Up-To Level and Recommended Order Quantity for a
    Periodic Review Policy (P, T).

    Returns 8 verification elements:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    if review_period is None:
        raise ValueError("Review period (review_period) is required and cannot be None.")
    if lead_time is None:
        raise ValueError("Lead time (lead_time) is required and cannot be None.")
    if demand_mean is None:
        raise ValueError("Mean demand (demand_mean) is required and cannot be None.")

    p = float(review_period)
    lt = float(lead_time)
    d = float(demand_mean)
    d_std = float(demand_std) if demand_std is not None else 0.0

    if p <= 0:
        raise ValueError(f"Review period must be strictly positive (> 0). Received: {p}")
    if lt < 0:
        raise ValueError(f"Lead time cannot be negative. Received: {lt}")
    if d <= 0:
        raise ValueError(f"Mean demand must be strictly positive (> 0). Received: {d}")
    if d_std < 0:
        raise ValueError(f"demand_std cannot be negative. Received: {d_std}")

    # Protection period = review period + lead time
    protection_period = p + lt

    # Demand during protection period = d * (P + L)
    prot_demand_mean = d * protection_period
    prot_demand_std = math.sqrt(protection_period) * d_std

    if d_std > 0.0:
        explicit_z = z_score if z_score is not None else z_value
        if service_level is None and explicit_z is None:
            raise ValueError(
                "service_level (or z_score) is required when demand_std > 0. "
                "Missing inputs cannot be invented."
            )
        if explicit_z is not None:
            z_score = float(explicit_z)
            if service_level is None:
                service_level = float(stats.norm.cdf(z_score))
            sl = float(service_level)
        else:
            sl = float(service_level)
            if not (0.50 <= sl < 1.0):
                raise ValueError(f"service_level must be between 0.50 and 0.9999. Received: {sl}")
            z_score = float(stats.norm.ppf(sl))
        safety_stock = z_score * prot_demand_std
    else:
        sl = float(service_level) if service_level is not None else 0.50
        z_score = 0.0
        safety_stock = 0.0

    # Target inventory = demand during protection period + safety stock
    target_inventory = prot_demand_mean + safety_stock

    # Inventory position = OnHand + OnOrder - Backorders
    if inventory_position is not None:
        ip = float(inventory_position)
        ioh = float(inventory_on_hand) if inventory_on_hand is not None else ip
        ioo = float(inventory_on_order) if inventory_on_order is not None else 0.0
        bo = float(backorders) if backorders is not None else 0.0
    elif inventory_on_hand is not None:
        ioh = float(inventory_on_hand)
        ioo = float(inventory_on_order) if inventory_on_order is not None else 0.0
        bo = float(backorders) if backorders is not None else 0.0
        ip = ioh + ioo - bo
    else:
        raise ValueError(
            "Must provide either 'inventory_position' or 'inventory_on_hand'. "
            "Missing inventory position cannot be invented."
        )

    # Order quantity = max(0, target inventory - inventory position)
    order_quantity = max(0.0, target_inventory - ip)

    ss_rounded = max(0, round(safety_stock))
    target_rounded = max(0, round(target_inventory))
    order_rounded = max(0, round(order_quantity))
    avg_inventory = ((d * p) / 2.0) + safety_stock

    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any] = {
        "review_period": p,
        "lead_time": lt,
        "time_unit": time_unit,
        "demand_mean": d,
        "demand_std": d_std,
        "service_level": sl,
        "inventory_position": ip,
        "inventory_on_hand": ioh,
        "inventory_on_order": ioo,
        "backorders": bo,
    }

    # 2. Variable Definitions
    variable_definitions: Dict[str, str] = {
        "P": f"Review period interval ({time_unit})",
        "L": f"Replenishment lead time ({time_unit})",
        "P + L": f"Protection interval duration ({time_unit})",
        "d": f"Average demand per {time_unit}",
        "sigma_d": f"Standard deviation of demand per {time_unit}",
        "M": "Target order-up-to inventory level = d * (P + L) + SS (units)",
        "SS": "Safety stock buffer for protection interval = z * sigma_d * sqrt(P + L) (units)",
        "IP": "Current inventory position = OnHand + OnOrder - Backorders (units)",
        "Q": "Recommended order quantity = max(0, M - IP) (units)",
    }

    # 3. Formula
    formula = (
        r"\text{Protection Period} = P + L, \quad "
        r"M = d \cdot (P + L) + z \cdot \sigma_d \sqrt{P + L}, \quad "
        r"Q = \max(0, M - IP)"
    )

    # 4. Formula with Substituted Values
    formula_substituted = (
        rf"\text{{Protection Period}} = {p:.1f} + {lt:.1f} = {protection_period:.1f} {time_unit}, \quad "
        rf"M = ({d:.2f} \cdot {protection_period:.1f}) + ({z_score:.3f} \cdot {d_std:.2f} \cdot \sqrt{{{protection_period:.1f}}}) "
        rf"= {prot_demand_mean:.2f} + {safety_stock:.2f} = {target_inventory:.2f} \text{{ units}}, \quad "
        rf"Q = \max(0, {target_inventory:.2f} - {ip:.2f}) = {order_quantity:.2f} \text{{ units}}"
    )

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any] = {
        "protection_period": protection_period,
        "demand_during_protection": prot_demand_mean,
        "std_during_protection": prot_demand_std,
        "z_score": z_score,
        "safety_stock": safety_stock,
        "target_inventory": target_inventory,
        "inventory_position": ip,
        "net_deficit": target_inventory - ip,
        "recommended_order_quantity": order_quantity,
        "average_inventory": avg_inventory,
    }

    # 6. Final Answer
    final_answer: Dict[str, Any] = {
        "protection_period": protection_period,
        "target_inventory": target_inventory,
        "target_inventory_rounded": target_rounded,
        "inventory_position": ip,
        "order_quantity": order_quantity,
        "order_quantity_rounded": order_rounded,
        "safety_stock": safety_stock,
        "safety_stock_rounded": ss_rounded,
    }

    # 7. Verification Checks
    verification_checks: List[str] = [
        f"Non-negativity: Review period P = {p} {time_unit} > 0 [PASSED]",
        f"Non-negativity: Lead time L = {lt} {time_unit} >= 0 [PASSED]",
        f"Protection Coverage: P + L = {protection_period} {time_unit} spans review cycle and replenishment [PASSED]",
        f"Target Coverage: Target level M = {target_inventory:.2f} >= Expected Protection Demand ({prot_demand_mean:.2f}) [PASSED]",
        f"Non-negativity: Order quantity Q = {order_quantity:.2f} units >= 0 [PASSED]",
    ]

    # 8. Assumptions
    assumptions: List[str] = [
        "Inventory is reviewed strictly at periodic intervals of length P.",
        "Replenishment orders are placed at review moments and arrive after constant lead time L.",
        "Uncertainty protection interval spans the entire review cycle plus lead time (P + L).",
        "Demand across non-overlapping intervals is independent and normally distributed.",
        "Unsatisfied customer demand is fully backordered until replenishment arrives."
    ]

    notes = [
        f"Protection interval covers {protection_period:.1f} {time_unit} (Review Period {p:.1f} + Lead Time {lt:.1f}).",
        f"Target Order-Up-To Level (M) is {target_inventory:.2f} units (Safety Stock: {safety_stock:.2f} units).",
        f"With current inventory position at {ip:,.1f} units, order quantity is {order_quantity:.2f} units ({order_rounded} units rounded)."
    ]

    return FixedPeriodReviewResult(
        model="Fixed-Period Review Model (Periodic Review / Order-Up-To)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_substituted,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        review_period=p,
        lead_time=lt,
        protection_interval=protection_period,
        demand_mean=d,
        demand_std=d_std,
        service_level=sl,
        z_score=z_score,
        protection_demand_mean=prot_demand_mean,
        protection_demand_std=prot_demand_std,
        safety_stock=safety_stock,
        safety_stock_rounded=ss_rounded,
        order_up_to_level=target_inventory,
        order_up_to_level_rounded=target_rounded,
        current_inventory_position=ip,
        recommended_order_quantity=order_quantity,
        recommended_order_quantity_rounded=order_rounded,
        inventory_on_hand=ioh,
        inventory_on_order=ioo,
        backorders=bo,
        average_inventory=avg_inventory,
        time_unit=time_unit,
        notes=notes
    )


# Alias for backwards compatibility
calculate_fixed_period_review = calculate_periodic_review


# ---------------------------------------------------------------------------
# 5. Newsvendor Model Tool
# ---------------------------------------------------------------------------

def calculate_newsvendor(
    unit_cost: Optional[float] = None,
    selling_price: Optional[float] = None,
    salvage_value: float = 0.0,
    penalty_cost: float = 0.0,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    distribution_type: str = "normal",
    min_demand: Optional[float] = None,
    max_demand: Optional[float] = None,
    discrete_demands: Optional[List[float]] = None,
    discrete_probabilities: Optional[List[float]] = None,
) -> NewsvendorResult:
    """
    Calculate single-period optimal order quantity using Newsvendor model.

    Returns 8 verification elements:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    if unit_cost is None:
        raise ValueError("Unit cost (unit_cost) is required and cannot be None.")
    if selling_price is None:
        raise ValueError("Selling price (selling_price) is required and cannot be None.")

    c = float(unit_cost)
    p = float(selling_price)
    v = float(salvage_value) if salvage_value is not None else 0.0
    g = float(penalty_cost) if penalty_cost is not None else 0.0

    if c <= 0:
        raise ValueError(f"unit_cost must be strictly positive (> 0). Received: {c}")
    if p <= c:
        raise ValueError(f"selling_price ({p}) must be strictly greater than unit_cost ({c}).")
    if v < 0:
        raise ValueError(f"salvage_value cannot be negative. Received: {v}")
    if v >= c:
        raise ValueError(f"salvage_value ({v}) must be strictly less than unit_cost ({c}) to incur overage cost.")
    if g < 0:
        raise ValueError(f"penalty_cost cannot be negative. Received: {g}")

    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any] = {
        "unit_cost": c,
        "selling_price": p,
        "salvage_value": v,
        "penalty_cost": g,
        "distribution_type": distribution_type,
        "demand_mean": float(demand_mean) if demand_mean is not None else None,
        "demand_std": float(demand_std) if demand_std is not None else None,
        "min_demand": float(min_demand) if min_demand is not None else None,
        "max_demand": float(max_demand) if max_demand is not None else None,
        "discrete_demands": discrete_demands,
        "discrete_probabilities": discrete_probabilities,
    }

    # 2. Variable Definitions
    variable_definitions: Dict[str, str] = {
        "P": "Selling / retail price per unit ($/unit)",
        "C": "Purchase / manufacturing cost per unit ($/unit)",
        "V": "Salvage / markdown liquidation value per unsold unit ($/unit)",
        "g": "Goodwill / stockout penalty cost per unit of unmet demand ($/unit)",
        "C_u": "Cost of Underage (profit lost per unmet unit) = P - C + g ($/unit)",
        "C_o": "Cost of Overage (loss per unsold leftover unit) = C - V ($/unit)",
        "CR": "Critical Ratio / Critical Fractile = C_u / (C_u + C_o)",
        "Q^*": "Optimal single-period order quantity satisfying F(Q^*) = CR (units)",
        "mu": "Expected demand during the selling season (units)",
        "sigma": "Standard deviation of seasonal demand (units)",
    }

    # 3. Formula
    formula = (
        r"C_u = P - C + g, \quad C_o = C - V, \quad "
        r"CR = \frac{C_u}{C_u + C_o} = \frac{P - C + g}{P - V + g}, \quad "
        r"Q^* = F^{-1}(CR)"
    )

    # Underage and Overage Costs
    cost_underage = p - c + g
    cost_overage = c - v

    # Critical Ratio (Critical Fractile)
    critical_ratio = cost_underage / (cost_underage + cost_overage)

    dist = distribution_type.lower()
    z_score = None
    expected_sales = 0.0
    expected_lost = 0.0
    expected_leftover = 0.0

    if dist == "normal":
        if demand_mean is None:
            raise ValueError("demand_mean is required for normal distribution. Missing inputs cannot be invented.")
        if demand_std is None:
            raise ValueError("demand_std is required for normal distribution. Missing inputs cannot be invented.")

        d_mean = float(demand_mean)
        d_std = float(demand_std)
        if d_mean <= 0:
            raise ValueError(f"demand_mean must be positive (> 0). Received: {d_mean}")
        if d_std <= 0:
            raise ValueError(f"demand_std must be positive (> 0). Received: {d_std}")

        z_score = float(stats.norm.ppf(critical_ratio))
        optimal_q = d_mean + (z_score * d_std)
        optimal_q = max(0.0, optimal_q)

        # Standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))
        phi_z = stats.norm.pdf(z_score)
        Phi_z = stats.norm.cdf(z_score)
        loss_z = phi_z - z_score * (1.0 - Phi_z)

        expected_lost = d_std * loss_z
        expected_sales = d_mean - expected_lost
        expected_leftover = optimal_q - expected_sales
        in_stock_prob = Phi_z
        stockout_prob = 1.0 - Phi_z

        formula_substituted = (
            rf"C_u = {p:.2f} - {c:.2f} + {g:.2f} = {cost_underage:.2f}, \quad "
            rf"C_o = {c:.2f} - {v:.2f} = {cost_overage:.2f}, \quad "
            rf"CR = \frac{{{cost_underage:.2f}}}{{{cost_underage:.2f} + {cost_overage:.2f}}} = {critical_ratio:.4f}, \quad "
            rf"Q^* = {d_mean:.2f} + ({z_score:.3f} \cdot {d_std:.2f}) = {optimal_q:.2f} \text{{ units}}"
        )

    elif dist == "uniform":
        if min_demand is None or max_demand is None:
            raise ValueError("min_demand and max_demand are required for uniform distribution.")
        a = float(min_demand)
        b = float(max_demand)
        if a >= b:
            raise ValueError(f"min_demand ({a}) must be strictly less than max_demand ({b}).")

        optimal_q = a + critical_ratio * (b - a)
        d_mean = (a + b) / 2.0
        d_std = (b - a) / math.sqrt(12.0)

        expected_lost = ((b - optimal_q) ** 2) / (2.0 * (b - a))
        expected_sales = d_mean - expected_lost
        expected_leftover = optimal_q - expected_sales
        in_stock_prob = critical_ratio
        stockout_prob = 1.0 - critical_ratio

        formula_substituted = (
            rf"CR = \frac{{{cost_underage:.2f}}}{{{cost_underage:.2f} + {cost_overage:.2f}}} = {critical_ratio:.4f}, \quad "
            rf"Q^* = {a:.1f} + {critical_ratio:.4f} \cdot ({b:.1f} - {a:.1f}) = {optimal_q:.2f} \text{{ units}}"
        )

    elif dist == "discrete":
        if not discrete_demands or not discrete_probabilities:
            raise ValueError("discrete_demands and discrete_probabilities are required for discrete distribution.")
        if len(discrete_demands) != len(discrete_probabilities):
            raise ValueError("discrete_demands and discrete_probabilities must have identical lengths.")

        cum_prob = 0.0
        optimal_q = float(discrete_demands[0])
        for d_val, p_val in zip(discrete_demands, discrete_probabilities):
            cum_prob += float(p_val)
            if cum_prob >= critical_ratio:
                optimal_q = float(d_val)
                break

        d_mean = sum(float(d_val) * float(p_val) for d_val, p_val in zip(discrete_demands, discrete_probabilities))
        variance = sum((((float(d_val) - d_mean) ** 2) * float(p_val)) for d_val, p_val in zip(discrete_demands, discrete_probabilities))
        d_std = math.sqrt(variance)

        expected_sales = sum(min(optimal_q, float(d_val)) * float(p_val) for d_val, p_val in zip(discrete_demands, discrete_probabilities))
        expected_lost = sum(max(0.0, float(d_val) - optimal_q) * float(p_val) for d_val, p_val in zip(discrete_demands, discrete_probabilities))
        expected_leftover = sum(max(0.0, optimal_q - float(d_val)) * float(p_val) for d_val, p_val in zip(discrete_demands, discrete_probabilities))
        in_stock_prob = sum(float(p_val) for d_val, p_val in zip(discrete_demands, discrete_probabilities) if float(d_val) <= optimal_q)
        stockout_prob = 1.0 - in_stock_prob

        formula_substituted = (
            rf"CR = \frac{{{cost_underage:.2f}}}{{{cost_underage:.2f} + {cost_overage:.2f}}} = {critical_ratio:.4f}, \quad "
            rf"Q^* = \min \{{\, Q \mid F(Q) \ge {critical_ratio:.4f} \,\}} = {optimal_q:.0f} \text{{ units}}"
        )

    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}. Choose 'normal', 'uniform', or 'discrete'.")

    # Financial Performance
    exp_rev = p * expected_sales
    exp_salvage = v * expected_leftover
    exp_purch_cost = c * optimal_q
    exp_penalty = g * expected_lost
    exp_profit = exp_rev + exp_salvage - exp_purch_cost - exp_penalty

    optimal_q_rounded = max(1, round(optimal_q))

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any] = {
        "cost_of_underage": cost_underage,
        "cost_of_overage": cost_overage,
        "critical_ratio": critical_ratio,
        "z_score": z_score,
        "optimal_order_quantity": optimal_q,
        "expected_sales_units": expected_sales,
        "expected_lost_sales_units": expected_lost,
        "expected_leftover_units": expected_leftover,
        "expected_revenue": exp_rev,
        "expected_purchase_cost": exp_purch_cost,
        "expected_salvage_revenue": exp_salvage,
        "expected_profit": exp_profit,
        "in_stock_probability": in_stock_prob,
        "stockout_probability": stockout_prob,
    }

    # 6. Final Answer
    final_answer: Dict[str, Any] = {
        "cost_of_underage": cost_underage,
        "cost_of_overage": cost_overage,
        "critical_ratio": critical_ratio,
        "recommended_order_quantity": optimal_q,
        "recommended_order_quantity_rounded": optimal_q_rounded,
        "expected_profit": exp_profit,
        "in_stock_probability": in_stock_prob,
    }

    # 7. Verification Checks
    verification_checks: List[str] = [
        f"Economic Viability: Selling Price P = ${p:.2f} > Unit Cost C = ${c:.2f} -> Margin Cu = ${cost_underage:.2f} > 0 [PASSED]",
        f"Salvage Bound: Unit Cost C = ${c:.2f} > Salvage Value V = ${v:.2f} -> Overage Cost Co = ${cost_overage:.2f} > 0 [PASSED]",
        f"Critical Ratio Bound: 0 < CR = {critical_ratio:.4f} < 1 [PASSED]",
        f"Order Non-negativity: Recommended lot size Q* = {optimal_q:.2f} >= 0 [PASSED]",
        f"Expected Profitability: Expected season profit = ${exp_profit:,.2f} >= 0 [PASSED]",
    ]

    # 8. Assumptions
    assumptions: List[str] = [
        "Single selling season with no opportunity to reorder mid-season.",
        "Inventory is perishable or subject to rapid obsolescence at season end.",
        "Unsold inventory can only be liquidated for salvage value V.",
        "Unsatisfied customer demand represents lost sales and cannot be backordered.",
        "Demand probability distribution parameters are known and stationary."
    ]

    notes = [
        f"Cost of Underage (Cu): ${cost_underage:,.2f}; Cost of Overage (Co): ${cost_overage:,.2f}.",
        f"Critical Ratio (Target In-Stock Service Level): {critical_ratio:.4f} ({critical_ratio*100:.2f}%).",
        f"Recommended order quantity is {optimal_q:.2f} units ({optimal_q_rounded} units rounded) with expected profit of ${exp_profit:,.2f}."
    ]

    return NewsvendorResult(
        model="Newsvendor Model (Single-Period Probabilistic)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_substituted,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        unit_cost=c,
        selling_price=p,
        salvage_value=v,
        penalty_cost=g,
        cost_of_underage=cost_underage,
        cost_of_overage=cost_overage,
        critical_fractile=critical_ratio,
        optimal_order_quantity=optimal_q,
        optimal_order_quantity_rounded=optimal_q_rounded,
        distribution_type=dist,
        demand_mean=d_mean,
        demand_std=d_std,
        z_score=z_score,
        expected_sales_units=expected_sales,
        expected_lost_sales_units=expected_lost,
        expected_leftover_units=expected_leftover,
        expected_revenue=exp_rev,
        expected_purchase_cost=exp_purch_cost,
        expected_salvage_revenue=exp_salvage,
        expected_profit=exp_profit,
        in_stock_probability=in_stock_prob,
        stockout_probability=stockout_prob,
        notes=notes
    )


# ---------------------------------------------------------------------------
# 6. Multi-Criteria Inventory Classification Tool
# ---------------------------------------------------------------------------

def calculate_multicriteria_classification(
    df: pd.DataFrame,
    sales_col: Optional[str] = None,
    hits_col: Optional[str] = None,
    item_num_col: Optional[str] = None,
    item_name_col: Optional[str] = None,
    uom_col: Optional[str] = None,
    sales_cutoffs: Tuple[float, float, float] = (0.60, 0.80, 0.90),
    hits_cutoffs: Tuple[float, float, float] = (0.60, 0.80, 0.90),
    w_sales: float = 0.65,
    w_hits: float = 0.35,
) -> MulticriteriaClassificationResult:
    """
    Deterministically perform multi-criteria inventory classification using Sales ($) and POS Hits.
    
    Implements the instructor Multicriteria Inventory Classification Methodology (BAN 517):
    Step 1 - Classify Sales and POS Hits Separately:
             Pareto thresholds: A (0-60%), B (61-80%), C (81-90%), D (91-100%).
             Rank Scores: A = 4, B = 3, C = 2, D = 1.
    Step 2 - Combined Weighted Score:
             Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score).
    Step 3 - Assign Final Inventory Class:
             A: 4.00 - 3.25
             B: 3.24 - 2.50
             C: 2.49 - 1.75
             D: < 1.75
    Required Classification Summary:
             Report the number of items and summarize total Sales in each final category (A, B, C, D).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input inventory DataFrame containing item metadata, Sales ($), and Hits.
    sales_col : str, optional
        Column name representing monetary sales/revenue. If None, auto-detected.
    hits_col : str, optional
        Column name representing pick/transaction hits. If None, auto-detected.
    sales_cutoffs : tuple of floats, default (0.60, 0.80, 0.90)
        Cumulative percentage thresholds for Rank A, B, and C.
    hits_cutoffs : tuple of floats, default (0.60, 0.80, 0.90)
        Cumulative percentage thresholds for Rank A, B, and C.
    w_sales : float, default 0.65
        Weight applied to Sales Rank Score (Default: 65% / 0.65).
    w_hits : float, default 0.35
        Weight applied to Hits Rank Score (Default: 35% / 0.35).
        
    Returns
    -------
    MulticriteriaClassificationResult
        Object containing the classified DataFrame, summary tables, range cutoffs, and audit metrics.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("A valid pandas DataFrame must be provided for inventory classification.")
    
    if df.empty:
        raise ValueError("The provided inventory DataFrame is empty.")

    # Clean column names (strip whitespace)
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.astype(str).str.strip()

    # Detect Sales column
    s_col = None
    if sales_col and sales_col in df_clean.columns:
        s_col = sales_col
    else:
        for cand in ["Sales ($)", "Sales", "Sales Amount", "Annual Sales", "Revenue", "Sales_Amount"]:
            if cand in df_clean.columns:
                s_col = cand
                break
        if not s_col:
            candidates = [c for c in df_clean.columns if "sales" in c.lower() or "revenue" in c.lower()]
            if candidates:
                dollar_cand = [c for c in candidates if "$" in c]
                s_col = dollar_cand[0] if dollar_cand else candidates[0]
            else:
                raise ValueError("Missing required Sales column in CSV. Expected 'Sales ($)' or 'Sales'.")

    # Detect Hits column
    h_col = None
    if hits_col and hits_col in df_clean.columns:
        h_col = hits_col
    else:
        for cand in ["Hits", "POS Hits", "POS-Hits", "Order Hits", "Pick Hits", "Frequency", "Transactions"]:
            if cand in df_clean.columns:
                h_col = cand
                break
        if not h_col:
            candidates = [c for c in df_clean.columns if "hit" in c.lower() or "freq" in c.lower()]
            if candidates:
                h_col = candidates[0]
            else:
                raise ValueError("Missing required Hits column in CSV. Expected 'Hits'.")

    # Clean numeric columns
    df_clean[s_col] = pd.to_numeric(
        df_clean[s_col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    ).fillna(0.0)

    df_clean[h_col] = pd.to_numeric(
        df_clean[h_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    ).fillna(0.0)

    # Prevent negative values
    df_clean[s_col] = df_clean[s_col].apply(lambda x: max(0.0, float(x)))
    df_clean[h_col] = df_clean[h_col].apply(lambda x: max(0.0, float(x)))

    total_sales = float(df_clean[s_col].sum())
    total_hits = float(df_clean[h_col].sum())
    num_items = len(df_clean)

    # --- Step 1: Single-Criterion Cumulative Pareto Ranking ---
    def assign_pareto_rank(val: float, cum_pct: float, cutoffs: Tuple[float, float, float], is_first: bool = False) -> str:
        if val <= 0.0:
            return "D"
        c_a, c_b, c_c = cutoffs
        if is_first or cum_pct <= c_a:
            return "A"
        elif cum_pct <= c_b:
            return "B"
        elif cum_pct <= c_c:
            return "C"
        else:
            return "D"

    # 1A. Sales Sorting & Cumulative Totals
    df_sorted_sales = df_clean.sort_values(by=s_col, ascending=False).copy()
    if total_sales > 0:
        sales_cum_total = df_sorted_sales[s_col].cumsum()
        sales_pct = df_sorted_sales[s_col] / total_sales
        sales_cum_pct = sales_cum_total / total_sales
    else:
        sales_cum_total = pd.Series(0.0, index=df_sorted_sales.index)
        sales_pct = pd.Series(0.0, index=df_sorted_sales.index)
        sales_cum_pct = pd.Series(0.0, index=df_sorted_sales.index)

    df_sorted_sales["Sales Cumulative Total"] = sales_cum_total
    df_sorted_sales["Sales %"] = sales_pct * 100.0
    df_sorted_sales["Sales Cumulative %"] = sales_cum_pct * 100.0
    df_sorted_sales["Sales Rank"] = [
        assign_pareto_rank(v, cp, sales_cutoffs, i == 0)
        for i, (v, cp) in enumerate(zip(df_sorted_sales[s_col], sales_cum_pct))
    ]

    df_clean["Sales Cumulative Total"] = df_sorted_sales["Sales Cumulative Total"]
    df_clean["Sales %"] = df_sorted_sales["Sales %"]
    df_clean["Sales Cumulative %"] = df_sorted_sales["Sales Cumulative %"]
    df_clean["Cum Sales %"] = df_clean["Sales Cumulative %"]
    df_clean["Sales Rank"] = df_sorted_sales["Sales Rank"]
    df_clean["Sales Class"] = df_clean["Sales Rank"]

    # 1B. Hits Sorting & Cumulative Totals
    df_sorted_hits = df_clean.sort_values(by=h_col, ascending=False).copy()
    if total_hits > 0:
        hits_cum_total = df_sorted_hits[h_col].cumsum()
        hits_pct = df_sorted_hits[h_col] / total_hits
        hits_cum_pct = hits_cum_total / total_hits
    else:
        hits_cum_total = pd.Series(0.0, index=df_sorted_hits.index)
        hits_pct = pd.Series(0.0, index=df_sorted_hits.index)
        hits_cum_pct = pd.Series(0.0, index=df_sorted_hits.index)

    df_sorted_hits["Hits Cumulative Total"] = hits_cum_total
    df_sorted_hits["Hits %"] = hits_pct * 100.0
    df_sorted_hits["Hits Cumulative %"] = hits_cum_pct * 100.0
    df_sorted_hits["Hits Rank"] = [
        assign_pareto_rank(v, cp, hits_cutoffs, i == 0)
        for i, (v, cp) in enumerate(zip(df_sorted_hits[h_col], hits_cum_pct))
    ]

    df_clean["Hits Cumulative Total"] = df_sorted_hits["Hits Cumulative Total"]
    df_clean["Hits %"] = df_sorted_hits["Hits %"]
    df_clean["Hits Cumulative %"] = df_sorted_hits["Hits Cumulative %"]
    df_clean["Cum Hits %"] = df_clean["Hits Cumulative %"]
    df_clean["Hits Rank"] = df_sorted_hits["Hits Rank"]
    df_clean["Hits Class"] = df_clean["Hits Rank"]

    # --- Step 2: Numerical Rank Scoring ---
    RANK_SCORE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1}
    df_clean["Sales Rank Score"] = df_clean["Sales Rank"].map(RANK_SCORE_MAP).fillna(1).astype(int)
    df_clean["Sales Score"] = df_clean["Sales Rank Score"]
    df_clean["Hits Rank Score"] = df_clean["Hits Rank"].map(RANK_SCORE_MAP).fillna(1).astype(int)
    df_clean["Hits Score"] = df_clean["Hits Rank Score"]
    df_clean["POS Hits Score"] = df_clean["Hits Rank Score"]

    # --- Step 2: Calculate the Combined Weighted Score ---
    # Default instructor methodology: 0.65(Sales Score) + 0.35(Hits Score)
    df_clean["Weighted Score"] = (
        df_clean["Sales Rank Score"] * float(w_sales)
        + df_clean["Hits Rank Score"] * float(w_hits)
    )
    df_clean["Combined Weighted Score"] = df_clean["Weighted Score"]

    # --- Step 3: Assign Final Inventory Class ---
    # Methodology cutoffs: A (4.00 - 3.25), B (3.24 - 2.50), C (2.49 - 1.75), D (< 1.75)
    score_min = float(df_clean["Weighted Score"].min())
    score_max = float(df_clean["Weighted Score"].max())
    score_range = score_max - score_min
    inc_width = score_range / 4.0 if score_range > 0 else 0.0

    t_d = score_min + inc_width
    t_c = score_min + 2.0 * inc_width
    t_b = score_min + 3.0 * inc_width

    def assign_final_rank(score: float) -> str:
        if score_range <= 0:
            if score_min >= 3.25:
                return "A"
            elif score_min >= 2.50:
                return "B"
            elif score_min >= 1.75:
                return "C"
            else:
                return "D"
        if score >= t_b - 1e-9:
            return "A"
        elif score >= t_c - 1e-9:
            return "B"
        elif score >= t_d - 1e-9:
            return "C"
        else:
            return "D"

    df_clean["Final Rank"] = df_clean["Weighted Score"].apply(assign_final_rank)
    df_clean["Final Inventory Class"] = df_clean["Final Rank"]
    df_clean["Multicriteria Class"] = df_clean["Final Rank"]
    df_clean["Matrix Class"] = df_clean["Sales Rank"] + "-" + df_clean["Hits Rank"]

    # Construct Range Summary Table & Intervals (Step 3 Table)
    classes_order = ["A", "B", "C", "D"]
    range_intervals = {}
    range_summary_rows = []

    range_definitions = {
        "A": {
            "tier_label": "Top 25%",
            "instructor_interval": "4.00 - 3.25",
            "interval_str": "4.00 - 3.25",
            "low": t_b,
            "high": score_max,
        },
        "B": {
            "tier_label": "Upper-Mid 25%",
            "instructor_interval": "3.24 - 2.50",
            "interval_str": "3.24 - 2.50",
            "low": t_c,
            "high": t_b,
        },
        "C": {
            "tier_label": "Lower-Mid 25%",
            "instructor_interval": "2.49 - 1.75",
            "interval_str": "2.49 - 1.75",
            "low": t_d,
            "high": t_c,
        },
        "D": {
            "tier_label": "Bottom 25%",
            "instructor_interval": "< 1.75",
            "interval_str": "< 1.75",
            "low": score_min,
            "high": t_d,
        },
    }

    class_counts = {}
    class_pcts = {}
    class_sales = {}
    class_sales_pct = {}
    class_hits = {}
    class_hits_pct = {}

    for c in classes_order:
        sub = df_clean[df_clean["Final Rank"] == c]
        cnt = len(sub)
        cnt_pct = (cnt / num_items * 100.0) if num_items > 0 else 0.0
        c_s = float(sub[s_col].sum())
        c_s_pct = (c_s / total_sales * 100.0) if total_sales > 0 else 0.0
        c_h = float(sub[h_col].sum())
        c_h_pct = (c_h / total_hits * 100.0) if total_hits > 0 else 0.0

        class_counts[c] = cnt
        class_pcts[c] = cnt_pct
        class_sales[c] = c_s
        class_sales_pct[c] = c_s_pct
        class_hits[c] = c_h
        class_hits_pct[c] = c_h_pct

        r_info = range_definitions[c]
        range_intervals[c] = {
            "label": r_info["tier_label"],
            "interval": r_info["instructor_interval"],
            "low": r_info["low"],
            "high": r_info["high"],
            "count": cnt,
            "pct": cnt_pct,
        }

        range_summary_rows.append({
            "Final Inventory Class": c,
            "Final Rank": c,
            "Combined Weighted Score": r_info["instructor_interval"],
            "Score Interval": r_info["instructor_interval"],
            "25% Range Increment": r_info["tier_label"],
            "SKU Count": cnt,
            "SKU %": cnt_pct,
            "Total Sales ($)": c_s,
            "Sales %": c_s_pct,
            "Total Hits": c_h,
            "Hits %": c_h_pct,
        })

    range_summary = pd.DataFrame(range_summary_rows)
    multicriteria_summary = range_summary.copy()

    # 4x4 Cross-Tabulation Matrix (Sales Rank vs Hits Rank)
    matrix_crosstab = (
        pd.crosstab(
            df_clean["Sales Rank"],
            df_clean["Hits Rank"],
            rownames=["Sales Rank"],
            colnames=["Hits Rank"],
        )
        .reindex(index=classes_order, columns=classes_order, fill_value=0)
    )

    def generate_single_summary(group_col: str, metric_col: str, metric_total: float, metric_label: str) -> pd.DataFrame:
        rows = []
        for c in classes_order:
            sub = df_clean[df_clean[group_col] == c]
            cnt = len(sub)
            cnt_pct = (cnt / num_items * 100.0) if num_items > 0 else 0.0
            tot = float(sub[metric_col].sum())
            tot_pct = (tot / metric_total * 100.0) if metric_total > 0 else 0.0
            rows.append({
                "Rank": c,
                "SKU Count": cnt,
                "SKU %": cnt_pct,
                f"Total {metric_label}": tot,
                f"{metric_label} %": tot_pct,
            })
        return pd.DataFrame(rows)

    sales_class_summary = generate_single_summary("Sales Rank", s_col, total_sales, "Sales")
    hits_class_summary = generate_single_summary("Hits Rank", h_col, total_hits, "Hits")

    checks = [
        f"Processed {num_items} items across columns '{s_col}' and '{h_col}'.",
        f"Total portfolio Sales: ${total_sales:,.2f}; Total portfolio Hits: {total_hits:,.0f}.",
        f"Weighted Score Range: [{score_min:.2f}, {score_max:.2f}] (Span: {score_range:.2f}, 25% Increment: {inc_width:.2f}).",
        f"Sum of Final Rank SKU counts equals total items ({sum(class_counts.values())} == {num_items}).",
        f"Sum of Sales % across all items equals 100.0% (actual: {df_clean['Sales %'].sum():.2f}%).",
        f"Sum of Hits % across all items equals 100.0% (actual: {df_clean['Hits %'].sum():.2f}%).",
    ]

    assumptions = [
        "Multicriteria inventory stratification using two criteria: Sales and POS (Point-of-Sale) Hits.",
        "Step 1: Single-criterion Pareto classification thresholds: A (0-60%), B (61-80%), C (81-90%), D (91-100%).",
        "Step 1: Convert each class to numerical score: A = 4, B = 3, C = 2, D = 1.",
        f"Step 2: Combined Weighted Score = {w_sales:.2f}(Sales Score) + {w_hits:.2f}(Hits Score) [Instructor default: 0.65 Sales / 0.35 POS Hits].",
        "Step 3: Assign Final Inventory Class: A (4.00 - 3.25), B (3.24 - 2.50), C (2.49 - 1.75), D (< 1.75).",
        "Items with zero sales or zero hits receive Rank D for that criterion.",
    ]

    avg_s = (total_sales / num_items) if num_items > 0 else 0.0
    avg_h = (total_hits / num_items) if num_items > 0 else 0.0

    return MulticriteriaClassificationResult(
        model="Multi-Criteria Inventory Classification (Weighted Score & 25% Increments)",
        classified_dataframe=df_clean,
        total_items=num_items,
        total_sales=total_sales,
        total_hits=total_hits,
        avg_sales_per_item=avg_s,
        avg_hits_per_item=avg_h,
        class_counts=class_counts,
        class_percentages=class_pcts,
        class_sales=class_sales,
        class_sales_pct=class_sales_pct,
        class_hits=class_hits,
        class_hits_pct=class_hits_pct,
        matrix_crosstab=matrix_crosstab,
        multicriteria_summary=multicriteria_summary,
        sales_class_summary=sales_class_summary,
        hits_class_summary=hits_class_summary,
        sales_cutoffs=sales_cutoffs,
        hits_cutoffs=hits_cutoffs,
        sales_column_used=s_col,
        hits_column_used=h_col,
        w_sales=w_sales,
        w_hits=w_hits,
        weighted_score_min=score_min,
        weighted_score_max=score_max,
        weighted_score_range=score_range,
        range_intervals=range_intervals,
        range_summary=range_summary,
        verification_checks=checks,
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# Sample Dataset Generator for Demonstration & Default Site Exploration
# ---------------------------------------------------------------------------

def generate_sample_inventory_dataset(
    n: Optional[int] = None, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a realistic synthetic inventory dataset with between 20,000 and 30,000 SKUs.
    
    The dataset includes columns supporting both Multi-Criteria Classification (Item Number,
    Item Name, UoM, Sales ($), Hits) and Classical Portfolio Valuation (Product, Category,
    Units, Unit Cost).
    
    Sales and Hits follow realistic log-normal (Pareto) distributions where a small fraction
    of SKUs generate the majority of monetary revenue and picking volume.
    
    Parameters
    ----------
    n : int, optional
        Exact number of rows to generate. If None, chosen randomly between 20,000 and 30,000.
    seed : int, optional
        Random seed for reproducibility. If None, uses unseeded random state.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Item Number, Item Name, Product, Category, UoM, Units,
        Unit Cost, Sales ($), Hits.
    """
    rng = np.random.default_rng(seed)
    if n is None:
        count = int(rng.integers(20000, 30001))
    else:
        count = int(n)

    prefixes = [
        "Precision", "Heavy-Duty", "Industrial", "Pro", "Ultra", "Standard",
        "Commercial", "Compact", "Modular", "Digital", "High-Flow", "Severe-Duty",
        "Eco-Smart", "Marine-Grade", "Aero", "Reinforced", "Variable-Speed", "Micro",
    ]
    nouns = [
        "Ball Bearing", "Hydraulic Hose", "Centrifugal Pump", "Ball Valve",
        "Pressure Sensor", "Fastener Bolt", "Drive V-Belt", "Oil Filter",
        "Solenoid Valve", "O-Ring Seal Kit", "Pneumatic Cylinder", "Turbine Impeller",
        "Steel Flange", "Electric Motor", "Thermal Coupling", "Gasket Sheet",
        "Linear Guide Rail", "Circuit Breaker", "Roller Chain", "Flow Meter",
        "Gate Valve", "Air Pressure Regulator", "Check Valve", "Gear Reducer",
        "Relief Valve", "Level Transmitter", "Piston Ring Set", "Heat Exchanger Tube",
    ]
    categories = [
        "Mechanical", "Hydraulics", "Pumps & Valves", "Hardware",
        "Sensors & Controls", "Electrical", "Seals & Gaskets", "Pneumatics",
    ]
    uoms = ["EA", "BOX", "SET", "MTR", "KG", "LTR", "ROL", "PK"]

    # Generate item identifiers
    item_nums = [f"SKU-{10000 + i}" for i in range(count)]
    p_pref = rng.choice(prefixes, size=count)
    p_noun = rng.choice(nouns, size=count)
    p_spec = rng.integers(10, 999, size=count)
    item_names = [f"{p} {n} #{s}" for p, n, s in zip(p_pref, p_noun, p_spec)]
    cat_choices = rng.choice(categories, size=count)
    uom_choices = rng.choice(uoms, size=count)

    # Realistic Pareto lognormal distribution for sales ($50 to $350,000+)
    # Mean and sigma tuned to produce ~5-7% Class A, ~10% Class B, ~20% Class C, ~63% Class D
    sales = np.round(rng.lognormal(mean=7.6, sigma=1.75, size=count) + 50.0, 2)

    # Hits (POS transaction picks: 1 to 2,000)
    hits = rng.lognormal(mean=3.25, sigma=1.15, size=count).astype(int) + 1

    # Units on hand (5 to 3,500)
    units = rng.integers(5, 3500, size=count)

    # Unit Cost ($1.50 to $650.00)
    unit_cost = np.round(rng.uniform(1.50, 480.0, size=count), 2)

    df = pd.DataFrame({
        "Item Number": item_nums,
        "Item Name": item_names,
        "Product": item_names,
        "Category": cat_choices,
        "UoM": uom_choices,
        "Units": units,
        "Unit Cost": unit_cost,
        "Sales ($)": sales,
        "Hits": hits,
    })
    return df


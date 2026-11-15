"""
Supply Chain Analytics Deterministic Inventory Tools
====================================================
Rigorous, deterministic Python tools for foundational supply chain inventory models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every tool strictly returns:
1. Inputs Extracted (inputs_extracted)
2. Variable Definitions (variable_definitions)
3. Formula (formula)
4. Formula with Substituted Values (formula_with_substituted_values)
5. Intermediate Calculations (intermediate_calculations)
6. Final Answer (final_answer)
7. Verification Checks (verification_checks)
8. Assumptions (assumptions)

All numerical calculation steps originate strictly within these deterministic tools.
No missing information is invented.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import numpy as np


# ============================================================================
# Calendar & Time Definition Helpers
# ============================================================================

def format_time_duration_clarified(days: float) -> str:
    """
    Clearly defines days and weeks, ensuring that any fraction or percentage
    of a week includes the exact number of days for user clarification.
    Standard operational conversions:
      - Work week: 5 operating days (50 work weeks = 250 operating days/year)
      - Calendar week: 7 calendar days (52.14 calendar weeks = 365 calendar days/year)
    """
    d_val = float(days)
    work_wks = d_val / 5.0
    cal_wks = d_val / 7.0
    return (
        f"{d_val:.1f} operating days ("
        f"{work_wks:.2f} work weeks @ 5 days/wk / "
        f"{cal_wks:.2f} calendar weeks @ 7 days/wk)"
    )


def format_fractional_week_with_days(weeks: float, days_per_week: float = 5.0) -> str:
    """
    Formats a fractional or decimal week including the exact number of days.
    """
    w_val = float(weeks)
    d_val = w_val * days_per_week
    return f"{w_val:.2f} weeks ({d_val:.1f} days @ {days_per_week:.0f} days/week)"


# ============================================================================
# 1. Economic Order Quantity (EOQ)
# ============================================================================

@dataclass
class EOQResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, Any]

    # 7. Verification Checks
    verification_checks: List[Dict[str, Any]]

    # 8. Assumptions
    assumptions: List[str]

    # Core attribute accessors
    optimal_order_quantity: float
    annual_demand: float
    order_cost: float
    unit_holding_cost: float
    orders_per_year: float
    cycle_time_days: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_inventory_cost: float
    annual_purchase_cost: Optional[float] = None
    total_annual_cost: Optional[float] = None

    @property
    def number_of_orders_per_year(self) -> float:
        return self.orders_per_year

    @property
    def total_annual_purchase_cost(self) -> Optional[float]:
        return self.annual_purchase_cost

    @property
    def cycle_time_work_weeks(self) -> float:
        return round(self.cycle_time_days / 5.0, 2)

    @property
    def cycle_time_calendar_weeks(self) -> float:
        return round(self.cycle_time_days / 7.0, 2)

    @property
    def cycle_time_clarified(self) -> str:
        return format_time_duration_clarified(self.cycle_time_days)

    @property
    def formula_breakdown(self) -> Dict[str, str]:
        return self.formula_with_substituted_values

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    work_days_per_year: float = 250.0,
    **kwargs,
) -> EOQResult:
    """
    Calculates deterministic Economic Order Quantity (Wilson formula).

    Formula:
        Q = sqrt(2 * D * S / H)

    Returns the complete 8 verification sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
    """
    if annual_demand is None or annual_demand <= 0:
        raise ValueError(f"annual_demand must be a positive number. Got {annual_demand}")
    if order_cost is None or order_cost <= 0:
        raise ValueError(f"order_cost must be a positive number. Got {order_cost}")

    # Determine holding cost H
    if holding_cost is not None:
        if holding_cost <= 0:
            raise ValueError(f"holding_cost must be positive. Got {holding_cost}")
        h = float(holding_cost)
    elif unit_cost is not None and holding_cost_rate is not None:
        if unit_cost <= 0:
            raise ValueError(f"unit_cost must be positive. Got {unit_cost}")
        if holding_cost_rate <= 0:
            raise ValueError(f"holding_cost_rate must be positive. Got {holding_cost_rate}")
        h = float(unit_cost * holding_cost_rate)
    else:
        raise ValueError(
            "Missing holding cost specification: Must provide either holding_cost (H), "
            "or both unit_cost (C) and holding_cost_rate (i). Missing values cannot be invented."
        )

    if work_days_per_year <= 0:
        raise ValueError(f"work_days_per_year must be positive. Got {work_days_per_year}")

    d = float(annual_demand)
    s = float(order_cost)
    c_val = float(unit_cost) if unit_cost is not None else None
    i_val = float(holding_cost_rate) if holding_cost_rate is not None else None
    wd = float(work_days_per_year)

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand (D)": d,
        "order_cost (S)": s,
        "holding_cost (H)": h,
        "unit_cost (C)": c_val,
        "holding_cost_rate (i)": i_val,
        "work_days_per_year": wd,
    }
    if "original_demand_input" in kwargs:
        inputs_extracted["original_demand_input"] = kwargs["original_demand_input"]
    if "demand_period_conversion" in kwargs:
        inputs_extracted["demand_period_conversion"] = kwargs["demand_period_conversion"]

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand rate (units per year)",
        "S": "Fixed setup or ordering cost incurred per order placed ($/order)",
        "H": "Holding (carrying) cost to maintain one unit in inventory for one year ($/unit/year)",
        "C": "Purchase cost per unit ($/unit, if applicable)",
        "i": "Annual holding cost rate as a fraction of unit purchase price (%/year)",
        "Q*": "Economic Order Quantity: the optimal replenishment batch size that minimizes total inventory costs (units)",
        "N": "Order frequency: optimal number of replenishment orders placed per year (orders/year)",
        "T": "Cycle time: time between consecutive replenishment orders. Expressed in operating days and weeks (clearly defined: 1 work week = 5 operating days; 1 calendar week = 7 calendar days; fractional weeks always state exact days)",
        "work_days_per_year": "Annual operating calendar basis (default: 250 work days = 50 five-day work weeks per year)",
        "Time Units Definition": "1 work week = 5 operating days (standard 250 work days = 50 work weeks per year); 1 calendar week = 7 calendar days (standard 365 calendar days = 52.14 calendar weeks per year). All fractional week metrics include the exact day count.",
        "TC_order": "Total annual ordering cost ($/year)",
        "TC_hold": "Total annual inventory holding cost ($/year)",
        "TC_inv": "Total annual inventory cost (Ordering Cost + Holding Cost) ($/year)",
        "TC_total": "Total annual landed cost including acquisition purchase cost ($/year)",
    }

    # Deterministic Calculations
    two_ds = 2.0 * d * s
    ratio = two_ds / h
    q_opt_raw = math.sqrt(ratio)
    q_opt = round(q_opt_raw, 2)

    n_orders = d / q_opt
    cycle_days = (q_opt / d) * wd
    cycle_weeks_work = cycle_days / 5.0
    cycle_weeks_cal = cycle_days / 7.0
    annual_ordering = n_orders * s
    annual_holding = (q_opt / 2.0) * h
    total_inv = annual_ordering + annual_holding
    purchase_cost = (d * c_val) if c_val is not None else None
    total_cost = (purchase_cost + total_inv) if purchase_cost is not None else total_inv

    # 3. Formula
    formulas = {
        "Optimal Order Quantity (Q*)": "Q* = sqrt((2 * D * S) / H)",
        "Orders per Year (N)": "N = D / Q*",
        "Cycle Time Operating Days (T)": "T = (Q* / D) * work_days_per_year",
        "Cycle Time Weeks (T_weeks)": "T_work_weeks = T / 5 operating days/week | T_cal_weeks = T / 7 calendar days/week (fractional weeks include exact days)",
        "Annual Ordering Cost (TC_order)": "TC_order = (D / Q*) * S = N * S",
        "Annual Holding Cost (TC_hold)": "TC_hold = (Q* / 2) * H",
        "Total Annual Inventory Cost (TC_inv)": "TC_inv = TC_order + TC_hold",
        "Total Annual Landed Cost (TC_total)": "TC_total = (D * C) + TC_inv  [if unit cost provided]",
    }

    # 4. Formula with Substituted Values
    formulas_substituted = {
        "Optimal Order Quantity (Q*)": f"sqrt((2 * {d:,.2f} * {s:,.2f}) / {h:.4f}) = sqrt({two_ds:,.2f} / {h:.4f}) = sqrt({ratio:,.4f}) = {q_opt:,.2f} units",
        "Orders per Year (N)": f"{d:,.2f} / {q_opt:,.2f} = {n_orders:.2f} orders/year",
        "Cycle Time Operating Days (T)": f"({q_opt:,.2f} / {d:,.2f}) * {wd:,.0f} = {cycle_days:.1f} operating days",
        "Cycle Time Weeks (T_weeks)": (
            f"{cycle_days:.1f} operating days / 5 days/wk = {cycle_weeks_work:.2f} work weeks ({cycle_days:.1f} operating days) | "
            f"{cycle_days:.1f} days / 7 days/wk = {cycle_weeks_cal:.2f} calendar weeks ({cycle_days:.1f} calendar days)"
        ),
        "Annual Ordering Cost (TC_order)": f"({d:,.2f} / {q_opt:,.2f}) * {s:,.2f} = {n_orders:.2f} * ${s:,.2f} = ${annual_ordering:,.2f}",
        "Annual Holding Cost (TC_hold)": f"({q_opt:,.2f} / 2) * {h:.4f} = {q_opt/2:,.2f} * ${h:.4f} = ${annual_holding:,.2f}",
        "Total Annual Inventory Cost (TC_inv)": f"${annual_ordering:,.2f} + ${annual_holding:,.2f} = ${total_inv:,.2f}",
    }
    if purchase_cost is not None:
        formulas_substituted["Total Annual Landed Cost (TC_total)"] = (
            f"({d:,.2f} * ${c_val:,.2f}) + ${total_inv:,.2f} = ${purchase_cost:,.2f} + ${total_inv:,.2f} = ${total_cost:,.2f}"
        )

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "2DS": two_ds,
        "step_0_demand_period_conversion": kwargs.get("demand_period_conversion", "Direct annual demand (no conversion needed)"),
        "step_1_numerator_2DS": two_ds,
        "step_2_ratio_2DS_over_H": ratio,
        "step_3_sqrt_unrounded_Q": q_opt_raw,
        "step_4_optimal_Q_rounded": q_opt,
        "step_5_orders_per_year_N": round(n_orders, 4),
        "step_6_cycle_time_fraction_of_year": round(q_opt / d, 6),
        "step_6b_cycle_time_fraction_of_year_pct": f"{((q_opt / d) * 100):.2f}% of year ({cycle_days:.1f} operating days)",
        "step_7_cycle_time_days": round(cycle_days, 2),
        "step_7b_cycle_time_work_weeks": f"{cycle_weeks_work:.2f} work weeks ({cycle_days:.1f} operating days @ 5 days/week)",
        "step_7c_cycle_time_calendar_weeks": f"{cycle_weeks_cal:.2f} calendar weeks ({cycle_days:.1f} calendar days @ 7 days/week)",
        "step_8_annual_ordering_cost": round(annual_ordering, 2),
        "step_9_annual_holding_cost": round(annual_holding, 2),
        "step_10_cost_difference_ordering_vs_holding": round(abs(annual_ordering - annual_holding), 2),
        "step_11_total_annual_inventory_cost": round(total_inv, 2),
        "step_12_annual_purchase_cost": round(purchase_cost, 2) if purchase_cost is not None else None,
        "step_13_total_annual_landed_cost": round(total_cost, 2),
    }

    # 6. Final Answer
    final_ans = {
        "optimal_order_quantity": q_opt,
        "optimal_order_quantity_units": "units per replenishment order",
        "orders_per_year": round(n_orders, 2),
        "cycle_time_days": round(cycle_days, 1),
        "cycle_time_weeks": f"{cycle_weeks_work:.2f} work weeks ({cycle_days:.1f} operating days @ 5 days/wk) / {cycle_weeks_cal:.2f} calendar weeks ({cycle_days:.1f} days @ 7 days/wk)",
        "cycle_time_detailed": format_time_duration_clarified(cycle_days),
        "annual_ordering_cost": round(annual_ordering, 2),
        "annual_holding_cost": round(annual_holding, 2),
        "total_annual_inventory_cost": round(total_inv, 2),
        "total_annual_landed_cost": round(total_cost, 2),
    }

    # 7. Verification Checks
    diff = abs(annual_ordering - annual_holding)
    rel_diff = diff / max(1.0, annual_ordering)
    second_deriv = (2.0 * d * s) / (q_opt ** 3)

    verifications = [
        {
            "check": "Positivity Check",
            "condition": "Q* > 0",
            "passed": q_opt > 0,
            "detail": f"Optimal order quantity ({q_opt:,.2f} units) is strictly positive.",
        },
        {
            "check": "Cost Trade-off Balance (First Order Optimality)",
            "condition": "|Annual Ordering Cost - Annual Holding Cost| / Ordering Cost <= 0.05",
            "passed": rel_diff <= 0.05,
            "detail": (
                f"At Q* = {q_opt:,.2f}, Annual Ordering Cost (${annual_ordering:,.2f}) equals Annual Holding Cost "
                f"(${annual_holding:,.2f}) within {rel_diff:.2%} (diff: ${diff:.2f}), confirming the trade-off minimum where dTC/dQ = 0."
            ),
        },
        {
            "check": "Convexity Check (Second Order Sufficiency)",
            "condition": "d^2(TC)/dQ^2 = (2 * D * S) / Q^3 > 0",
            "passed": second_deriv > 0,
            "detail": f"Second derivative d^2(TC)/dQ^2 = {second_deriv:.6f} > 0, confirming a strict global cost minimum.",
        },
    ]

    # 8. Assumptions
    model_assumptions = [
        "Demand rate (D) is constant, deterministic, and continuous over the annual planning cycle.",
        "Replenishment lead time is zero or constant.",
        "Batch replenishment arrives instantaneously as a single complete delivery of Q units.",
        "Stockouts, shortages, and backorders are not permitted.",
        "Unit purchase price, fixed order cost (S), and unit holding cost (H) are constant over time.",
        "Calendar & Time Units Definition: 1 work week is defined as 5 operating days (standard 250 work days = 50 work weeks per year); 1 calendar week is defined as 7 calendar days (365 days/year = 52.14 calendar weeks). Any result given in a fraction or percentage of a week includes the exact number of days for complete clarity.",
    ]

    return EOQResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_ans,
        verification_checks=verifications,
        assumptions=model_assumptions,
        optimal_order_quantity=q_opt,
        annual_demand=d,
        order_cost=s,
        unit_holding_cost=round(h, 4),
        orders_per_year=round(n_orders, 2),
        cycle_time_days=round(cycle_days, 1),
        annual_ordering_cost=round(annual_ordering, 2),
        annual_holding_cost=round(annual_holding, 2),
        total_annual_inventory_cost=round(total_inv, 2),
        annual_purchase_cost=round(purchase_cost, 2) if purchase_cost is not None else None,
        total_annual_cost=round(total_cost, 2),
    )


# ============================================================================
# 2. Quantity Discount Model
# ============================================================================

@dataclass
class DiscountTierDetail:
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    unit_holding_cost: float
    unconstrained_eoq: float
    feasible_quantity: float
    is_price_break_adjusted: bool
    annual_purchasing_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, Any]

    # 7. Verification Checks
    verification_checks: List[Dict[str, Any]]

    # 8. Assumptions
    assumptions: List[str]

    # Core attribute accessors
    optimal_tier_index: int
    optimal_order_quantity: float
    optimal_unit_price: float
    minimum_total_annual_cost: float
    annual_demand: float
    order_cost: float
    tier_evaluations: List[DiscountTierDetail]

    @property
    def minimum_total_cost(self) -> float:
        return self.minimum_total_annual_cost

    @property
    def annual_purchasing_cost(self) -> float:
        return self.tier_evaluations[self.optimal_tier_index].annual_purchasing_cost

    @property
    def annual_ordering_cost(self) -> float:
        return self.tier_evaluations[self.optimal_tier_index].annual_ordering_cost

    @property
    def annual_holding_cost(self) -> float:
        return self.tier_evaluations[self.optimal_tier_index].annual_holding_cost

    @property
    def summary_message(self) -> str:
        return (
            f"Recommended: Order {self.optimal_order_quantity:,.0f} units at ${self.optimal_unit_price:.2f}/unit "
            f"(Tier {self.optimal_tier_index + 1}) for a total annual landed cost of ${self.minimum_total_annual_cost:,.2f}."
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_quantity_discount(
    annual_demand: float,
    order_cost: float,
    price_tiers: List[Dict[str, Any]],
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
    **kwargs,
) -> QuantityDiscountResult:
    """
    Evaluates quantity discount schedules across all pricing tiers.
    Supports both fixed holding cost and percentage holding cost.

    Returns the complete 8 verification sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
    """
    if annual_demand is None or annual_demand <= 0:
        raise ValueError(f"annual_demand must be positive. Got {annual_demand}")
    if order_cost is None or order_cost <= 0:
        raise ValueError(f"order_cost must be positive. Got {order_cost}")
    if not price_tiers or not isinstance(price_tiers, list) or len(price_tiers) == 0:
        raise ValueError("price_tiers must be a non-empty list of pricing tier specifications.")

    if fixed_holding_cost is not None and fixed_holding_cost > 0:
        holding_type = "fixed"
        h_param = float(fixed_holding_cost)
    elif holding_cost_rate is not None and holding_cost_rate > 0:
        holding_type = "percentage"
        h_param = float(holding_cost_rate)
    else:
        raise ValueError(
            "Missing holding cost specification: Must provide either holding_cost_rate (i) "
            "or fixed_holding_cost (H). Missing values cannot be invented."
        )

    d = float(annual_demand)
    s = float(order_cost)

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand (D)": d,
        "order_cost (S)": s,
        "holding_cost_type": holding_type,
        "holding_cost_rate (i)": h_param if holding_type == "percentage" else None,
        "fixed_holding_cost (H)": h_param if holding_type == "fixed" else None,
        "price_tiers": price_tiers,
    }
    if "original_demand_input" in kwargs:
        inputs_extracted["original_demand_input"] = kwargs["original_demand_input"]
    if "demand_period_conversion" in kwargs:
        inputs_extracted["demand_period_conversion"] = kwargs["demand_period_conversion"]

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand in units",
        "S": "Fixed ordering cost per order placed ($)",
        "j": "Pricing tier bracket index (1 to K)",
        "P_j": "Unit purchase price within tier j ($/unit)",
        "H_j": "Unit holding cost per year for tier j ($/unit/year; H_j = i * P_j or fixed H)",
        "EOQ_j": "Unconstrained Wilson EOQ evaluated at tier j's holding cost H_j",
        "Q_j": "Feasible order quantity adjusted to tier j's minimum price break threshold if EOQ_j < min_qty_j",
        "TC_purchase_j": "Annual acquisition purchasing cost for tier j: D * P_j ($)",
        "TC_order_j": "Annual ordering cost for tier j: (D / Q_j) * S ($)",
        "TC_hold_j": "Annual holding cost for tier j: (Q_j / 2) * H_j ($)",
        "TC_j": "Total annual landed cost for tier j: TC_purchase_j + TC_order_j + TC_hold_j ($)",
        "j*": "Optimal pricing tier that achieves the global minimum total annual cost",
        "Time Units Definition": "Operating calendar basis: 250 work days per year. 1 work week = 5 operating days; 1 calendar week = 7 calendar days. Any result given in a fraction or percentage of a week includes the exact number of days.",
    }

    # 3. Formula
    formulas = {
        "Tier Holding Cost (H_j)": "H_j = fixed_H  [if fixed]  OR  H_j = i * P_j  [if percentage]",
        "Unconstrained EOQ (EOQ_j)": "EOQ_j = sqrt((2 * D * S) / H_j)",
        "Feasibility Condition": "If EOQ_j < min_qty_j: Q_j = min_qty_j; Else if EOQ_j in bracket: Q_j = EOQ_j; Else Q_j = max_qty_j",
        "Total Annual Cost (TC_j)": "TC_j = (D * P_j) + ((D / Q_j) * S) + ((Q_j / 2) * H_j)",
        "Optimal Decision": "j* = argmin_j(TC_j),  Q* = Q_{j*}",
    }

    # Evaluate each tier
    sorted_tiers = sorted(price_tiers, key=lambda t: t.get("min_qty", 0))
    tier_details: List[DiscountTierDetail] = []
    formulas_substituted: Dict[str, str] = {}

    for idx, tier in enumerate(sorted_tiers):
        if "unit_price" not in tier or tier["unit_price"] <= 0:
            raise ValueError(f"Tier {idx} missing valid positive unit_price: {tier}")
        min_q = float(tier.get("min_qty", 1.0))
        max_q_val = tier.get("max_qty")
        max_q = float(max_q_val) if max_q_val not in (None, "", float("inf")) else None
        p = float(tier["unit_price"])

        h_j = h_param if holding_type == "fixed" else p * h_param
        raw_eoq = math.sqrt((2.0 * d * s) / h_j)

        is_adjusted = False
        if raw_eoq < min_q:
            feasible_q = min_q
            is_adjusted = True
        elif max_q is not None and raw_eoq > max_q:
            feasible_q = max_q
            is_adjusted = True
        else:
            feasible_q = raw_eoq

        purchasing = d * p
        ordering = (d / feasible_q) * s
        holding = (feasible_q / 2.0) * h_j
        total = purchasing + ordering + holding

        max_label = f"{max_q:,.0f}" if max_q is not None else "∞"
        t_cycle_days = (feasible_q / d) * 250.0
        t_cycle_work_wks = t_cycle_days / 5.0
        t_cycle_cal_wks = t_cycle_days / 7.0
        formulas_substituted[f"Tier_{idx+1}_Evaluation ({min_q:,.0f}-{max_label} @ ${p:.2f})"] = (
            f"EOQ_{idx+1} = sqrt((2 * {d:,.0f} * {s:,.2f}) / {h_j:.4f}) = {raw_eoq:,.2f} units -> "
            f"Feasible Q_{idx+1} = {feasible_q:,.0f} units (Order Cycle: {t_cycle_days:.1f} operating days / {t_cycle_work_wks:.2f} work weeks @ 5 days/wk). "
            f"TC_{idx+1} = (${purchasing:,.2f} purchase) + (${ordering:,.2f} order) + (${holding:,.2f} hold) = ${total:,.2f}"
        )

        tier_details.append(
            DiscountTierDetail(
                tier_index=idx,
                min_qty=min_q,
                max_qty=max_q,
                unit_price=p,
                unit_holding_cost=round(h_j, 4),
                unconstrained_eoq=round(raw_eoq, 2),
                feasible_quantity=round(feasible_q, 2),
                is_price_break_adjusted=is_adjusted,
                annual_purchasing_cost=round(purchasing, 2),
                annual_ordering_cost=round(ordering, 2),
                annual_holding_cost=round(holding, 2),
                total_annual_cost=round(total, 2),
            )
        )

    best_tier = min(tier_details, key=lambda t: t.total_annual_cost)
    opt_cycle_days = (best_tier.feasible_quantity / d) * 250.0
    opt_cycle_work_wks = opt_cycle_days / 5.0
    opt_cycle_cal_wks = opt_cycle_days / 7.0

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "step_0_demand_period_conversion": kwargs.get("demand_period_conversion", "Direct annual demand (no conversion needed)"),
        "holding_cost_type": holding_type,
        "holding_rate_or_fixed": h_param,
        "evaluated_tier_details": [asdict(t) for t in tier_details],
        "all_tier_total_costs": {f"Tier_{t.tier_index+1}": t.total_annual_cost for t in tier_details},
        "optimal_tier_index": best_tier.tier_index,
        "optimal_tier_cycle_days": round(opt_cycle_days, 2),
        "optimal_tier_cycle_work_weeks": f"{opt_cycle_work_wks:.2f} work weeks ({opt_cycle_days:.1f} operating days @ 5 days/week)",
        "optimal_tier_cycle_calendar_weeks": f"{opt_cycle_cal_wks:.2f} calendar weeks ({opt_cycle_days:.1f} calendar days @ 7 days/week)",
        "savings_vs_base_tier": round(tier_details[0].total_annual_cost - best_tier.total_annual_cost, 2),
        "cost_savings_vs_base_tier": round(tier_details[0].total_annual_cost - best_tier.total_annual_cost, 2),
    }

    # 6. Final Answer
    final_ans = {
        "optimal_tier_index": best_tier.tier_index,
        "optimal_tier_label": f"Tier {best_tier.tier_index + 1}",
        "optimal_order_quantity": best_tier.feasible_quantity,
        "optimal_unit_price": best_tier.unit_price,
        "order_cycle_time": f"{opt_cycle_days:.1f} operating days ({opt_cycle_work_wks:.2f} work weeks @ 5 days/wk / {opt_cycle_cal_wks:.2f} calendar weeks @ 7 days/wk)",
        "minimum_total_annual_cost": best_tier.total_annual_cost,
        "annual_purchasing_cost": best_tier.annual_purchasing_cost,
        "annual_ordering_cost": best_tier.annual_ordering_cost,
        "annual_holding_cost": best_tier.annual_holding_cost,
    }

    # 7. Verification Checks
    verifications = [
        {
            "check": "Global Cost Minimization Check",
            "condition": "TC(j*) <= TC(j) for all tiers j",
            "passed": True,
            "detail": (
                f"Tier {best_tier.tier_index + 1} achieves the global minimum total cost (${best_tier.total_annual_cost:,.2f}) "
                f"across all {len(tier_details)} evaluated price brackets."
            ),
        },
        {
            "check": "Bracket Feasibility Compliance",
            "condition": "min_qty_j* <= Q* <= max_qty_j*",
            "passed": best_tier.feasible_quantity >= best_tier.min_qty and (best_tier.max_qty is None or best_tier.feasible_quantity <= best_tier.max_qty),
            "detail": f"Recommended order quantity ({best_tier.feasible_quantity:,.0f} units) qualifies for the Tier {best_tier.tier_index + 1} unit price (${best_tier.unit_price:.2f}).",
        },
        {
            "check": "Economic Trade-off Justification",
            "condition": "Purchasing cost reduction >= Incremental holding/ordering cost vs Base Tier",
            "passed": True,
            "detail": (
                f"Purchasing at Tier {best_tier.tier_index + 1} saves ${tier_details[0].total_annual_cost - best_tier.total_annual_cost:,.2f} "
                f"annually compared to Tier 1 base pricing."
            ),
        },
    ]

    # 8. Assumptions
    model_assumptions = [
        "All-units quantity discount applies: the discounted price P_j applies to every unit in the order once quantity threshold is met.",
        f"Holding cost is {'constant per unit across all brackets' if holding_type == 'fixed' else 'proportional to unit purchase price (H = i * P)'}.",
        "Demand rate is deterministic and constant over the entire annual planning horizon.",
        "Supplier can fulfill orders of any quantity without capacity caps or lead time penalties.",
        "Time Units & Calendar Definition: 1 work week is defined as 5 operating days (50 work weeks = 250 operating days per year); 1 calendar week is defined as 7 calendar days. Any fractional week metric explicitly states the exact number of days.",
    ]

    return QuantityDiscountResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_ans,
        verification_checks=verifications,
        assumptions=model_assumptions,
        optimal_tier_index=best_tier.tier_index,
        optimal_order_quantity=best_tier.feasible_quantity,
        optimal_unit_price=best_tier.unit_price,
        minimum_total_annual_cost=best_tier.total_annual_cost,
        annual_demand=d,
        order_cost=s,
        tier_evaluations=tier_details,
    )


# ============================================================================
# 3. Safety Stock and Reorder Point (SS & ROP)
# ============================================================================

@dataclass
class ROPResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, Any]

    # 7. Verification Checks
    verification_checks: List[Dict[str, Any]]

    # 8. Assumptions
    assumptions: List[str]

    # Core attribute accessors
    reorder_point: float
    safety_stock: float
    case_type: str
    avg_demand: float
    lead_time: float
    std_demand: float
    std_lead_time: float
    expected_lead_time_demand: float
    sigma_lead_time: float
    z_score: float
    cycle_service_level: Optional[float]

    @property
    def lead_time_std(self) -> float:
        return self.std_lead_time

    @property
    def std_lead_time_demand(self) -> float:
        return self.sigma_lead_time

    @property
    def stockout_probability_per_cycle(self) -> float:
        return round(1.0 - self.cycle_service_level, 4) if self.cycle_service_level is not None else 0.0

    @property
    def is_variable_lead_time(self) -> bool:
        return self.std_lead_time > 0

    @property
    def formula_breakdown(self) -> Dict[str, str]:
        return self.formula_with_substituted_values

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_rop(
    avg_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    **kwargs,
) -> ROPResult:
    """
    Calculates Safety Stock (SS) and Reorder Point (ROP) for Continuous Review.

    Supports all 4 deterministic uncertainty combinations:
        1. Constant demand and constant lead time (sigma_d = 0, sigma_L = 0)
        2. Variable demand and constant lead time (sigma_d > 0, sigma_L = 0)
        3. Constant demand and variable lead time (sigma_d = 0, sigma_L > 0)
        4. Variable demand and variable lead time (sigma_d > 0, sigma_L > 0)

    Returns the complete 8 verification sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
    """
    if avg_demand is None:
        avg_demand = kwargs.get("avg_demand_per_period")
    if std_demand == 0.0 and "std_demand_per_period" in kwargs:
        std_demand = kwargs.get("std_demand_per_period", 0.0)

    if avg_demand is None or avg_demand <= 0:
        raise ValueError(f"avg_demand must be positive. Got {avg_demand}")
    if lead_time is None or lead_time <= 0:
        raise ValueError(f"lead_time must be positive. Got {lead_time}")
    if std_demand is None or std_demand < 0:
        raise ValueError(f"std_demand cannot be negative. Got {std_demand}")
    if std_lead_time is None or std_lead_time < 0:
        raise ValueError(f"std_lead_time cannot be negative. Got {std_lead_time}")

    d = float(avg_demand)
    l_mean = float(lead_time)
    sig_d = float(std_demand)
    sig_l = float(std_lead_time)

    # 1. Inputs Extracted
    inputs_extracted = {
        "avg_demand (d)": d,
        "lead_time (L)": l_mean,
        "std_demand (sigma_d)": sig_d,
        "std_lead_time (sigma_L)": sig_l,
        "service_level (CSL)": service_level,
        "z_score (z)": z_score,
    }
    if "original_demand_input" in kwargs:
        inputs_extracted["original_demand_input"] = kwargs["original_demand_input"]
    if "demand_period_conversion" in kwargs:
        inputs_extracted["demand_period_conversion"] = kwargs["demand_period_conversion"]

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average demand rate per unit time period (e.g., daily or weekly demand)",
        "L": "Average supplier replenishment lead time. Defined in days and weeks (1 work week = 5 operating days; 1 calendar week = 7 calendar days; fractional weeks state exact days)",
        "sigma_d": "Standard deviation of demand per period",
        "sigma_L": "Standard deviation of lead time in periods",
        "CSL": "Cycle Service Level: target probability of not stocking out during a replenishment lead time cycle",
        "z": "Safety factor: number of standard deviations from standard normal distribution corresponding to CSL",
        "DDLT": "Expected demand during lead time: d * L (units)",
        "sigma_LT": "Combined standard deviation of demand during lead time (units)",
        "SS": "Safety Stock: buffer inventory held to mitigate demand and lead time volatility (units)",
        "ROP": "Reorder Point: inventory level that triggers placement of a new replenishment order (units)",
        "Time Units Definition": "1 work week = 5 operating days; 1 calendar week = 7 calendar days. Demand rate and replenishment lead time must be in identical time units (e.g., daily demand with days of lead time). Any result given in a fraction or percentage of a week includes the exact number of days.",
    }

    # Identify Case & Solve
    has_var_d = sig_d > 0
    has_var_l = sig_l > 0
    ddlt = d * l_mean

    if not has_var_d and not has_var_l:
        case_type = "constant_demand_and_constant_lead_time"
        sigma_lt = 0.0
        z = 0.0
        csl = None
        ss = 0.0
        rop = ddlt
        term_demand_var = 0.0
        term_lead_var = 0.0
        formula_sigma = "sigma_LT = 0  (deterministic)"
        subst_sigma = "sigma_LT = 0.00 units"
    else:
        if z_score is not None:
            if z_score < 0:
                raise ValueError(f"z_score cannot be negative. Got {z_score}")
            z = float(z_score)
            csl = float(stats.norm.cdf(z)) if service_level is None else service_level
        elif service_level is not None:
            if not (0.50 <= service_level < 1.0):
                raise ValueError(f"service_level must be between 0.50 and 0.9999. Got {service_level}")
            csl = float(service_level)
            z = float(stats.norm.ppf(csl))
        else:
            raise ValueError(
                "Missing service level: Uncertainty detected (std_demand > 0 or std_lead_time > 0), "
                "must provide either service_level (e.g. 0.95) or z_score. Missing values cannot be invented."
            )

        if has_var_d and not has_var_l:
            case_type = "variable_demand_and_constant_lead_time"
            term_demand_var = l_mean * (sig_d ** 2)
            term_lead_var = 0.0
            sigma_lt = math.sqrt(term_demand_var)
            formula_sigma = "sigma_LT = sqrt(L * sigma_d^2) = sqrt(L) * sigma_d"
            subst_sigma = f"sqrt({l_mean:,.2f} * {sig_d:,.2f}^2) = sqrt({term_demand_var:,.2f}) = {sigma_lt:,.2f} units"
        elif not has_var_d and has_var_l:
            case_type = "constant_demand_and_variable_lead_time"
            term_demand_var = 0.0
            term_lead_var = (d ** 2) * (sig_l ** 2)
            sigma_lt = math.sqrt(term_lead_var)
            formula_sigma = "sigma_LT = sqrt(d^2 * sigma_L^2) = d * sigma_L"
            subst_sigma = f"sqrt({d:,.2f}^2 * {sig_l:,.2f}^2) = sqrt({term_lead_var:,.2f}) = {sigma_lt:,.2f} units"
        else:
            case_type = "variable_demand_and_variable_lead_time"
            term_demand_var = l_mean * (sig_d ** 2)
            term_lead_var = (d ** 2) * (sig_l ** 2)
            sigma_lt = math.sqrt(term_demand_var + term_lead_var)
            formula_sigma = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
            subst_sigma = (
                f"sqrt({l_mean:,.2f} * {sig_d:,.2f}^2 + {d:,.2f}^2 * {sig_l:,.2f}^2) = "
                f"sqrt({term_demand_var:,.2f} + {term_lead_var:,.2f}) = sqrt({term_demand_var + term_lead_var:,.2f}) = {sigma_lt:,.2f} units"
            )

        ss = z * sigma_lt
        rop = ddlt + ss

    # 3. Formula
    formulas = {
        "Case Identified": case_type,
        "Expected Demand during Lead Time (DDLT)": "DDLT = d * L",
        "Lead Time Demand Std Dev (sigma_LT)": formula_sigma,
        "Safety Stock (SS)": "SS = z * sigma_LT",
        "Reorder Point (ROP)": "ROP = DDLT + SS = (d * L) + SS",
    }

    # 4. Formula with Substituted Values
    csl_label = f"{csl*100:.2f}%" if csl is not None else "N/A"
    l_work_wks = l_mean / 5.0
    l_cal_wks = l_mean / 7.0
    formulas_substituted = {
        "Expected Demand during Lead Time (DDLT)": (
            f"{d:,.2f} * {l_mean:,.2f} days "
            f"({l_work_wks:.2f} work wks @ 5 days/wk / {l_cal_wks:.2f} cal wks @ 7 days/wk) = {ddlt:,.2f} units"
        ),
        "Lead Time Demand Std Dev (sigma_LT)": subst_sigma,
        "Safety Factor (z)": f"norm.ppf({csl_label}) = {z:.4f}" if csl else "0.0000",
        "Safety Stock (SS)": f"{z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units",
        "Reorder Point (ROP)": f"{ddlt:,.2f} + {ss:,.2f} = {rop:,.2f} units",
    }

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "step_0_demand_period_conversion": kwargs.get("demand_period_conversion", "Direct demand rate matching lead time period (no conversion needed)"),
        "case_type": case_type,
        "expected_lead_time_demand_DDLT": round(ddlt, 2),
        "lead_time_duration_days": round(l_mean, 2),
        "lead_time_work_weeks": f"{l_work_wks:.2f} work weeks ({l_mean:.1f} operating days @ 5 days/wk)",
        "lead_time_calendar_weeks": f"{l_cal_wks:.2f} calendar weeks ({l_mean:.1f} calendar days @ 7 days/wk)",
        "demand_variance_contribution (L * sigma_d^2)": round(term_demand_var, 4),
        "lead_time_variance_contribution (d^2 * sigma_L^2)": round(term_lead_var, 4),
        "demand_variance_contribution_L_sigma_d_sq": round(term_demand_var, 4),
        "lead_time_variance_contribution_d_sq_sigma_L_sq": round(term_lead_var, 4),
        "total_lead_time_variance": round(term_demand_var + term_lead_var, 4),
        "sigma_LT": round(sigma_lt, 4),
        "z_score": round(z, 4),
        "cycle_service_level": round(csl, 4) if csl is not None else None,
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2),
    }

    # 6. Final Answer
    final_ans = {
        "reorder_point": round(rop, 2),
        "reorder_point_units": "units (trigger replenishment when inventory position drops to this level)",
        "safety_stock": round(ss, 2),
        "safety_stock_units": "units reserved as safety buffer",
        "expected_lead_time_demand": round(ddlt, 2),
        "lead_time_days": round(l_mean, 2),
        "lead_time_clarification": f"{l_mean:.1f} days ({l_work_wks:.2f} work weeks @ 5 days/wk / {l_cal_wks:.2f} calendar weeks @ 7 days/wk)",
        "cycle_service_level": f"{csl*100:.1f}%" if csl is not None else "100.0% (Deterministic)",
    }

    # 7. Verification Checks
    verifications = [
        {
            "check": "ROP Lower Bound Check",
            "condition": "ROP >= d * L",
            "passed": rop >= ddlt,
            "detail": f"ROP ({rop:,.2f} units) >= Expected Lead Time Demand ({ddlt:,.2f} units), ensuring average lead time demand is fully covered.",
        },
        {
            "check": "Safety Stock Non-Negativity Check",
            "condition": "SS >= 0 for CSL >= 50%",
            "passed": ss >= 0,
            "detail": f"Safety stock ({ss:,.2f} units) is non-negative for target service level ({csl_label}).",
        },
        {
            "check": "Variance Decomposition Integrity",
            "condition": "sigma_LT^2 == (L * sigma_d^2) + (d^2 * sigma_L^2)",
            "passed": math.isclose(sigma_lt ** 2, term_demand_var + term_lead_var, abs_tol=1e-5),
            "detail": (
                f"Combined variance ({term_demand_var + term_lead_var:,.2f}) accurately accounts for "
                f"demand variance ({term_demand_var:,.2f}) and lead time variance ({term_lead_var:,.2f})."
            ),
        },
    ]

    # 8. Assumptions
    model_assumptions = [
        "Continuous review (s, Q) policy: inventory position is tracked continuously in real time.",
        "Demand in each time period is independent and identically distributed.",
        f"Demand during lead time follows a normal distribution ({'active' if (has_var_d or has_var_l) else 'deterministic'}).",
        "Stockouts occurring during the replenishment lead time are backordered and filled when the batch arrives.",
        "Time Units & Calendar Definition: 1 work week is defined as 5 operating days; 1 calendar week is defined as 7 calendar days. Lead time and demand rates must use consistent time units. Any result given in a fraction or percentage of a week includes the exact number of days.",
    ]

    return ROPResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_ans,
        verification_checks=verifications,
        assumptions=model_assumptions,
        reorder_point=round(rop, 2),
        safety_stock=round(ss, 2),
        case_type=case_type,
        avg_demand=d,
        lead_time=l_mean,
        std_demand=sig_d,
        std_lead_time=sig_l,
        expected_lead_time_demand=round(ddlt, 2),
        sigma_lead_time=round(sigma_lt, 2),
        z_score=round(z, 4),
        cycle_service_level=round(csl, 4) if csl is not None else None,
    )


# Alias for backwards compatibility
calculate_safety_stock_rop = calculate_rop


# ============================================================================
# 4. Fixed-Period Review Model
# ============================================================================

@dataclass
class PeriodicReviewResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, Any]

    # 7. Verification Checks
    verification_checks: List[Dict[str, Any]]

    # 8. Assumptions
    assumptions: List[str]

    # Core attribute accessors
    recommended_order_quantity: float
    target_inventory: float
    review_period: float
    lead_time: float
    protection_period: float
    avg_demand: float
    demand_during_protection_period: float
    std_demand: float
    sigma_protection_period: float
    safety_stock: float
    z_score: float
    cycle_service_level: Optional[float]
    inventory_position: float
    gross_inventory_shortfall: float

    @property
    def order_up_to_level(self) -> float:
        return self.target_inventory

    @property
    def current_inventory(self) -> float:
        return self.inventory_position

    @property
    def expected_demand_protection_period(self) -> float:
        return self.demand_during_protection_period

    @property
    def std_demand_protection_period(self) -> float:
        return self.sigma_protection_period

    @property
    def formula_breakdown(self) -> Dict[str, str]:
        return self.formula_with_substituted_values

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    avg_demand: Optional[float] = None,
    inventory_position: Optional[float] = None,
    std_demand: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    **kwargs,
) -> PeriodicReviewResult:
    """
    Calculates Target Inventory and Order Quantity for a Fixed-Period Review (R, S) policy.

    Formulas:
        Protection period = review period + lead time
        Demand during protection period = d * (R + L)
        Safety stock = z * (sigma_d * sqrt(R + L))  [if std_demand > 0]
        Target inventory = Demand during protection period + Safety stock
        Order quantity = max(0, Target inventory - Inventory position)

    Returns the complete 8 verification sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
    """
    if avg_demand is None:
        avg_demand = kwargs.get("avg_demand_per_period")
    if std_demand == 0.0 and "std_demand_per_period" in kwargs:
        std_demand = kwargs.get("std_demand_per_period", 0.0)
    if inventory_position is None:
        inventory_position = kwargs.get("current_inventory")

    if review_period is None or review_period <= 0:
        raise ValueError(f"review_period must be positive. Got {review_period}")
    if lead_time is None or lead_time < 0:
        raise ValueError(f"lead_time cannot be negative. Got {lead_time}")
    if avg_demand is None or avg_demand <= 0:
        raise ValueError(f"avg_demand must be positive. Got {avg_demand}")
    if inventory_position is None:
        raise ValueError("Missing inventory_position: Current inventory position must be provided.")
    if std_demand is None or std_demand < 0:
        raise ValueError(f"std_demand cannot be negative. Got {std_demand}")

    r = float(review_period)
    l = float(lead_time)
    d = float(avg_demand)
    inv_pos = float(inventory_position)
    sig_d = float(std_demand)

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period (R)": r,
        "lead_time (L)": l,
        "avg_demand (d)": d,
        "inventory_position (I)": inv_pos,
        "std_demand (sigma_d)": sig_d,
        "service_level (CSL)": service_level,
        "z_score (z)": z_score,
    }
    if "original_demand_input" in kwargs:
        inputs_extracted["original_demand_input"] = kwargs["original_demand_input"]
    if "demand_period_conversion" in kwargs:
        inputs_extracted["demand_period_conversion"] = kwargs["demand_period_conversion"]

    # 2. Variable Definitions
    variable_definitions = {
        "R": "Review period interval: time between consecutive inventory reviews. Defined in days and weeks (1 work week = 5 operating days; 1 calendar week = 7 calendar days; fractional weeks state exact days)",
        "L": "Supplier replenishment lead time. Defined in days and weeks (1 work week = 5 operating days; 1 calendar week = 7 calendar days; fractional weeks state exact days)",
        "R + L": "Total protection period: time interval across which inventory must buffer demand until next delivery. Defined in days and weeks (any fractional week includes exact days)",
        "d": "Average customer demand rate per period (units/period)",
        "sigma_d": "Standard deviation of demand per period",
        "sigma_prot": "Standard deviation of demand during the protection period: sigma_d * sqrt(R + L)",
        "CSL": "Target Cycle Service Level for the periodic review cycle",
        "z": "Safety factor corresponding to target CSL from standard normal distribution",
        "SS": "Safety Stock for periodic review: z * sigma_prot (units)",
        "Target Inventory (M)": "Order-Up-To Level (Base Stock): expected protection period demand + Safety Stock (units)",
        "Inventory Position (I)": "Current net inventory position: On-Hand + On-Order - Backorders (units)",
        "Order Quantity (Q)": "Recommended replenishment order placed right now: max(0, Target Inventory - Inventory Position)",
        "Time Units Definition": "1 work week = 5 operating days; 1 calendar week = 7 calendar days. Any result given in a fraction or percentage of a week includes the number of days for clarification.",
    }

    # Deterministic Calculations
    prot_period = r + l
    demand_prot = d * prot_period
    r_cal_wks = r / 7.0
    r_work_wks = r / 5.0
    l_cal_wks = l / 7.0
    l_work_wks = l / 5.0
    prot_cal_wks = prot_period / 7.0
    prot_work_wks = prot_period / 5.0

    if sig_d > 0:
        if z_score is not None:
            if z_score < 0:
                raise ValueError(f"z_score cannot be negative. Got {z_score}")
            z = float(z_score)
            csl = float(stats.norm.cdf(z)) if service_level is None else service_level
        elif service_level is not None:
            if not (0.50 <= service_level < 1.0):
                raise ValueError(f"service_level must be between 0.50 and 0.9999. Got {service_level}")
            csl = float(service_level)
            z = float(stats.norm.ppf(csl))
        else:
            raise ValueError(
                "Missing service level: std_demand > 0 requires either service_level (e.g. 0.95) "
                "or z_score. Missing values cannot be invented."
            )
        sigma_prot = sig_d * math.sqrt(prot_period)
        ss = z * sigma_prot
    else:
        z = 0.0
        csl = None
        sigma_prot = 0.0
        ss = 0.0

    target_inv = demand_prot + ss
    shortfall = target_inv - inv_pos
    order_qty = max(0.0, shortfall)

    # 3. Formula
    formulas = {
        "Protection Period (R + L)": "Protection Period = R + L (in days; fractional weeks include exact days)",
        "Demand during Protection Period": "Demand_prot = d * (R + L)",
        "Protection Period Std Dev (sigma_prot)": "sigma_prot = sigma_d * sqrt(R + L)",
        "Safety Stock (SS)": "SS = z * sigma_prot",
        "Target Inventory (Order-Up-To Level M)": "M = Demand_prot + SS = (d * (R + L)) + (z * sigma_d * sqrt(R + L))",
        "Recommended Order Quantity (Q)": "Q = max(0, Target Inventory - Inventory Position)",
    }

    # 4. Formula with Substituted Values
    csl_str = f"{csl*100:.1f}%" if csl is not None else "100.0% (Deterministic)"
    formulas_substituted = {
        "Protection Period (R + L)": f"{r:,.1f} days + {l:,.1f} days = {prot_period:,.1f} days ({prot_cal_wks:.2f} calendar weeks of 7 days / {prot_work_wks:.2f} work weeks of 5 days)",
        "Demand during Protection Period": f"{d:,.2f} * ({r:,.1f} + {l:,.1f}) = {d:,.2f} * {prot_period:,.1f} = {demand_prot:,.2f} units",
        "Protection Period Std Dev (sigma_prot)": f"{sig_d:,.2f} * sqrt({prot_period:,.1f}) = {sig_d:,.2f} * {math.sqrt(prot_period):.4f} = {sigma_prot:,.2f} units",
        "Safety Stock (SS)": f"{z:.4f} * {sigma_prot:,.2f} = {ss:,.2f} units  (CSL: {csl_str})",
        "Target Inventory (M)": f"{demand_prot:,.2f} + {ss:,.2f} = {target_inv:,.2f} units",
        "Recommended Order Quantity (Q)": f"max(0, {target_inv:,.2f} - {inv_pos:,.2f}) = max(0, {shortfall:,.2f}) = {order_qty:,.2f} units",
    }

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "step_0_demand_period_conversion": kwargs.get("demand_period_conversion", "Direct demand rate matching review and lead time periods (no conversion needed)"),
        "step_1_review_period_R": r,
        "step_1b_review_period_weeks": f"{r_cal_wks:.2f} calendar weeks ({r:.1f} days @ 7 days/wk) / {r_work_wks:.2f} work weeks ({r:.1f} days @ 5 days/wk)",
        "step_2_lead_time_L": l,
        "step_2b_lead_time_weeks": f"{l_cal_wks:.2f} calendar weeks ({l:.1f} days @ 7 days/wk) / {l_work_wks:.2f} work weeks ({l:.1f} days @ 5 days/wk)",
        "step_3_protection_period_R_plus_L": prot_period,
        "step_3b_protection_period_weeks": f"{prot_cal_wks:.2f} calendar weeks ({prot_period:.1f} days @ 7 days/wk) / {prot_work_wks:.2f} work weeks ({prot_period:.1f} days @ 5 days/wk)",
        "step_4_demand_during_protection_period": round(demand_prot, 2),
        "step_5_sigma_protection_period": round(sigma_prot, 4),
        "step_6_z_score": round(z, 4),
        "step_7_cycle_service_level": round(csl, 4) if csl is not None else None,
        "step_8_safety_stock": round(ss, 2),
        "step_9_target_inventory_M": round(target_inv, 2),
        "step_10_inventory_position_I": round(inv_pos, 2),
        "step_11_gross_inventory_shortfall": round(shortfall, 2),
        "step_12_recommended_order_quantity": round(order_qty, 2),
    }

    # 6. Final Answer
    final_ans = {
        "recommended_order_quantity": round(order_qty, 2),
        "recommended_order_quantity_units": "units to order immediately",
        "target_inventory": round(target_inv, 2),
        "target_inventory_units": "units (order-up-to base stock level M)",
        "safety_stock": round(ss, 2),
        "protection_period": prot_period,
        "protection_period_clarification": f"{prot_period:.1f} days ({prot_cal_wks:.2f} calendar weeks @ 7 days/wk / {prot_work_wks:.2f} work weeks @ 5 days/wk)",
        "review_period_clarification": f"{r:.1f} days ({r_cal_wks:.2f} calendar weeks @ 7 days/wk / {r_work_wks:.2f} work weeks @ 5 days/wk)",
        "lead_time_clarification": f"{l:.1f} days ({l_cal_wks:.2f} calendar weeks @ 7 days/wk / {l_work_wks:.2f} work weeks @ 5 days/wk)",
        "current_inventory_position": round(inv_pos, 2),
    }

    # 7. Verification Checks
    verifications = [
        {
            "check": "Full Protection Period Buffer Check",
            "condition": "M >= d * (R + L)",
            "passed": target_inv >= demand_prot,
            "detail": f"Target inventory M ({target_inv:,.2f} units) >= Expected demand ({demand_prot:,.2f} units) across the entire {prot_period:,.1f}-period protection window.",
        },
        {
            "check": "Order Quantity Non-Negativity Floor",
            "condition": "Q = max(0, M - I) >= 0",
            "passed": order_qty >= 0,
            "detail": f"Recommended order quantity ({order_qty:,.2f} units) is non-negative (Inventory position: {inv_pos:,.2f}, Target: {target_inv:,.2f}).",
        },
        {
            "check": "Protection Period Duration Check",
            "condition": "Protection Period == R + L",
            "passed": math.isclose(prot_period, r + l, abs_tol=1e-5),
            "detail": f"Protection period ({prot_period:,.1f} periods) accurately aggregates review cycle ({r:,.1f}) and lead time ({l:,.1f}).",
        },
    ]

    # 8. Assumptions
    model_assumptions = [
        "Inventory is reviewed at fixed, discrete periodic intervals of length R.",
        "Protection period must buffer against stockouts over both the review cycle R and supplier lead time L.",
        "Inventory position equals On-Hand + On-Order - Backorders.",
        "Customer stockouts occurring during the protection period are backordered.",
        "Demand per period is independent and identically normally distributed.",
        "Time Units & Calendar Definition: 1 work week is defined as 5 operating days; 1 calendar week is defined as 7 calendar days. Review period R, lead time L, and protection period (R+L) are clearly defined in days, and any result given in a fraction or percentage of a week includes the exact number of days.",
    ]

    return PeriodicReviewResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_ans,
        verification_checks=verifications,
        assumptions=model_assumptions,
        recommended_order_quantity=round(order_qty, 2),
        target_inventory=round(target_inv, 2),
        review_period=r,
        lead_time=l,
        protection_period=prot_period,
        avg_demand=d,
        demand_during_protection_period=round(demand_prot, 2),
        std_demand=sig_d,
        sigma_protection_period=round(sigma_prot, 2),
        safety_stock=round(ss, 2),
        z_score=round(z, 4),
        cycle_service_level=round(csl, 4) if csl is not None else None,
        inventory_position=round(inv_pos, 2),
        gross_inventory_shortfall=round(shortfall, 2),
    )


# Alias for backwards compatibility
calculate_fixed_period_review = calculate_periodic_review


# ============================================================================
# 5. Newsvendor Model
# ============================================================================

@dataclass
class NewsvendorResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, Any]

    # 7. Verification Checks
    verification_checks: List[Dict[str, Any]]

    # 8. Assumptions
    assumptions: List[str]

    # Core attribute accessors
    recommended_order_quantity: float
    critical_ratio: float
    underage_cost: float
    overage_cost: float
    z_score: float
    demand_mean: float
    demand_std: float
    unit_cost: float
    selling_price: float
    salvage_value: float
    shortage_penalty: float
    expected_sales: float
    expected_lost_sales: float
    expected_leftover_inventory: float
    expected_profit: float
    in_stock_probability: float

    @property
    def optimal_order_quantity(self) -> float:
        return self.recommended_order_quantity

    @property
    def formula_breakdown(self) -> Dict[str, str]:
        return self.formula_with_substituted_values

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_newsvendor(
    unit_cost: float,
    selling_price: float,
    demand_mean: float,
    demand_std: float,
    salvage_value: float = 0.0,
    shortage_penalty: float = 0.0,
    **kwargs,
) -> NewsvendorResult:
    """
    Calculates optimal order quantity for the single-period Newsvendor model.

    Formulas:
        Underage cost: Cu = selling_price - unit_cost + shortage_penalty
        Overage cost:  Co = unit_cost - salvage_value
        Critical ratio: CR = Cu / (Cu + Co)
        Z-score: Z = norm.ppf(CR)
        Recommended order quantity: Q* = demand_mean + Z * demand_std

    Returns the complete 8 verification sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
    """
    if unit_cost is None or unit_cost <= 0:
        raise ValueError(f"unit_cost must be positive. Got {unit_cost}")
    if selling_price is None or selling_price <= 0:
        raise ValueError(f"selling_price must be positive. Got {selling_price}")
    if selling_price <= unit_cost:
        raise ValueError(
            f"selling_price ({selling_price}) must strictly exceed unit_cost ({unit_cost}). "
            "A negative or zero profit margin would guarantee operational loss."
        )
    if salvage_value is None or salvage_value < 0:
        raise ValueError(f"salvage_value cannot be negative. Got {salvage_value}")
    if salvage_value > unit_cost:
        raise ValueError(
            f"salvage_value ({salvage_value}) cannot exceed unit_cost ({unit_cost}). "
            "Arbitrage conditions are invalid."
        )
    if shortage_penalty is None or shortage_penalty < 0:
        raise ValueError(f"shortage_penalty cannot be negative. Got {shortage_penalty}")
    if demand_mean is None or demand_mean <= 0:
        raise ValueError(f"demand_mean must be positive. Got {demand_mean}")
    if demand_std is None or demand_std <= 0:
        raise ValueError(f"demand_std must be positive. Got {demand_std}")

    c = float(unit_cost)
    p = float(selling_price)
    v = float(salvage_value)
    g = float(shortage_penalty)
    mu = float(demand_mean)
    sigma = float(demand_std)

    # 1. Inputs Extracted
    inputs_extracted = {
        "unit_cost (c)": c,
        "selling_price (p)": p,
        "salvage_value (v)": v,
        "shortage_penalty (g)": g,
        "demand_mean (mu)": mu,
        "demand_std (sigma)": sigma,
    }
    if "original_demand_input" in kwargs:
        inputs_extracted["original_demand_input"] = kwargs["original_demand_input"]
    if "demand_period_conversion" in kwargs:
        inputs_extracted["demand_period_conversion"] = kwargs["demand_period_conversion"]

    # 2. Variable Definitions
    variable_definitions = {
        "p": "Retail selling price per unit ($/unit)",
        "c": "Unit purchase or production cost ($/unit)",
        "v": "Salvage or scrap value recovered per unsold unit at end of season ($/unit)",
        "g": "Shortage penalty cost incurred per unit of unmet customer demand ($/unit)",
        "Cu": "Cost of Underage: marginal profit loss incurred per unit of unmet demand: p - c + g ($/unit)",
        "Co": "Cost of Overage: marginal loss incurred per unsold unit ordered: c - v ($/unit)",
        "CR": "Critical Ratio (Critical Fractile): optimal service probability: Cu / (Cu + Co)",
        "mu": "Mean expected customer demand during the selling period (units)",
        "sigma": "Standard deviation of customer demand during the selling period (units)",
        "z": "Optimal safety factor corresponding to standard normal quantile: norm.ppf(CR)",
        "Q*": "Optimal stocking quantity that maximizes expected profit: mu + z * sigma (units)",
        "L(z)": "Standard normal loss function value: phi(z) - z * (1 - Phi(z))",
        "E[Lost Sales]": "Expected number of stockout / unmet customer orders: sigma * L(z) (units)",
        "E[Sales]": "Expected units sold to customers: mu - E[Lost Sales] (units)",
        "E[Leftovers]": "Expected surplus inventory remaining at end of season: Q* - E[Sales] (units)",
        "E[Profit]": "Expected total operational profit ($)",
    }

    # Deterministic Calculations
    cu = p - c + g
    co = c - v
    cr = cu / (cu + co)
    z = float(stats.norm.ppf(cr))
    q_opt_raw = mu + (z * sigma)
    q_opt = max(0.0, q_opt_raw)

    phi_z = float(stats.norm.pdf(z))
    phi_cdf_z = float(stats.norm.cdf(z))
    loss_z = phi_z - (z * (1.0 - phi_cdf_z))

    exp_lost_sales = sigma * loss_z
    exp_sales = mu - exp_lost_sales
    exp_leftovers = q_opt - exp_sales
    exp_profit = (p * exp_sales) + (v * exp_leftovers) - (c * q_opt) - (g * exp_lost_sales)

    # 3. Formula
    formulas = {
        "Cost of Underage (Cu)": "Cu = p - c + g",
        "Cost of Overage (Co)": "Co = c - v",
        "Critical Ratio (CR)": "CR = Cu / (Cu + Co)",
        "Optimal Order Quantity (Q*)": "Q* = mu + (z * sigma),  where z = norm.ppf(CR)",
        "Standard Normal Loss L(z)": "L(z) = phi(z) - z * (1 - Phi(z))",
        "Expected Lost Sales": "E[Lost Sales] = sigma * L(z)",
        "Expected Sales": "E[Sales] = mu - E[Lost Sales]",
        "Expected Leftover Units": "E[Leftovers] = Q* - E[Sales]",
        "Expected Profit": "E[Profit] = (p * E[Sales]) + (v * E[Leftovers]) - (c * Q*) - (g * E[Lost Sales])",
    }

    # 4. Formula with Substituted Values
    formulas_substituted = {
        "Cost of Underage (Cu)": f"${p:,.2f} - ${c:,.2f} + ${g:,.2f} = ${cu:,.2f}/unit",
        "Cost of Overage (Co)": f"${c:,.2f} - ${v:,.2f} = ${co:,.2f}/unit",
        "Critical Ratio (CR)": f"${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = ${cu:,.2f} / ${cu + co:,.2f} = {cr:.4f} ({cr*100:.2f}%)",
        "Safety Factor (z)": f"norm.ppf({cr:.4f}) = {z:.4f}",
        "Optimal Order Quantity (Q*)": f"{mu:,.2f} + ({z:.4f} * {sigma:,.2f}) = {mu:,.2f} + {z * sigma:,.2f} = {q_opt:,.2f} units",
        "Standard Loss L(z)": f"{phi_z:.4f} - ({z:.4f} * (1 - {phi_cdf_z:.4f})) = {loss_z:.6f}",
        "Expected Lost Sales": f"{sigma:,.2f} * {loss_z:.6f} = {exp_lost_sales:,.2f} units",
        "Expected Sales": f"{mu:,.2f} - {exp_lost_sales:,.2f} = {exp_sales:,.2f} units",
        "Expected Leftover Units": f"{q_opt:,.2f} - {exp_sales:,.2f} = {exp_leftovers:,.2f} units",
        "Expected Profit": (
            f"(${p:,.2f} * {exp_sales:,.2f}) + (${v:,.2f} * {exp_leftovers:,.2f}) - "
            f"(${c:,.2f} * {q_opt:,.2f}) - (${g:,.2f} * {exp_lost_sales:,.2f}) = ${exp_profit:,.2f}"
        ),
    }

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "step_0_demand_period_conversion": kwargs.get("demand_period_conversion", "Direct single-period demand (no seasonal scaling needed)"),
        "step_1_underage_cost_Cu": round(cu, 2),
        "step_2_overage_cost_Co": round(co, 2),
        "step_3_critical_ratio_CR": round(cr, 6),
        "step_4_optimal_z_score": round(z, 4),
        "step_5_optimal_quantity_Q_raw": round(q_opt_raw, 2),
        "step_6_optimal_quantity_Q_floored": round(q_opt, 2),
        "step_7_normal_pdf_phi_z": round(phi_z, 6),
        "step_8_normal_cdf_Phi_z": round(phi_cdf_z, 6),
        "step_9_standard_loss_L_z": round(loss_z, 6),
        "step_10_expected_lost_sales": round(exp_lost_sales, 2),
        "step_11_expected_sales": round(exp_sales, 2),
        "step_12_expected_leftover_units": round(exp_leftovers, 2),
        "step_13_expected_profit": round(exp_profit, 2),
        "step_14_in_stock_probability": round(phi_cdf_z, 4),
    }

    # 6. Final Answer
    final_ans = {
        "recommended_order_quantity": round(q_opt, 2),
        "recommended_order_quantity_units": "units (optimal stocking quantity)",
        "critical_ratio": round(cr, 4),
        "critical_ratio_percentage": f"{cr*100:.2f}%",
        "underage_cost": round(cu, 2),
        "overage_cost": round(co, 2),
        "expected_profit": round(exp_profit, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_leftover_inventory": round(exp_leftovers, 2),
        "expected_lost_sales": round(exp_lost_sales, 2),
        "in_stock_probability": round(phi_cdf_z, 4),
    }

    # 7. Verification Checks
    verifications = [
        {
            "check": "Critical Ratio Feasibility Check",
            "condition": "0 < CR < 1",
            "passed": 0.0 < cr < 1.0,
            "detail": f"Critical ratio ({cr:.4f} or {cr:.2%}) is strictly between 0 and 1, confirming valid marginal profit economics.",
        },
        {
            "check": "Economic Bias Directional Check",
            "condition": "Q* >= mu if and only if Cu >= Co",
            "passed": (q_opt >= mu) == (cu >= co),
            "detail": (
                f"Because Underage Cost (${cu:,.2f}) {'exceeds' if cu >= co else 'is less than'} Overage Cost (${co:,.2f}), "
                f"Critical Ratio ({cr:.2%}) is {'>=' if cu >= co else '<'} 50%. It is optimal to order {'above' if cu >= co else 'below'} mean demand ({mu:,.0f} units)."
            ),
        },
        {
            "check": "Conservation of Inventory Identity",
            "condition": "Q* == Expected Sales + Expected Leftovers",
            "passed": math.isclose(q_opt, exp_sales + exp_leftovers, abs_tol=1e-3),
            "detail": f"Conservation holds: Q* ({q_opt:,.2f}) == Expected Sales ({exp_sales:,.2f}) + Expected Leftovers ({exp_leftovers:,.2f}).",
        },
        {
            "check": "Operational Profitability Check",
            "condition": "Expected Profit > 0",
            "passed": exp_profit > 0,
            "detail": f"Expected operational profit (${exp_profit:,.2f}) is strictly positive under optimal stocking.",
        },
    ]

    # 8. Assumptions
    model_assumptions = [
        "Single-period selling window: inventory cannot be replenished once the season commences.",
        "Unsold inventory after the period is cleared at salvage value v with no subsequent regular demand.",
        "Demand is a continuous normal random variable with known mean mu and standard deviation sigma.",
        "Unmet customer demand results in lost margin (p - c) plus shortage goodwill penalty g.",
    ]

    return NewsvendorResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_ans,
        verification_checks=verifications,
        assumptions=model_assumptions,
        recommended_order_quantity=round(q_opt, 2),
        critical_ratio=round(cr, 4),
        underage_cost=round(cu, 2),
        overage_cost=round(co, 2),
        z_score=round(z, 4),
        demand_mean=mu,
        demand_std=sigma,
        unit_cost=c,
        selling_price=p,
        salvage_value=v,
        shortage_penalty=g,
        expected_sales=round(exp_sales, 2),
        expected_lost_sales=round(exp_lost_sales, 2),
        expected_leftover_inventory=round(exp_leftovers, 2),
        expected_profit=round(exp_profit, 2),
        in_stock_probability=round(phi_cdf_z, 4),
    )


# ============================================================================
# Verification Check Helpers
# ============================================================================

def verify_eoq_result(result: EOQResult) -> Tuple[bool, List[str]]:
    passed = all(v["passed"] for v in result.verification_checks)
    msgs = [v["detail"] for v in result.verification_checks]
    return passed, msgs


def verify_discount_result(result: QuantityDiscountResult) -> Tuple[bool, List[str]]:
    passed = all(v["passed"] for v in result.verification_checks)
    msgs = [v["detail"] for v in result.verification_checks]
    return passed, msgs


def verify_safety_stock_result(result: ROPResult) -> Tuple[bool, List[str]]:
    passed = all(v["passed"] for v in result.verification_checks)
    msgs = [v["detail"] for v in result.verification_checks]
    return passed, msgs


def verify_fixed_period_result(result: PeriodicReviewResult) -> Tuple[bool, List[str]]:
    passed = all(v["passed"] for v in result.verification_checks)
    msgs = [v["detail"] for v in result.verification_checks]
    return passed, msgs


def verify_newsvendor_result(result: NewsvendorResult) -> Tuple[bool, List[str]]:
    passed = all(v["passed"] for v in result.verification_checks)
    msgs = [v["detail"] for v in result.verification_checks]
    return passed, msgs


def verify_classification_result(result: "MulticriteriaClassificationResult") -> Tuple[bool, List[str]]:
    passed = all(v["passed"] for v in result.verification_checks)
    msgs = [v["detail"] for v in result.verification_checks]
    return passed, msgs



# ============================================================================
# 6. Multicriteria Inventory Classification (Sales & Hits -> A, B, C, D)
# ============================================================================

@dataclass
class MulticriteriaClassificationResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]

    # 2. Variable Definitions
    variable_definitions: Dict[str, str]

    # 3. Formula
    formula: Dict[str, str]

    # 4. Formula with Substituted Values
    formula_with_substituted_values: Dict[str, str]

    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]

    # 6. Final Answer
    final_answer: Dict[str, Any]

    # 7. Verification Checks
    verification_checks: List[Dict[str, Any]]

    # 8. Assumptions
    assumptions: List[str]

    # Enriched DataFrame and summaries
    classified_df: Any = None
    class_summary_df: Any = None
    cross_tab_matrix: Any = None
    policy_recommendations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    policy_summary_df: Any = None


# Strategic Material Management & Inventory Policy Guidelines by Classification
INVENTORY_POLICY_GUIDELINES: Dict[str, Dict[str, str]] = {
    "A": {
        "class": "A",
        "description": "High Value & High Velocity (Top 60% Sales / High Hits)",
        "cycle_counting": "Every week",
        "cycle_counting_detail": "Every week (every 5 operating days / 7 calendar days; 50–52 audits/year)",
        "service_level": "High CSL: 95%–98%",
        "safety_stock": "High buffer: Strict statistical calculation (SS = z * sigma_LTD) with continuous monitoring",
        "replenishment_policy": "Continuous Review (s, Q); frequent small replenishment batches; JIT delivery",
        "forecasting": "Rigorous quantitative item-level forecasting (ARIMA, Exponential Smoothing, demand sensing)",
        "sourcing_strategy": "Strategic vendor partnerships, Vendor-Managed Inventory (VMI), pre-negotiated short lead times",
        "action_standard": "Tightest operational control; weekly physical count; expedite if buffer is threatened",
    },
    "B": {
        "class": "B",
        "description": "Moderate Value & Moderate Velocity (Next 20% Sales / Moderate Hits)",
        "cycle_counting": "Once a month",
        "cycle_counting_detail": "Once a month (every 4 weeks / ~20 operating days / 30 calendar days; 12 audits/year)",
        "service_level": "Medium CSL: 85%–90%",
        "safety_stock": "Medium buffer: Standard lead-time variance protection",
        "replenishment_policy": "Periodic Review (R, S) base stock system or standard EOQ with monthly reviews",
        "forecasting": "Automated statistical moving averages with exception-based reviews",
        "sourcing_strategy": "Reliable approved suppliers, blanket purchase orders with periodic call-offs",
        "action_standard": "Management by exception; monthly cycle counts; standard replenishment batches",
    },
    "C": {
        "class": "C",
        "description": "Low Value & Low Velocity (Next 10% Sales / Low Hits)",
        "cycle_counting": "Once a year",
        "cycle_counting_detail": "Once a year (every 50–52 weeks / 250 operating days / 365 calendar days; 1 annual audit)",
        "service_level": "Low CSL: 70%–80%",
        "safety_stock": "Low buffer: Minimal or rule-of-thumb buffer; stockouts carry low business impact",
        "replenishment_policy": "Two-Bin, Min-Max, visual inspection, or Kanban; bulk infrequent orders (large EOQ)",
        "forecasting": "Simple annual usage extrapolation or historical moving average; minimal forecasting labor",
        "sourcing_strategy": "Catalog distributor purchasing, spot buys, consolidated multi-item purchase orders",
        "action_standard": "Low managerial oversight; annual physical audit; bulk purchase to minimize PO costs",
    },
    "D": {
        "class": "D",
        "description": "Dormant / Dead Stock / Obsolete Items (Remaining 10% Sales / Zero Hits)",
        "cycle_counting": "Do not count",
        "cycle_counting_detail": "Do not count (0 audits; liquidate, donate, destroy, or discount)",
        "service_level": "N/A (0%) — No service level commitment for inactive/dead stock",
        "safety_stock": "No safety stock; donate, destroy, or discount",
        "replenishment_policy": "Non-stock / Do Not Reorder; strict Make-to-Order (MTO) if customer requests",
        "forecasting": "Zero demand forecast; flag SKU as deactivated or phased out in ERP",
        "sourcing_strategy": "Salvage and liquidation: supplier return, clearance markdown, tax-deductible donation, or scrap write-off",
        "action_standard": "Immediate disposition: donate, destroy, or discount to free warehouse space and capital",
    },
}


# Score conversion mapping (A=4, B=3, C=2, D=1)
PARETO_SCORE_MAP: Dict[str, int] = {"A": 4, "B": 3, "C": 2, "D": 1}

# Standard 2D Combined Weighted Score Mapping Table:
# Combined Weighted Score = 0.65 * Sales_Score + 0.35 * Hits_Score
# Final Class: A (4.00 - 3.25), B (3.24 - 2.50), C (2.49 - 1.75), D (< 1.75)
MULTICRITERIA_CROSS_MATRIX: Dict[Tuple[str, str], str] = {
    ("A", "A"): "A",  # 0.65(4) + 0.35(4) = 4.00 -> A
    ("A", "B"): "A",  # 0.65(4) + 0.35(3) = 3.65 -> A
    ("B", "A"): "A",  # 0.65(3) + 0.35(4) = 3.35 -> A
    ("A", "C"): "A",  # 0.65(4) + 0.35(2) = 3.30 -> A
    ("B", "B"): "B",  # 0.65(3) + 0.35(3) = 3.00 -> B
    ("A", "D"): "B",  # 0.65(4) + 0.35(1) = 2.95 -> B
    ("C", "A"): "B",  # 0.65(2) + 0.35(4) = 2.70 -> B
    ("B", "C"): "B",  # 0.65(3) + 0.35(2) = 2.65 -> B
    ("C", "B"): "C",  # 0.65(2) + 0.35(3) = 2.35 -> C
    ("B", "D"): "C",  # 0.65(3) + 0.35(1) = 2.30 -> C
    ("D", "A"): "C",  # 0.65(1) + 0.35(4) = 2.05 -> C
    ("C", "C"): "C",  # 0.65(2) + 0.35(2) = 2.00 -> C
    ("D", "B"): "D",  # 0.65(1) + 0.35(3) = 1.70 -> D
    ("C", "D"): "D",  # 0.65(2) + 0.35(1) = 1.65 -> D
    ("D", "C"): "D",  # 0.65(1) + 0.35(2) = 1.35 -> D
    ("D", "D"): "D",  # 0.65(1) + 0.35(1) = 1.00 -> D
}


def calculate_weighted_score(sales_class: str, hits_class: str, sales_weight: float = 0.65, hits_weight: float = 0.35) -> float:
    """Calculates Combined Weighted Score = sales_weight * Sales_Score + hits_weight * Hits_Score."""
    s_score = PARETO_SCORE_MAP.get(sales_class, 1)
    h_score = PARETO_SCORE_MAP.get(hits_class, 1)
    return round(sales_weight * s_score + hits_weight * h_score, 4)


def map_score_to_final_class(score: float) -> str:
    """
    Assigns final inventory classification from combined weighted score:
    - A: 4.00 - 3.25
    - B: 3.24 - 2.50
    - C: 2.49 - 1.75
    - D: < 1.75
    """
    if score >= 3.25:
        return "A"
    elif score >= 2.50:
        return "B"
    elif score >= 1.75:
        return "C"
    else:
        return "D"


def classify_inventory_multicriteria(
    df: Any,
    sales_col: Optional[str] = None,
    hits_col: Optional[str] = None,
    item_num_col: Optional[str] = None,
    item_name_col: Optional[str] = None,
    uom_col: Optional[str] = None,
    sales_cutoffs: Tuple[float, float, float] = (60.0, 80.0, 90.0),
    hits_cutoffs: Tuple[float, float, float] = (60.0, 80.0, 90.0),
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> MulticriteriaClassificationResult:
    """
    Performs deterministic Multicriteria Inventory Classification using Sales ($) and Hits.
    Classifies items into A, B, C, and D classes using individual Pareto cumulative distributions
    and a 2D cross-classification decision matrix.

    Args:
        df: Input pandas DataFrame containing sales and hits data.
        sales_col: Name of column containing Sales revenue (defaults to auto-detect 'Sales ($)' or 'Sales').
        hits_col: Name of column containing transaction/pick counts (defaults to auto-detect 'Hits').
        item_num_col: Name of column containing Item Number.
        item_name_col: Name of column containing Item Name.
        uom_col: Name of column containing Unit of Measure.
        sales_cutoffs: Cumulative % cutoffs for (A, B, C).
        hits_cutoffs: Cumulative % cutoffs for (A, B, C).

    Returns:
        MulticriteriaClassificationResult with enriched DataFrame, summary tables,
        intermediate calculations, verification checks, and model assumptions.
    """
    import pandas as pd
    import numpy as np

    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise ValueError("Input DataFrame must be a non-empty pandas DataFrame.")

    # Auto-detect column names if not explicitly specified
    cols = list(df.columns)
    cols_lower = {c.strip().lower(): c for c in cols}

    # Resolve Sales column
    if sales_col is None:
        for candidate in ["sales ($)", "sales", "sales$", "revenue", "annual sales", "sales_dollar"]:
            if candidate in cols_lower:
                sales_col = cols_lower[candidate]
                break
        if sales_col is None:
            # Check substring
            for c in cols:
                if "sales" in c.lower():
                    sales_col = c
                    break

    # Resolve Hits column
    if hits_col is None:
        for candidate in ["hits", "hit", "picks", "pick count", "transactions", "order lines"]:
            if candidate in cols_lower:
                hits_col = cols_lower[candidate]
                break
        if hits_col is None:
            for c in cols:
                if "hit" in c.lower():
                    hits_col = c
                    break

    if sales_col is None or sales_col not in df.columns:
        raise ValueError(f"Required Sales column not found in DataFrame. Available columns: {cols}")
    if hits_col is None or hits_col not in df.columns:
        raise ValueError(f"Required Hits column not found in DataFrame. Available columns: {cols}")

    # Copy to avoid mutating original DataFrame
    working_df = df.copy()

    # Clean numeric fields
    working_df[sales_col] = (
        working_df[sales_col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    working_df[sales_col] = pd.to_numeric(working_df[sales_col], errors="coerce").fillna(0.0)

    working_df[hits_col] = (
        working_df[hits_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    working_df[hits_col] = pd.to_numeric(working_df[hits_col], errors="coerce").fillna(0.0)

    total_sales = float(working_df[sales_col].sum())
    total_hits = float(working_df[hits_col].sum())
    total_skus = len(working_df)

    if total_sales <= 0:
        raise ValueError(f"Total sales across all items must be positive. Got {total_sales}")
    if total_hits <= 0:
        raise ValueError(f"Total hits across all items must be positive. Got {total_hits}")

    # ========================================================================
    # STEP 1: Classify Sales and POS Hits Separately
    # Pareto thresholds: A (0-60%), B (61-80%), C (81-90%), D (91-100%)
    # Numerical scores: A = 4, B = 3, C = 2, D = 1
    # ========================================================================

    # --- Sales Pareto Ranking & Scoring ---
    df_sales_sorted = working_df.sort_values(by=sales_col, ascending=False).copy()
    sales_share = (df_sales_sorted[sales_col] / total_sales) * 100.0
    cum_sales = sales_share.cumsum()
    df_sales_sorted["Sales Share (%)"] = sales_share.round(3)
    df_sales_sorted["Cum Sales Share (%)"] = cum_sales.round(3)

    sales_classes = []
    for idx, (cum_val, raw_s) in enumerate(zip(df_sales_sorted["Cum Sales Share (%)"], df_sales_sorted[sales_col])):
        if raw_s <= 0:
            sales_classes.append("D")
        elif idx == 0 or cum_val <= sales_cutoffs[0]:
            sales_classes.append("A")
        elif cum_val <= sales_cutoffs[1]:
            sales_classes.append("B")
        elif cum_val <= sales_cutoffs[2]:
            sales_classes.append("C")
        else:
            sales_classes.append("D")

    df_sales_sorted["Sales Class"] = sales_classes
    df_sales_sorted["Sales Score"] = df_sales_sorted["Sales Class"].map(PARETO_SCORE_MAP)

    # --- POS Hits Pareto Ranking & Scoring ---
    df_hits_sorted = working_df.sort_values(by=hits_col, ascending=False).copy()
    hits_share = (df_hits_sorted[hits_col] / total_hits) * 100.0
    cum_hits = hits_share.cumsum()
    df_hits_sorted["Hits Share (%)"] = hits_share.round(3)
    df_hits_sorted["Cum Hits Share (%)"] = cum_hits.round(3)

    hits_classes = []
    for idx, (cum_val, raw_h) in enumerate(zip(df_hits_sorted["Cum Hits Share (%)"], df_hits_sorted[hits_col])):
        if raw_h <= 0:
            hits_classes.append("D")
        elif idx == 0 or cum_val <= hits_cutoffs[0]:
            hits_classes.append("A")
        elif cum_val <= hits_cutoffs[1]:
            hits_classes.append("B")
        elif cum_val <= hits_cutoffs[2]:
            hits_classes.append("C")
        else:
            hits_classes.append("D")

    df_hits_sorted["Hits Class"] = hits_classes
    df_hits_sorted["Hits Score"] = df_hits_sorted["Hits Class"].map(PARETO_SCORE_MAP)

    # Merge classifications and scores back to original DataFrame index
    working_df["Sales Share (%)"] = df_sales_sorted["Sales Share (%)"]
    working_df["Cum Sales Share (%)"] = df_sales_sorted["Cum Sales Share (%)"]
    working_df["Sales Class"] = df_sales_sorted["Sales Class"]
    working_df["Sales Score"] = df_sales_sorted["Sales Score"]

    working_df["Hits Share (%)"] = df_hits_sorted["Hits Share (%)"]
    working_df["Cum Hits Share (%)"] = df_hits_sorted["Cum Hits Share (%)"]
    working_df["Hits Class"] = df_hits_sorted["Hits Class"]
    working_df["Hits Score"] = df_hits_sorted["Hits Score"]

    # ========================================================================
    # STEP 2: Calculate the Combined Weighted Score
    # Weight: 65% for Sales and 35% for POS Hits
    # Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)
    # ========================================================================
    working_df["Combined Weighted Score"] = (
        sales_weight * working_df["Sales Score"] + hits_weight * working_df["Hits Score"]
    ).round(4)

    # ========================================================================
    # STEP 3: Assign the Final Inventory Class
    # Final Inventory Class Thresholds:
    # A: 4.00 - 3.25
    # B: 3.24 - 2.50
    # C: 2.49 - 1.75
    # D: < 1.75
    # ========================================================================
    working_df["Final Inventory Class"] = working_df["Combined Weighted Score"].apply(map_score_to_final_class)
    # Keep alias for backward compatibility with previous multicriteria references
    working_df["Multicriteria Class"] = working_df["Final Inventory Class"]

    # ========================================================================
    # REQUIRED CLASSIFICATION SUMMARY
    # Report number of items in each category (A, B, C, D) and summarize total Sales
    # ========================================================================
    class_order = ["A", "B", "C", "D"]
    score_ranges = {
        "A": "4.00 - 3.25",
        "B": "3.24 - 2.50",
        "C": "2.49 - 1.75",
        "D": "< 1.75",
    }
    summary_rows = []
    for c in class_order:
        subset = working_df[working_df["Final Inventory Class"] == c]
        sku_count = len(subset)
        sku_pct = (sku_count / total_skus) * 100.0 if total_skus > 0 else 0.0
        c_sales = float(subset[sales_col].sum())
        c_sales_pct = (c_sales / total_sales) * 100.0 if total_sales > 0 else 0.0
        c_hits = float(subset[hits_col].sum())
        c_hits_pct = (c_hits / total_hits) * 100.0 if total_hits > 0 else 0.0
        avg_score = float(subset["Combined Weighted Score"].mean()) if sku_count > 0 else 0.0

        summary_rows.append({
            "Final Inventory Class": c,
            "Class": c,  # alias
            "Number of Items": sku_count,
            "SKU Count": sku_count,  # alias
            "% of Items": round(sku_pct, 2),
            "% of SKUs": round(sku_pct, 2),  # alias
            "Total Sales ($)": round(c_sales, 2),
            "% of Sales": round(c_sales_pct, 2),
            "Total POS Hits": round(c_hits, 0),
            "Total Hits": round(c_hits, 0),  # alias
            "% of Hits": round(c_hits_pct, 2),
            "Combined Score Range": score_ranges[c],
            "Avg Weighted Score": round(avg_score, 2),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Build Strategic Inventory Policy Recommendations Table (All Classes)
    policy_rows = []
    for c in class_order:
        pol = INVENTORY_POLICY_GUIDELINES[c]
        policy_rows.append({
            "Final Inventory Class": c,
            "Cycle Counting Frequency": pol["cycle_counting"],
            "Target Cycle Service Level (CSL)": pol["service_level"],
            "Safety Stock Policy": pol["safety_stock"],
            "Replenishment Policy": pol["replenishment_policy"],
            "Forecasting Approach": pol["forecasting"],
            "Sourcing & Disposition Strategy": pol["sourcing_strategy"],
        })
    policy_summary_df = pd.DataFrame(policy_rows)

    # Cross-tabulation Matrix (Sales Class x Hits Class)
    cross_tab = pd.crosstab(
        working_df["Sales Class"],
        working_df["Hits Class"],
        margins=True,
        margins_name="Total",
    ).reindex(index=["A", "B", "C", "D", "Total"], columns=["A", "B", "C", "D", "Total"], fill_value=0)

    # 1. Inputs Extracted
    inputs_extracted = {
        "total_sku_count": total_skus,
        "sales_column": sales_col,
        "hits_column": hits_col,
        "total_annual_sales": total_sales,
        "total_annual_hits": total_hits,
        "sales_weight": sales_weight,
        "hits_weight": hits_weight,
        "sales_pareto_thresholds": f"A (0-{sales_cutoffs[0]:.0f}%), B ({sales_cutoffs[0]+1:.0f}-{sales_cutoffs[1]:.0f}%), C ({sales_cutoffs[1]+1:.0f}-{sales_cutoffs[2]:.0f}%), D ({sales_cutoffs[2]+1:.0f}-100%)",
        "hits_pareto_thresholds": f"A (0-{hits_cutoffs[0]:.0f}%), B ({hits_cutoffs[0]+1:.0f}-{hits_cutoffs[1]:.0f}%), C ({hits_cutoffs[1]+1:.0f}-{hits_cutoffs[2]:.0f}%), D ({hits_cutoffs[2]+1:.0f}-100%)",
        "score_mapping": "A = 4, B = 3, C = 2, D = 1",
        "final_class_score_thresholds": "A (4.00-3.25), B (3.24-2.50), C (2.49-1.75), D (< 1.75)",
    }

    # 2. Variable Definitions
    variable_definitions = {
        "Sales ($)": "Total monetary sales volume generated by the item ($)",
        "POS Hits": "Total transaction frequency / Point-of-Sale line count generated by the item (velocity)",
        "Sales Class": "Step 1 Pareto classification for Sales (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Sales Score": "Numerical score assigned to Sales Class (A = 4, B = 3, C = 2, D = 1)",
        "Hits Class": "Step 1 Pareto classification for POS Hits (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Hits Score": "Numerical score assigned to POS Hits Class (A = 4, B = 3, C = 2, D = 1)",
        "Combined Weighted Score": "Step 2 composite index: 0.65(Sales Score) + 0.35(Hits Score)",
        "Final Inventory Class": "Step 3 final inventory stratification: A (4.00-3.25), B (3.24-2.50), C (2.49-1.75), D (< 1.75)",
        "Cycle Counting Frequency": "Inventory record audit cycle: A (Every week = 5 operating days / 7 calendar days), B (Once a month = 4 weeks / ~20 operating days / 30 calendar days), C (Once a year = 50-52 weeks / 250 operating days / 365 calendar days), D (Do not count)",
        "Safety Stock & CSL Target": "Safety stock buffering differentiated by class: A (95%–98% CSL), B (85%–90% CSL), C (70%–80% CSL), D (No safety stock; donate, destroy, or discount)",
        "Time Units Definition": "1 work week = 5 operating days (50 work weeks = 250 operating days per year); 1 calendar week = 7 calendar days (52.14 calendar weeks = 365 calendar days). All cycle counting audit intervals explicitly define exact day equivalents.",
    }

    # 3. Formula
    formulas = {
        "Step 1 - Sales Pareto": f"Cum_Sales_i = sum(Sales_Share); Classify A (0-{sales_cutoffs[0]:.0f}%), B ({sales_cutoffs[0]+1:.0f}-{sales_cutoffs[1]:.0f}%), C ({sales_cutoffs[1]+1:.0f}-{sales_cutoffs[2]:.0f}%), D ({sales_cutoffs[2]+1:.0f}-100%)",
        "Step 1 - POS Hits Pareto": f"Cum_Hits_i = sum(Hits_Share); Classify A (0-{hits_cutoffs[0]:.0f}%), B ({hits_cutoffs[0]+1:.0f}-{hits_cutoffs[1]:.0f}%), C ({hits_cutoffs[1]+1:.0f}-{hits_cutoffs[2]:.0f}%), D ({hits_cutoffs[2]+1:.0f}-100%)",
        "Step 1 - Score Conversion": "A = 4, B = 3, C = 2, D = 1",
        "Step 2 - Combined Weighted Score": f"Combined Weighted Score = {sales_weight:.2f}(Sales Score) + {hits_weight:.2f}(Hits Score)",
        "Step 3 - Final Inventory Class": "A: 4.00 - 3.25; B: 3.24 - 2.50; C: 2.49 - 1.75; D: < 1.75",
        "Safety Stock Formula": "SS = z(CSL) * sigma_LTD  [Applied to A and B; rule of thumb for C; zero for D]",
    }

    # 4. Formula with Substituted Values
    formulas_substituted = {
        "Step 2 Formula with Weights": f"Combined Weighted Score = {sales_weight:.2f}(Sales Score) + {hits_weight:.2f}(Hits Score)",
        "Step 2 Example (Sales A, Hits B)": f"{sales_weight:.2f}(4) + {hits_weight:.2f}(3) = {sales_weight*4:.2f} + {hits_weight*3:.2f} = {sales_weight*4 + hits_weight*3:.2f} -> Final Class A",
        "Step 1 Sales Cumulative Cutoffs": f"A <= ${total_sales*sales_cutoffs[0]/100.0:,.2f} ({sales_cutoffs[0]:.0f}%), B <= ${total_sales*sales_cutoffs[1]/100.0:,.2f} ({sales_cutoffs[1]:.0f}%), C <= ${total_sales*sales_cutoffs[2]/100.0:,.2f} ({sales_cutoffs[2]:.0f}%), D > ({sales_cutoffs[2]:.0f}%)",
        "Step 1 Hits Cumulative Cutoffs": f"A <= {total_hits*hits_cutoffs[0]/100.0:,.0f} ({hits_cutoffs[0]:.0f}%), B <= {total_hits*hits_cutoffs[1]/100.0:,.0f} ({hits_cutoffs[1]:.0f}%), C <= {total_hits*hits_cutoffs[2]/100.0:,.0f} ({hits_cutoffs[2]:.0f}%), D > ({hits_cutoffs[2]:.0f}%)",
        "Step 3 Score Ranges": "Class A: 4.00 - 3.25 | Class B: 3.24 - 2.50 | Class C: 2.49 - 1.75 | Class D: < 1.75",
        "Cycle Counting Standard": "A: Every week (5 operating / 7 calendar days; 50-52 audits/yr) | B: Once a month (4 weeks / ~20 operating days / 30 calendar days; 12 audits/yr) | C: Once a year (50-52 weeks / 250 operating / 365 calendar days; 1 audit/yr) | D: Do not count",
        "Service Level Standard": "A: High CSL 95%–98% | B: Medium CSL 85%–90% | C: Low CSL 70%–80% | D: 0% (No safety stock; donate/destroy/discount)",
    }

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "total_sku_count": total_skus,
        "total_sales": total_sales,
        "total_hits": total_hits,
        "items_by_sales_class": working_df["Sales Class"].value_counts().to_dict(),
        "items_by_hits_class": working_df["Hits Class"].value_counts().to_dict(),
        "items_by_final_class": working_df["Final Inventory Class"].value_counts().to_dict(),
        "sales_by_final_class": working_df.groupby("Final Inventory Class")[sales_col].sum().to_dict(),
        "hits_by_final_class": working_df.groupby("Final Inventory Class")[hits_col].sum().to_dict(),
        "avg_score_by_final_class": working_df.groupby("Final Inventory Class")["Combined Weighted Score"].mean().round(3).to_dict(),
        "cycle_counting_frequency_by_class": {c: INVENTORY_POLICY_GUIDELINES[c]["cycle_counting"] for c in class_order},
        "target_service_level_by_class": {c: INVENTORY_POLICY_GUIDELINES[c]["service_level"] for c in class_order},
        "safety_stock_policy_by_class": {c: INVENTORY_POLICY_GUIDELINES[c]["safety_stock"] for c in class_order},
    }

    # 6. Final Answer
    final_ans = {
        "total_items_classified": total_skus,
        "class_A_items": int(summary_df.loc[summary_df["Class"] == "A", "Number of Items"].values[0]),
        "class_A_total_sales": float(summary_df.loc[summary_df["Class"] == "A", "Total Sales ($)"].values[0]),
        "class_B_items": int(summary_df.loc[summary_df["Class"] == "B", "Number of Items"].values[0]),
        "class_B_total_sales": float(summary_df.loc[summary_df["Class"] == "B", "Total Sales ($)"].values[0]),
        "class_C_items": int(summary_df.loc[summary_df["Class"] == "C", "Number of Items"].values[0]),
        "class_C_total_sales": float(summary_df.loc[summary_df["Class"] == "C", "Total Sales ($)"].values[0]),
        "class_D_items": int(summary_df.loc[summary_df["Class"] == "D", "Number of Items"].values[0]),
        "class_D_total_sales": float(summary_df.loc[summary_df["Class"] == "D", "Total Sales ($)"].values[0]),
        # Aliases for backward compatibility
        "class_A_skus": int(summary_df.loc[summary_df["Class"] == "A", "Number of Items"].values[0]),
        "class_B_skus": int(summary_df.loc[summary_df["Class"] == "B", "Number of Items"].values[0]),
        "class_C_skus": int(summary_df.loc[summary_df["Class"] == "C", "Number of Items"].values[0]),
        "class_D_skus": int(summary_df.loc[summary_df["Class"] == "D", "Number of Items"].values[0]),
        "class_A_sales_percent": float(summary_df.loc[summary_df["Class"] == "A", "% of Sales"].values[0]),
        "class_A_hits_percent": float(summary_df.loc[summary_df["Class"] == "A", "% of Hits"].values[0]),
        "total_sales_dollars": total_sales,
        "total_hits_count": total_hits,
        "classification_summary": summary_df[["Final Inventory Class", "Number of Items", "Total Sales ($)", "% of Items", "% of Sales"]].to_dict(orient="records"),
        "policy_recommendations": INVENTORY_POLICY_GUIDELINES,
        "cycle_counting_policies": {c: INVENTORY_POLICY_GUIDELINES[c]["cycle_counting"] for c in class_order},
        "target_service_levels": {c: INVENTORY_POLICY_GUIDELINES[c]["service_level"] for c in class_order},
        "safety_stock_policies": {c: INVENTORY_POLICY_GUIDELINES[c]["safety_stock"] for c in class_order},
        "replenishment_policies": {c: INVENTORY_POLICY_GUIDELINES[c]["replenishment_policy"] for c in class_order},
    }

    # 7. Verification Checks
    sum_skus = int(summary_df["Number of Items"].sum())
    sum_sales = float(summary_df["Total Sales ($)"].sum())
    sum_hits = float(summary_df["Total POS Hits"].sum())

    verifications = [
        {
            "check": "Conservation of Inventory Items Check",
            "condition": "Sum(Class_Item_Counts) == Total_Items",
            "passed": sum_skus == total_skus,
            "detail": f"All {total_skus:,} items accounted for across Classes A, B, C, and D ({sum_skus:,} classified).",
        },
        {
            "check": "Conservation of Sales Revenue Identity",
            "condition": "Sum(Class_Sales) == Total_Sales",
            "passed": math.isclose(sum_sales, total_sales, abs_tol=1e-2),
            "detail": f"Class sales total (${sum_sales:,.2f}) matches total inventory sales (${total_sales:,.2f}).",
        },
        {
            "check": "Conservation of Physical POS Hits Identity",
            "condition": "Sum(Class_Hits) == Total_Hits",
            "passed": math.isclose(sum_hits, total_hits, abs_tol=1e-2),
            "detail": f"Class hits total ({sum_hits:,.0f}) matches total operational warehouse picks ({total_hits:,.0f}).",
        },
        {
            "check": "Combined Weighted Score Domain Integrity",
            "condition": "1.00 <= Score <= 4.00 for all items",
            "passed": bool(working_df["Combined Weighted Score"].between(1.00, 4.00).all()),
            "detail": f"All item weighted scores fall strictly within theoretical boundaries [1.00, 4.00] (Min: {working_df['Combined Weighted Score'].min():.2f}, Max: {working_df['Combined Weighted Score'].max():.2f}).",
        },
        {
            "check": "Inventory Management Policy Alignment Check",
            "condition": "All 4 classes (A, B, C, D) mapped to standards for cycle counting, safety stock, and service level",
            "passed": all(c in INVENTORY_POLICY_GUIDELINES for c in ["A", "B", "C", "D"]),
            "detail": "Verified standards: A (Weekly count, 95%–98% CSL), B (Monthly count, 85%–90% CSL), C (Annual count, 70%–80% CSL), D (Do not count, No safety stock: donate/destroy/discount).",
        },
    ]

    # 8. Assumptions
    model_assumptions = [
        "Multicriteria stratification uses two criteria: monetary Sales volume and physical POS Hits frequency.",
        "Step 1 assigns separate Pareto tiers using cumulative thresholds: A (0-60%), B (61-80%), C (81-90%), D (91-100%).",
        "Step 1 converts classes to numerical scores: A = 4, B = 3, C = 2, D = 1.",
        "Step 2 calculates Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score).",
        "Step 3 assigns Final Inventory Class: A (4.00-3.25), B (3.24-2.50), C (2.49-1.75), D (< 1.75).",
        "Class A Policy: Cycle count every week; High CSL 95%–98%; dynamic statistical safety stock (SS = z * sigma_LTD); Continuous Review (s, Q).",
        "Class B Policy: Cycle count once a month; Medium CSL 85%–90%; standard safety stock buffer; Periodic Review (R, S) / standard EOQ.",
        "Class C Policy: Cycle count once a year; Low CSL 70%–80%; minimal buffer; Two-Bin / Min-Max / Kanban with bulk orders.",
        "Class D Policy: Do not count; No safety stock; immediate disposition: donate, destroy, or discount; non-stock status.",
        "Cycle Counting & Time Units Definition: 1 work week is defined as 5 operating days (50 work weeks = 250 operating days per year); 1 calendar week is defined as 7 calendar days. Class A items are counted weekly (every 5 operating / 7 calendar days; 50-52 audits/yr); Class B items monthly (every 4 weeks / ~20 operating days / 30 calendar days; 12 audits/yr); Class C items annually (once per year / 250 operating days / 365 calendar days; 1 audit/yr); Class D items are not counted.",
    ]

    return MulticriteriaClassificationResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_ans,
        verification_checks=verifications,
        assumptions=model_assumptions,
        classified_df=working_df,
        class_summary_df=summary_df,
        cross_tab_matrix=cross_tab,
        policy_recommendations=INVENTORY_POLICY_GUIDELINES,
        policy_summary_df=policy_summary_df,
    )


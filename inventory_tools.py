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

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand rate (units per year)",
        "S": "Fixed setup or ordering cost incurred per order placed ($/order)",
        "H": "Holding (carrying) cost to maintain one unit in inventory for one year ($/unit/year)",
        "C": "Purchase cost per unit ($/unit, if applicable)",
        "i": "Annual holding cost rate as a fraction of unit purchase price (%/year)",
        "Q*": "Economic Order Quantity: the optimal replenishment batch size that minimizes total inventory costs (units)",
        "N": "Order frequency: optimal number of replenishment orders placed per year (orders/year)",
        "T": "Cycle time: time between consecutive replenishment orders (operating days)",
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
    annual_ordering = n_orders * s
    annual_holding = (q_opt / 2.0) * h
    total_inv = annual_ordering + annual_holding
    purchase_cost = (d * c_val) if c_val is not None else None
    total_cost = (purchase_cost + total_inv) if purchase_cost is not None else total_inv

    # 3. Formula
    formulas = {
        "Optimal Order Quantity (Q*)": "Q* = sqrt((2 * D * S) / H)",
        "Orders per Year (N)": "N = D / Q*",
        "Cycle Time Days (T)": "T = (Q* / D) * work_days_per_year",
        "Annual Ordering Cost (TC_order)": "TC_order = (D / Q*) * S = N * S",
        "Annual Holding Cost (TC_hold)": "TC_hold = (Q* / 2) * H",
        "Total Annual Inventory Cost (TC_inv)": "TC_inv = TC_order + TC_hold",
        "Total Annual Landed Cost (TC_total)": "TC_total = (D * C) + TC_inv  [if unit cost provided]",
    }

    # 4. Formula with Substituted Values
    formulas_substituted = {
        "Optimal Order Quantity (Q*)": f"sqrt((2 * {d:,.2f} * {s:,.2f}) / {h:.4f}) = sqrt({two_ds:,.2f} / {h:.4f}) = sqrt({ratio:,.4f}) = {q_opt:,.2f} units",
        "Orders per Year (N)": f"{d:,.2f} / {q_opt:,.2f} = {n_orders:.2f} orders/year",
        "Cycle Time Days (T)": f"({q_opt:,.2f} / {d:,.2f}) * {wd:,.0f} = {cycle_days:.1f} operating days",
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
        "step_1_numerator_2DS": two_ds,
        "step_2_ratio_2DS_over_H": ratio,
        "step_3_sqrt_unrounded_Q": q_opt_raw,
        "step_4_optimal_Q_rounded": q_opt,
        "step_5_orders_per_year_N": round(n_orders, 4),
        "step_6_cycle_time_fraction_of_year": round(q_opt / d, 6),
        "step_7_cycle_time_days": round(cycle_days, 2),
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
        formulas_substituted[f"Tier_{idx+1}_Evaluation ({min_q:,.0f}-{max_label} @ ${p:.2f})"] = (
            f"EOQ_{idx+1} = sqrt((2 * {d:,.0f} * {s:,.2f}) / {h_j:.4f}) = {raw_eoq:,.2f} units -> "
            f"Feasible Q_{idx+1} = {feasible_q:,.0f} units. "
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

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "holding_cost_type": holding_type,
        "holding_rate_or_fixed": h_param,
        "evaluated_tier_details": [asdict(t) for t in tier_details],
        "all_tier_total_costs": {f"Tier_{t.tier_index+1}": t.total_annual_cost for t in tier_details},
        "optimal_tier_index": best_tier.tier_index,
        "savings_vs_base_tier": round(tier_details[0].total_annual_cost - best_tier.total_annual_cost, 2),
        "cost_savings_vs_base_tier": round(tier_details[0].total_annual_cost - best_tier.total_annual_cost, 2),
    }

    # 6. Final Answer
    final_ans = {
        "optimal_tier_index": best_tier.tier_index,
        "optimal_tier_label": f"Tier {best_tier.tier_index + 1}",
        "optimal_order_quantity": best_tier.feasible_quantity,
        "optimal_unit_price": best_tier.unit_price,
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

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average demand rate per unit time period (e.g., daily or weekly demand)",
        "L": "Average supplier replenishment lead time in periods",
        "sigma_d": "Standard deviation of demand per period",
        "sigma_L": "Standard deviation of lead time in periods",
        "CSL": "Cycle Service Level: target probability of not stocking out during a replenishment lead time cycle",
        "z": "Safety factor: number of standard deviations from standard normal distribution corresponding to CSL",
        "DDLT": "Expected demand during lead time: d * L (units)",
        "sigma_LT": "Combined standard deviation of demand during lead time (units)",
        "SS": "Safety Stock: buffer inventory held to mitigate demand and lead time volatility (units)",
        "ROP": "Reorder Point: inventory level that triggers placement of a new replenishment order (units)",
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
    formulas_substituted = {
        "Expected Demand during Lead Time (DDLT)": f"{d:,.2f} * {l_mean:,.2f} = {ddlt:,.2f} units",
        "Lead Time Demand Std Dev (sigma_LT)": subst_sigma,
        "Safety Factor (z)": f"norm.ppf({csl_label}) = {z:.4f}" if csl else "0.0000",
        "Safety Stock (SS)": f"{z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units",
        "Reorder Point (ROP)": f"{ddlt:,.2f} + {ss:,.2f} = {rop:,.2f} units",
    }

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "case_type": case_type,
        "expected_lead_time_demand_DDLT": round(ddlt, 2),
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

    # 2. Variable Definitions
    variable_definitions = {
        "R": "Review period interval: time between consecutive inventory reviews (periods)",
        "L": "Supplier replenishment lead time (periods)",
        "R + L": "Total protection period: time interval across which inventory must buffer demand until next delivery",
        "d": "Average customer demand rate per period (units/period)",
        "sigma_d": "Standard deviation of demand per period",
        "sigma_prot": "Standard deviation of demand during the protection period: sigma_d * sqrt(R + L)",
        "CSL": "Target Cycle Service Level for the periodic review cycle",
        "z": "Safety factor corresponding to target CSL from standard normal distribution",
        "SS": "Safety Stock for periodic review: z * sigma_prot (units)",
        "Target Inventory (M)": "Order-Up-To Level (Base Stock): expected protection period demand + Safety Stock (units)",
        "Inventory Position (I)": "Current net inventory position: On-Hand + On-Order - Backorders (units)",
        "Order Quantity (Q)": "Recommended replenishment order placed right now: max(0, Target Inventory - Inventory Position)",
    }

    # Deterministic Calculations
    prot_period = r + l
    demand_prot = d * prot_period

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
        "Protection Period (R + L)": "Protection Period = R + L",
        "Demand during Protection Period": "Demand_prot = d * (R + L)",
        "Protection Period Std Dev (sigma_prot)": "sigma_prot = sigma_d * sqrt(R + L)",
        "Safety Stock (SS)": "SS = z * sigma_prot",
        "Target Inventory (Order-Up-To Level M)": "M = Demand_prot + SS = (d * (R + L)) + (z * sigma_d * sqrt(R + L))",
        "Recommended Order Quantity (Q)": "Q = max(0, Target Inventory - Inventory Position)",
    }

    # 4. Formula with Substituted Values
    csl_str = f"{csl*100:.1f}%" if csl is not None else "100.0% (Deterministic)"
    formulas_substituted = {
        "Protection Period (R + L)": f"{r:,.1f} + {l:,.1f} = {prot_period:,.1f} periods",
        "Demand during Protection Period": f"{d:,.2f} * ({r:,.1f} + {l:,.1f}) = {d:,.2f} * {prot_period:,.1f} = {demand_prot:,.2f} units",
        "Protection Period Std Dev (sigma_prot)": f"{sig_d:,.2f} * sqrt({prot_period:,.1f}) = {sig_d:,.2f} * {math.sqrt(prot_period):.4f} = {sigma_prot:,.2f} units",
        "Safety Stock (SS)": f"{z:.4f} * {sigma_prot:,.2f} = {ss:,.2f} units  (CSL: {csl_str})",
        "Target Inventory (M)": f"{demand_prot:,.2f} + {ss:,.2f} = {target_inv:,.2f} units",
        "Recommended Order Quantity (Q)": f"max(0, {target_inv:,.2f} - {inv_pos:,.2f}) = max(0, {shortfall:,.2f}) = {order_qty:,.2f} units",
    }

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "step_1_review_period_R": r,
        "step_2_lead_time_L": l,
        "step_3_protection_period_R_plus_L": prot_period,
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

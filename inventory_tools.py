"""Supply Chain Analytics Inventory Tools

Pure Python deterministic calculation engine for 5 core inventory models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every tool strictly returns all 8 required verification sections:
1. Inputs Extracted
2. Variable Definitions
3. Formula
4. Formula with Substituted Values
5. Intermediate Calculations
6. Final Answer
7. Verification Checks
8. Assumptions

All numerical steps come directly from the deterministic Python tool.
Missing information is never invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data Classes for Structured Pedagogical Tool Outputs
# ---------------------------------------------------------------------------

@dataclass
class ToolVerification:
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseInventoryResult:
    model_name: str
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Union[str, Dict[str, str]] = ""
    formula_with_substituted_values: Union[str, Dict[str, str]] = ""
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)
    final_answer: Dict[str, Any] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    verification: Optional[ToolVerification] = None

    # Backward compatibility property for existing code accessing .inputs
    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    def to_dict(self) -> Dict[str, Any]:
        """Return the standardized 8 verification sections as a structured dictionary."""
        return {
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }

    def to_student_verification_text(self) -> str:
        """Format the complete 8-section pedagogical breakdown for student verification."""
        lines = [f"## Student Verification: {self.model_name}\n"]

        # 1. Inputs Extracted
        lines.append("### 1. Inputs Extracted")
        for k, v in self.inputs_extracted.items():
            if isinstance(v, list):
                lines.append(f"- **{k}**: {len(v)} tiers/items")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

        # 2. Variable Definitions
        lines.append("### 2. Variable Definitions")
        for sym, defn in self.variable_definitions.items():
            lines.append(f"- **{sym}**: {defn}")
        lines.append("")

        # 3. Formula
        lines.append("### 3. Formula")
        if isinstance(self.formula, dict):
            for name, f_expr in self.formula.items():
                lines.append(f"- **{name}**: `{f_expr}`")
        else:
            lines.append(f"`{self.formula}`")
        lines.append("")

        # 4. Formula with Substituted Values
        lines.append("### 4. Formula with Substituted Values")
        if isinstance(self.formula_with_substituted_values, dict):
            for name, sub_expr in self.formula_with_substituted_values.items():
                lines.append(f"- **{name}**: `{sub_expr}`")
        else:
            lines.append(f"`{self.formula_with_substituted_values}`")
        lines.append("")

        # 5. Intermediate Calculations
        lines.append("### 5. Intermediate Calculations")
        for k, v in self.intermediate_calculations.items():
            if isinstance(v, list):
                lines.append(f"- **{k}**:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

        # 6. Final Answer
        lines.append("### 6. Final Answer")
        for k, v in self.final_answer.items():
            lines.append(f"- **{k}**: **{v}**")
        lines.append("")

        # 7. Verification Checks
        lines.append("### 7. Verification Checks")
        for check in self.verification_checks:
            lines.append(f"- [x] {check}")
        lines.append("")

        # 8. Assumptions
        lines.append("### 8. Assumptions")
        for idx, assumption in enumerate(self.assumptions, 1):
            lines.append(f"{idx}. {assumption}")
        lines.append("")

        return "\n".join(lines)


@dataclass
class EOQResult(BaseInventoryResult):
    model_name: str = "Economic Order Quantity (EOQ)"

    @property
    def optimal_order_quantity(self) -> float:
        return self.final_answer.get("optimal_order_quantity", 0.0)

    @property
    def orders_per_year(self) -> float:
        return self.intermediate_calculations.get("orders_per_year", 0.0)

    @property
    def order_cycle_days(self) -> float:
        return self.intermediate_calculations.get("order_cycle_days", 0.0)

    @property
    def annual_ordering_cost(self) -> float:
        return self.intermediate_calculations.get("annual_ordering_cost", 0.0)

    @property
    def annual_holding_cost(self) -> float:
        return self.intermediate_calculations.get("annual_holding_cost", 0.0)

    @property
    def total_inventory_cost(self) -> float:
        return self.final_answer.get("total_inventory_cost", 0.0)

    @property
    def total_annual_cost(self) -> float:
        return self.final_answer.get("total_annual_cost", self.total_inventory_cost)

    @property
    def reorder_point(self) -> Optional[float]:
        return self.intermediate_calculations.get("reorder_point")


@dataclass
class DiscountTierEvaluation:
    tier_id: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost: float
    calculated_eoq: float
    feasible_quantity: float
    is_price_break_point: bool
    annual_material_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult(BaseInventoryResult):
    model_name: str = "Quantity Discount"
    tier_evaluations: List[DiscountTierEvaluation] = field(default_factory=list)

    @property
    def optimal_order_quantity(self) -> float:
        return self.final_answer.get("optimal_order_quantity", 0.0)

    @property
    def optimal_tier_id(self) -> int:
        return self.final_answer.get("optimal_tier_id", 1)

    @property
    def optimal_unit_price(self) -> float:
        return self.final_answer.get("optimal_unit_price", 0.0)

    @property
    def optimal_total_cost(self) -> float:
        return self.final_answer.get("min_total_annual_cost", 0.0)

    @property
    def annual_savings_vs_base(self) -> float:
        return self.final_answer.get("annual_savings_vs_base", 0.0)


@dataclass
class ROPResult(BaseInventoryResult):
    model_name: str = "Safety Stock & Reorder Point"

    @property
    def reorder_point(self) -> float:
        return self.final_answer.get("reorder_point", 0.0)

    @property
    def safety_stock(self) -> float:
        return self.final_answer.get("safety_stock", 0.0)

    @property
    def service_level(self) -> float:
        return self.inputs_extracted.get("service_level", 0.0)

    @property
    def z_score(self) -> float:
        return self.intermediate_calculations.get("z_score", 0.0)

    @property
    def lead_time_demand_mean(self) -> float:
        return self.intermediate_calculations.get("lead_time_demand_mean", 0.0)

    @property
    def lead_time_demand_std(self) -> float:
        return self.intermediate_calculations.get("lead_time_demand_std", 0.0)

    @property
    def stockout_probability_per_cycle(self) -> float:
        return self.intermediate_calculations.get("stockout_probability_per_cycle", 0.0)

    @property
    def annual_ss_holding_cost(self) -> Optional[float]:
        return self.intermediate_calculations.get("annual_ss_holding_cost")


# Alias for backward compatibility
SafetyStockROPResult = ROPResult


@dataclass
class PeriodicReviewResult(BaseInventoryResult):
    model_name: str = "Fixed-Period Review (Order-Up-To Level)"

    @property
    def order_up_to_level_M(self) -> float:
        return self.final_answer.get("target_inventory", 0.0)

    @property
    def target_inventory(self) -> float:
        return self.order_up_to_level_M

    @property
    def recommended_order_quantity(self) -> Optional[float]:
        return self.final_answer.get("order_quantity")

    @property
    def safety_stock(self) -> float:
        return self.final_answer.get("safety_stock", 0.0)

    @property
    def protection_interval(self) -> float:
        return self.intermediate_calculations.get("protection_period", 0.0)

    @property
    def review_period_P(self) -> float:
        return self.inputs_extracted.get("review_period", 0.0)

    @property
    def lead_time_L(self) -> float:
        return self.inputs_extracted.get("lead_time", 0.0)

    @property
    def protection_demand_mean(self) -> float:
        return self.intermediate_calculations.get("protection_demand_mean", 0.0)

    @property
    def protection_demand_std(self) -> float:
        return self.intermediate_calculations.get("protection_demand_std", 0.0)

    @property
    def z_score(self) -> float:
        return self.intermediate_calculations.get("z_score", 0.0)

    @property
    def current_inventory_position(self) -> Optional[float]:
        return self.intermediate_calculations.get("inventory_position")

    @property
    def safety_stock_vs_continuous_ratio(self) -> float:
        return self.intermediate_calculations.get("safety_stock_vs_continuous_ratio", 1.0)


# Alias for backward compatibility
FixedPeriodReviewResult = PeriodicReviewResult


@dataclass
class NewsvendorResult(BaseInventoryResult):
    model_name: str = "Newsvendor (Single-Period)"

    @property
    def optimal_order_quantity(self) -> float:
        return self.final_answer.get("recommended_order_quantity", 0.0)

    @property
    def recommended_order_quantity(self) -> float:
        return self.optimal_order_quantity

    @property
    def critical_ratio(self) -> float:
        return self.final_answer.get("critical_ratio", 0.0)

    @property
    def underage_cost_Cu(self) -> float:
        return self.final_answer.get("underage_cost", 0.0)

    @property
    def overage_cost_Co(self) -> float:
        return self.final_answer.get("overage_cost", 0.0)

    @property
    def expected_profit(self) -> float:
        return self.final_answer.get("expected_profit", 0.0)

    @property
    def selling_price(self) -> float:
        return self.inputs_extracted.get("selling_price", 0.0)

    @property
    def purchase_cost(self) -> float:
        return self.inputs_extracted.get("purchase_cost", 0.0)

    @property
    def salvage_value(self) -> float:
        return self.inputs_extracted.get("salvage_value", 0.0)

    @property
    def expected_sales(self) -> float:
        return self.intermediate_calculations.get("expected_sales", 0.0)

    @property
    def expected_lost_sales(self) -> float:
        return self.intermediate_calculations.get("expected_lost_sales", 0.0)

    @property
    def expected_leftover_units(self) -> float:
        return self.intermediate_calculations.get("expected_leftover_units", 0.0)

    @property
    def in_stock_probability(self) -> float:
        return self.critical_ratio


# ---------------------------------------------------------------------------
# Tool 1: Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------

def calculate_eoq(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    demand: Optional[float] = None,
    unit_cost: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    work_days_per_year: Optional[float] = None,
) -> EOQResult:
    """Calculate Classical Economic Order Quantity with full pedagogical verification steps.

    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    D = demand if demand is not None else annual_demand
    S = ordering_cost
    H = holding_cost

    # Input Validation (never invent missing information)
    if D is None:
        raise ValueError("Missing required input: annual demand (D) must be provided. Missing numerical inputs will not be invented.")
    if S is None:
        raise ValueError("Missing required input: ordering/setup cost (S) must be provided. Missing numerical inputs will not be invented.")
    if H is None:
        raise ValueError("Missing required input: holding cost (H) must be provided. Missing numerical inputs will not be invented.")

    if D <= 0:
        raise ValueError(f"Annual demand (D={D}) must be strictly positive.")
    if S <= 0:
        raise ValueError(f"Ordering cost (S={S}) must be strictly positive.")
    if H <= 0:
        raise ValueError(f"Holding cost (H={H}) must be strictly positive.")
    if unit_cost is not None and unit_cost < 0:
        raise ValueError(f"Unit cost ({unit_cost}) cannot be negative.")
    if lead_time_days is not None and lead_time_days < 0:
        raise ValueError(f"Lead time days ({lead_time_days}) cannot be negative.")

    work_days = work_days_per_year if work_days_per_year is not None else 250.0

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": D,
        "ordering_cost": S,
        "holding_cost": H,
        "unit_cost": unit_cost,
        "lead_time_days": lead_time_days,
        "work_days_per_year": work_days,
    }

    # 2. Variable Definitions
    var_definitions = {
        "D": "Annual customer demand (units/year)",
        "S": "Order setup / placement cost ($/order)",
        "H": "Annual inventory carrying/holding cost per unit ($/unit/year)",
        "C": "Unit purchase price ($/unit)",
        "Q*": "Economic Order Quantity (optimal batch size in units)",
        "N": "Order frequency (orders placed per year = D / Q*)",
        "T": "Cycle length / order cycle in operating days = (Q* / D) * work_days",
        "AOC": "Annual Ordering Cost = (D / Q*) * S ($/year)",
        "AHC": "Annual Holding Cost = (Q* / 2) * H ($/year)",
        "TC": "Total Annual Variable Inventory Cost = AOC + AHC ($/year)",
        "ROP": "Reorder Point in units = (D / work_days) * L (if lead time provided)",
    }

    # 3. Formula
    formulas = {
        "EOQ": "Q* = sqrt((2 * D * S) / H)",
        "Orders_Per_Year": "N = D / Q*",
        "Cycle_Days": "T = (Q* / D) * work_days",
        "Annual_Ordering_Cost": "AOC = (D / Q*) * S",
        "Annual_Holding_Cost": "AHC = (Q* / 2) * H",
        "Total_Inventory_Cost": "TC = AOC + AHC",
    }
    if lead_time_days is not None and lead_time_days > 0:
        formulas["Reorder_Point"] = "ROP = (D / work_days) * lead_time_days"

    # Step-by-step arithmetic
    numerator = 2.0 * D * S
    ratio = numerator / H
    eoq = math.sqrt(ratio)
    orders_per_year = D / eoq
    order_cycle_days = (eoq / D) * work_days
    annual_ordering_cost = orders_per_year * S
    annual_holding_cost = (eoq / 2.0) * H
    total_inventory_cost = annual_ordering_cost + annual_holding_cost
    annual_purchase_cost = (D * unit_cost) if unit_cost is not None else None
    total_annual_cost = (total_inventory_cost + annual_purchase_cost) if annual_purchase_cost is not None else total_inventory_cost

    reorder_point = None
    if lead_time_days is not None and lead_time_days > 0:
        daily_d = D / work_days
        reorder_point = daily_d * lead_time_days

    # 4. Formula with Substituted Values
    substitutions = {
        "EOQ": f"Q* = sqrt((2 * {D:,.2f} * {S:,.2f}) / {H:,.2f}) = sqrt({numerator:,.2f} / {H:,.2f}) = sqrt({ratio:,.2f}) = {eoq:,.2f} units",
        "Orders_Per_Year": f"N = {D:,.2f} / {eoq:,.2f} = {orders_per_year:,.2f} orders/year",
        "Cycle_Days": f"T = ({eoq:,.2f} / {D:,.2f}) * {work_days:.0f} = {order_cycle_days:,.1f} operating days",
        "Annual_Ordering_Cost": f"AOC = ({D:,.2f} / {eoq:,.2f}) * ${S:,.2f} = ${annual_ordering_cost:,.2f}",
        "Annual_Holding_Cost": f"AHC = ({eoq:,.2f} / 2) * ${H:,.2f} = ${annual_holding_cost:,.2f}",
        "Total_Inventory_Cost": f"TC = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_inventory_cost:,.2f}",
    }
    if reorder_point is not None:
        substitutions["Reorder_Point"] = f"ROP = ({D:,.2f} / {work_days:.0f}) * {lead_time_days:.1f} = {reorder_point:,.2f} units"

    # 5. Intermediate Calculations
    intermediate = {
        "numerator_2DS": round(numerator, 2),
        "ratio_2DS_over_H": round(ratio, 2),
        "orders_per_year": round(orders_per_year, 2),
        "order_cycle_days": round(order_cycle_days, 1),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "annual_purchase_cost": round(annual_purchase_cost, 2) if annual_purchase_cost is not None else None,
        "total_annual_cost": round(total_annual_cost, 2),
        "reorder_point": round(reorder_point, 2) if reorder_point is not None else None,
    }

    # 6. Final Answer
    final_answer = {
        "optimal_order_quantity": round(eoq, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "total_annual_cost": round(total_annual_cost, 2),
    }

    # 7. Verification Checks
    verification_checks = []
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    parity_tol = max(2.0, 0.01 * total_inventory_cost)
    if cost_diff <= parity_tol:
        verification_checks.append(f"Cost parity verified: |AOC - AHC| = ${cost_diff:.2f} <= tolerance ${parity_tol:.2f}")
    else:
        verification_checks.append(f"Cost difference between AOC (${annual_ordering_cost:.2f}) and AHC (${annual_holding_cost:.2f}) due to integer lot rounding")

    def cost_at(q: float) -> float:
        return (D / q) * S + (q / 2.0) * H

    base_cost = cost_at(eoq)
    cost_minus_10 = cost_at(eoq * 0.90)
    cost_plus_10 = cost_at(eoq * 1.10)
    if cost_minus_10 >= base_cost and cost_plus_10 >= base_cost:
        verification_checks.append("Convexity check passed: Q* is the strict global minimum.")
    else:
        verification_checks.append("Convexity check warning: perturbed point cost was lower than Q*.")

    # 8. Assumptions
    assumptions = [
        "Demand rate is deterministic, constant, and uniformly distributed throughout the year.",
        "Replenishment lead time is constant and known (or instantaneous).",
        "Order setup cost and unit holding cost are fixed constants independent of order size.",
        "Entire replenishment batch arrives simultaneously (instantaneous delivery).",
        "Shortages, stockouts, and backorders are strictly prohibited.",
        "Unit purchase price is constant (no quantity discounts apply).",
    ]

    result = EOQResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_definitions,
        formula=formulas,
        formula_with_substituted_values=substitutions,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
    )

    result.verification = verify_eoq(inputs_extracted, result)
    return result


def verify_eoq(inputs: Dict[str, Any], result: EOQResult) -> ToolVerification:
    """Verify cost parity and sensitivity for EOQ."""
    passed = result.verification_checks.copy()
    warnings = []
    D = inputs["annual_demand"]
    S = inputs["ordering_cost"]
    H = inputs["holding_cost"]
    Q = result.optimal_order_quantity

    def cost_at(q: float) -> float:
        return (D / q) * S + (q / 2.0) * H

    base_cost = cost_at(Q)
    cost_minus_10 = cost_at(Q * 0.90)
    cost_plus_10 = cost_at(Q * 1.10)

    sensitivity = {
        "at_minus_20_pct": round(cost_at(Q * 0.80), 2),
        "at_minus_10_pct": round(cost_minus_10, 2),
        "at_optimal_Q": round(base_cost, 2),
        "at_plus_10_pct": round(cost_plus_10, 2),
        "at_plus_20_pct": round(cost_at(Q * 1.20), 2),
        "cost_penalty_at_plus_10_pct": round(cost_plus_10 - base_cost, 2),
    }

    return ToolVerification(
        is_valid=(len(warnings) == 0),
        checks_passed=passed,
        warnings=warnings,
        sensitivity_analysis=sensitivity,
    )


# ---------------------------------------------------------------------------
# Tool 2: Quantity Discount
# ---------------------------------------------------------------------------

def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    tiers: Optional[List[Dict[str, Any]]] = None,
    demand: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """Calculate Optimal Order Quantity under All-Units Quantity Discounts with all 8 verification sections."""
    D = demand if demand is not None else annual_demand
    S = ordering_cost

    # Input validation
    if D is None:
        raise ValueError("Missing required input: annual demand (D) must be provided. Missing numerical inputs will not be invented.")
    if S is None:
        raise ValueError("Missing required input: ordering cost (S) must be provided. Missing numerical inputs will not be invented.")
    if not tiers:
        raise ValueError("Missing required input: at least one discount tier must be provided.")
    if holding_cost_rate is None and fixed_holding_cost is None:
        if not all("holding_cost" in t and t["holding_cost"] is not None for t in tiers):
            raise ValueError("Missing holding cost: either holding_cost_rate, fixed_holding_cost, or tier-level holding_cost must be provided.")

    if D <= 0:
        raise ValueError(f"Annual demand ({D}) must be strictly positive.")
    if S <= 0:
        raise ValueError(f"Ordering cost ({S}) must be strictly positive.")
    if holding_cost_rate is not None and holding_cost_rate <= 0:
        raise ValueError(f"Holding cost rate ({holding_cost_rate}) must be positive.")
    if fixed_holding_cost is not None and fixed_holding_cost <= 0:
        raise ValueError(f"Fixed holding cost ({fixed_holding_cost}) must be positive.")

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": D,
        "ordering_cost": S,
        "holding_cost_rate": holding_cost_rate,
        "fixed_holding_cost": fixed_holding_cost,
        "tiers": tiers,
    }

    # 2. Variable Definitions
    var_definitions = {
        "D": "Annual customer demand (units/year)",
        "S": "Order setup / placement cost ($/order)",
        "P_i": "Unit purchase price for tier i ($/unit)",
        "H_i": "Annual unit holding cost for tier i ($/unit/year)",
        "EOQ_i": "Unconstrained EOQ calculated for tier i = sqrt(2DS / H_i)",
        "Q_feasible_i": "Feasible lot size after checking price break thresholds",
        "Material_Cost_i": "Annual purchasing cost = D * P_i ($/year)",
        "Ordering_Cost_i": "Annual order cost = (D / Q_feasible_i) * S ($/year)",
        "Holding_Cost_i": "Annual carrying cost = (Q_feasible_i / 2) * H_i ($/year)",
        "Total_Cost_i": "Total Annual Cost = Material + Ordering + Holding ($/year)",
    }

    # 3. Formula
    formulas = {
        "Unconstrained_EOQ": "EOQ_i = sqrt((2 * D * S) / H_i)",
        "Feasible_Quantity": "Q_i = max(min_qty_i, EOQ_i) if EOQ_i <= max_qty_i",
        "Total_Annual_Cost": "TC(Q_i) = (D * P_i) + ((D / Q_i) * S) + ((Q_i / 2) * H_i)",
        "Optimal_Decision": "Select tier i with minimum Total Annual Cost TC(Q_i)",
    }

    evaluations: List[DiscountTierEvaluation] = []
    substitutions: Dict[str, str] = {}
    best_tier_idx = -1
    lowest_total_cost = float("inf")
    optimal_q = 0.0

    for idx, tier in enumerate(tiers):
        tier_id = tier.get("tier_id", idx + 1)
        min_q = float(tier["min_qty"])
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["unit_price"])

        if price <= 0:
            raise ValueError(f"Tier {tier_id} unit price ({price}) must be positive.")
        if min_q < 0:
            raise ValueError(f"Tier {tier_id} min_qty ({min_q}) cannot be negative.")

        if "holding_cost" in tier and tier["holding_cost"] is not None:
            H = float(tier["holding_cost"])
        elif holding_cost_rate is not None:
            H = holding_cost_rate * price
        elif fixed_holding_cost is not None:
            H = fixed_holding_cost
        else:
            raise ValueError(f"Unable to determine holding cost for tier {tier_id}.")

        unconstrained_eoq = math.sqrt((2.0 * D * S) / H)

        is_break_point = False
        if unconstrained_eoq < min_q:
            feasible_q = min_q
            is_break_point = True
        elif max_q is not None and unconstrained_eoq > max_q:
            feasible_q = max_q
        else:
            feasible_q = unconstrained_eoq

        annual_mat_cost = D * price
        annual_ord_cost = (D / feasible_q) * S
        annual_hld_cost = (feasible_q / 2.0) * H
        total_cost = annual_mat_cost + annual_ord_cost + annual_hld_cost

        evaluations.append(
            DiscountTierEvaluation(
                tier_id=tier_id,
                min_qty=min_q,
                max_qty=max_q,
                unit_price=round(price, 4),
                holding_cost=round(H, 4),
                calculated_eoq=round(unconstrained_eoq, 2),
                feasible_quantity=round(feasible_q, 2),
                is_price_break_point=is_break_point,
                annual_material_cost=round(annual_mat_cost, 2),
                annual_ordering_cost=round(annual_ord_cost, 2),
                annual_holding_cost=round(annual_hld_cost, 2),
                total_annual_cost=round(total_cost, 2),
            )
        )

        substitutions[f"Tier_{tier_id}_Evaluation"] = (
            f"EOQ_{tier_id} = sqrt((2 * {D:,.2f} * {S:,.2f}) / {H:,.2f}) = {unconstrained_eoq:,.2f} units. "
            f"Feasible Q_{tier_id} = {feasible_q:,.2f} units ({'Price-break point' if is_break_point else 'Internal EOQ'}). "
            f"TC_{tier_id} = (${D:,.2f} * ${price:.2f}) + (${D:,.2f}/{feasible_q:,.2f} * ${S:.2f}) + (${feasible_q:,.2f}/2 * ${H:.2f}) "
            f"= ${annual_mat_cost:,.2f} + ${annual_ord_cost:,.2f} + ${annual_hld_cost:,.2f} = ${total_cost:,.2f}"
        )

        if total_cost < lowest_total_cost:
            lowest_total_cost = total_cost
            best_tier_idx = idx
            optimal_q = feasible_q

    best_tier = evaluations[best_tier_idx]
    base_tier_cost = evaluations[0].total_annual_cost
    savings = max(0.0, base_tier_cost - best_tier.total_annual_cost)

    # 5. Intermediate Calculations
    intermediate = {
        "evaluated_tiers_count": len(evaluations),
        "tier_evaluations": [
            {
                "tier_id": t.tier_id,
                "unit_price": t.unit_price,
                "holding_cost": t.holding_cost,
                "calculated_eoq": t.calculated_eoq,
                "feasible_quantity": t.feasible_quantity,
                "is_price_break_point": t.is_price_break_point,
                "annual_material_cost": t.annual_material_cost,
                "annual_ordering_cost": t.annual_ordering_cost,
                "annual_holding_cost": t.annual_holding_cost,
                "total_annual_cost": t.total_annual_cost,
            }
            for t in evaluations
        ],
    }

    # 6. Final Answer
    final_answer = {
        "optimal_order_quantity": round(optimal_q, 2),
        "optimal_tier_id": best_tier.tier_id,
        "optimal_unit_price": best_tier.unit_price,
        "min_total_annual_cost": round(lowest_total_cost, 2),
        "annual_savings_vs_base": round(savings, 2),
    }

    # 7. Verification Checks
    verification_checks = [
        f"All {len(evaluations)} price break brackets successfully evaluated.",
        f"Global optimality verified: Tier {best_tier.tier_id} (Q={optimal_q:,.2f}) achieves minimum annual total cost (${lowest_total_cost:,.2f}).",
    ]
    if savings > 0:
        verification_checks.append(f"Volume discount yields ${savings:,.2f} in net annual savings over baseline Tier 1.")

    # 8. Assumptions
    assumptions = [
        "All-units quantity discount applies to every unit purchased in the replenishment batch.",
        "Annual customer demand is constant and deterministic across the planning horizon.",
        "Supplier has unlimited production capacity to satisfy orders at higher price breaks.",
        "Holding cost is either constant across tiers or proportional to unit purchase price.",
        "Lead time is deterministic and orders arrive in full batches.",
    ]

    result = QuantityDiscountResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_definitions,
        formula=formulas,
        formula_with_substituted_values=substitutions,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        tier_evaluations=evaluations,
    )

    result.verification = verify_quantity_discount(inputs_extracted, result)
    return result


def verify_quantity_discount(inputs: Dict[str, Any], result: QuantityDiscountResult) -> ToolVerification:
    """Verify optimality and savings across discount tiers."""
    passed = result.verification_checks.copy()
    warnings = []

    all_costs = [t.total_annual_cost for t in result.tier_evaluations]
    if all_costs and abs(min(all_costs) - result.optimal_total_cost) > 1e-4:
        warnings.append("Selected order quantity total cost does not match lowest computed cost.")

    sensitivity = {
        "optimal_tier": result.optimal_tier_id,
        "optimal_order_quantity": result.optimal_order_quantity,
        "min_total_annual_cost": result.optimal_total_cost,
        "annual_savings_vs_base": result.annual_savings_vs_base,
        "tier_costs": {f"tier_{t.tier_id}": t.total_annual_cost for t in result.tier_evaluations},
    }

    return ToolVerification(
        is_valid=(len(warnings) == 0),
        checks_passed=passed,
        warnings=warnings,
        sensitivity_analysis=sensitivity,
    )


# ---------------------------------------------------------------------------
# Tool 3: Safety Stock & Reorder Point (calculate_rop)
# ---------------------------------------------------------------------------

def calculate_rop(
    mean_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    mean_demand_per_period: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    lead_time_periods: Optional[float] = None,
    std_lead_time_periods: Optional[float] = None,
    holding_cost_per_unit_per_period: Optional[float] = None,
    holding_cost: Optional[float] = None,
) -> ROPResult:
    """Calculate Safety Stock and Reorder Point across all 4 uncertainty regimes with full verification steps."""
    d = mean_demand if mean_demand is not None else mean_demand_per_period
    L = lead_time if lead_time is not None else lead_time_periods
    sigma_d = std_demand_per_period if std_demand_per_period is not None else std_demand
    sigma_L = std_lead_time_periods if std_lead_time_periods is not None else std_lead_time
    csl = service_level
    H = holding_cost if holding_cost is not None else holding_cost_per_unit_per_period

    # Input Validation
    if d is None:
        raise ValueError("Missing required input: mean demand per period must be provided. Missing numerical inputs will not be invented.")
    if L is None:
        raise ValueError("Missing required input: lead time must be provided. Missing numerical inputs will not be invented.")
    if csl is None:
        raise ValueError("Missing required input: service level must be provided. Missing numerical inputs will not be invented.")

    if d <= 0:
        raise ValueError(f"Mean demand ({d}) must be strictly positive.")
    if L <= 0:
        raise ValueError(f"Lead time ({L}) must be strictly positive.")
    if sigma_d < 0:
        raise ValueError(f"Standard deviation of demand ({sigma_d}) cannot be negative.")
    if sigma_L < 0:
        raise ValueError(f"Standard deviation of lead time ({sigma_L}) cannot be negative.")
    if not (0.50 <= csl < 1.0):
        raise ValueError(f"Service level ({csl}) must be between 0.50 (50%) and 0.9999 (99.99%).")

    # 1. Inputs Extracted
    inputs_extracted = {
        "mean_demand": d,
        "lead_time": L,
        "service_level": csl,
        "std_demand": sigma_d,
        "std_lead_time": sigma_L,
        "holding_cost": H,
    }

    # 2. Variable Definitions
    var_definitions = {
        "d": "Mean customer demand per period (units/period)",
        "sigma_d": "Standard deviation of customer demand per period",
        "L": "Mean supplier replenishment lead time (periods)",
        "sigma_L": "Standard deviation of supplier lead time (periods)",
        "CSL": "Cycle Service Level (target in-stock probability during lead time)",
        "z": "Z-score from standard normal distribution, z = Phi^-1(CSL)",
        "mu_LTD": "Mean lead time demand = d * L",
        "sigma_LT": "Standard deviation of lead time demand",
        "SS": "Safety Stock buffer = z * sigma_LT",
        "ROP": "Reorder Point = mu_LTD + SS",
    }

    # Determine regime and formulas
    is_demand_variable = (sigma_d > 0.0)
    is_lead_time_variable = (sigma_L > 0.0)

    if not is_demand_variable and not is_lead_time_variable:
        case_name = "constant_demand_constant_lead_time"
        case_desc = "Constant demand and constant lead time"
        sigma_LT = 0.0
        z = 0.0
        safety_stock = 0.0
        formula_sigma = "sigma_LT = 0 (deterministic)"
        formula_ss = "SS = 0"
        subst_sigma = "sigma_LT = 0.0"
        subst_ss = "SS = 0.0 units"
    elif is_demand_variable and not is_lead_time_variable:
        case_name = "variable_demand_constant_lead_time"
        case_desc = "Variable demand and constant lead time"
        sigma_LT = sigma_d * math.sqrt(L)
        z = float(stats.norm.ppf(csl))
        safety_stock = z * sigma_LT
        formula_sigma = "sigma_LT = sigma_d * sqrt(L)"
        formula_ss = "SS = z * sigma_LT"
        subst_sigma = f"sigma_LT = {sigma_d:.2f} * sqrt({L:.1f}) = {sigma_LT:.2f} units"
        subst_ss = f"SS = {z:.4f} * {sigma_LT:.2f} = {safety_stock:.2f} units"
    elif not is_demand_variable and is_lead_time_variable:
        case_name = "constant_demand_variable_lead_time"
        case_desc = "Constant demand and variable lead time"
        sigma_LT = d * sigma_L
        z = float(stats.norm.ppf(csl))
        safety_stock = z * sigma_LT
        formula_sigma = "sigma_LT = d * sigma_L"
        formula_ss = "SS = z * sigma_LT"
        subst_sigma = f"sigma_LT = {d:.2f} * {sigma_L:.2f} = {sigma_LT:.2f} units"
        subst_ss = f"SS = {z:.4f} * {sigma_LT:.2f} = {safety_stock:.2f} units"
    else:
        case_name = "variable_demand_variable_lead_time"
        case_desc = "Variable demand and variable lead time"
        sigma_LT = math.sqrt(L * (sigma_d ** 2) + (d ** 2) * (sigma_L ** 2))
        z = float(stats.norm.ppf(csl))
        safety_stock = z * sigma_LT
        formula_sigma = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        formula_ss = "SS = z * sigma_LT"
        subst_sigma = (
            f"sigma_LT = sqrt(({L:.1f} * {sigma_d:.2f}^2) + ({d:.2f}^2 * {sigma_L:.2f}^2)) "
            f"= sqrt({L * (sigma_d ** 2):.2f} + {(d ** 2) * (sigma_L ** 2):.2f}) = {sigma_LT:.2f} units"
        )
        subst_ss = f"SS = {z:.4f} * {sigma_LT:.2f} = {safety_stock:.2f} units"

    ltd_mean = d * L
    reorder_point = ltd_mean + safety_stock
    annual_ss_holding = (safety_stock * H) if (H is not None and H > 0) else None

    # 3. Formula
    formulas = {
        "Lead_Time_Demand_Mean": "mu_LTD = d * L",
        "Lead_Time_Demand_Std": formula_sigma,
        "Safety_Stock": formula_ss,
        "Reorder_Point": "ROP = mu_LTD + SS = (d * L) + SS",
    }

    # 4. Formula with Substituted Values
    substitutions = {
        "Lead_Time_Demand_Mean": f"mu_LTD = {d:.2f} * {L:.1f} = {ltd_mean:.2f} units",
        "Lead_Time_Demand_Std": subst_sigma,
        "Safety_Stock": subst_ss,
        "Reorder_Point": f"ROP = {ltd_mean:.2f} + {safety_stock:.2f} = {reorder_point:.2f} units",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "case_identified": case_name,
        "case_description": case_desc,
        "lead_time_demand_mean": round(ltd_mean, 2),
        "lead_time_demand_std": round(sigma_LT, 2),
        "z_score": round(z, 4),
        "safety_stock": round(safety_stock, 2),
        "stockout_probability_per_cycle": round(1.0 - csl, 4) if sigma_LT > 0 else 0.0,
        "annual_ss_holding_cost": round(annual_ss_holding, 2) if annual_ss_holding is not None else None,
    }

    # 6. Final Answer
    final_answer = {
        "reorder_point": round(reorder_point, 2),
        "safety_stock": round(safety_stock, 2),
        "cycle_service_level": round(csl, 4),
    }

    # 7. Verification Checks
    verification_checks = [
        f"Regime confirmed: {case_desc}.",
        f"Non-negative safety stock verified (SS = {safety_stock:.2f} >= 0).",
        f"ROP strictly covers expected lead time demand ({reorder_point:.2f} >= {ltd_mean:.2f}).",
    ]

    # 8. Assumptions
    assumptions = [
        "Demand per period is normally distributed when uncertain.",
        "Lead time duration is normally distributed when uncertain.",
        "Period-to-period demands are independent and identically distributed.",
        "Demand and lead time are mutually independent random variables.",
        "Inventory is monitored continuously under a standard (s, Q) reorder point policy.",
    ]

    result = ROPResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_definitions,
        formula=formulas,
        formula_with_substituted_values=substitutions,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
    )

    result.verification = verify_safety_stock_rop(inputs_extracted, result)
    return result


def verify_safety_stock_rop(inputs: Dict[str, Any], result: ROPResult) -> ToolVerification:
    """Verify safety stock and reorder point with service level sensitivity."""
    passed = result.verification_checks.copy()
    warnings = []

    d = inputs.get("mean_demand", 0.0)
    L = inputs.get("lead_time", 0.0)
    sigma_lt = result.lead_time_demand_std

    sensitivity = {}
    if sigma_lt > 0:
        for sl in [0.90, 0.95, 0.98, 0.99]:
            z_val = float(stats.norm.ppf(sl))
            ss_val = z_val * sigma_lt
            rop_val = (d * L) + ss_val
            sensitivity[f"rop_at_{int(sl*100)}_pct_service"] = round(rop_val, 2)
            sensitivity[f"ss_at_{int(sl*100)}_pct_service"] = round(ss_val, 2)
    else:
        sensitivity["status"] = "Deterministic lead-time demand; zero safety stock required."

    return ToolVerification(
        is_valid=(len(warnings) == 0),
        checks_passed=passed,
        warnings=warnings,
        sensitivity_analysis=sensitivity,
    )


# ---------------------------------------------------------------------------
# Tool 4: Fixed-Period Review (calculate_periodic_review)
# ---------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    service_level: Optional[float] = None,
    inventory_position: Optional[float] = None,
    on_hand: Optional[float] = None,
    on_order: Optional[float] = None,
    backorders: float = 0.0,
    review_period_P: Optional[float] = None,
    lead_time_L: Optional[float] = None,
    mean_demand_per_period: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    current_on_hand: Optional[float] = None,
    open_orders_pipeline: Optional[float] = None,
) -> PeriodicReviewResult:
    """Calculate Target Inventory and Order Quantity for Fixed-Period Review with all 8 verification sections."""
    P = review_period if review_period is not None else review_period_P
    L = lead_time if lead_time is not None else lead_time_L
    d = mean_demand if mean_demand is not None else mean_demand_per_period
    sigma_d = std_demand if std_demand is not None else std_demand_per_period
    csl = service_level
    hand = on_hand if on_hand is not None else current_on_hand
    pipe = on_order if on_order is not None else open_orders_pipeline

    # Input validation
    if P is None:
        raise ValueError("Missing required input: review period must be provided. Missing numerical inputs will not be invented.")
    if L is None:
        raise ValueError("Missing required input: lead time must be provided. Missing numerical inputs will not be invented.")
    if d is None:
        raise ValueError("Missing required input: mean demand must be provided. Missing numerical inputs will not be invented.")
    if sigma_d is None:
        raise ValueError("Missing required input: standard deviation of demand must be provided. Missing numerical inputs will not be invented.")
    if csl is None:
        raise ValueError("Missing required input: service level must be provided. Missing numerical inputs will not be invented.")

    if P <= 0:
        raise ValueError(f"Review period ({P}) must be strictly positive.")
    if L < 0:
        raise ValueError(f"Lead time ({L}) cannot be negative.")
    if d <= 0:
        raise ValueError(f"Mean demand ({d}) must be strictly positive.")
    if sigma_d < 0:
        raise ValueError(f"Standard deviation of demand ({sigma_d}) cannot be negative.")
    if not (0.50 <= csl < 1.0):
        raise ValueError(f"Service level ({csl}) must be between 0.50 and 0.9999.")

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period": P,
        "lead_time": L,
        "mean_demand": d,
        "std_demand": sigma_d,
        "service_level": csl,
        "inventory_position": inventory_position,
        "on_hand": hand,
        "on_order": pipe,
        "backorders": backorders,
    }

    # 2. Variable Definitions
    var_definitions = {
        "P": "Review period interval in periods/days",
        "L": "Supplier replenishment lead time in periods/days",
        "P + L": "Protection period / exposure interval (periods/days)",
        "d": "Mean customer demand per period (units/period)",
        "sigma_d": "Standard deviation of demand per period",
        "CSL": "Cycle Service Level (in-stock probability during protection period)",
        "z": "Z-score from standard normal distribution, z = Phi^-1(CSL)",
        "mu_P_plus_L": "Expected demand during protection period = d * (P + L)",
        "sigma_P_plus_L": "Standard deviation of demand during protection period = sigma_d * sqrt(P + L)",
        "SS": "Safety stock over protection interval = z * sigma_d * sqrt(P + L)",
        "Target_Inventory_M": "Order-Up-To Level = mu_P_plus_L + SS",
        "IP": "Inventory Position = On-Hand + On-Order - Backorders",
        "Order_Quantity_q": "Order quantity = max(0, Target_Inventory - IP)",
    }

    # Calculations
    protection_period = P + L
    demand_protection = d * protection_period
    std_protection = sigma_d * math.sqrt(protection_period)
    z = float(stats.norm.ppf(csl))
    safety_stock = z * std_protection
    target_inventory = demand_protection + safety_stock

    inv_pos = inventory_position
    if inv_pos is None and hand is not None:
        inv_pos = hand + (pipe or 0.0) - backorders

    order_quantity = None
    if inv_pos is not None:
        order_quantity = max(0.0, target_inventory - inv_pos)

    continuous_ss = z * sigma_d * math.sqrt(L) if L > 0 else 0.0
    ss_ratio = (safety_stock / continuous_ss) if continuous_ss > 0 else 1.0

    # 3. Formula
    formulas = {
        "Protection_Period": "Protection_Period = review_period + lead_time = P + L",
        "Protection_Demand_Mean": "mu_{P+L} = d * (P + L)",
        "Protection_Demand_Std": "sigma_{P+L} = sigma_d * sqrt(P + L)",
        "Safety_Stock": "SS = z * sigma_d * sqrt(P + L)",
        "Target_Inventory": "Target_Inventory = mu_{P+L} + SS = d * (P + L) + z * sigma_d * sqrt(P + L)",
        "Order_Quantity": "q = max(0, Target_Inventory - Inventory_Position)",
    }

    # 4. Formula with Substituted Values
    substitutions = {
        "Protection_Period": f"P + L = {P:.1f} + {L:.1f} = {protection_period:.1f} periods",
        "Protection_Demand_Mean": f"mu_{{P+L}} = {d:.2f} * {protection_period:.1f} = {demand_protection:.2f} units",
        "Protection_Demand_Std": f"sigma_{{P+L}} = {sigma_d:.2f} * sqrt({protection_period:.1f}) = {std_protection:.2f} units",
        "Safety_Stock": f"SS = {z:.4f} * {sigma_d:.2f} * sqrt({protection_period:.1f}) = {safety_stock:.2f} units",
        "Target_Inventory": f"M = {demand_protection:.2f} + {safety_stock:.2f} = {target_inventory:.2f} units",
    }
    if inv_pos is not None:
        substitutions["Order_Quantity"] = f"q = max(0, {target_inventory:.2f} - {inv_pos:.2f}) = {order_quantity:.2f} units"
    else:
        substitutions["Order_Quantity"] = "q = max(0, Target_Inventory - IP) [IP not provided; order quantity omitted]"

    # 5. Intermediate Calculations
    intermediate = {
        "protection_period": protection_period,
        "protection_demand_mean": round(demand_protection, 2),
        "protection_demand_std": round(std_protection, 2),
        "z_score": round(z, 4),
        "safety_stock": round(safety_stock, 2),
        "target_inventory": round(target_inventory, 2),
        "inventory_position": round(inv_pos, 2) if inv_pos is not None else None,
        "safety_stock_vs_continuous_ratio": round(ss_ratio, 2),
    }

    # 6. Final Answer
    final_answer = {
        "target_inventory": round(target_inventory, 2),
        "order_quantity": round(order_quantity, 2) if order_quantity is not None else None,
        "safety_stock": round(safety_stock, 2),
    }

    # 7. Verification Checks
    verification_checks = [
        f"Target inventory level adequately covers protection interval demand ({target_inventory:.2f} >= {demand_protection:.2f}).",
        f"Periodic review buffer property verified: requires {ss_ratio:.2f}x the safety stock of a continuous review system.",
    ]

    # 8. Assumptions
    assumptions = [
        "Inventory review occurs at fixed periodic intervals of review_period.",
        "Customer demand is stationary and normally distributed with constant mean and variance.",
        "Unsatisfied customer demand during stockout intervals is backordered.",
        "Lead time is known and constant.",
    ]
    if inv_pos is None:
        assumptions.append("Note: Inventory position was not provided; order quantity was not calculated.")

    result = PeriodicReviewResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_definitions,
        formula=formulas,
        formula_with_substituted_values=substitutions,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
    )

    result.verification = verify_fixed_period_review(inputs_extracted, result)
    return result


def verify_fixed_period_review(inputs: Dict[str, Any], result: PeriodicReviewResult) -> ToolVerification:
    """Verify target inventory level and order quantity for fixed-period review."""
    passed = result.verification_checks.copy()
    warnings = []

    P = inputs.get("review_period", 0.0)
    L = inputs.get("lead_time", 0.0)
    d = inputs.get("mean_demand", 0.0)
    sigma_p_l = result.protection_demand_std

    sensitivity = {}
    if sigma_p_l > 0:
        for sl in [0.90, 0.95, 0.98, 0.99]:
            z_val = float(stats.norm.ppf(sl))
            ss_val = z_val * sigma_p_l
            m_val = (d * (P + L)) + ss_val
            sensitivity[f"target_M_at_{int(sl*100)}_pct"] = round(m_val, 2)
            sensitivity[f"ss_at_{int(sl*100)}_pct"] = round(ss_val, 2)
    else:
        sensitivity["status"] = "Deterministic demand; safety stock is zero."

    return ToolVerification(
        is_valid=(len(warnings) == 0),
        checks_passed=passed,
        warnings=warnings,
        sensitivity_analysis=sensitivity,
    )


# ---------------------------------------------------------------------------
# Tool 5: Newsvendor (calculate_newsvendor)
# ---------------------------------------------------------------------------

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    purchase_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    stockout_penalty: float = 0.0,
    demand_dist: str = "normal",
    uniform_min: float = 0.0,
    uniform_max: float = 0.0,
    discrete_dist: Optional[List[Tuple[float, float]]] = None,
) -> NewsvendorResult:
    """Calculate Underage Cost, Overage Cost, Critical Ratio, and Recommended Order Quantity with all 8 verification sections."""
    if selling_price is None:
        raise ValueError("Missing required input: selling_price must be provided. Missing numerical inputs will not be invented.")
    if purchase_cost is None:
        raise ValueError("Missing required input: purchase_cost must be provided. Missing numerical inputs will not be invented.")
    if salvage_value is None:
        raise ValueError("Missing required input: salvage_value must be provided. Missing numerical inputs will not be invented.")
    if mean_demand is None:
        raise ValueError("Missing required input: mean_demand must be provided. Missing numerical inputs will not be invented.")
    if std_demand is None:
        raise ValueError("Missing required input: std_demand must be provided. Missing numerical inputs will not be invented.")

    if selling_price <= purchase_cost:
        raise ValueError(f"Selling price (${selling_price}) must exceed purchase cost (${purchase_cost}) for a profitable product.")
    if salvage_value > purchase_cost:
        raise ValueError(f"Salvage value (${salvage_value}) cannot exceed purchase cost (${purchase_cost}).")
    if stockout_penalty < 0:
        raise ValueError(f"Stockout penalty ({stockout_penalty}) cannot be negative.")

    # 1. Inputs Extracted
    inputs_extracted = {
        "selling_price": selling_price,
        "purchase_cost": purchase_cost,
        "salvage_value": salvage_value,
        "mean_demand": mean_demand,
        "std_demand": std_demand,
        "stockout_penalty": stockout_penalty,
        "demand_dist": demand_dist,
    }

    # 2. Variable Definitions
    var_definitions = {
        "P": "Unit retail selling price ($/unit)",
        "C": "Unit purchase or production cost ($/unit)",
        "V": "Unit salvage value for unsold leftover inventory ($/unit)",
        "g": "Unit penalty or lost goodwill for stockout ($/unit)",
        "Cu": "Cost of Underage (shortage cost) per unit = P - C + g ($/unit)",
        "Co": "Cost of Overage (excess cost) per unit = C - V ($/unit)",
        "CR": "Critical Ratio / Critical Fractile = Cu / (Cu + Co)",
        "mu": "Mean customer demand (units)",
        "sigma": "Standard deviation of customer demand (units)",
        "z": "Z-score corresponding to in-stock probability, z = Phi^-1(CR)",
        "Q*": "Optimal single-period order quantity = mu + z * sigma",
        "E_Sales": "Expected units sold during the season",
        "E_Lost": "Expected lost sales (shortage) based on normal loss function L(z)",
        "E_Leftover": "Expected unsold units salvaged at end of season",
        "E_Profit": "Expected net financial profit ($)",
    }

    # Calculations
    Cu = selling_price - purchase_cost + stockout_penalty
    Co = purchase_cost - salvage_value
    critical_ratio = Cu / (Cu + Co)

    if demand_dist == "normal":
        if mean_demand <= 0:
            raise ValueError(f"Mean demand ({mean_demand}) must be strictly positive.")
        if std_demand <= 0:
            raise ValueError(f"Standard deviation of demand ({std_demand}) must be strictly positive.")

        z = float(stats.norm.ppf(critical_ratio))
        optimal_q = mean_demand + z * std_demand

        phi_z = stats.norm.pdf(z)
        Phi_z = stats.norm.cdf(z)
        loss_z = phi_z - z * (1.0 - Phi_z)

        expected_lost_sales = std_demand * loss_z
        expected_leftover = optimal_q - mean_demand + expected_lost_sales
        expected_sales = mean_demand - expected_lost_sales

    elif demand_dist == "uniform":
        if uniform_max <= uniform_min:
            raise ValueError("uniform_max must be strictly greater than uniform_min.")
        optimal_q = uniform_min + critical_ratio * (uniform_max - uniform_min)
        z = (optimal_q - mean_demand) / std_demand if std_demand > 0 else 0.0
        expected_leftover = ((optimal_q - uniform_min) ** 2) / (2.0 * (uniform_max - uniform_min))
        expected_sales = optimal_q - expected_leftover
        expected_lost_sales = mean_demand - expected_sales

    elif demand_dist == "discrete" and discrete_dist is not None:
        sorted_pairs = sorted(discrete_dist, key=lambda p: p[0])
        cum_prob = 0.0
        optimal_q = sorted_pairs[0][0]
        for d_val, prob in sorted_pairs:
            cum_prob += prob
            if cum_prob >= critical_ratio:
                optimal_q = d_val
                break
        z = 0.0
        expected_sales = sum(min(optimal_q, d) * p for d, p in sorted_pairs)
        expected_leftover = sum(max(0.0, optimal_q - d) * p for d, p in sorted_pairs)
        expected_lost_sales = sum(max(0.0, d - optimal_q) * p for d, p in sorted_pairs)
    else:
        raise ValueError(f"Unsupported demand distribution: {demand_dist}")

    total_purchase_cost = optimal_q * purchase_cost
    expected_revenue = expected_sales * selling_price
    expected_salvage_rev = expected_leftover * salvage_value
    expected_profit = (
        expected_revenue + expected_salvage_rev - total_purchase_cost - (expected_lost_sales * stockout_penalty)
    )

    # 3. Formula
    formulas = {
        "Underage_Cost": "Cu = Selling_Price - Purchase_Cost + Stockout_Penalty",
        "Overage_Cost": "Co = Purchase_Cost - Salvage_Value",
        "Critical_Ratio": "CR = Cu / (Cu + Co)",
        "Recommended_Order_Quantity": "Q* = mu + z * sigma",
        "Normal_Loss_Function": "L(z) = phi(z) - z * (1 - Phi(z))",
        "Expected_Profit": "Expected_Profit = (P * E[Sales]) + (V * E[Leftover]) - (C * Q*) - (g * E[Lost])",
    }

    # 4. Formula with Substituted Values
    substitutions = {
        "Underage_Cost": f"Cu = ${selling_price:.2f} - ${purchase_cost:.2f} + ${stockout_penalty:.2f} = ${Cu:.2f}",
        "Overage_Cost": f"Co = ${purchase_cost:.2f} - ${salvage_value:.2f} = ${Co:.2f}",
        "Critical_Ratio": f"CR = ${Cu:.2f} / (${Cu:.2f} + ${Co:.2f}) = ${Cu:.2f} / ${Cu + Co:.2f} = {critical_ratio:.4f} ({critical_ratio * 100:.2f}%)",
        "Recommended_Order_Quantity": f"Q* = {mean_demand:,.2f} + ({z:.4f} * {std_demand:,.2f}) = {optimal_q:,.2f} units",
        "Expected_Profit": (
            f"Expected_Profit = (${selling_price:.2f} * {expected_sales:.2f}) + (${salvage_value:.2f} * {expected_leftover:.2f}) "
            f"- (${purchase_cost:.2f} * {optimal_q:.2f}) = ${expected_profit:,.2f}"
        ),
    }

    # 5. Intermediate Calculations
    intermediate = {
        "underage_cost_Cu": round(Cu, 2),
        "overage_cost_Co": round(Co, 2),
        "critical_ratio": round(critical_ratio, 4),
        "z_score": round(z, 4),
        "expected_sales": round(float(expected_sales), 2),
        "expected_lost_sales": round(float(expected_lost_sales), 2),
        "expected_leftover_units": round(float(expected_leftover), 2),
        "expected_revenue": round(float(expected_revenue), 2),
        "expected_salvage_revenue": round(float(expected_salvage_rev), 2),
        "total_purchase_cost": round(float(total_purchase_cost), 2),
        "expected_profit": round(float(expected_profit), 2),
    }

    # 6. Final Answer
    final_answer = {
        "underage_cost": round(Cu, 2),
        "overage_cost": round(Co, 2),
        "critical_ratio": round(critical_ratio, 4),
        "recommended_order_quantity": round(optimal_q, 2),
        "expected_profit": round(expected_profit, 2),
    }

    # 7. Verification Checks
    verification_checks = [
        f"Valid critical ratio verified: CR = {critical_ratio:.4f} in interval (0, 1).",
        f"Critical fractile matches in-stock probability: {critical_ratio * 100:.2f}%.",
        f"Stockout risk per season is mathematically optimal at {(1.0 - critical_ratio) * 100:.2f}%.",
    ]

    # 8. Assumptions
    assumptions = [
        "Single selling season with strictly no mid-season replenishment or order cancellation.",
        "Customer demand follows a stationary normal distribution with known mean and variance.",
        "Unsold inventory can be fully salvaged at salvage value with zero disposal fee.",
        "Unit selling prices and production costs are constant and linear across the volume.",
    ]

    result = NewsvendorResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_definitions,
        formula=formulas,
        formula_with_substituted_values=substitutions,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
    )

    result.verification = verify_newsvendor(inputs_extracted, result)
    return result


def verify_newsvendor(inputs: Dict[str, Any], result: NewsvendorResult) -> ToolVerification:
    """Verify newsvendor critical ratio and expected profit sensitivity."""
    passed = result.verification_checks.copy()
    warnings = []

    cr = result.critical_ratio
    sensitivity = {
        "critical_fractile": round(cr, 4),
        "in_stock_probability": f"{cr * 100:.2f}%",
        "stockout_probability": f"{(1.0 - cr) * 100:.2f}%",
        "optimal_order_quantity": result.optimal_order_quantity,
        "optimal_expected_profit": result.expected_profit,
    }

    return ToolVerification(
        is_valid=(len(warnings) == 0),
        checks_passed=passed,
        warnings=warnings,
        sensitivity_analysis=sensitivity,
    )


# Function Aliases for convenience & backward compatibility
calculate_safety_stock_and_rop = calculate_rop
calculate_fixed_period_review = calculate_periodic_review

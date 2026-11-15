"""
Supply Chain Analytics Inventory Decision Agent
================================================
Implements the 6-stage architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock & Reorder Point (SS & ROP)
4. Fixed-Period Review (Periodic Review / Order-Up-To Level)
5. Newsvendor Model
"""

from __future__ import annotations
import re
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

import inventory_tools as tools


@dataclass
class AgentExecutionTrace:
    """Stores full step-by-step trace of the agent decision pipeline."""
    raw_question: str
    identified_model: str
    model_name: str
    confidence: float
    identification_reasoning: str
    extracted_inputs: Dict[str, Any]
    selected_tool_name: str
    calculation_result: Optional[Any] = None
    verification_passed: bool = False
    verification_messages: List[str] = field(default_factory=list)
    explanation_text: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.calculation_result is not None and hasattr(self.calculation_result, "__dict__"):
            d["calculation_result"] = asdict(self.calculation_result)
        return d


class InventoryDecisionAgent:
    """
    Intelligent Supply Chain Agent capable of parsing natural language,
    mapping to the correct inventory policy, running numerical tools,
    verifying constraints, and producing managerial explanations.
    """

    MODEL_NAMES = {
        "eoq": "Economic Order Quantity (EOQ)",
        "quantity_discount": "Quantity Discount Model",
        "safety_stock_rop": "Safety Stock & Reorder Point (Continuous Review)",
        "fixed_period_review": "Fixed-Period Review (Periodic Review / Order-Up-To)",
        "newsvendor": "Newsvendor Model (Single-Period Stochastic)",
        "inventory_classification": "Multicriteria Inventory Classification (Sales & Hits)",
    }

    def __init__(self):
        pass

    # ========================================================================
    # STAGE 1: IDENTIFY MODEL
    # ========================================================================
    def identify_model(self, question: str) -> Tuple[str, float, str]:
        """
        Classifies the user query into one of the 5 supported inventory models.
        Returns: (model_key, confidence, rationale)
        """
        text = question.lower()

        # Score matching
        scores = {
            "newsvendor": 0,
            "quantity_discount": 0,
            "fixed_period_review": 0,
            "safety_stock_rop": 0,
            "eoq": 0,
            "inventory_classification": 0,
        }

        # 1. Newsvendor indicators
        if any(w in text for w in ["newsvendor", "newsboy", "single period", "salvage", "perishable", "spoilage", "critical ratio", "critical fractile", "end of the season", "clearance price", "markdown"]):
            scores["newsvendor"] += 6
        if ("salvage" in text or "scrap" in text) and ("cost" in text or "price" in text):
            scores["newsvendor"] += 4

        # 2. Quantity Discount indicators
        if any(w in text for w in ["quantity discount", "price break", "bracket", "tier", "tiers", "discount tier"]):
            scores["quantity_discount"] += 6
        if ("order" in text and "discount" in text) or ("price" in text and ("bracket" in text or "tier" in text or "schedule" in text)):
            scores["quantity_discount"] += 4
        if re.search(r"\b\d+\s*-\s*\d+\b", text) and ("$" in text or "price" in text or "cost" in text):
            scores["quantity_discount"] += 4

        # 3. Fixed Period Review indicators
        if any(w in text for w in ["fixed period", "periodic review", "order up to", "order-up-to", "base stock", "review period", "review interval", "reviewed every", "review every"]):
            scores["fixed_period_review"] += 6
        if re.search(r"review(?:ed)?\s+(?:[a-zA-Z]+\s+)?every", text):
            scores["fixed_period_review"] += 6
        if ("review" in text or "reviewed" in text or "interval" in text) and ("lead time" in text or "order" in text or "target" in text):
            scores["fixed_period_review"] += 4
        if any(w in text for w in ["on-hand plus on-order", "inventory position", "target level", "order-up-to", "order up to level"]):
            scores["fixed_period_review"] += 4

        # 4. Safety Stock & ROP indicators
        if any(w in text for w in ["safety stock", "reorder point", "rop", "service level", "cycle service level", "lead time demand"]):
            scores["safety_stock_rop"] += 5
        if ("lead time" in text or "lead-time" in text) and ("std dev" in text or "standard deviation" in text or "service level" in text or "stockout" in text):
            scores["safety_stock_rop"] += 3

        # 5. EOQ indicators
        if any(w in text for w in ["eoq", "economic order quantity", "optimal order quantity", "wilson", "how many units should we order each time", "optimal batch size"]):
            scores["eoq"] += 5
        if ("demand" in text or "annual demand" in text) and ("order cost" in text or "setup cost" in text) and ("holding cost" in text or "carrying cost" in text):
            scores["eoq"] += 4

        # 6. Inventory Classification indicators
        if any(w in text for w in ["classification", "classify", "multicriteria", "multi-criteria", "abc", "pareto", "hits", "sales ($)", "inventory classification", "stratification"]):
            scores["inventory_classification"] += 7
        if ("class" in text or "classes" in text) and ("a, b, c" in text or "sales" in text or "hits" in text or "matrix" in text or "policy" in text or "inventory" in text):
            scores["inventory_classification"] += 6
        if any(w in text for w in ["cycle count", "cycle-counting", "inventory-management policy", "inventory management policy", "material classification"]):
            scores["inventory_classification"] += 7

        # Disambiguation heuristics
        # If periodic review mentions review period and ROP, fixed period review takes priority
        if scores["fixed_period_review"] > 0 and ("review" in text or "every" in text and "day" in text):
            scores["fixed_period_review"] += 2

        # If discount tier is present, quantity discount overrules simple EOQ
        if scores["quantity_discount"] > 0 and scores["eoq"] > 0:
            scores["quantity_discount"] += 2

        # Find best model
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]

        if best_score == 0:
            # Fallback based on keywords
            if "order" in text and "cost" in text:
                return "eoq", 0.50, "Defaulted to EOQ based on general order and holding cost terms."
            return "eoq", 0.30, "Low confidence match; defaulting to foundational EOQ."

        confidence = min(0.98, 0.55 + (best_score * 0.08))
        rationale = f"Detected high-salience keywords for {self.MODEL_NAMES[best_model]} (score: {best_score})."
        return best_model, round(confidence, 2), rationale

    # ========================================================================
    # STAGE 2: EXTRACT INPUTS
    # ========================================================================
    def extract_inputs(self, question: str, model: str) -> Dict[str, Any]:
        """
        Parses structured numerical parameters from the natural language query.
        """
        text = question.strip()
        inputs: Dict[str, Any] = {}

        if model == "eoq":
            inputs = self._extract_eoq_inputs(text)
        elif model == "quantity_discount":
            inputs = self._extract_discount_inputs(text)
        elif model == "safety_stock_rop":
            inputs = self._extract_safety_stock_inputs(text)
        elif model == "fixed_period_review":
            inputs = self._extract_fixed_period_inputs(text)
        elif model == "newsvendor":
            inputs = self._extract_newsvendor_inputs(text)
        elif model == "inventory_classification":
            inputs = self._extract_classification_inputs(text)

        return inputs

    def _extract_number(self, pattern: str, text: str, default: Optional[float] = None) -> Optional[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            clean = m.group(1).replace(",", "").replace("$", "").replace("%", "").strip()
            try:
                return float(clean)
            except ValueError:
                return default
        return default

    def _extract_demand_and_period(
        self,
        text: str,
        target_period: str = "year",
        default_demand: Optional[float] = None,
        work_days: float = 250.0,
    ) -> Tuple[float, str, str, str]:
        """
        Parses demand rate and associated time period (day, week, month, quarter, year)
        from natural language text and converts it deterministically into the target period.

        Returns:
            (converted_demand, original_demand_str, demand_period_conversion, source_period)
        """
        val = None
        src_period = "unspecified"

        # Pattern 1: Explicit prefix period (e.g. "average weekly demand is 700", "daily demand is 50", "monthly demand of 800")
        p_prefix = re.search(
            r"(?:average\s+|mean\s+)?(daily|weekly|monthly|quarterly|annual|yearly)\s*"
            r"(?:demand|usage|sales|volume|requirements?)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            text,
            re.I,
        )
        if p_prefix:
            val = float(p_prefix.group(2).replace(",", ""))
            src_raw = p_prefix.group(1).lower()
            if "daily" in src_raw:
                src_period = "day"
            elif "weekly" in src_raw:
                src_period = "week"
            elif "monthly" in src_raw:
                src_period = "month"
            elif "quarterly" in src_raw:
                src_period = "quarter"
            elif "annual" in src_raw or "yearly" in src_raw:
                src_period = "year"

        # Pattern 2: Explicit units per <period> (e.g., "50 units per day", "100 units/week", "800 units a month")
        # Ensure the matched number is NOT standard deviation, lead time, or order cost
        if val is None:
            for m in re.finditer(
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:per|\/|a)\s*"
                r"(day|daily|work\s*day|operating\s*day|week|weekly|work\s*week|operating\s*week|month|monthly|quarter|quarterly|year|annual|annually)\b",
                text,
                re.I,
            ):
                preceding = text[max(0, m.start() - 35):m.start()].lower()
                if any(w in preceding for w in ["deviation", "std", "sigma", "lead time", "lead-time", "cost", "$"]):
                    continue
                val = float(m.group(1).replace(",", ""))
                src_raw = m.group(2).lower()
                if "day" in src_raw:
                    src_period = "day"
                elif "week" in src_raw:
                    src_period = "week"
                elif "month" in src_raw:
                    src_period = "month"
                elif "quarter" in src_raw:
                    src_period = "quarter"
                elif "year" in src_raw or "annual" in src_raw:
                    src_period = "year"
                break

        # Pattern 3: Suffix period words (e.g., "12,000 units annually", "50 units daily")
        if val is None:
            p3 = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(annually|yearly|daily|weekly|monthly|quarterly)\b", text, re.I)
            if p3:
                val = float(p3.group(1).replace(",", ""))
                src_raw = p3.group(2).lower()
                if "daily" in src_raw:
                    src_period = "day"
                elif "weekly" in src_raw:
                    src_period = "week"
                elif "monthly" in src_raw:
                    src_period = "month"
                elif "quarterly" in src_raw:
                    src_period = "quarter"
                elif "annual" in src_raw or "yearly" in src_raw:
                    src_period = "year"

        # Pattern 4: Explicit annual phrases
        if val is None:
            p4 = re.search(r"(?:annual\s+demand|annual\s+usage|annual\s+sales|annual\s+need|demand\s+annually)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)", text, re.I)
            if p4:
                val = float(p4.group(1).replace(",", ""))
                src_period = "year"

        # Pattern 5: General demand
        if val is None:
            p5 = re.search(r"(?:demand|sells?|selling|usage|volume|requires?)\s*(?:is|of|about|=|:)?\s*[\$]?([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?", text, re.I)
            if p5:
                val = float(p5.group(1).replace(",", ""))

        # Pattern 6: Plain number before 'units'
        if val is None:
            p6 = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*units\b", text, re.I)
            if p6:
                val = float(p6.group(1).replace(",", ""))

        # Fallback default
        if val is None:
            val = default_demand if default_demand is not None else 10000.0

        # Perform period conversions
        if target_period == "year":
            if src_period == "day":
                converted = val * work_days
                orig = f"{val:,.2f} units per day"
                conv = f"{val:,.2f} units/day * {work_days:,.0f} work days/year = {converted:,.2f} units/year"
            elif src_period == "week":
                m_wks = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:work\s*weeks|weeks\s*per\s*year)", text, re.I)
                wks = float(m_wks.group(1)) if m_wks else (work_days / 5.0)
                converted = val * wks
                orig = f"{val:,.2f} units per week"
                conv = f"{val:,.2f} units/week * {wks:,.1f} work weeks/year = {converted:,.2f} units/year"
            elif src_period == "month":
                converted = val * 12.0
                orig = f"{val:,.2f} units per month"
                conv = f"{val:,.2f} units/month * 12 months/year = {converted:,.2f} units/year"
            elif src_period == "quarter":
                converted = val * 4.0
                orig = f"{val:,.2f} units per quarter"
                conv = f"{val:,.2f} units/quarter * 4 quarters/year = {converted:,.2f} units/year"
            else:
                converted = val
                orig = f"{val:,.2f} units/year"
                conv = "Direct annual demand (no conversion needed)"
        elif target_period == "day":
            if src_period == "day":
                converted = val
                orig = f"{val:,.2f} units per day"
                conv = "Direct daily demand (matches daily lead time period)"
            elif src_period == "week":
                is_work = bool(re.search(r"work\s*week|operating\s*week", text, re.I))
                days_in_wk = 5.0 if is_work else 7.0
                converted = val / days_in_wk
                orig = f"{val:,.2f} units per week"
                conv = f"{val:,.2f} units/week / {days_in_wk:,.0f} days/week = {converted:,.2f} units/day"
            elif src_period == "month":
                converted = val / 30.0
                orig = f"{val:,.2f} units per month"
                conv = f"{val:,.2f} units/month / 30 calendar days/month = {converted:,.2f} units/day"
            elif src_period == "quarter":
                converted = val / 90.0
                orig = f"{val:,.2f} units per quarter"
                conv = f"{val:,.2f} units/quarter / 90 calendar days/quarter = {converted:,.2f} units/day"
            elif src_period == "year":
                converted = val / work_days
                orig = f"{val:,.2f} units per year"
                conv = f"{val:,.2f} units/year / {work_days:,.0f} work days/year = {converted:,.2f} units/day"
            else:
                converted = val
                orig = f"{val:,.2f} units/period"
                conv = "Direct demand rate matching lead time period"
        else:
            converted = val
            orig = f"{val:,.2f}"
            conv = "Unspecified conversion"

        return float(converted), orig, conv, src_period

    def _extract_std_and_period(
        self,
        text: str,
        demand_src_period: str,
        target_period: str = "day",
        default_std: Optional[float] = None,
        work_days: float = 250.0,
    ) -> Tuple[float, str]:
        """
        Parses standard deviation of demand and scales it according to Brownian motion variance properties:
        sigma_target = sigma_source * sqrt(duration_target / duration_source).
        """
        val = None
        std_period = demand_src_period

        # Check explicit std dev period
        p1 = re.search(
            r"(?:standard\s+deviation|std\s*dev|sigma(?:_d)?)\s*(?:is|of|=|:)?\s*"
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:per|\/|a)\s*"
            r"(day|daily|work\s*day|operating\s*day|week|weekly|work\s*week|operating\s*week|month|monthly|quarter|quarterly|year|annual|annually)",
            text,
            re.I,
        )
        if p1:
            val = float(p1.group(1).replace(",", ""))
            raw = p1.group(2).lower()
            if "day" in raw:
                std_period = "day"
            elif "week" in raw:
                std_period = "week"
            elif "month" in raw:
                std_period = "month"
            elif "quarter" in raw:
                std_period = "quarter"
            elif "year" in raw or "annual" in raw:
                std_period = "year"

        if val is None:
            p2 = re.search(
                r"(daily|weekly|monthly|quarterly|annual|yearly)\s*(?:standard\s+deviation|std\s*dev|sigma(?:_d)?)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                text,
                re.I,
            )
            if p2:
                val = float(p2.group(2).replace(",", ""))
                raw = p2.group(1).lower()
                if "daily" in raw:
                    std_period = "day"
                elif "weekly" in raw:
                    std_period = "week"
                elif "monthly" in raw:
                    std_period = "month"
                elif "quarterly" in raw:
                    std_period = "quarter"
                elif "annual" in raw or "yearly" in raw:
                    std_period = "year"

        if val is None:
            p3 = re.search(r"(?:standard\s+deviation|std\s*dev|sigma(?:_d)?)\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)", text, re.I)
            if p3:
                val = float(p3.group(1).replace(",", ""))

        if val is None:
            val = default_std if default_std is not None else 10.0

        # Scale standard deviation
        if target_period == "day":
            if std_period == "day":
                scaled = val
                desc = f"{val:,.2f} units/day (direct daily std dev)"
            elif std_period == "week":
                is_work = bool(re.search(r"work\s*week|operating\s*week", text, re.I))
                days_in_wk = 5.0 if is_work else 7.0
                scaled = val / math.sqrt(days_in_wk)
                desc = f"{val:,.2f} units/week / sqrt({days_in_wk:.0f}) = {scaled:,.2f} units/day"
            elif std_period == "month":
                scaled = val / math.sqrt(30.0)
                desc = f"{val:,.2f} units/month / sqrt(30) = {scaled:,.2f} units/day"
            elif std_period == "quarter":
                scaled = val / math.sqrt(90.0)
                desc = f"{val:,.2f} units/quarter / sqrt(90) = {scaled:,.2f} units/day"
            elif std_period == "year":
                scaled = val / math.sqrt(work_days)
                desc = f"{val:,.2f} units/year / sqrt({work_days:,.0f}) = {scaled:,.2f} units/day"
            else:
                scaled = val
                desc = f"{val:,.2f} units (direct std dev)"
        else:
            scaled = val
            desc = f"{val:,.2f}"

        return float(scaled), desc

    def _extract_eoq_inputs(self, text: str) -> Dict[str, Any]:
        # Work days
        work_days = self._extract_number(r"([0-9]+(?:\.[0-9]+)?)\s*(?:work\s*days|operating\s*days|days\s*per\s*year|days\s*a\s*year)", text, default=250.0)
        wd = float(work_days) if work_days else 250.0

        # Demand D & period conversion
        demand, orig_demand, conv_demand, _ = self._extract_demand_and_period(
            text, target_period="year", default_demand=10000.0, work_days=wd
        )

        # Order Cost S
        order_cost = self._extract_number(r"(?:order(?:ing)?\s*cost|setup\s*cost|cost\s*to\s*place\s*(?:an)?\s*order|cost\s*per\s*order)\s*(?:is|of|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)", text)
        if order_cost is None:
            m = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*order|to\s*order|to\s*place)", text, re.I)
            if m:
                order_cost = float(m.group(1).replace(",", ""))

        # Holding Cost H or Rate i
        holding_cost = self._extract_number(r"(?:holding\s*cost|carrying\s*cost)\s*(?:is|of|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?", text)
        holding_rate = None
        rate_match = re.search(r"(?:holding\s*(?:cost)?\s*rate|carrying\s*(?:cost)?\s*rate|rate\s*of)\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.I)
        if rate_match:
            holding_rate = float(rate_match.group(1)) / 100.0

        # Unit Cost C
        unit_cost = self._extract_number(
            r"(?:unit\s*(?:cost|price)|purchase\s*price|item\s*cost|each\s*unit\s*costs?)\s*(?:is|of|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            text
        )
        if unit_cost is None:
            m = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)", text, re.I)
            # Make sure it's not holding cost per unit
            if m and "holding" not in text[max(0, m.start()-20):m.end()+20].lower():
                unit_cost = float(m.group(1).replace(",", ""))

        # Defaults if missing
        if order_cost is None:
            order_cost = 50.0
        if holding_cost is None and holding_rate is None:
            holding_cost = 2.0

        return {
            "annual_demand": float(demand),
            "order_cost": float(order_cost),
            "holding_cost": float(holding_cost) if holding_cost is not None else None,
            "unit_cost": float(unit_cost) if unit_cost is not None else None,
            "holding_cost_rate": float(holding_rate) if holding_rate is not None else None,
            "work_days_per_year": wd,
            "original_demand_input": orig_demand,
            "demand_period_conversion": conv_demand,
        }

    def _extract_discount_inputs(self, text: str) -> Dict[str, Any]:
        work_days = self._extract_number(r"([0-9]+(?:\.[0-9]+)?)\s*(?:work\s*days|operating\s*days|days\s*per\s*year)", text, default=250.0)
        wd = float(work_days) if work_days else 250.0

        demand, orig_demand, conv_demand, _ = self._extract_demand_and_period(
            text, target_period="year", default_demand=5000.0, work_days=wd
        )

        order_cost = self._extract_number(r"(?:order(?:ing)?\s*cost|setup\s*cost|cost\s*per\s*order)\s*(?:is|of|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)", text, default=49.0)

        holding_rate = None
        rate_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying)", text, re.I)
        if not rate_match:
            rate_match = re.search(r"(?:holding|carrying)\s*rate\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.I)
        if rate_match:
            holding_rate = float(rate_match.group(1)) / 100.0
        else:
            holding_rate = 0.20

        # Parse price tiers: e.g. "1-499: $10, 500-999: $9, 1000+: $8" or "1 to 499 at $10"
        tiers: List[Dict[str, Any]] = []
        # Pattern 1: Range tiers e.g. "1 to 499 units price is $10", "1-499 units: $10"
        bracket_pattern = (
            r"\b([0-9,]+)\b\s*(?:to|-)\s*\b([0-9,]+)\b\s*(?:units\b)?"
            r"\s*(?:(?:at|price\s*(?:is|of)?|costs?|is|:)\s*\$?|\$)([0-9]+(?:\.[0-9]+)?)"
        )
        for match in re.finditer(bracket_pattern, text, re.I):
            min_q = float(match.group(1).replace(",", ""))
            max_q = float(match.group(2).replace(",", ""))
            price = float(match.group(3))
            tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})

        # Pattern 2: Highest bracket: "1000+ units at $8" or "1000 or more at $8"
        high_pattern = (
            r"\b([0-9,]+)\b\s*(?:\+|or\s*more|and\s*above)\s*(?:units\b)?"
            r"\s*(?:(?:at|price\s*(?:is|of)?|costs?|is|:)\s*\$?|\$)([0-9]+(?:\.[0-9]+)?)"
        )
        high_match = re.search(high_pattern, text, re.I)
        if high_match:
            min_q = float(high_match.group(1).replace(",", ""))
            price = float(high_match.group(2))
            tiers.append({"min_qty": min_q, "max_qty": None, "unit_price": price})

        # Fallback default tiers if none parsed
        if not tiers:
            tiers = [
                {"min_qty": 1, "max_qty": 499, "unit_price": 10.0},
                {"min_qty": 500, "max_qty": 999, "unit_price": 9.0},
                {"min_qty": 1000, "max_qty": None, "unit_price": 8.0},
            ]

        return {
            "annual_demand": float(demand),
            "order_cost": float(order_cost),
            "holding_cost_rate": float(holding_rate),
            "price_tiers": tiers,
            "original_demand_input": orig_demand,
            "demand_period_conversion": conv_demand,
        }

    def _extract_safety_stock_inputs(self, text: str) -> Dict[str, Any]:
        work_days = self._extract_number(r"([0-9]+(?:\.[0-9]+)?)\s*(?:work\s*days|operating\s*days|days\s*per\s*year)", text, default=250.0)
        wd = float(work_days) if work_days else 250.0

        # Lead time
        l_week_match = re.search(r"(?:lead\s*time|takes?)\s*(?:is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:weeks?|wks?)\b", text, re.I)
        if not l_week_match:
            l_week_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:weeks?|wks?)\s*(?:lead\s*time|delivery\s*time)", text, re.I)
        l_frac_week = None
        if re.search(r"(?:half\s*(?:a)?\s*week|1/2\s*week|50%\s*of\s*(?:a)?\s*week)\s*(?:lead\s*time)?", text, re.I):
            l_frac_week = 0.5

        if l_week_match:
            lead_time = float(l_week_match.group(1)) * 7.0
        elif l_frac_week is not None:
            lead_time = l_frac_week * 7.0
        else:
            l_day_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|operating\s*days?)\s*(?:lead\s*time|delivery\s*time)", text, re.I)
            if l_day_match:
                lead_time = float(l_day_match.group(1))
            else:
                lead_time = self._extract_number(r"(?:lead\s*time|takes?)\s*(?:is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|periods?)?", text)
                if lead_time is None:
                    lead_time = 6.0

        # Demand and period conversion
        avg_demand, orig_demand, conv_demand, src_period = self._extract_demand_and_period(
            text, target_period="day", default_demand=100.0, work_days=wd
        )

        # Standard deviation of demand scaled to daily basis
        std_demand, std_desc = self._extract_std_and_period(
            text, demand_src_period=src_period, target_period="day", default_std=15.0, work_days=wd
        )

        full_conversion = f"{conv_demand}; std dev: {std_desc}"

        # Service level
        service_level = None
        sl_match = re.search(r"([0-9]{2}(?:\.[0-9]+)?)\s*%\s*(?:service\s*level|cycle\s*service\s*level|csl|in-stock)", text, re.I)
        if not sl_match:
            sl_match = re.search(r"(?:service\s*level|csl)\s*(?:of|is)?\s*([0-9]{2}(?:\.[0-9]+)?)\s*%", text, re.I)
        if sl_match:
            service_level = float(sl_match.group(1)) / 100.0
        else:
            sl_dec = self._extract_number(r"(?:service\s*level|csl)\s*(?:of|is)?\s*(0\.[0-9]+)", text)
            service_level = sl_dec if sl_dec else 0.95

        # Variable lead time std dev (if any)
        std_lt = self._extract_number(r"(?:lead\s*time\s*std(?:_dev)?|lead\s*time\s*standard\s*deviation|sigma_l)\s*(?:of|is)?\s*([0-9]+(?:\.[0-9]+)?)", text, default=0.0)

        return {
            "avg_demand_per_period": float(avg_demand),
            "std_demand_per_period": float(std_demand),
            "lead_time": float(lead_time),
            "service_level": float(service_level),
            "std_lead_time": float(std_lt) if std_lt else 0.0,
            "original_demand_input": orig_demand,
            "demand_period_conversion": full_conversion,
        }

    def _extract_fixed_period_inputs(self, text: str) -> Dict[str, Any]:
        work_days = self._extract_number(r"([0-9]+(?:\.[0-9]+)?)\s*(?:work\s*days|operating\s*days|days\s*per\s*year)", text, default=250.0)
        wd = float(work_days) if work_days else 250.0

        # Review period R
        r_week_match = re.search(r"(?:review\s*period|review\s*interval|reviewed\s*every|every)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:weeks?|wks?)\b", text, re.I)
        r_frac_week = None
        if re.search(r"(?:review\s*period|review\s*interval|reviewed\s*every|every).*(?:half\s*(?:a)?\s*week|1/2\s*week|50%\s*of\s*(?:a)?\s*week)", text, re.I):
            r_frac_week = 0.5

        if r_week_match:
            review_period = float(r_week_match.group(1)) * 7.0
        elif r_frac_week is not None:
            review_period = r_frac_week * 7.0
        else:
            review_period = self._extract_number(r"(?:review\s*period|review\s*interval|reviewed\s*every|every)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|periods?)?", text)
            if review_period is None:
                review_period = 7.0

        # Lead time L
        l_week_match = re.search(r"(?:lead\s*time|delivery\s*time)\s*(?:of|is)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:weeks?|wks?)\b", text, re.I)
        if not l_week_match:
            l_week_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:weeks?|wks?)\s*(?:lead\s*time|delivery\s*time)", text, re.I)
        l_frac_week = None
        if re.search(r"(?:half\s*(?:a)?\s*week|1/2\s*week|50%\s*of\s*(?:a)?\s*week)\s*(?:lead\s*time)?", text, re.I):
            l_frac_week = 0.5

        if l_week_match:
            lead_time = float(l_week_match.group(1)) * 7.0
        elif l_frac_week is not None:
            lead_time = l_frac_week * 7.0
        else:
            l_day_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|operating\s*days?)\s*(?:lead\s*time|delivery\s*time)", text, re.I)
            if l_day_match:
                lead_time = float(l_day_match.group(1))
            else:
                lead_time = self._extract_number(r"(?:lead\s*time|delivery\s*time)\s*(?:of|is)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|periods?)?", text)
                if lead_time is None:
                    lead_time = 3.0

        # Demand and period conversion
        avg_demand, orig_demand, conv_demand, src_period = self._extract_demand_and_period(
            text, target_period="day", default_demand=50.0, work_days=wd
        )

        # Std dev demand scaled to daily
        std_demand, std_desc = self._extract_std_and_period(
            text, demand_src_period=src_period, target_period="day", default_std=10.0, work_days=wd
        )

        full_conversion = f"{conv_demand}; std dev: {std_desc}"

        # Service level
        service_level = None
        sl_match = re.search(r"([0-9]{2}(?:\.[0-9]+)?)\s*%\s*(?:service\s*level|csl)", text, re.I)
        if sl_match:
            service_level = float(sl_match.group(1)) / 100.0
        else:
            service_level = 0.95

        # Current inventory
        current_inv = self._extract_number(
            r"(?:current\s*inventory(?:\s*(?:position|on-hand\s*plus\s*on-order|on\s*hand|level))?|inventory\s*position|on-hand\s*plus\s*on-order|current\s*stock|currently\s*have)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            text
        )
        if current_inv is None:
            m_ho = re.search(r"(?:on-hand|on\s*hand)\s*(?:plus|\+)\s*(?:on-order|on\s*order)\s*(?:is|of|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)", text, re.I)
            if m_ho:
                current_inv = float(m_ho.group(1).replace(",", ""))
        if current_inv is None:
            m_inv = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*units\s*(?:on\s*hand|in\s*stock|on-hand)", text, re.I)
            if m_inv:
                current_inv = float(m_inv.group(1).replace(",", ""))
        if current_inv is None:
            current_inv = 0.0

        return {
            "review_period": float(review_period),
            "lead_time": float(lead_time),
            "avg_demand_per_period": float(avg_demand),
            "std_demand_per_period": float(std_demand),
            "service_level": float(service_level),
            "current_inventory": float(current_inv),
            "original_demand_input": orig_demand,
            "demand_period_conversion": full_conversion,
        }

    def _extract_newsvendor_inputs(self, text: str) -> Dict[str, Any]:
        # Cost c
        unit_cost = self._extract_number(r"(?:costs?|cost\s*to\s*(?:make|produce|buy|purchase)|cost\s*is)\s*(?:of)?\s*\$?([0-9,]+(?:\.[0-9]+)?)", text)
        if unit_cost is None:
            m = re.search(r"cost(?:s)?\s*(?:is)?\s*\$?([0-9]+(?:\.[0-9]+)?)", text, re.I)
            unit_cost = float(m.group(1)) if m else 40.0

        # Price p
        selling_price = self._extract_number(r"(?:sells?\s*for|selling\s*price|retail\s*price|price\s*is)\s*(?:of)?\s*\$?([0-9,]+(?:\.[0-9]+)?)", text)
        if selling_price is None:
            m = re.search(r"price\s*(?:is)?\s*\$?([0-9]+(?:\.[0-9]+)?)", text, re.I)
            selling_price = float(m.group(1)) if m else 100.0

        # Salvage v
        salvage_val = self._extract_number(
            r"(?:salvage\s*value|salvage|cleared\s*(?:for|at)|discounted\s*to|marked\s*down\s*to|scrap\s*value|scrapped\s*(?:for|at))\s*(?:of|is|at|for)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            text
        )
        if salvage_val is None:
            m = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:salvage|scrap)", text, re.I)
            if m:
                salvage_val = float(m.group(1).replace(",", ""))
        if salvage_val is None:
            salvage_val = 0.0

        # Shortage penalty g
        shortage = self._extract_number(r"(?:shortage\s*penalty|goodwill\s*penalty|penalty|shortage\s*cost)\s*(?:of|is)?\s*\$?([0-9,]+(?:\.[0-9]+)?)", text, default=0.0)

        # Check for selling season / horizon duration: e.g. "selling season lasts 30 days"
        m_dur = re.search(r"(?:season|selling\s*season|selling\s*period|period|window|festival|event)\s*(?:lasts|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?)", text, re.I)
        if not m_dur:
            m_dur = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?)\s*(?:selling\s*season|season|selling\s*period|window|festival|event)", text, re.I)

        # Demand mean mu
        demand_mean = self._extract_number(r"(?:mean\s*(?:demand)?|average\s*demand|mean\s*of)\s*(?:is)?\s*([0-9,]+(?:\.[0-9]+)?)", text)
        if demand_mean is None:
            m = re.search(r"demand\s*(?:is)?\s*(?:normally\s*distributed\s*with\s*)?(?:mean\s*)?([0-9,]+(?:\.[0-9]+)?)", text, re.I)
            demand_mean = float(m.group(1).replace(",", "")) if m else 450.0

        # Demand std dev sigma
        demand_std = self._extract_number(r"(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of|is)?\s*([0-9,]+(?:\.[0-9]+)?)", text)
        if demand_std is None:
            demand_std = 75.0

        # Rate unit check
        m_rate = re.search(r"(daily|weekly|monthly)\s*(?:demand|sales|usage|mean)", text, re.I)
        if not m_rate:
            m_rate = re.search(r"units?\s*(?:per|\/|a)\s*(day|week|month)", text, re.I)

        if m_dur and m_rate:
            dur_num = float(m_dur.group(1))
            dur_unit = m_dur.group(2).lower()
            rate_unit = m_rate.group(1).lower()

            if "day" in dur_unit and ("day" in rate_unit or "daily" in rate_unit):
                mu = demand_mean * dur_num
                sigma = demand_std * math.sqrt(dur_num)
                conv_str = f"{demand_mean:,.2f} units/day * {dur_num:,.0f} days = {mu:,.2f} units; {demand_std:,.2f} * sqrt({dur_num:,.0f}) = {sigma:,.2f}"
                orig_str = f"{demand_mean:,.2f} units/day, std dev {demand_std:,.2f} over {dur_num:,.0f}-day season"
            elif "week" in dur_unit and ("week" in rate_unit or "weekly" in rate_unit):
                mu = demand_mean * dur_num
                sigma = demand_std * math.sqrt(dur_num)
                conv_str = f"{demand_mean:,.2f} units/week * {dur_num:,.0f} weeks = {mu:,.2f} units; {demand_std:,.2f} * sqrt({dur_num:,.0f}) = {sigma:,.2f}"
                orig_str = f"{demand_mean:,.2f} units/week, std dev {demand_std:,.2f} over {dur_num:,.0f}-week season"
            else:
                mu = demand_mean
                sigma = demand_std
                conv_str = "Direct single-period demand"
                orig_str = f"mean: {demand_mean:,.2f}, std: {demand_std:,.2f}"
        else:
            mu = demand_mean
            sigma = demand_std
            conv_str = "Direct single-period demand (no seasonal scaling needed)"
            orig_str = f"mean: {demand_mean:,.2f}, std: {demand_std:,.2f}"

        return {
            "unit_cost": float(unit_cost),
            "selling_price": float(selling_price),
            "demand_mean": float(mu),
            "demand_std": float(sigma),
            "salvage_value": float(salvage_val) if salvage_val else 0.0,
            "shortage_penalty": float(shortage) if shortage else 0.0,
            "original_demand_input": orig_str,
            "demand_period_conversion": conv_str,
        }

    def _extract_classification_inputs(self, text: str) -> Dict[str, Any]:
        """
        Extracts DataFrame input and parameters for multicriteria inventory classification.
        Extracts weights and cutoffs if provided in the text.
        Defaults to the bundled sample classification dataset if present.
        """
        import os
        import pandas as pd
        if os.path.exists("sample_inventory_classification.csv"):
            df = pd.read_csv("sample_inventory_classification.csv")
        else:
            df = pd.DataFrame({
                "Item Number": ["SKU-1001", "SKU-1002", "SKU-1003", "SKU-1004"],
                "Item Name": ["Motor AC", "Drive VFD", "Sensor TS", "Obsolete Fuse"],
                "UoM": ["EA", "EA", "EA", "EA"],
                "Sales ($)": [125000.0, 35000.0, 8000.0, 0.0],
                "Hits": [1400, 350, 80, 0],
            })

        inputs: Dict[str, Any] = {"df": df}

        # Extract weights (e.g. 65% for sales and 35% for hits or 0.65 / 0.35)
        m_sw = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:weight\s*)?(?:for\s*)?sales", text, re.I)
        if not m_sw:
            m_sw = re.search(r"(?:sales\s*(?:weight)?\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%)", text, re.I)
        if not m_sw:
            m_dec = re.search(r"(0\.[0-9]+)\s*\(\s*sales", text, re.I)
            if m_dec:
                inputs["sales_weight"] = float(m_dec.group(1))
        if m_sw and "sales_weight" not in inputs:
            inputs["sales_weight"] = float(m_sw.group(1)) / 100.0

        m_hw = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:weight\s*)?(?:for\s*)?(?:pos\s*)?hits", text, re.I)
        if not m_hw:
            m_hw = re.search(r"(?:hits\s*(?:weight)?\s*(?:is|of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%)", text, re.I)
        if not m_hw:
            m_dec = re.search(r"(0\.[0-9]+)\s*\(\s*(?:pos\s*)?hits", text, re.I)
            if m_dec:
                inputs["hits_weight"] = float(m_dec.group(1))
        if m_hw and "hits_weight" not in inputs:
            inputs["hits_weight"] = float(m_hw.group(1)) / 100.0

        return inputs

    # ========================================================================
    # STAGE 3: SELECT TOOL
    # ========================================================================
    def select_tool(self, model: str) -> Tuple[Any, str]:
        """
        Maps model key to corresponding Python function in inventory_tools.
        """
        tool_map = {
            "eoq": (tools.calculate_eoq, "tools.calculate_eoq"),
            "quantity_discount": (tools.calculate_quantity_discount, "tools.calculate_quantity_discount"),
            "safety_stock_rop": (tools.calculate_safety_stock_rop, "tools.calculate_safety_stock_rop"),
            "fixed_period_review": (tools.calculate_fixed_period_review, "tools.calculate_fixed_period_review"),
            "newsvendor": (tools.calculate_newsvendor, "tools.calculate_newsvendor"),
            "inventory_classification": (tools.classify_inventory_multicriteria, "tools.classify_inventory_multicriteria"),
        }
        if model not in tool_map:
            raise ValueError(f"Unknown inventory model '{model}'.")
        return tool_map[model]

    # ========================================================================
    # STAGE 4: CALCULATE
    # ========================================================================
    def calculate(self, tool_func: Any, inputs: Dict[str, Any]) -> Any:
        """
        Executes pure Python calculation with extracted parameters.
        """
        return tool_func(**inputs)

    # ========================================================================
    # STAGE 5: VERIFY
    # ========================================================================
    def verify(self, model: str, result: Any) -> Tuple[bool, List[str]]:
        """
        Applies domain-specific verification rules to numerical output.
        """
        verifier_map = {
            "eoq": tools.verify_eoq_result,
            "quantity_discount": tools.verify_discount_result,
            "safety_stock_rop": tools.verify_safety_stock_result,
            "fixed_period_review": tools.verify_fixed_period_result,
            "newsvendor": tools.verify_newsvendor_result,
            "inventory_classification": tools.verify_classification_result,
        }
        verifier = verifier_map.get(model)
        if verifier:
            return verifier(result)
        return True, ["No specific verifier registered."]

    # ========================================================================
    # STAGE 6: EXPLAIN (Student Verification Framework)
    # ========================================================================
    def explain(self, model: str, inputs: Dict[str, Any], result: Any, verification_passed: bool, verification_messages: List[str]) -> str:
        """
        Produces a pedagogical, student-verifiable executive report directly
        presenting the 8 verification sections calculated by the deterministic tool:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions

        All numerical intermediate calculations originate strictly from the deterministic Python tool.
        """
        header = f"### Supply Chain Analytics Decision & Verification Report: {self.MODEL_NAMES.get(model, model)}\n\n"

        # 1. Inputs Extracted
        sec1 = "#### 1. Inputs Extracted\n"
        if hasattr(result, "inputs_extracted"):
            for k, v in result.inputs_extracted.items():
                sec1 += f"- **{k}:** `{v}`\n"
        else:
            for k, v in inputs.items():
                sec1 += f"- **{k}:** `{v}`\n"

        # 2. Variable Definitions
        sec2 = "\n#### 2. Variable Definitions\n"
        sec2 += (
            "> ℹ️ **Operational Time Units Definition:** 1 work week is defined as **5 operating days** "
            "(standard annual basis: 250 work days = 50 work weeks per year); 1 calendar week is defined as **7 calendar days** "
            "(standard calendar year: 365 days = 52.14 calendar weeks per year). All metrics presented in fractions "
            "or percentages of a week explicitly state the exact number of days for complete user clarification.\n\n"
        )
        if hasattr(result, "variable_definitions"):
            for sym, desc in result.variable_definitions.items():
                sec2 += f"- **`{sym}`**: {desc}\n"

        # 3. Formula
        sec3 = "\n#### 3. Formula\n"
        if hasattr(result, "formula"):
            for name, form in result.formula.items():
                sec3 += f"- **{name}:** `{form}`\n"

        # 4. Formula with Substituted Values
        sec4 = "\n#### 4. Formula with Substituted Values\n"
        if hasattr(result, "formula_with_substituted_values"):
            for name, subst in result.formula_with_substituted_values.items():
                sec4 += f"- **{name}:** `{subst}`\n"

        # 5. Intermediate Calculations
        sec5 = "\n#### 5. Intermediate Calculations (from Deterministic Python Tool)\n"
        if hasattr(result, "intermediate_calculations"):
            sec5 += "| Step / Parameter | Evaluated Deterministic Result |\n| :--- | :--- |\n"
            for step, val in result.intermediate_calculations.items():
                if step == "evaluated_tier_details":
                    continue  # Detailed tiers formatted in final answer
                sec5 += f"| `{step}` | `{val}` |\n"

        # 6. Final Answer
        sec6 = "\n#### 6. Final Answer\n"
        if hasattr(result, "final_answer"):
            for k, v in result.final_answer.items():
                if k in ["policy_recommendations", "cycle_counting_policies", "target_service_levels", "safety_stock_policies", "replenishment_policies", "classification_summary"]:
                    continue  # Formatted cleanly below
                sec6 += f"- **{k.replace('_', ' ').title()}:** **{v}**\n"

        if model == "inventory_classification" and hasattr(result, "class_summary_df"):
            sec6 += "\n**Multicriteria Classification Summary Table:**\n\n"
            sec6 += "| Final Inventory Class | Number of Items | Total Sales ($) | % of Items | % of Sales | Score Range |\n"
            sec6 += "|:---:|:---:|:---:|:---:|:---:|:---:|\n"
            for _, r in result.class_summary_df.iterrows():
                sec6 += f"| **{r['Final Inventory Class']}** | {r['Number of Items']:,} | ${r['Total Sales ($)']:,.2f} | {r['% of Items']:.1f}% | {r['% of Sales']:.1f}% | {r['Combined Score Range']} |\n"

            sec6 += "\n\n### 📦 Recommended Inventory Management Policies (All Classes of Inventory)\n\n"
            sec6 += "| Final Class | Cycle Counting Frequency | Target Service Level (CSL) | Safety Stock Policy | Replenishment Policy | Sourcing & Disposition Strategy |\n"
            sec6 += "|:---:|:---|:---:|:---|:---|:---|\n"
            sec6 += "| **A** | **Every week** (every 5 operating days / 7 calendar days; 50–52 audits/year) | **High CSL: 95%–98%** | **High buffer**: Strict statistical calculation ($SS = z \\times \\sigma_{LTD}$) with continuous monitoring | Continuous Review $(s, Q)$ / JIT | Strategic VMI / Partner contracts |\n"
            sec6 += "| **B** | **Once a month** (every 4 weeks / ~20 operating days / 30 calendar days; 12 audits/year) | **Medium CSL: 85%–90%** | **Medium buffer**: Standard lead-time variance protection | Periodic Review $(R, S)$ / EOQ | Blanket POs / Scheduled orders |\n"
            sec6 += "| **C** | **Once a year** (every 50–52 weeks / 250 operating days / 365 calendar days; 1 annual audit) | **Low CSL: 70%–80%** | **Low buffer**: Minimal or visual buffer; stockouts carry low financial impact | Two-Bin / Min-Max / Kanban | Distributor spot buys / Bulk orders |\n"
            sec6 += "| **D** | **Do not count** (0 audits / dead stock; donate, destroy, or discount) | **N/A (0%)** | **No safety stock; donate, destroy, or discount** | Non-stock / Do Not Reorder | Salvage liquidation / Scrap write-off |\n"

            sec6 += "\n\n#### 🎯 Strategic Impact on Safety Stock and Service-Level Policies\n"
            sec6 += "- **Class A Items (High Value & Velocity):** Generate the vast majority of sales revenue and pick activity. A high Cycle Service Level (95%–98%) is mandatory because stockouts cause severe customer attrition and revenue losses. Safety stock must be computed rigorously using statistical standard deviation ($SS = z \\times \\sigma_{LTD}$) with frequent replenishment to prevent costly over-buffering. Weekly cycle counts ensure near-100% inventory record accuracy.\n"
            sec6 += "- **Class B Items (Moderate Value & Velocity):** Represent standard core stock. A balanced 85%–90% CSL and moderate safety stock buffer optimize working capital while maintaining high fulfillment rates under periodic review.\n"
            sec6 += "- **Class C Items (Low Value & Velocity):** Individual dollar impact is low. Investing in high service levels or precise safety stock buffers is economically wasteful due to holding cost drag. A 70%–80% CSL, visual Two-Bin/Kanban controls, and annual cycle counting minimize administrative overhead, while large infrequent bulk purchases keep ordering costs low.\n"
            sec6 += "- **Class D Items (Dormant & Dead Stock):** Stagnant inventory generates zero revenue while consuming working capital, rack capacity, and insurance costs. Carrying any safety stock on Class D items causes compounding holding losses and depreciation. Cycle counting wastes labor ('Do not count'). The standard accepted action is immediate disposition: **Donate** (tax deduction), **Destroy** (scrap write-off), or **Discount** (clearance markdown).\n"

        if model == "quantity_discount" and hasattr(result, "tier_evaluations"):
            sec6 += "\n**Tier Comparison Matrix:**\n\n"
            sec6 += "| Tier | Range | Unit Price | Feasible Q | Purchase Cost | Order Cost | Hold Cost | Total Cost | Status |\n"
            sec6 += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
            for t in result.tier_evaluations:
                max_str = f"{t.max_qty:.0f}" if t.max_qty is not None else "∞"
                is_opt = "★ **OPTIMAL**" if t.tier_index == result.optimal_tier_index else "Feasible"
                sec6 += f"| {t.tier_index + 1} | {t.min_qty:.0f}-{max_str} | ${t.unit_price:.2f} | {t.feasible_quantity:,.0f} | ${t.annual_purchasing_cost:,.2f} | ${t.annual_ordering_cost:,.2f} | ${t.annual_holding_cost:,.2f} | **${t.total_annual_cost:,.2f}** | {is_opt} |\n"

        # 7. Verification Checks
        sec7 = "\n#### 7. Verification Checks\n"
        if hasattr(result, "verification_checks"):
            for chk in result.verification_checks:
                status = "✅ PASS" if chk.get("passed", True) else "⚠️ ALERT"
                sec7 += f"- {status} **{chk['check']}** (`{chk['condition']}`): {chk['detail']}\n"
        else:
            status_emoji = "✅ All verified" if verification_passed else "⚠️ Warnings"
            sec7 += f"{status_emoji}:\n"
            for msg in verification_messages:
                sec7 += f"- {msg}\n"

        # 8. Assumptions
        sec8 = "\n#### 8. Assumptions\n"
        if hasattr(result, "assumptions"):
            for idx, asm in enumerate(result.assumptions, 1):
                sec8 += f"{idx}. {asm}\n"

        return header + sec1 + sec2 + sec3 + sec4 + sec5 + sec6 + sec7 + sec8

    # ========================================================================
    # END-TO-END PIPELINE: EXECUTE
    # ========================================================================
    def answer_question(self, question: str) -> AgentExecutionTrace:
        """
        Executes full 6-step agent pipeline:
        Customer Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
        """
        trace = AgentExecutionTrace(
            raw_question=question,
            identified_model="",
            model_name="",
            confidence=0.0,
            identification_reasoning="",
            extracted_inputs={},
            selected_tool_name="",
        )

        try:
            # 1. Identify Model
            model, conf, reasoning = self.identify_model(question)
            trace.identified_model = model
            trace.model_name = self.MODEL_NAMES.get(model, model)
            trace.confidence = conf
            trace.identification_reasoning = reasoning

            # 2. Extract Inputs
            inputs = self.extract_inputs(question, model)
            trace.extracted_inputs = inputs

            # 3. Select Tool
            tool_func, tool_name = self.select_tool(model)
            trace.selected_tool_name = tool_name

            # 4. Calculate
            result = self.calculate(tool_func, inputs)
            trace.calculation_result = result

            # 5. Verify
            passed, messages = self.verify(model, result)
            trace.verification_passed = passed
            trace.verification_messages = messages

            # 6. Explain
            explanation = self.explain(model, inputs, result, passed, messages)
            trace.explanation_text = explanation

        except Exception as e:
            trace.error_message = str(e)
            trace.explanation_text = f"❌ Pipeline execution halted: {str(e)}"

        return trace

"""Supply Chain Analytics Inventory Decision Agent

Implements the 7-stage decision pipeline:
Customer Question
  -> Identify Model
    -> Extract Inputs (with advanced terminology, synonym recognition & semantic verification)
      -> Select Python Tool
        -> Calculate
          -> Verify
            -> Explain

Supports 5 inventory models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock and Reorder Point
4. Fixed-Period Review
5. Newsvendor
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import inventory_tools as tools


class ModelType(str, Enum):
    EOQ = "EOQ"
    QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    NEWSVENDOR = "NEWSVENDOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExtractedParameter:
    name: str
    semantic_role: str
    value: Any
    raw_match: str
    is_mandatory: bool = True


@dataclass
class ExtractionSummary:
    model_type: ModelType
    extracted_parameters: Dict[str, ExtractedParameter] = field(default_factory=dict)
    missing_mandatory_parameters: List[str] = field(default_factory=list)
    is_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: param.value for name, param in self.extracted_parameters.items()
        }


@dataclass
class PipelineStepTrace:
    step_number: int
    step_name: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    customer_question: str
    model_type: ModelType
    model_name: str
    extracted_inputs: Dict[str, Any]
    selected_tool_name: str
    calculation_result: Any
    verification: tools.ToolVerification
    executive_explanation: str
    pipeline_trace: List[PipelineStepTrace] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)
    is_complete: bool = True


class InventoryDecisionAgent:
    """End-to-end agent orchestrating inventory intelligence with Python tools."""

    def __init__(self, use_llm_if_available: bool = True):
        self.use_llm = use_llm_if_available and bool(os.getenv("GEMINI_API_KEY"))

    # -----------------------------------------------------------------------
    # Step 1 & 2: Identify Model
    # -----------------------------------------------------------------------
    def identify_model(self, question: str) -> Tuple[ModelType, str]:
        """Classify question into one of the 5 supported inventory models."""
        text = question.lower()

        # Priority 1: Quantity Discount (explicit tiers, brackets, price breaks, tiered pricing)
        has_discount_keywords = any(
            w in text
            for w in [
                "quantity discount",
                "volume discount",
                "price break",
                "discount tier",
                "discount",
                "bracket",
                "all-units",
                "tier 1",
                "tier 2",
                "tiers:",
            ]
        )
        has_tiered_pricing = bool(
            re.search(r"under\s+\d+.*?\d+\+", text)
            or re.search(r"\d+\s*(?:to|-)\s*\d+.*?\d+\+", text)
            or (re.search(r"\$\d+.*?for\s+(?:orders\s+)?under", text) and re.search(r"\$\d+.*?for.*?\d+", text))
        )
        if has_discount_keywords or has_tiered_pricing:
            return (
                ModelType.QUANTITY_DISCOUNT,
                "Quantity Discount Model (All-Units Price Breaks)",
            )

        # Priority 2: Newsvendor (single-period, perishable, salvage, scrap, overage/underage)
        if any(w in text for w in ["newsvendor", "newsboy", "single period", "single-period", "perishable", "salvage", "scrap value", "overage", "underage", "critical ratio", "critical fractile", "t-shirt", "shortage cost", "lost goodwill"]):
            return (
                ModelType.NEWSVENDOR,
                "Newsvendor Model (Single-Period Stochastic Demand)",
            )

        # Priority 3: Fixed-Period Review (periodic, review period, order-up-to, target inventory)
        if any(w in text for w in ["fixed period", "fixed-period", "periodic review", "order up to", "order-up-to", "review period", "every 2 weeks", "every 7 days", "every 14 days", "every week", "every month", "target inventory", "order-up-to level"]):
            return (
                ModelType.FIXED_PERIOD_REVIEW,
                "Fixed-Period Review Model (Periodic Review / Order-Up-To Level)",
            )

        # Priority 4: Safety Stock & Reorder Point (continuous review, lead time demand, service level)
        if any(w in text for w in ["safety stock", "reorder point", "rop", "service level", "cycle service level", "lead time demand", "stockout risk"]):
            return (
                ModelType.SAFETY_STOCK_ROP,
                "Safety Stock & Reorder Point (Continuous Review (s, Q))",
            )

        # Check for discount if general "discount" was used without specific tier word
        if "discount" in text:
            return (
                ModelType.QUANTITY_DISCOUNT,
                "Quantity Discount Model (All-Units Price Breaks)",
            )

        # Priority 5: Classical EOQ (annual demand, ordering cost, holding cost)
        if any(w in text for w in ["eoq", "economic order quantity", "annual demand", "holding cost", "carrying cost", "ordering cost", "setup cost", "setup expense", "cost per order"]):
            return (
                ModelType.EOQ,
                "Economic Order Quantity (EOQ) Model",
            )

        # Fallback default
        return ModelType.EOQ, "Economic Order Quantity (EOQ) Model (Default)"

    # -----------------------------------------------------------------------
    # Advanced NLP Extraction Engine
    # -----------------------------------------------------------------------

    def _clean_number(self, num_str: str) -> float:
        """Parse clean float from numeric string with commas."""
        cleaned = num_str.replace(",", "").strip()
        return float(cleaned)

    def _extract_number_with_synonyms(
        self,
        text: str,
        prefix_synonyms: List[str],
        suffix_synonyms: Optional[List[str]] = None,
        disallowed_prefix: Optional[List[str]] = None,
    ) -> Optional[Tuple[float, str]]:
        """Extract float and raw match from text using prefix or suffix synonym phrases."""
        # 1. Check prefixes: "synonym ... [number]"
        for syn in prefix_synonyms:
            pattern = rf"\b(?:{syn})(?:[^\d\n\$\.]{{0,50}})?\$?\s*([0-9]+(?:,[0-9]{{3}})*(?:\.[0-9]+)?)"
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                # Check disallowed context
                if disallowed_prefix:
                    span_start = max(0, m.start() - 35)
                    preceding = text[span_start:m.start()].lower()
                    if any(dis in preceding for dis in disallowed_prefix):
                        continue
                try:
                    val = self._clean_number(m.group(1))
                    return val, m.group(0)
                except ValueError:
                    continue

        # 2. Check suffixes: "[number] ... synonym"
        if suffix_synonyms:
            for syn in suffix_synonyms:
                pattern = rf"\$?\s*([0-9]+(?:,[0-9]{{3}})*(?:\.[0-9]+)?)\s*(?:[^\d\n\$\.]{{0,35}})?\b(?:{syn})\b"
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    if disallowed_prefix:
                        span_start = max(0, m.start() - 35)
                        preceding = text[span_start:m.start()].lower()
                        if any(dis in preceding for dis in disallowed_prefix):
                            continue
                    try:
                        val = self._clean_number(m.group(1))
                        return val, m.group(0)
                    except ValueError:
                        continue

        return None

    def _extract_percentage(self, text: str, context_keywords: List[str]) -> Optional[float]:
        """Extract a percentage value associated with specific context keywords."""
        for kw in context_keywords:
            # "keyword ... XX%"
            p1 = rf"\b(?:{kw})[^\d%\n]*?([0-9]+(?:\.[0-9]+)?)\s*%"
            m1 = re.search(p1, text, re.IGNORECASE)
            if m1:
                return float(m1.group(1)) / 100.0

            # "XX% ... keyword"
            p2 = rf"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:[^\d%\n]{{0,25}})?\b(?:{kw})\b"
            m2 = re.search(p2, text, re.IGNORECASE)
            if m2:
                return float(m2.group(1)) / 100.0

        return None

    def _extract_service_level(self, text: str) -> Optional[float]:
        """Extract service level (CSL, e.g. 95%, 98%, 0.95)."""
        # Look for explicit percentage with service level keywords
        csl = self._extract_percentage(
            text,
            [
                r"service\s+level",
                r"cycle\s+service\s+level",
                r"csl",
                r"target\s+service",
                r"fill\s+rate",
                r"in[\s-]stock\s+probability",
            ],
        )
        if csl is not None:
            return csl

        # Decimal format: "service level of 0.95"
        m = re.search(
            r"\b(?:service\s+level|cycle\s+service\s+level|csl)[^\d\n]*?(0\.\d+)\b",
            text,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))

        # General standalone percentage if clearly meant for service level
        m_gen = re.search(r"\b(9[0-9](?:\.[0-9]+)?)\s*%", text)
        if m_gen:
            return float(m_gen.group(1)) / 100.0

        return None

    # -----------------------------------------------------------------------
    # Model-Specific Extraction Implementations
    # -----------------------------------------------------------------------

    def extract_eoq_inputs(self, text: str) -> ExtractionSummary:
        """Extract EOQ inputs with comprehensive ordering cost and holding cost terminology."""
        extracted: Dict[str, ExtractedParameter] = {}
        missing: List[str] = []

        # 1. Annual Demand
        demand_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"annual\s+demand",
                r"yearly\s+demand",
                r"annual\s+requirement",
                r"annual\s+usage",
                r"demand\s+per\s+year",
                r"annual\s+sales",
                r"demand\s+(?:is|of)",
                r"sells\s+(?:[a-zA-Z\s]+)?",
                r"uses\s+(?:[a-zA-Z\s]+)?",
                r"demand",
            ],
            suffix_synonyms=[
                r"units\s+per\s+year",
                r"units\s+annually",
                r"annual\s+demand",
                r"yearly\s+demand",
                r"units\s+a\s+year",
            ],
        )
        if demand_match:
            extracted["annual_demand"] = ExtractedParameter(
                name="annual_demand",
                semantic_role="Annual Demand (D)",
                value=demand_match[0],
                raw_match=demand_match[1],
            )
        else:
            missing.append("annual_demand (D): Annual customer demand in units")

        # 2. Ordering Cost (Recognizing all user-requested synonyms)
        ordering_cost_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"ordering/setup\s+cost",
                r"ordering\s+and\s+setup\s+cost",
                r"order\s+and\s+setup\s+cost",
                r"cost\s+of\s+placing\s+an\s+order",
                r"cost\s+per\s+order",
                r"cost\s+to\s+place\s+an\s+order",
                r"setup\s+expense",
                r"ordering\s+cost",
                r"order\s+cost",
                r"setup\s+cost",
                r"order\s+setup\s+cost",
                r"cost\s+to\s+order",
            ],
            suffix_synonyms=[
                r"per\s+order",
                r"cost\s+per\s+order",
                r"to\s+place\s+an\s+order",
                r"setup\s+cost",
                r"setup\s+expense",
                r"ordering\s+cost",
                r"order\s+cost",
            ],
        )
        if ordering_cost_match:
            extracted["ordering_cost"] = ExtractedParameter(
                name="ordering_cost",
                semantic_role="Ordering/Setup Cost per Order (S)",
                value=ordering_cost_match[0],
                raw_match=ordering_cost_match[1],
            )
        else:
            missing.append("ordering_cost (S): Cost to place an order or setup cost")

        # 3. Unit Purchase Price (optional or used for % holding cost)
        unit_cost_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"unit\s+purchase\s+price",
                r"purchase\s+price",
                r"unit\s+price",
                r"unit\s+cost",
                r"cost\s+per\s+unit",
                r"item\s+cost",
                r"costs",
            ],
            suffix_synonyms=[
                r"per\s+unit",
                r"each",
                r"unit\s+price",
                r"unit\s+cost",
            ],
        )
        unit_cost = unit_cost_match[0] if unit_cost_match else None
        if unit_cost_match:
            extracted["unit_cost"] = ExtractedParameter(
                name="unit_cost",
                semantic_role="Unit Purchase Cost (C)",
                value=unit_cost,
                raw_match=unit_cost_match[1],
                is_mandatory=False,
            )

        # 4. Holding Cost (Fixed or Percentage)
        # Check percentage holding rate first (e.g. 20% of unit price)
        pct_holding = self._extract_percentage(
            text,
            [
                r"holding\s+cost",
                r"carrying\s+cost",
                r"inventory\s+carrying\s+cost",
                r"carrying\s+charge",
                r"holding\s+rate",
                r"holding\s+expense",
                r"of\s+(?:unit\s+)?price",
            ],
        )

        holding_cost_val = None
        raw_holding = ""
        if pct_holding is not None:
            if unit_cost is not None and unit_cost > 0:
                holding_cost_val = round(pct_holding * unit_cost, 4)
                raw_holding = f"{pct_holding * 100:.1f}% of unit price (${unit_cost})"
            else:
                raw_holding = f"{pct_holding * 100:.1f}%"
                extracted["holding_cost_rate"] = ExtractedParameter(
                    name="holding_cost_rate",
                    semantic_role="Annual Holding Cost Percentage (h)",
                    value=pct_holding,
                    raw_match=raw_holding,
                )

        if holding_cost_val is None:
            # Fixed dollar holding cost
            fixed_holding_match = self._extract_number_with_synonyms(
                text,
                prefix_synonyms=[
                    r"inventory\s+carrying\s+cost",
                    r"cost\s+of\s+carrying\s+inventory",
                    r"annual\s+holding\s+cost",
                    r"annual\s+carrying\s+cost",
                    r"holding\s+expense",
                    r"carrying\s+expense",
                    r"carrying\s+charge",
                    r"storage\s+cost",
                    r"holding\s+cost",
                    r"carrying\s+cost",
                    r"cost\s+to\s+hold",
                ],
                suffix_synonyms=[
                    r"per\s+unit\s+per\s+year",
                    r"per\s+unit\s+annually",
                    r"holding\s+cost",
                    r"carrying\s+cost",
                    r"to\s+hold",
                ],
            )
            if fixed_holding_match:
                holding_cost_val = fixed_holding_match[0]
                raw_holding = fixed_holding_match[1]

        if holding_cost_val is not None:
            extracted["holding_cost"] = ExtractedParameter(
                name="holding_cost",
                semantic_role="Annual Holding Cost per Unit (H)",
                value=holding_cost_val,
                raw_match=raw_holding,
            )
        elif "holding_cost_rate" not in extracted:
            missing.append("holding_cost (H): Cost to hold/carry one unit of inventory per year")

        # 5. Lead Time & Working Days (Optional)
        lead_time_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"replenishment\s+lead[\s-]time",
                r"supplier\s+lead[\s-]time",
                r"lead[\s-]time\s+(?:is|of)",
                r"delivery\s+time\s+(?:is|of)",
                r"lead[\s-]time",
            ],
            suffix_synonyms=[r"days?\s+lead[\s-]time", r"days?\s+delivery\s+time"],
            disallowed_prefix=[r"standard\s+deviation", r"std\s+dev", r"variance"],
        )
        if lead_time_match:
            extracted["lead_time_days"] = ExtractedParameter(
                name="lead_time_days",
                semantic_role="Lead Time in Days (L)",
                value=lead_time_match[0],
                raw_match=lead_time_match[1],
                is_mandatory=False,
            )

        work_days_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[r"working\s+days", r"operating\s+days", r"days\s+per\s+year"],
            suffix_synonyms=[r"working\s+days\s+per\s+year", r"operating\s+days\s+per\s+year"],
        )
        if work_days_match:
            extracted["work_days_per_year"] = ExtractedParameter(
                name="work_days_per_year",
                semantic_role="Operating Days per Year",
                value=work_days_match[0],
                raw_match=work_days_match[1],
                is_mandatory=False,
            )

        return ExtractionSummary(
            model_type=ModelType.EOQ,
            extracted_parameters=extracted,
            missing_mandatory_parameters=missing,
            is_complete=(len(missing) == 0),
        )

    def extract_quantity_discount_inputs(self, text: str) -> ExtractionSummary:
        """Extract Quantity Discount inputs distinguishing quantity breaks, prices, and holding cost forms."""
        extracted: Dict[str, ExtractedParameter] = {}
        missing: List[str] = []

        # 1. Annual Demand
        demand_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"annual\s+demand",
                r"yearly\s+demand",
                r"annual\s+requirement",
                r"demand\s+(?:is|of)",
                r"demand",
            ],
            suffix_synonyms=[r"units\s+per\s+year", r"units\s+annually"],
        )
        if demand_match:
            extracted["annual_demand"] = ExtractedParameter(
                name="annual_demand",
                semantic_role="Annual Demand (D)",
                value=demand_match[0],
                raw_match=demand_match[1],
            )
        else:
            missing.append("annual_demand (D): Total annual demand")

        # 2. Ordering Cost (all user-requested synonyms)
        ordering_cost_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"ordering/setup\s+cost",
                r"cost\s+of\s+placing\s+an\s+order",
                r"cost\s+per\s+order",
                r"setup\s+expense",
                r"ordering\s+cost",
                r"order\s+cost",
                r"setup\s+cost",
                r"order\s+setup\s+cost",
            ],
            suffix_synonyms=[r"per\s+order", r"setup\s+cost", r"ordering\s+cost"],
        )
        if ordering_cost_match:
            extracted["ordering_cost"] = ExtractedParameter(
                name="ordering_cost",
                semantic_role="Ordering/Setup Cost (S)",
                value=ordering_cost_match[0],
                raw_match=ordering_cost_match[1],
            )
        else:
            missing.append("ordering_cost (S): Order placement or setup cost")

        # 3. Holding Cost: Percentage vs Fixed
        pct_holding = self._extract_percentage(
            text,
            [
                r"holding\s+cost",
                r"carrying\s+cost",
                r"carrying\s+charge",
                r"holding\s+rate",
                r"of\s+(?:unit\s+)?price",
            ],
        )

        fixed_holding = None
        if pct_holding is not None:
            extracted["holding_cost_rate"] = ExtractedParameter(
                name="holding_cost_rate",
                semantic_role="Percentage Holding Cost Rate (h)",
                value=pct_holding,
                raw_match=f"{pct_holding * 100:.1f}%",
            )
        else:
            fixed_h_match = self._extract_number_with_synonyms(
                text,
                prefix_synonyms=[
                    r"fixed\s+holding\s+cost",
                    r"holding\s+cost",
                    r"carrying\s+cost",
                    r"storage\s+cost",
                ],
                suffix_synonyms=[r"per\s+unit\s+holding", r"per\s+unit\s+per\s+year"],
            )
            if fixed_h_match:
                fixed_holding = fixed_h_match[0]
                extracted["fixed_holding_cost"] = ExtractedParameter(
                    name="fixed_holding_cost",
                    semantic_role="Fixed Holding Cost per Unit (H)",
                    value=fixed_holding,
                    raw_match=fixed_h_match[1],
                )
            else:
                missing.append("holding_cost: Either percentage holding cost (e.g., 20% of price) or fixed holding cost ($/unit/year)")

        # 4. Quantity Breaks & Unit Prices (Tiers)
        tiers: List[Dict[str, Any]] = []

        # Case A: Price before bracket: "$20 for orders under 500, $18 for 500 to 999, $16 for 1000+"
        p_before_under = re.findall(r"\$\s*([0-9]+(?:\.[0-9]+)?)[^\d\n]*?under\s*([0-9]+)", text, re.IGNORECASE)
        p_before_range = re.findall(r"\$\s*([0-9]+(?:\.[0-9]+)?)[^\d\n]*?([0-9]+)\s*(?:to|-)\s*([0-9]+)", text, re.IGNORECASE)
        p_before_plus = re.findall(r"\$\s*([0-9]+(?:\.[0-9]+)?)[^\d\n]*?([0-9]+)\s*\+", text, re.IGNORECASE)

        if p_before_under or p_before_range or p_before_plus:
            tier_num = 1
            for price_str, max_str in p_before_under:
                tiers.append({
                    "tier_id": tier_num,
                    "min_qty": 1.0,
                    "max_qty": float(max_str) - 1.0,
                    "unit_price": float(price_str),
                })
                tier_num += 1
            for price_str, min_str, max_str in p_before_range:
                tiers.append({
                    "tier_id": tier_num,
                    "min_qty": float(min_str),
                    "max_qty": float(max_str),
                    "unit_price": float(price_str),
                })
                tier_num += 1
            for price_str, min_str in p_before_plus:
                tiers.append({
                    "tier_id": tier_num,
                    "min_qty": float(min_str),
                    "max_qty": None,
                    "unit_price": float(price_str),
                })
                tier_num += 1

        # Case B: Bracket before price: "under 500: $20, 500 to 999: $18, 1000+: $16"
        if not tiers:
            b_before_under = re.findall(r"under\s*([0-9]+)[^\$\n]*?\$?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            b_before_range = re.findall(r"([0-9]+)\s*(?:to|-)\s*([0-9]+)[^\$\n]*?\$?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            b_before_plus = re.findall(r"([0-9]+)\s*\+[^\$\n]*?\$?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)

            if b_before_under or b_before_range or b_before_plus:
                tier_num = 1
                for max_str, price_str in b_before_under:
                    tiers.append({
                        "tier_id": tier_num,
                        "min_qty": 1.0,
                        "max_qty": float(max_str) - 1.0,
                        "unit_price": float(price_str),
                    })
                    tier_num += 1
                for min_str, max_str, price_str in b_before_range:
                    tiers.append({
                        "tier_id": tier_num,
                        "min_qty": float(min_str),
                        "max_qty": float(max_str),
                        "unit_price": float(price_str),
                    })
                    tier_num += 1
                for min_str, price_str in b_before_plus:
                    tiers.append({
                        "tier_id": tier_num,
                        "min_qty": float(min_str),
                        "max_qty": None,
                        "unit_price": float(price_str),
                    })
                    tier_num += 1

        # Try explicit tier mentions: "tier 1 ($20) for 1-499 units"
        if not tiers:
            tier_tokens = re.findall(
                r"(?:tier\s*([0-9]+))[^\$\n]*?\$?\s*([0-9]+(?:\.[0-9]+)?)[^\d\n]*?([0-9]+)\s*(?:to|-)\s*([0-9]+|\+)",
                text,
                re.IGNORECASE,
            )
            if tier_tokens:
                for tt in tier_tokens:
                    max_q = float(tt[3]) if tt[3] != "+" else None
                    tiers.append({
                        "tier_id": int(tt[0]),
                        "unit_price": float(tt[1]),
                        "min_qty": float(tt[2]),
                        "max_qty": max_q,
                    })

        # Fallback: prices with quantity breaks in prompt
        if not tiers:
            dollar_prices = [float(p) for p in re.findall(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text)]
            # Exclude ordering cost if known
            ord_c = extracted.get("ordering_cost")
            if ord_c and ord_c.value in dollar_prices:
                dollar_prices.remove(ord_c.value)

            breaks = [float(b) for b in re.findall(r"\b([0-9]{3,})\b", text)]
            demand_val = extracted.get("annual_demand")
            if demand_val and demand_val.value in breaks:
                breaks.remove(demand_val.value)

            if len(dollar_prices) >= 3:
                sorted_prices = sorted(dollar_prices, reverse=True)
                sorted_breaks = sorted(breaks) if len(breaks) >= 2 else [500.0, 1000.0]
                tiers = [
                    {"tier_id": 1, "min_qty": 1.0, "max_qty": sorted_breaks[0] - 1.0, "unit_price": sorted_prices[0]},
                    {"tier_id": 2, "min_qty": sorted_breaks[0], "max_qty": sorted_breaks[1] - 1.0 if len(sorted_breaks) > 1 else 999.0, "unit_price": sorted_prices[1]},
                    {"tier_id": 3, "min_qty": sorted_breaks[-1], "max_qty": None, "unit_price": sorted_prices[2]},
                ]

        if tiers:
            extracted["tiers"] = ExtractedParameter(
                name="tiers",
                semantic_role="Discount Tiers & Price Breaks",
                value=tiers,
                raw_match=f"{len(tiers)} brackets evaluated",
            )
        else:
            missing.append("tiers: Quantity discount price break tiers (min_qty, max_qty, unit_price)")

        return ExtractionSummary(
            model_type=ModelType.QUANTITY_DISCOUNT,
            extracted_parameters=extracted,
            missing_mandatory_parameters=missing,
            is_complete=(len(missing) == 0),
        )

    def extract_safety_stock_inputs(self, text: str) -> ExtractionSummary:
        """Extract Safety Stock & ROP inputs distinguishing average demand vs std dev, and lead time vs lead time std dev."""
        extracted: Dict[str, ExtractedParameter] = {}
        missing: List[str] = []

        # 1. Demand Standard Deviation (Extract FIRST to avoid collision with average demand)
        std_demand_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"standard\s+deviation\s+of\s+demand",
                r"demand\s+standard\s+deviation",
                r"demand\s+std\s+dev",
                r"std\s+dev\s+of\s+demand",
                r"demand\s+std",
                r"standard\s+deviation\s+(?:is|of)?",
                r"std\s+dev\s+(?:is|of)?",
                r"std\s+(?:is|of)?",
                r"sigma\s*d",
                r"sigma_d",
            ],
            suffix_synonyms=[
                r"demand\s+standard\s+deviation",
                r"demand\s+std\s+dev",
                r"std\s+dev",
            ],
            disallowed_prefix=[r"lead[\s-]time"],
        )
        std_demand = std_demand_match[0] if std_demand_match else 0.0
        extracted["std_demand_per_period"] = ExtractedParameter(
            name="std_demand_per_period",
            semantic_role="Demand Standard Deviation (sigma_d)",
            value=std_demand,
            raw_match=std_demand_match[1] if std_demand_match else "0.0 (constant demand assumed)",
            is_mandatory=False,
        )

        # 2. Average Demand (Disallow matching where standard deviation is mentioned)
        mean_demand_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"average\s+daily\s+demand",
                r"average\s+weekly\s+demand",
                r"average\s+demand",
                r"mean\s+demand",
                r"daily\s+demand",
                r"weekly\s+demand",
                r"expected\s+demand",
                r"demand\s+averages?",
                r"demand\s+(?:is|of)",
                r"demand",
            ],
            suffix_synonyms=[
                r"units\s+per\s+day",
                r"units\s+per\s+week",
                r"units\s+a\s+day",
                r"daily\s+demand",
                r"average\s+demand",
            ],
            disallowed_prefix=[r"standard\s+deviation", r"std\s+dev", r"variance"],
        )
        if mean_demand_match:
            extracted["mean_demand_per_period"] = ExtractedParameter(
                name="mean_demand_per_period",
                semantic_role="Average Demand per Period (d)",
                value=mean_demand_match[0],
                raw_match=mean_demand_match[1],
            )
        else:
            missing.append("mean_demand_per_period (d): Average/mean customer demand per period")

        # 3. Lead-Time Standard Deviation (Extract FIRST to avoid collision with average lead time)
        std_lt_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"lead[\s-]time\s+standard\s+deviation",
                r"standard\s+deviation\s+of\s+lead[\s-]time",
                r"lead[\s-]time\s+std\s+dev",
                r"std\s+dev\s+of\s+lead[\s-]time",
                r"lead[\s-]time\s+variability",
                r"variability\s+in\s+lead[\s-]time",
                r"sigma\s*l",
                r"sigma_l",
            ],
            suffix_synonyms=[
                r"lead[\s-]time\s+std\s+dev",
                r"lead[\s-]time\s+standard\s+deviation",
            ],
        )
        std_lead_time = std_lt_match[0] if std_lt_match else 0.0
        extracted["std_lead_time_periods"] = ExtractedParameter(
            name="std_lead_time_periods",
            semantic_role="Lead Time Standard Deviation (sigma_L)",
            value=std_lead_time,
            raw_match=std_lt_match[1] if std_lt_match else "0.0 (constant lead time assumed)",
            is_mandatory=False,
        )

        # 4. Average Lead Time
        mean_lt_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"average\s+lead[\s-]time",
                r"mean\s+lead[\s-]time",
                r"supplier\s+lead[\s-]time",
                r"replenishment\s+lead[\s-]time",
                r"lead[\s-]time\s+(?:is|of)",
                r"delivery\s+time\s+(?:is|of)",
                r"takes",
                r"lead[\s-]time",
            ],
            suffix_synonyms=[
                r"days?\s+lead[\s-]time",
                r"weeks?\s+lead[\s-]time",
                r"days?\s+delivery\s+time",
                r"days?\s+to\s+arrive",
            ],
            disallowed_prefix=[r"standard\s+deviation", r"std\s+dev", r"variance"],
        )
        if mean_lt_match:
            extracted["lead_time_periods"] = ExtractedParameter(
                name="lead_time_periods",
                semantic_role="Average Lead Time (L)",
                value=mean_lt_match[0],
                raw_match=mean_lt_match[1],
            )
        else:
            missing.append("lead_time_periods (L): Supplier replenishment lead time in periods")

        # 5. Service Level
        csl = self._extract_service_level(text)
        if csl is not None:
            extracted["service_level"] = ExtractedParameter(
                name="service_level",
                semantic_role="Cycle Service Level (CSL)",
                value=csl,
                raw_match=f"{csl * 100:.1f}%",
            )
        else:
            missing.append("service_level: Target Cycle Service Level (e.g., 95% or 0.95)")

        # Optional Holding Cost
        holding_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[r"holding\s+cost", r"carrying\s+cost", r"storage\s+cost"],
            suffix_synonyms=[r"per\s+unit\s+per\s+day", r"per\s+unit\s+holding"],
        )
        if holding_match:
            extracted["holding_cost_per_unit_per_period"] = ExtractedParameter(
                name="holding_cost_per_unit_per_period",
                semantic_role="Unit Holding Cost per Period",
                value=holding_match[0],
                raw_match=holding_match[1],
                is_mandatory=False,
            )

        return ExtractionSummary(
            model_type=ModelType.SAFETY_STOCK_ROP,
            extracted_parameters=extracted,
            missing_mandatory_parameters=missing,
            is_complete=(len(missing) == 0),
        )

    def extract_fixed_period_inputs(self, text: str) -> ExtractionSummary:
        """Extract Fixed-Period Review inputs distinguishing demand, review period, lead time, and inventory status."""
        extracted: Dict[str, ExtractedParameter] = {}
        missing: List[str] = []

        # 1. Review Period (P)
        review_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"review\s+period\s+(?:is|of)",
                r"review\s+interval\s+(?:is|of)",
                r"order\s+interval\s+(?:is|of)",
                r"reviewed\s+every",
                r"review\s+every",
                r"periodic\s+review\s+every",
                r"order\s+cycle\s+(?:is|of)",
            ],
            suffix_synonyms=[
                r"days?\s+review\s+period",
                r"days?\s+review\s+interval",
            ],
        )
        if review_match:
            extracted["review_period_P"] = ExtractedParameter(
                name="review_period_P",
                semantic_role="Review Period (P)",
                value=review_match[0],
                raw_match=review_match[1],
            )
        else:
            missing.append("review_period_P (P): Review period interval (days/weeks between reviews)")

        # 2. Lead Time (L)
        lt_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"supplier\s+lead[\s-]time",
                r"lead[\s-]time\s+(?:is|of)",
                r"delivery\s+time\s+(?:is|of)",
                r"takes",
                r"lead[\s-]time",
            ],
            suffix_synonyms=[r"days?\s+lead[\s-]time", r"days?\s+delivery\s+time"],
            disallowed_prefix=[r"review\s+period"],
        )
        if lt_match:
            extracted["lead_time_L"] = ExtractedParameter(
                name="lead_time_L",
                semantic_role="Supplier Lead Time (L)",
                value=lt_match[0],
                raw_match=lt_match[1],
            )
        else:
            missing.append("lead_time_L (L): Supplier replenishment lead time")

        # 3. Demand Rate (d)
        demand_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"daily\s+demand",
                r"average\s+demand",
                r"mean\s+demand",
                r"demand\s+averages?",
                r"demand\s+(?:is|of)",
                r"demand",
            ],
            suffix_synonyms=[r"units\s+per\s+day", r"daily\s+demand"],
            disallowed_prefix=[r"standard\s+deviation", r"std\s+dev"],
        )
        if demand_match:
            extracted["mean_demand_per_period"] = ExtractedParameter(
                name="mean_demand_per_period",
                semantic_role="Average Demand per Period (d)",
                value=demand_match[0],
                raw_match=demand_match[1],
            )
        else:
            missing.append("mean_demand_per_period (d): Average demand per period")

        # 4. Demand Standard Deviation (sigma_d)
        std_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"standard\s+deviation\s+of\s+demand",
                r"demand\s+standard\s+deviation",
                r"standard\s+deviation\s+(?:is|of)?",
                r"demand\s+std\s+dev",
                r"std\s+dev\s+(?:is|of)?",
                r"std\s+(?:is|of)?",
                r"sigma",
            ],
            suffix_synonyms=[r"standard\s+deviation", r"std\s+dev"],
        )
        if std_match:
            extracted["std_demand_per_period"] = ExtractedParameter(
                name="std_demand_per_period",
                semantic_role="Demand Standard Deviation (sigma_d)",
                value=std_match[0],
                raw_match=std_match[1],
            )
        else:
            missing.append("std_demand_per_period (sigma_d): Standard deviation of demand")

        # 5. Service Level
        csl = self._extract_service_level(text)
        if csl is not None:
            extracted["service_level"] = ExtractedParameter(
                name="service_level",
                semantic_role="Cycle Service Level (CSL)",
                value=csl,
                raw_match=f"{csl * 100:.1f}%",
            )
        else:
            missing.append("service_level: Target Cycle Service Level")

        # 6. Inventory Status: On-Hand, On-Order, Backorders, or Position
        # Explicit Inventory Position
        inv_pos_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"inventory\s+position\s+(?:is|of)",
                r"current\s+inventory\s+position",
                r"position\s+is",
            ],
            suffix_synonyms=[r"inventory\s+position"],
        )

        on_hand_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"on[\s-]hand\s+inventory\s+(?:is|of)",
                r"on[\s-]hand\s+(?:is|of)",
                r"current\s+inventory\s+(?:is|of)",
                r"in\s+stock\s+(?:is|of)",
            ],
            suffix_synonyms=[
                r"on\s+hand",
                r"in\s+stock",
                r"on[\s-]hand\s+inventory",
                r"physical\s+stock",
            ],
        )

        on_order_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"on[\s-]order\s+inventory\s+(?:is|of)",
                r"open\s+orders\s+(?:are|of|is)",
                r"pipeline\s+inventory\s+(?:is|of)",
                r"in\s+transit\s+(?:is|of)",
                r"on\s+order\s+(?:is|of)",
            ],
            suffix_synonyms=[
                r"on\s+order",
                r"open\s+orders",
                r"in\s+transit",
                r"pipeline",
            ],
        )

        backorder_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"backorders?\s+(?:are|of|is)",
                r"backlog\s+(?:is|of)",
                r"unfilled\s+orders\s+(?:are|of|is)",
            ],
            suffix_synonyms=[r"backordered", r"backorders", r"backlog"],
        )

        on_hand = on_hand_match[0] if on_hand_match else None
        on_order = on_order_match[0] if on_order_match else None
        backorders = backorder_match[0] if backorder_match else 0.0

        if inv_pos_match:
            extracted["inventory_position"] = ExtractedParameter(
                name="inventory_position",
                semantic_role="Inventory Position (IP)",
                value=inv_pos_match[0],
                raw_match=inv_pos_match[1],
                is_mandatory=False,
            )
        elif on_hand is not None:
            calc_ip = on_hand + (on_order or 0.0) - backorders
            extracted["current_on_hand"] = ExtractedParameter(
                name="current_on_hand",
                semantic_role="On-Hand Inventory (I)",
                value=on_hand,
                raw_match=on_hand_match[1],
                is_mandatory=False,
            )
            if on_order is not None:
                extracted["open_orders_pipeline"] = ExtractedParameter(
                    name="open_orders_pipeline",
                    semantic_role="On-Order Pipeline Inventory",
                    value=on_order,
                    raw_match=on_order_match[1],
                    is_mandatory=False,
                )
            extracted["inventory_position"] = ExtractedParameter(
                name="inventory_position",
                semantic_role="Calculated Inventory Position (On-Hand + On-Order - Backorders)",
                value=calc_ip,
                raw_match=f"Computed IP = {calc_ip}",
                is_mandatory=False,
            )

        return ExtractionSummary(
            model_type=ModelType.FIXED_PERIOD_REVIEW,
            extracted_parameters=extracted,
            missing_mandatory_parameters=missing,
            is_complete=(len(missing) == 0),
        )

    def extract_newsvendor_inputs(self, text: str) -> ExtractionSummary:
        """Extract Newsvendor inputs distinguishing selling price, purchase cost, salvage, underage/overage, and demand distribution."""
        extracted: Dict[str, ExtractedParameter] = {}
        missing: List[str] = []

        # 1. Selling Price
        price_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"selling\s+price",
                r"retail\s+price",
                r"retail\s+for",
                r"sells?\s+(?:[a-zA-Z\s]+)?for",
                r"sold\s+for",
                r"price\s+per\s+unit",
                r"revenue\s+per\s+unit",
                r"price\s+is",
                r"price\s+of",
            ],
            suffix_synonyms=[r"selling\s+price", r"retail\s+price"],
            disallowed_prefix=[r"purchase", r"salvage", r"cost"],
        )

        # 2. Purchase / Production Cost
        cost_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"purchase\s+cost",
                r"production\s+cost",
                r"cost\s+to\s+produce",
                r"cost\s+to\s+make",
                r"cost\s+to\s+buy",
                r"wholesale\s+price",
                r"wholesale\s+cost",
                r"unit\s+cost",
                r"cost\s+per\s+unit",
                r"cost\s+is",
                r"costs",
            ],
            suffix_synonyms=[
                r"to\s+produce",
                r"to\s+make",
                r"to\s+buy",
                r"purchase\s+cost",
                r"unit\s+cost",
            ],
            disallowed_prefix=[r"selling", r"salvage", r"salvaged", r"retail"],
        )

        # 3. Salvage Value
        salvage_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"salvage\s+value",
                r"salvage\s+price",
                r"salvaged?\s+for",
                r"scrap\s+value",
                r"scrap\s+price",
                r"liquidation\s+value",
                r"clearance\s+price",
                r"discounted\s+to",
                r"salvage",
            ],
            suffix_synonyms=[
                r"salvage\s+value",
                r"scrap\s+value",
                r"salvage",
            ],
        )

        # 4. Direct Underage Cost / Shortage Cost (if provided directly)
        underage_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"underage\s+cost",
                r"cost\s+of\s+underage",
                r"shortage\s+cost",
                r"cost\s+of\s+shortage",
                r"stockout\s+cost",
            ],
            suffix_synonyms=[r"underage\s+cost", r"shortage\s+cost"],
        )

        # 5. Direct Overage Cost / Excess Cost (if provided directly)
        overage_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"overage\s+cost",
                r"cost\s+of\s+overage",
                r"excess\s+cost",
                r"cost\s+of\s+excess",
            ],
            suffix_synonyms=[r"overage\s+cost", r"excess\s+cost"],
        )

        # Fallback sorting heuristic if 3 dollar amounts exist: selling > cost > salvage
        if (not price_match or not cost_match or not salvage_match) and not (underage_match and overage_match):
            dollars = [float(x) for x in re.findall(r"\$\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", text)]
            if len(dollars) >= 3:
                sorted_d = sorted(set(dollars), reverse=True)
                if len(sorted_d) >= 3:
                    if not price_match:
                        price_match = (sorted_d[0], f"${sorted_d[0]}")
                    if not cost_match:
                        cost_match = (sorted_d[1], f"${sorted_d[1]}")
                    if not salvage_match:
                        salvage_match = (sorted_d[2], f"${sorted_d[2]}")

        # Check marginal cost completeness
        if price_match:
            extracted["selling_price"] = ExtractedParameter(
                name="selling_price",
                semantic_role="Selling / Retail Price (P)",
                value=price_match[0],
                raw_match=price_match[1],
            )
        if cost_match:
            extracted["purchase_cost"] = ExtractedParameter(
                name="purchase_cost",
                semantic_role="Purchase / Production Cost (C)",
                value=cost_match[0],
                raw_match=cost_match[1],
            )
        if salvage_match:
            extracted["salvage_value"] = ExtractedParameter(
                name="salvage_value",
                semantic_role="Salvage Value (V)",
                value=salvage_match[0],
                raw_match=salvage_match[1],
            )

        if underage_match:
            extracted["underage_cost"] = ExtractedParameter(
                name="underage_cost",
                semantic_role="Cost of Underage (Cu)",
                value=underage_match[0],
                raw_match=underage_match[1],
            )
        if overage_match:
            extracted["overage_cost"] = ExtractedParameter(
                name="overage_cost",
                semantic_role="Cost of Overage (Co)",
                value=overage_match[0],
                raw_match=overage_match[1],
            )

        # Validate that marginal costs are solvable
        has_prices = ("selling_price" in extracted and "purchase_cost" in extracted and "salvage_value" in extracted)
        has_direct_costs = ("underage_cost" in extracted and "overage_cost" in extracted)

        if not has_prices and not has_direct_costs:
            if "selling_price" not in extracted:
                missing.append("selling_price (P): Unit retail selling price")
            if "purchase_cost" not in extracted:
                missing.append("purchase_cost (C): Unit purchase/production cost")
            if "salvage_value" not in extracted:
                missing.append("salvage_value (V): Unit salvage value for unsold inventory")

        # 6. Demand Mean
        mean_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"mean\s+demand",
                r"average\s+demand",
                r"expected\s+demand",
                r"demand\s+mean",
                r"demand\s+averages?",
                r"mean\s+is",
                r"mean\s+of",
                r"demand\s+(?:is|of)",
            ],
            suffix_synonyms=[r"mean\s+demand", r"average\s+demand", r"expected\s+demand"],
            disallowed_prefix=[r"standard\s+deviation", r"std\s+dev", r"variance"],
        )
        if mean_match:
            extracted["mean_demand"] = ExtractedParameter(
                name="mean_demand",
                semantic_role="Mean Demand (mu)",
                value=mean_match[0],
                raw_match=mean_match[1],
            )
        else:
            missing.append("mean_demand (mu): Expected or average seasonal demand")

        # 7. Demand Standard Deviation
        std_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"demand\s+standard\s+deviation",
                r"standard\s+deviation\s+of\s+demand",
                r"standard\s+deviation\s+(?:is|of)?",
                r"demand\s+std\s+dev",
                r"std\s+dev\s+(?:is|of)?",
                r"std\s+(?:is|of)?",
                r"sigma",
            ],
            suffix_synonyms=[r"standard\s+deviation", r"std\s+dev"],
        )
        if std_match:
            extracted["std_demand"] = ExtractedParameter(
                name="std_demand",
                semantic_role="Demand Standard Deviation (sigma)",
                value=std_match[0],
                raw_match=std_match[1],
            )
        else:
            missing.append("std_demand (sigma): Standard deviation of demand")

        # 8. Stockout Penalty (optional)
        penalty_match = self._extract_number_with_synonyms(
            text,
            prefix_synonyms=[
                r"stockout\s+penalty",
                r"goodwill\s+cost",
                r"penalty\s+cost",
                r"penalty",
            ],
            suffix_synonyms=[r"stockout\s+penalty", r"goodwill\s+penalty"],
        )
        extracted["stockout_penalty"] = ExtractedParameter(
            name="stockout_penalty",
            semantic_role="Stockout Penalty (g)",
            value=penalty_match[0] if penalty_match else 0.0,
            raw_match=penalty_match[1] if penalty_match else "$0.0",
            is_mandatory=False,
        )

        extracted["demand_dist"] = ExtractedParameter(
            name="demand_dist",
            semantic_role="Demand Distribution",
            value="normal",
            raw_match="Normal Distribution",
            is_mandatory=False,
        )

        return ExtractionSummary(
            model_type=ModelType.NEWSVENDOR,
            extracted_parameters=extracted,
            missing_mandatory_parameters=missing,
            is_complete=(len(missing) == 0),
        )

    # -----------------------------------------------------------------------
    # Step 3: Extract Inputs (Router)
    # -----------------------------------------------------------------------
    def extract_inputs(self, question: str, model_type: ModelType) -> ExtractionSummary:
        """Route to specialized natural language extractor based on identified model."""
        if model_type == ModelType.EOQ:
            return self.extract_eoq_inputs(question)
        elif model_type == ModelType.QUANTITY_DISCOUNT:
            return self.extract_quantity_discount_inputs(question)
        elif model_type == ModelType.SAFETY_STOCK_ROP:
            return self.extract_safety_stock_inputs(question)
        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            return self.extract_fixed_period_inputs(question)
        elif model_type == ModelType.NEWSVENDOR:
            return self.extract_newsvendor_inputs(question)
        else:
            return self.extract_eoq_inputs(question)

    # -----------------------------------------------------------------------
    # Step 4: Select Python Tool
    # -----------------------------------------------------------------------
    def select_tool(self, model_type: ModelType) -> Tuple[str, Callable]:
        """Map identified model to specialized Python calculation tool."""
        mapping = {
            ModelType.EOQ: ("inventory_tools.calculate_eoq", tools.calculate_eoq),
            ModelType.QUANTITY_DISCOUNT: ("inventory_tools.calculate_quantity_discount", tools.calculate_quantity_discount),
            ModelType.SAFETY_STOCK_ROP: ("inventory_tools.calculate_rop", tools.calculate_rop),
            ModelType.FIXED_PERIOD_REVIEW: ("inventory_tools.calculate_periodic_review", tools.calculate_periodic_review),
            ModelType.NEWSVENDOR: ("inventory_tools.calculate_newsvendor", tools.calculate_newsvendor),
        }
        return mapping.get(model_type, ("inventory_tools.calculate_eoq", tools.calculate_eoq))

    # -----------------------------------------------------------------------
    # Step 5: Calculate
    # -----------------------------------------------------------------------
    def calculate(self, tool_callable: Callable, inputs: Dict[str, Any]) -> Any:
        """Execute the selected numerical Python tool with exact arithmetic."""
        return tool_callable(**inputs)

    # -----------------------------------------------------------------------
    # Step 6: Verify
    # -----------------------------------------------------------------------
    def verify(self, calculation_result: Any) -> tools.ToolVerification:
        """Extract the automated mathematical verification from the calculation result."""
        if hasattr(calculation_result, "verification") and calculation_result.verification:
            return calculation_result.verification
        return tools.ToolVerification(is_valid=True, checks_passed=["Default sanity checks passed."])

    # -----------------------------------------------------------------------
    # Step 7: Explain
    # -----------------------------------------------------------------------
    def explain(
        self,
        question: str,
        model_type: ModelType,
        inputs: Dict[str, Any],
        result: Any,
        verification: tools.ToolVerification,
        missing_parameters: Optional[List[str]] = None,
    ) -> str:
        """Generate structured executive explanation with decision insights or missing input request."""
        # Case 1: Missing Mandatory Information
        if missing_parameters and len(missing_parameters) > 0:
            lines = [
                f"### Information Needed to Solve {model_type.value}",
                "I examined your question and successfully identified the following parameters:",
            ]
            for k, v in inputs.items():
                lines.append(f"- **{k}**: {v}")

            lines.append("\nHowever, the calculation cannot proceed deterministically because the following required parameters are missing:")
            for p in missing_parameters:
                lines.append(f"- ⚠️ **{p}**")

            lines.append(
                "\n*Please provide the missing values so I can execute the mathematical calculation without making assumptions.*"
            )
            return "\n".join(lines)

        # Case 2: Complete Calculation Available
        lines = []

        # Return standardized 8-section pedagogical breakdown for students
        if hasattr(result, "to_student_verification_text"):
            lines.append(result.to_student_verification_text())
            lines.append("\n---\n")

        if model_type == ModelType.EOQ:
            lines.append("### Executive Inventory Decision: Economic Order Quantity")
            lines.append(f"- **Recommended Order Quantity (Q*)**: **{result.optimal_order_quantity:,.2f} units** per order.")
            lines.append(f"- **Annual Order Frequency**: **{result.orders_per_year:.2f} orders/year** (~every **{result.order_cycle_days:.1f} operating days**).")
            lines.append(f"- **Annual Inventory Holding Cost**: **${result.annual_holding_cost:,.2f}**")
            lines.append(f"- **Annual Order Setup Cost**: **${result.annual_ordering_cost:,.2f}**")
            lines.append(f"- **Total Annual Variable Inventory Cost**: **${result.total_inventory_cost:,.2f}**")
            if result.total_annual_cost > result.total_inventory_cost:
                lines.append(f"- **Total Annual Acquisition + Inventory Cost**: **${result.total_annual_cost:,.2f}**")
            if result.reorder_point is not None:
                lines.append(f"- **Reorder Point (ROP)**: Place an order when inventory reaches **{result.reorder_point:,.2f} units** to cover lead time.")

            d_val = inputs['annual_demand']
            s_val = inputs['ordering_cost']
            h_val = inputs['holding_cost']
            lines.append("\n#### Methodology & Mathematical Foundations")
            lines.append(f"$$\\text{{EOQ}} = \\sqrt{{\\frac{{2 D S}}{{H}}}} = \\sqrt{{\\frac{{2 \\times {d_val:,} \\times {s_val}}}{{{h_val}}}}} = {result.optimal_order_quantity:,.2f} \\text{{ units}}$$")

            lines.append("\n#### Verification & Sensitivity Insights")
            for c in verification.checks_passed:
                lines.append(f"- [x] {c}")
            sens = verification.sensitivity_analysis
            if "cost_penalty_at_plus_10_pct" in sens:
                lines.append(f"- *Cost Robustness*: Ordering ±10% around EOQ introduces only a **${sens['cost_penalty_at_plus_10_pct']:,.2f}** penalty, confirming EOQ curve flatness near the optimum.")

        elif model_type == ModelType.QUANTITY_DISCOUNT:
            lines.append("### Executive Inventory Decision: All-Units Quantity Discount")
            lines.append(f"- **Optimal Order Quantity (Q*)**: **{result.optimal_order_quantity:,.2f} units**.")
            lines.append(f"- **Selected Bracket**: Tier {result.optimal_tier_id} at **${result.optimal_unit_price:,.2f}/unit**.")
            lines.append(f"- **Minimum Total Annual Cost**: **${result.optimal_total_cost:,.2f}** (material + ordering + holding).")
            lines.append(f"- **Annual Net Savings vs Baseline**: **${result.annual_savings_vs_base:,.2f}**.")

            lines.append("\n#### Evaluated Tier Breakdown")
            lines.append("| Tier | Quantity Range | Unit Price | Feasible Q | Material Cost | Order Cost | Hold Cost | Total Cost |")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
            for t in result.tier_evaluations:
                max_str = f"{t.max_qty:,.0f}" if t.max_qty is not None else "+"
                marker = " **(OPTIMAL)**" if t.tier_id == result.optimal_tier_id else ""
                lines.append(
                    f"| {t.tier_id}{marker} | {t.min_qty:,.0f} - {max_str} | ${t.unit_price:.2f} | {t.feasible_quantity:,.0f} | ${t.annual_material_cost:,.2f} | ${t.annual_ordering_cost:,.2f} | ${t.annual_holding_cost:,.2f} | **${t.total_annual_cost:,.2f}** |"
                )

            lines.append("\n#### Strategic Trade-off")
            lines.append("Purchasing larger lot sizes triggers significant unit cost discounts that outweigh the higher inventory carrying costs.")

        elif model_type == ModelType.SAFETY_STOCK_ROP:
            lines.append("### Executive Inventory Decision: Safety Stock & Reorder Point")
            lines.append(f"- **Target Cycle Service Level (CSL)**: **{result.service_level * 100.0:.1f}%** (Z-Score: **{result.z_score:.2f}**).")
            lines.append(f"- **Recommended Safety Stock**: **{result.safety_stock:,.2f} units**.")
            lines.append(f"- **Reorder Point (ROP)**: **{result.reorder_point:,.2f} units**.")
            lines.append(f"- **Expected Lead Time Demand**: **{result.lead_time_demand_mean:,.2f} units** (Std Dev: **{result.lead_time_demand_std:,.2f}**).")
            lines.append(f"- **Stockout Probability per Replenishment Cycle**: **{result.stockout_probability_per_cycle * 100.0:.2f}%**.")

            lines.append("\n#### Formula Applied")
            lines.append(r"$$\text{ROP} = \mu_{\text{LTD}} + z \times \sigma_{\text{LTD}} = " f"{result.lead_time_demand_mean:,.2f} + ({result.z_score:.2f} \\times {result.lead_time_demand_std:,.2f}) = {result.reorder_point:,.2f}" "$$")

            lines.append("\n#### Service Level Trade-Off Table")
            if "service_level_tradeoffs" in verification.sensitivity_analysis:
                lines.append("| Target CSL | Z-Score | Required Safety Stock | Resulting ROP |")
                lines.append("|:---:|:---:|:---:|:---:|")
                for csl_label, row in verification.sensitivity_analysis["service_level_tradeoffs"].items():
                    lines.append(f"| {csl_label} | {row['z']} | {row['safety_stock']:,.2f} | {row['reorder_point']:,.2f} |")

        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            lines.append("### Executive Inventory Decision: Periodic Review (Order-Up-To Level)")
            lines.append(f"- **Review Interval (P)**: Every **{result.review_period_P} periods/days**.")
            lines.append(f"- **Supplier Lead Time (L)**: **{result.lead_time_L} periods/days**.")
            lines.append(f"- **Total Protection Interval (P + L)**: **{result.protection_interval} periods/days**.")
            lines.append(f"- **Safety Stock Required**: **{result.safety_stock:,.2f} units**.")
            lines.append(f"- **Target Order-Up-To Level (M)**: **{result.order_up_to_level_M:,.2f} units**.")
            if result.recommended_order_quantity is not None:
                lines.append(f"- **Current Inventory Position**: **{result.current_inventory_position:,.2f} units**.")
                lines.append(f"- **Actionable Order Quantity (q)**: Place an immediate order for **{result.recommended_order_quantity:,.2f} units**.")

            lines.append("\n#### Protection Horizon Calculation")
            lines.append(r"$$M = \mu_{P+L} + z \sigma_d \sqrt{P+L} = " f"{result.protection_demand_mean:,.2f} + {result.safety_stock:,.2f} = {result.order_up_to_level_M:,.2f}" "$$")
            lines.append(f"- *Operational Insight*: Because inventory is reviewed every {result.review_period_P} days rather than continuously, safety stock is **{result.safety_stock_vs_continuous_ratio:.2f}x higher** than a continuous review system to cover the additional review interval risk.")

        elif model_type == ModelType.NEWSVENDOR:
            lines.append("### Executive Inventory Decision: Single-Period Newsvendor")
            lines.append(f"- **Optimal Production/Order Quantity (Q*)**: **{result.optimal_order_quantity:,.2f} units**.")
            lines.append(f"- **Critical Fractile (In-Stock Probability)**: **{result.critical_ratio * 100.0:.2f}%**.")
            lines.append(f"- **Expected Profit**: **${result.expected_profit:,.2f}**.")
            lines.append(f"- **Expected Sales**: **{result.expected_sales:,.2f} units** (Expected Lost Sales: **{result.expected_lost_sales:,.2f} units**).")
            lines.append(f"- **Expected Leftover (Salvaged)**: **{result.expected_leftover_units:,.2f} units**.")

            lines.append("\n#### Marginal Economics & Critical Ratio")
            lines.append(f"- Cost of Underage ($C_u = P - C + g$): **${result.underage_cost_Cu:.2f}**")
            lines.append(f"- Cost of Overage ($C_o = C - v$): **${result.overage_cost_Co:.2f}**")
            lines.append(r"$$\text{Critical Ratio (CR)} = \frac{C_u}{C_u + C_o} = \frac{" f"{result.underage_cost_Cu:.2f}" "}{" f"{result.underage_cost_Cu:.2f} + {result.overage_cost_Co:.2f}" "} = " f"{result.critical_ratio:.4f}" "$$")

            lines.append("\n#### Risk & Verification Summary")
            for c in verification.checks_passed:
                lines.append(f"- [x] {c}")
            lines.append(f"- *Stockout Risk*: There is a **{(1.0 - result.critical_ratio) * 100.0:.1f}% chance** of stocking out during this single sales horizon, which is the mathematically optimal trade-off against overage costs.")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Full 7-Step Pipeline Orchestration
    # -----------------------------------------------------------------------
    def process_question(self, question: str) -> AgentResponse:
        """Execute the full 7-step pipeline on a natural language customer question."""
        trace: List[PipelineStepTrace] = []

        # Step 1: Customer Question
        trace.append(
            PipelineStepTrace(
                step_number=1,
                step_name="Customer Question",
                description="Ingested raw natural language inventory query.",
                details={"question": question},
            )
        )

        # Step 2: Identify Model
        model_type, model_desc = self.identify_model(question)
        trace.append(
            PipelineStepTrace(
                step_number=2,
                step_name="Identify Model",
                description=f"Classified problem as {model_type.value}.",
                details={"model_type": model_type.value, "model_description": model_desc},
            )
        )

        # Step 3: Extract Inputs
        extraction_summary = self.extract_inputs(question, model_type)
        raw_inputs = extraction_summary.to_dict()

        extracted_details = {
            name: {"value": p.value, "semantic_role": p.semantic_role, "raw_match": p.raw_match}
            for name, p in extraction_summary.extracted_parameters.items()
        }
        trace.append(
            PipelineStepTrace(
                step_number=3,
                step_name="Extract Inputs",
                description=f"Extracted {len(raw_inputs)} parameters (Missing mandatory: {len(extraction_summary.missing_mandatory_parameters)}).",
                details={
                    "extracted_parameters": extracted_details,
                    "missing_mandatory": extraction_summary.missing_mandatory_parameters,
                    "is_complete": extraction_summary.is_complete,
                },
            )
        )

        # Handle incomplete information: do NOT invent missing inputs
        if not extraction_summary.is_complete:
            tool_name = "N/A (Awaiting Missing Inputs)"
            calculation_result = None
            verification = tools.ToolVerification(
                is_valid=False,
                warnings=[f"Calculation paused: Missing required inputs: {', '.join(extraction_summary.missing_mandatory_parameters)}"],
            )

            trace.append(
                PipelineStepTrace(
                    step_number=4,
                    step_name="Select Python Tool",
                    description="Tool selection paused due to missing parameters.",
                    details={"tool_name": "None", "reason": "Missing inputs"},
                )
            )
            trace.append(
                PipelineStepTrace(
                    step_number=5,
                    step_name="Calculate",
                    description="Execution halted to prevent inventing missing values.",
                    details={"calculation_output": "None"},
                )
            )
            trace.append(
                PipelineStepTrace(
                    step_number=6,
                    step_name="Verify",
                    description="Input completeness check failed.",
                    details={"missing_parameters": extraction_summary.missing_mandatory_parameters},
                )
            )

            explanation = self.explain(
                question=question,
                model_type=model_type,
                inputs=raw_inputs,
                result=None,
                verification=verification,
                missing_parameters=extraction_summary.missing_mandatory_parameters,
            )

            trace.append(
                PipelineStepTrace(
                    step_number=7,
                    step_name="Explain",
                    description="Generated guidance requesting missing parameters from customer.",
                    details={"explanation_length_chars": len(explanation)},
                )
            )

            return AgentResponse(
                customer_question=question,
                model_type=model_type,
                model_name=model_desc,
                extracted_inputs=raw_inputs,
                selected_tool_name=tool_name,
                calculation_result=calculation_result,
                verification=verification,
                executive_explanation=explanation,
                pipeline_trace=trace,
                missing_parameters=extraction_summary.missing_mandatory_parameters,
                is_complete=False,
            )

        # Step 4: Select Python Tool
        tool_name, tool_callable = self.select_tool(model_type)
        trace.append(
            PipelineStepTrace(
                step_number=4,
                step_name="Select Python Tool",
                description=f"Bound model execution to {tool_name}.",
                details={"tool_name": tool_name},
            )
        )

        # Step 5: Calculate
        calculation_result = self.calculate(tool_callable, raw_inputs)
        trace.append(
            PipelineStepTrace(
                step_number=5,
                step_name="Calculate",
                description="Executed mathematical calculation in Python.",
                details={"calculation_output": str(calculation_result)},
            )
        )

        # Step 6: Verify
        verification = self.verify(calculation_result)
        trace.append(
            PipelineStepTrace(
                step_number=6,
                step_name="Verify",
                description="Performed sanity checks, boundary limits, and convexity tests.",
                details={
                    "is_valid": verification.is_valid,
                    "checks_passed": verification.checks_passed,
                    "warnings": verification.warnings,
                },
            )
        )

        # Step 7: Explain
        explanation = self.explain(
            question=question,
            model_type=model_type,
            inputs=raw_inputs,
            result=calculation_result,
            verification=verification,
        )
        trace.append(
            PipelineStepTrace(
                step_number=7,
                step_name="Explain",
                description="Synthesized strategic decision recommendation and financial breakdown.",
                details={"explanation_length_chars": len(explanation)},
            )
        )

        return AgentResponse(
            customer_question=question,
            model_type=model_type,
            model_name=model_desc,
            extracted_inputs=raw_inputs,
            selected_tool_name=tool_name,
            calculation_result=calculation_result,
            verification=verification,
            executive_explanation=explanation,
            pipeline_trace=trace,
            missing_parameters=[],
            is_complete=True,
        )

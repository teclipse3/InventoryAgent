"""
Supply Chain Analytics Inventory Decision Agent
----------------------------------------------
Implements the 7-step decision architecture:
1. Customer Question
2. Identify Models
3. Extract Inputs
4. Select Python Tool
5. Calculate
6. Verify
7. Explain

All numerical calculations are executed strictly via deterministic Python
functions in `inventory_tools.py`.
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple, Union

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_periodic_review,
    calculate_newsvendor,
    calculate_safety_stock_and_rop,
    calculate_fixed_period_review,
    EOQResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
)


# ---------------------------------------------------------------------------
# Structured Pipeline Step Outputs
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    is_valid: bool
    status: str  # "PASSED", "WARNING", "FAILED"
    checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationResult:
    executive_summary: str
    mathematical_formula: str
    intermediate_steps: List[str] = field(default_factory=list)
    cost_or_risk_breakdown: Dict[str, Any] = field(default_factory=dict)
    managerial_insights: List[str] = field(default_factory=list)
    formula_with_substituted_values: Optional[str] = None
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    verification_checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPipelineResponse:
    # 1. Customer Question
    customer_question: str

    # 2. Identify Models
    identified_model: str
    model_key: str  # 'eoq', 'quantity_discount', 'safety_stock_rop', 'fixed_period_review', 'newsvendor'
    model_confidence: float
    identification_rationale: str

    # 3. Extract Inputs
    extracted_inputs: Dict[str, Any]

    # 4. Select Python Tool
    selected_tool_name: str
    tool_module: str = "inventory_tools"
    input_extraction_notes: List[str] = field(default_factory=list)

    # 5. Calculate
    raw_calculation_result: Dict[str, Any] = field(default_factory=dict)

    # 6. Verify
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True, "PASSED"))

    # 7. Explain
    explanation: ExplanationResult = field(default_factory=lambda: ExplanationResult("", ""))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Natural Language Parsing & Heuristic Extraction Engine
# ---------------------------------------------------------------------------

class InventoryNLPParser:
    """
    Domain-specific Natural Language Understanding engine for Supply Chain
    Analytics inventory queries. Recognizes terminology and synonyms across:
    1. Economic Order Quantity (EOQ)
    2. Quantity Discount Model
    3. Safety Stock and Reorder Point (ROP)
    4. Fixed-Period Review Model
    5. Newsvendor Model
    """

    MODEL_PATTERNS = {
        "quantity_discount": [
            r"quant[i|y]ty\s+discou?nt",
            r"discou?nt\s+\bt[i|e]ers?\b",
            r"pr[i|y]ce\s+bre[a]?k",
            r"quant[i|y]ty\s+bre[a]?k",
            r"volum[e]?\s+discou?nt",
            r"bulk\s+discou?nt",
            r"pr[i|y]ce\s+\bt[i|e]ers?\b",
            r"\bt[i|e]ers?:",
            r"(?:[\$€£₹¥]|rs\.?|inr|cad\s*\$?|aud\s*\$?|usd\s*\$?|chf|gbp|eur)?\s*\d+(\.\d+)?\s+if\s+ord[e]?r\s+at\s+least",
            r"percent[a|e]?ge\s+hold[i]?ng\s+costs?t?",
            r"fixed\s+hold[i]?ng\s+costs?t?",
        ],
        "fixed_period_review": [
            r"f[i|y]xed[\s-]per[i]?od",
            r"per[i]?od[i]?c\s+re[v]?e?i[e]?w",
            r"ord[e]?r[\s-]up[\s-]to",
            r"base\s+stock",
            r"re[v]?e?i[e]?w\s+per[i]?od",
            r"re[v]?e?i[e]?wed\s+ev[e]?ry",
            r"re[v]?e?i[e]?ws?\s+inve?nt[o|e]ry\s+ev[e]?ry",
            r"re[v]?e?i[e]?w\s+int[e]?rval",
            r"targ[e]?t\s+inve?nt[o|e]ry\s+lev[e]?l",
            r"targ[e]?t\s+inve?nt[o|e]ry",
            r"inve?nt[o|e]ry\s+pos[i]?t[i]?on",
            r"on[\s-]hand.*?on[\s-]ord[e]?r",
        ],
        "newsvendor": [
            r"newsv[e|a]nd[e|o]r",
            r"newsboy",
            r"s[i|y]ngle[\s-]per[i]?od",
            r"cr[i]?t[i]?cal\s+fract[i]?le",
            r"cr[i]?t[i]?cal\s+rat[i]?o",
            r"salv[a|i]?ge\s+val[u]?e",
            r"salv[a|i]?ge",
            r"mark[\s-]?down\s+pr[i|y]ce",
            r"per[i]?sh[a|i]ble",
            r"cost\s+of\s+und[e]?rage",
            r"und[e]?rage\s+costs?t?",
            r"cost\s+of\s+ov[e]?rage",
            r"ov[e]?rage\s+costs?t?",
            r"exc[e]?ss\s+costs?t?",
            r"season[a]?l\s+(?:item|demm?and|product)",
            r"hol[i]?day\s+season",
        ],
        "safety_stock_rop": [
            r"saf[e|t]ty\s+stock",
            r"safty\s+stock",
            r"safety\s+buffer",
            r"re[\s-]?ord[e]?r\s+po[i]?nt",
            r"\brop\b",
            r"serv[i|e]?[s|c]e\s+lev[e|a]l",
            r"cyc[l|e]e\s+serv[i|e]?[s|c]e",
            r"lead[\s-]t[i|y]me\s+demm?and",
            r"f[i]?ll\s+rate",
            r"stock[\s-]?out\s+probab[i]?l[i]?ty",
            r"lead[\s-]t[i|y]me\s+stand[a]?rd\s+dev[i]?at[i]?on",
            r"lead[\s-]t[i|y]me\s+std\s+dev",
        ],
        "eoq": [
            r"econom[i]?c\s+ord[e]?r\s+quant[i]?ty",
            r"\beoq\b",
            r"opt[i]?mal\s+ord[e]?r\s+quant[i]?ty",
            r"opt[i]?mal\s+lot\s+s[i]?ze",
            r"ord[e]?r\s+quant[i]?ty\s+should\s+we\s+place",
            r"how\s+many\s+units\s+should\s+(?:we|they)\s+ord[e]?r",
            r"batch\s+s[i]?ze",
            r"set[\s-]?up\s+exp[e]?ns[e]?",
            r"costs?t?\s+of\s+plac[i]?ng\s+an?\s+ord[e]?r",
        ],
    }

    @classmethod
    def identify_model(cls, text: str) -> Tuple[str, str, float, str]:
        """
        Identify the most appropriate inventory model from user query text.
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {
            "quantity_discount": 0.0,
            "fixed_period_review": 0.0,
            "newsvendor": 0.0,
            "safety_stock_rop": 0.0,
            "eoq": 0.0,
        }

        for model_key, patterns in cls.MODEL_PATTERNS.items():
            for pat in patterns:
                matches = re.findall(pat, text_lower)
                if matches:
                    scores[model_key] += len(matches) * 2.5

        # Secondary context cues
        if re.search(r"salv[a|i]?ge|unsold|und[e]?rage|ov[e]?rage|perish[a|i]ble|season[a]?l", text_lower):
            scores["newsvendor"] += 4.0
        if re.search(r"\bt[i|e]ers?\b|pr[i|y]ce\s*bre[a]?k|quant[i|y]ty\s*bre[a]?k|discou?nt", text_lower):
            scores["quantity_discount"] += 4.0
        if re.search(r"re[v]?e?i[e]?w\s*per[i]?od|re[v]?e?i[e]?wed\s*ev[e]?ry|ord[e]?r[\s-]up[\s-]to|targ[e]?t\s*inve?nt[o|e]ry|re[v]?e?i[e]?ws?\s+inve?nt[o|e]ry", text_lower):
            scores["fixed_period_review"] += 4.0
        if re.search(r"re[\s-]?ord[e]?r\s*po[i]?nt|saf[e|t]ty\s*stock|safty\s*stock|safety\s*buffer", text_lower) and not re.search(r"per[i]?od[i]?c|re[v]?e?i[e]?w", text_lower):
            scores["safety_stock_rop"] += 4.0
        if re.search(r"set[\s-]?up\s*costs?t?|ord[e]?r(?:ing)?\s*costs?t?|set[\s-]?up\s*exp[e]?ns[e]?|costs?t?\s*per\s*ord[e]?r|batch\s*s[i]?ze", text_lower) and not re.search(r"\bt[i|e]ers?\b", text_lower):
            scores["eoq"] += 2.0

        best_model = max(scores, key=scores.get)
        max_score = scores[best_model]

        model_display_names = {
            "eoq": "Economic Order Quantity (EOQ)",
            "quantity_discount": "Quantity Discount Model",
            "safety_stock_rop": "Safety Stock and Reorder Point (ROP)",
            "fixed_period_review": "Fixed-Period Review Model",
            "newsvendor": "Newsvendor Model",
        }

        if max_score == 0:
            best_model = "eoq"
            confidence = 0.50
            rationale = "General inventory optimization query detected. Defaulting to basic Economic Order Quantity (EOQ)."
        else:
            confidence = min(0.99, 0.60 + (max_score * 0.06))
            rationale = f"Detected key patterns corresponding to {model_display_names[best_model]} (score: {max_score:.1f})."

        return model_display_names[best_model], best_model, confidence, rationale

    @classmethod
    def extract_numbers_with_context(cls, text: str) -> Dict[str, Any]:
        """
        Examine the entire question and extract numerical parameters, mapping them
        to their exact semantic meaning before declaring any information missing.
        """
        inputs: Dict[str, Any] = {}
        # Clean text by removing commas inside numbers (e.g. 10,000 -> 10000) while preserving clause commas
        text_clean = re.sub(r'(?<=\d),(?=\d)', '', text)

        # Generalized vocabulary tokens
        unit_nouns = (
            r"(?:units?|components?|parts?|items?|pieces?|cases?|boxes?|cartons?|packages?|products?|"
            r"assemblies|devices?|kits?|bottles?|gallons?|pounds?|lbs?|pallets?|widgets?|jackets?|coats?|"
            r"motors?|cycles?|bikes?|computers?|liters?|litres?|tons?|metric\s+tons?|kilograms?|kg|barrels?|"
            r"barels?|bbl|meters?|metres?|spools?|drums?|crates?|sacks?|bags?|rolls?|sheets?|tubes?|vials?|"
            r"hectoliters?|hectolitres?)"
        )
        curr_sym = r"(?:[\$€£₹¥]|rs\.?|inr|cad\s*\$?|aud\s*\$?|usd\s*\$?|chf|gbp|eur)?"
        pct_word = r"(?:%|percent|percentage)"
        annual_timeframes = r"(?<!semi[\s-])(?<!semi)(?<!bi[\s-])(?<!bi)\b(?:annually|anuall?y|annual|anual|yearly|yeerly|per\s+year|each\s+year|a\s+year|every\s+year|per\s+annum|\/\s*year|\/\s*yr\b)"
        monthly_timeframes = r"(?:monthly|mothly|montly|per\s+month|each\s+month|a\s+month|every\s+month|\/\s*month|\/\s*mo\b)"
        weekly_timeframes = r"(?<!bi[\s-])(?<!bi)\b(?:weekly|wekly|per\s+week|each\s+week|a\s+week|every\s+week|\/\s*week|\/\s*wk\b)"
        biweekly_timeframes = r"(?:bi[\s-]weekly|biweekly|every\s+two\s+weeks|every\s+2\s+weeks|per\s+two\s+weeks|fortnightly|per\s+fortnight)"
        quarterly_timeframes = r"(?:quarterly|quarterley|per\s+quarter|each\s+quarter|a\s+quarter|every\s+quarter|\/\s*quarter|\/\s*qtr\b)"
        semiannual_timeframes = r"(?:semi[\s-]annually|semiannually|semi[\s-]annual|bi[\s-]annually|biannually|every\s+6\s+months|every\s+six\s+months|per\s+half[\s-]year|half[\s-]yearly)"
        daily_timeframes = r"(?:daily|dialy|dayly|per\s+day|each\s+day|a\s+day|every\s+day|\/\s*day|per\s+diem)"

        need_verbs = r"(?:needs?|requir[e]?s?|uses?|consumes?|demm?ands?|ord[e]?rs?|purchases?|sells?|experiences?|expects?|goes\s+through)"
        need_nouns = r"(?:demm?ands?|requir[e]?ments?|usage|needs?|volume|consumption|sales?|purchases?|throughput)"
        # Variable symbols often written in parentheses or brackets e.g. (D), (d), (S), (H), (C), (L), (P), (M), (IP), (σd), (σL), [L]
        var_sym = r"(?:\s*[\(\[]\s*[\w\s_\'\"\u0370-\u03ff\u1d00-\u1d7f\u2070-\u209f]+\s*[\)\]])?"
        # Flexible intervening item descriptor that ignores operational schedule and carrying rate tokens
        item_filler = r"(?:(?!(?:business\s+|working\s+|operating\s+)?days\b|per\s+unit\b)[a-zA-Z\s-]){0,40}?\s*"

        # -------------------------------------------------------------
        # 0. Operational Schedule (Working days per year)
        # Extracted early to support daily-to-annual demand normalization
        # -------------------------------------------------------------
        work_days_patterns = [
            r"(?:operate|operating|open|works?|working)\s*(\d+(?:\.\d+)?)\s*(?:days?\s+(?:a|per)\s+year|days?\s*\/\s*year|days?\s+annually)",
            r"(\d+(?:\.\d+)?)\s*(?:working|operating|business)?\s*days\s+(?:a|per)\s+year",
            r"(\d+(?:\.\d+)?)\s*days\s+of\s+operation\s+per\s+year",
        ]
        w_days = 365.0
        for pat in work_days_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                w_days = float(m.group(1))
                inputs["working_days_per_year"] = w_days
                break

        # -------------------------------------------------------------
        # 1. Ordering Cost Synonyms (EOQ & Quantity Discount)
        # -------------------------------------------------------------
        order_cost_patterns = [
            rf"(?:set[\s-]?up\s+expens[e]?|set[\s-]?up\s+costs?t?|expens[e]?|costs?t?)\s+(?:for|to)\s+(?:plac(?:ing)?|replenish(?:ing)?)(?:\s+each|\s+an?)?\s+(?:replenishm?ent\s+)?ord[e]?r{var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:incurs\s+(?:an?\s+)?(?:fixed\s+)?)?(?:ord[e]?r\s+placement|ord[e]?r(?:ing)?|set[\s-]?up|replenishm?ent|ord[e]?r)(?:[\w\s,/-]*?)\s+costs?t?{var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:costs?t?|expens[e]?)\s*(?:of|is)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:each\s+time|every\s+time|whenever)\s+(?:an?\s+)?ord[e]?r\s+is\s+placed",
            rf"(?:the\s+)?costs?t?\s+to\s+(?:place|process|place\s+and\s+process|issue|procure)\s+(?:an?|one|each|every|\d+)?\s*ord[e]?r{var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:the\s+)?costs?t?\s+of\s+(?:plac(?:ing)?|process(?:ing)?|plac(?:ing)?\s+and\s+process(?:ing)?)\s+(?:an?|one|each|every|\d+)?\s*ord[e]?r{var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:each\s+)?ord[e]?r\s+costs?t?{var_sym}\s*{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:to\s+(?:place|ord[e]?r))?",
            rf"costs?t?\s*{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:per\s+ord[e]?r|to\s+place(?:\s+an?\s+ord[e]?r|\s+with\s+(?:the\s+)?supplier)?|(?:each|every)\s+time\s+an?\s+ord[e]?r\s+is\s+placed)",
            rf"(?:ord[e]?r(?:ing)?\s*/\s*set[\s-]?up\s*costs?t?|set[\s-]?up\s*/\s*ord[e]?r(?:ing)?\s*costs?t?){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:costs?t?\s+of\s+plac(?:ing)?\s+(?:an?\s+)?ord[e]?r|costs?t?\s+to\s+place\s+(?:an?\s+)?ord[e]?r|costs?t?\s+per\s+ord[e]?r|ord[e]?r\s+costs?t?|ord[e]?r(?:ing)?\s+costs?t?|ord[e]?r\s+placement\s+costs?t?|ord[e]?r\s+processing\s+costs?t?|procurement\s+costs?t?\s+per\s+ord[e]?r){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:set[\s-]?up\s+costs?t?|set[\s-]?up\s+expens[e]?|ord[e]?r\s+expens[e]?|replenishm?ent\s+costs?t?|ord[e]?r\s+charge){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:ord[e]?r\s+costs?t?|ord[e]?r(?:ing)?\s+costs?t?){var_sym}\s*(?:would\s+be|of|is|are|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:per\s+ord[e]?r)?",
            rf"(?:with\s+(?:an?\s+)?)?(?:ord[e]?r(?:ing)?\s+costs?t?|ord[e]?r\s+costs?t?|set[\s-]?up\s+costs?t?|set[\s-]?up\s+expens[e]?|costs?t?\s+per\s+ord[e]?r){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:per\s+ord[e]?r|costs?t?\s+per\s+ord[e]?r|ord[e]?r(?:ing)?\s+costs?t?|ord[e]?r\s+costs?t?|set[\s-]?up\s+costs?t?|set[\s-]?up\s+expens[e]?|to\s+place\s+(?:an?\s+)?ord[e]?r)",
        ]
        for pat in order_cost_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["ordering_cost"] = float(m.group(1))
                break

        # -------------------------------------------------------------
        # 2. Demand Mean & Demand Standard Deviation (with Universal Timeframe Conversion)
        # Converts any given units (monthly, weekly, quarterly, bi-weekly, semi-annual, daily)
        # into a normalized annual demand rate for inventory calculations.
        # -------------------------------------------------------------
        # 2a. Demand standard deviation
        demand_std_patterns = [
            rf"(?:standard\s+deviation\s+of\s+(?:daily\s+|weekly\s+|monthly\s+|annual\s+)?demm?and|std\s+dev\s+of\s+(?:daily\s+|weekly\s+|monthly\s+|annual\s+)?demm?and|(?:daily\s+|weekly\s+|monthly\s+|annual\s+)?demm?and\s+standard\s+deviation|(?:daily\s+|weekly\s+|monthly\s+|annual\s+)?demm?and\s+std\s+dev|sigma[\s_]?d|\u03c3[\s_]?d){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            rf"(?:standard\s+deviation|std\s+dev){var_sym}\s*(?:of|is|equals)?\s*(\d+(?:\.\d+)?)\s*(?:(?:{unit_nouns})(?:\/day|\/week|\/month|\/year)?)?(?!\s*(?:of\s+lead[d]?[\s-]t[i|y]me|days?|weeks?))",
            rf"(\d+(?:\.\d+)?)\s*(?:{unit_nouns})(?:\/day|\/week|\/month|\/year)?\s+(?:standard\s+deviation|std\s+dev)",
            r"with\s+(?:a\s+)?std\s+dev\s+(?:of\s+)?(\d+(?:\.\d+)?)",
            r"with\s+(?:a\s+)?standard\s+deviation\s+(?:of\s+)?(\d+(?:\.\d+)?)",
            rf"std\s+dev{var_sym}\s*[:=]?\s*(\d+(?:\.\d+)?)",
        ]
        for pat in demand_std_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["demand_std"] = float(m.group(1))
                break

        demand_identified = False

        # 2b. Bi-Weekly Demand Patterns (Factor: 26.0) — Evaluated before weekly to avoid substring collisions
        biweekly_demand_patterns = [
            rf"(?:bi[\s-]weekly|biweekly|fortnightly)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{biweekly_timeframes}\b",
            rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{biweekly_timeframes}\b",
            rf"{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{biweekly_timeframes})",
        ]
        for pat in biweekly_demand_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                raw_d = float(m.group(1))
                ann_d = raw_d * 26.0
                inputs["biweekly_demand"] = raw_d
                inputs["raw_demand"] = raw_d
                inputs["demand_time_unit"] = "biweek"
                inputs["demand_conversion_factor"] = 26.0
                inputs["demand"] = ann_d
                inputs["annual_demand"] = ann_d
                if "demand_mean" not in inputs:
                    inputs["demand_mean"] = ann_d
                inputs["demand_conversion_note"] = f"Converted bi-weekly demand of {raw_d:,.1f} units to annual demand of {ann_d:,.1f} units ({raw_d:,.1f} x 26 periods/year)."
                demand_identified = True
                break

        # 2c. Semi-Annual Demand Patterns (Factor: 2.0) — Evaluated before annual to avoid substring collisions
        if not demand_identified:
            semiannual_demand_patterns = [
                rf"(?:semi[\s-]annual|semiannual|half[\s-]yearly)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{semiannual_timeframes}\b",
                rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{semiannual_timeframes}\b",
                rf"{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{semiannual_timeframes})",
            ]
            for pat in semiannual_demand_patterns:
                m = re.search(pat, text_clean, re.IGNORECASE)
                if m:
                    raw_d = float(m.group(1))
                    ann_d = raw_d * 2.0
                    inputs["semiannual_demand"] = raw_d
                    inputs["raw_demand"] = raw_d
                    inputs["demand_time_unit"] = "semiannual"
                    inputs["demand_conversion_factor"] = 2.0
                    inputs["demand"] = ann_d
                    inputs["annual_demand"] = ann_d
                    if "demand_mean" not in inputs:
                        inputs["demand_mean"] = ann_d
                    inputs["demand_conversion_note"] = f"Converted semi-annual demand of {raw_d:,.1f} units to annual demand of {ann_d:,.1f} units ({raw_d:,.1f} x 2 periods/year)."
                    demand_identified = True
                    break

        # 2d. Monthly Demand Patterns (Factor: 12.0)
        if not demand_identified:
            monthly_demand_patterns = [
                rf"(?:monthly|per[\s-]month|mothly|montly)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{monthly_timeframes}\b",
                rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{monthly_timeframes}\b",
                rf"{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{monthly_timeframes})",
                rf"(\d+(?:\.\d+)?)\s*{item_filler}(?:needed|required|used|demanded|consumed|ordered|sold)\s*{monthly_timeframes}",
                rf"sells?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{monthly_timeframes})",
                rf"(\d+(?:\.\d+)?)\s*{item_filler}(?:a\s+month|each\s+month|every\s+month|per\s+month|\/\s*month|\/\s*mo\b)",
            ]
            for pat in monthly_demand_patterns:
                m = re.search(pat, text_clean, re.IGNORECASE)
                if m:
                    raw_d = float(m.group(1))
                    ann_d = raw_d * 12.0
                    inputs["monthly_demand"] = raw_d
                    inputs["raw_demand"] = raw_d
                    inputs["demand_time_unit"] = "month"
                    inputs["demand_conversion_factor"] = 12.0
                    inputs["demand"] = ann_d
                    inputs["annual_demand"] = ann_d
                    if "demand_mean" not in inputs:
                        inputs["demand_mean"] = ann_d
                    inputs["demand_conversion_note"] = f"Converted monthly demand of {raw_d:,.1f} units to annual demand of {ann_d:,.1f} units ({raw_d:,.1f} x 12 months/year)."
                    demand_identified = True
                    break

        # 2e. Quarterly Demand Patterns (Factor: 4.0)
        if not demand_identified:
            quarterly_demand_patterns = [
                rf"(?:quarterly|per[\s-]quarter|quarterley)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{quarterly_timeframes}\b",
                rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{quarterly_timeframes}\b",
                rf"{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{quarterly_timeframes})",
                rf"(\d+(?:\.\d+)?)\s*{item_filler}(?:needed|required|used|demanded|consumed|ordered|sold)\s*{quarterly_timeframes}",
                rf"sells?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{quarterly_timeframes})",
            ]
            for pat in quarterly_demand_patterns:
                m = re.search(pat, text_clean, re.IGNORECASE)
                if m:
                    raw_d = float(m.group(1))
                    ann_d = raw_d * 4.0
                    inputs["quarterly_demand"] = raw_d
                    inputs["raw_demand"] = raw_d
                    inputs["demand_time_unit"] = "quarter"
                    inputs["demand_conversion_factor"] = 4.0
                    inputs["demand"] = ann_d
                    inputs["annual_demand"] = ann_d
                    if "demand_mean" not in inputs:
                        inputs["demand_mean"] = ann_d
                    inputs["demand_conversion_note"] = f"Converted quarterly demand of {raw_d:,.1f} units to annual demand of {ann_d:,.1f} units ({raw_d:,.1f} x 4 quarters/year)."
                    demand_identified = True
                    break

        # 2f. Weekly Demand Patterns (Factor: 52.0)
        if not demand_identified:
            weekly_demand_patterns = [
                rf"\b(?<!bi[\s-])(?<!bi)(?:weekly|per[\s-]week|wekly)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{weekly_timeframes}\b",
                rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{weekly_timeframes}\b",
                rf"{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{weekly_timeframes})",
                rf"(\d+(?:\.\d+)?)\s*{item_filler}(?:needed|required|used|demanded|consumed|ordered|sold)\s*{weekly_timeframes}",
                rf"sells?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{weekly_timeframes})",
                rf"(\d+(?:\.\d+)?)\s*{item_filler}(?:a\s+week|each\s+week|every\s+week|per\s+week|\/\s*week|\/\s*wk\b)",
            ]
            for pat in weekly_demand_patterns:
                m = re.search(pat, text_clean, re.IGNORECASE)
                if m:
                    raw_d = float(m.group(1))
                    ann_d = raw_d * 52.0
                    inputs["weekly_demand"] = raw_d
                    inputs["raw_demand"] = raw_d
                    inputs["demand_time_unit"] = "week"
                    inputs["demand_conversion_factor"] = 52.0
                    inputs["demand"] = ann_d
                    inputs["annual_demand"] = ann_d
                    if "demand_mean" not in inputs:
                        inputs["demand_mean"] = ann_d
                    inputs["demand_conversion_note"] = f"Converted weekly demand of {raw_d:,.1f} units to annual demand of {ann_d:,.1f} units ({raw_d:,.1f} x 52 weeks/year)."
                    demand_identified = True
                    break

        # 2g. Daily Demand Patterns (Factor: w_days) — Evaluated when explicitly stated
        if not demand_identified:
            daily_demand_patterns = [
                rf"(?:daily\s+demm?and\s+mean|mean\s+daily\s+demm?and|daily\s+mean\s+demm?and){var_sym}\s*(?:of|is|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"(?:average\s+daily\s+demm?and|daily\s+demm?and\s+averages?|average\s+daily\s+usage|daily\s+usage|mean\s+daily\s+usage){var_sym}\s*(?:of|is|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"(?:have|experiences?|has|expects?)\s+(?:an?\s+)?daily\s+{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"(?:daily|per[\s-]day|dialy|dayly)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{daily_timeframes}\b",
                rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{daily_timeframes}\b",
            ]
            for pat in daily_demand_patterns:
                m = re.search(pat, text_clean, re.IGNORECASE)
                if m:
                    d_val = float(m.group(1))
                    inputs["daily_demand"] = d_val
                    inputs["raw_demand"] = d_val
                    inputs["demand_time_unit"] = "day"
                    inputs["demand_conversion_factor"] = w_days
                    ann_d = d_val * w_days
                    inputs["demand"] = ann_d
                    inputs["annual_demand"] = ann_d
                    inputs["demand_conversion_note"] = f"Converted daily demand of {d_val:,.1f} units into an annual demand of {ann_d:,.1f} units ({d_val:,.1f} x {w_days:.0f} operating days/year)."
                    if "demand_mean" not in inputs:
                        inputs["demand_mean"] = d_val
                    demand_identified = True
                    break

        # 2h. Annual Demand Patterns (Factor: 1.0)
        if not demand_identified:
            annual_demand_patterns = [
                rf"\b(?<!semi[\s-])(?<!semi)(?<!bi[\s-])(?<!bi)(?:annual|anual|yearly|yeerly|per[\s-]year|per[\s-]annum)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"(?:have|experiences?|has|expects?)\s+(?:an?\s+)?(?:annual|anual|yearly|yeerly)\s*{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{annual_timeframes}\b",
                rf"(?<!\$)\b(\d+(?:\.\d+)?)\s*{item_filler}{annual_timeframes}\b",
                rf"{need_nouns}{var_sym}\s*(?:of|is|are|equals|averages)?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{annual_timeframes})",
                rf"(\d+(?:\.\d+)?)\s*{item_filler}(?:needed|required|used|demanded|consumed|ordered|sold)\s*{annual_timeframes}",
                rf"sells?\s*(\d+(?:\.\d+)?)\s*{item_filler}(?:{annual_timeframes})",
            ]
            for pat in annual_demand_patterns:
                m = re.search(pat, text_clean, re.IGNORECASE)
                if m:
                    d_val = float(m.group(1))
                    inputs["demand"] = d_val
                    inputs["annual_demand"] = d_val
                    inputs["raw_demand"] = d_val
                    inputs["demand_time_unit"] = "year"
                    inputs["demand_conversion_factor"] = 1.0
                    if "demand_mean" not in inputs:
                        inputs["demand_mean"] = d_val
                    demand_identified = True
                    break

        # 2i. General / Unspecified Mean Demand (e.g. Newsvendor or verb + noun without timeframe)
        if "demand" not in inputs:
            general_mean_patterns = [
                rf"{need_verbs}\s*(?:an?\s+)?(?:average\s+|mean\s+|estimated\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*{unit_nouns}\b(?!\s*%)",
                rf"(?:demm?and{var_sym}\s+is\s+normal\s+with\s+mean|with\s+(?:a\s+)?mean\s+demm?and{var_sym}\s+of|(?:with\s+)?mean\s*(?:demm?and{var_sym})?\s*(?:of|is|equals)?)\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"(?:average\s+demm?and{var_sym}|mean\s+demm?and{var_sym}|expected\s+demm?and{var_sym})\s*(?:of|is|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                rf"\bdemm?and{var_sym}\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:{unit_nouns})?(?!\s*(?:standard\s+deviation|std\s+dev))",
            ]
            for pat in general_mean_patterns:
                m = re.search(pat, text_clean, re.IGNORECASE)
                if m:
                    d_val = float(m.group(1))
                    inputs["demand"] = d_val
                    inputs["annual_demand"] = d_val
                    inputs["raw_demand"] = d_val
                    inputs["demand_time_unit"] = "unspecified"
                    inputs["demand_conversion_factor"] = 1.0
                    if "demand_mean" not in inputs:
                        inputs["demand_mean"] = d_val
                    break

        # -------------------------------------------------------------
        # 3. Average Lead Time & Lead-Time Standard Deviation
        # Distinguish:
        # - average lead time (lead time, mean lead time, delivery time, etc.)
        # - lead-time standard deviation (lead-time std dev, sigma_L, etc.)
        # -------------------------------------------------------------
        # 3a. Lead-time standard deviation first
        lt_std_patterns = [
            rf"(?:standard\s+deviation\s+of\s+lead[d]?[\s-]t[i|y]me|std\s+dev\s+of\s+lead[d]?[\s-]t[i|y]me|lead[d]?[\s-]t[i|y]me\s+standard\s+deviation|lead[d]?[\s-]t[i|y]me\s+std\s+dev|lead[d]?[\s-]t[i|y]me\s+variability|sigma[\s_]?l|\u03c3[\s_]?l){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            rf"lead[d]?[\s-]t[i|y]me\s+(?:varies|fluctuates)\s+with\s+(?:a\s+)?standard\s+deviation\s+(?:of|is)?\s*(\d+(?:\.\d+)?)",
            rf"(?:with\s+(?:a\s+)?)lead[d]?[\s-]t[i|y]me\s+(?:standard\s+deviation|std\s+dev){var_sym}\s*(?:of|is)?\s*(\d+(?:\.\d+)?)",
        ]
        for pat in lt_std_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["lead_time_std"] = float(m.group(1))
                break

        # 3b. Average lead time
        lead_time_patterns = [
            rf"(?:deliv[e]?ry\s+lead[d]?[\s-]t[i|y]me|replenishm?ent\s+lead[d]?[\s-]t[i|y]me|lead[d]?[\s-]t[i|y]me|deliv[e]?ry\s+t[i|y]me){var_sym}\s*averages?\s*(\d+(?:\.\d+)?)\s*(?:business\s+|working\s+)?(days?|weeks?|months?)?",
            rf"(?:average\s+lead[d]?[\s-]t[i|y]me|mean\s+lead[d]?[\s-]t[i|y]me|expected\s+lead[d]?[\s-]t[i|y]me|deliv[e]?ry\s+lead[d]?[\s-]t[i|y]me|replenishm?ent\s+lead[d]?[\s-]t[i|y]me|lead[d]?[\s-]t[i|y]me){var_sym}\s*(?:of|is|are|equals|averages)?\s*(?:constant\s+at|fixed\s+at|estimated\s+at|approximately)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:business\s+|working\s+)?(days?|weeks?|months?)?\b(?!\s*(?:standard\s+deviation|std\s+dev))",
            rf"supplier\s+(?:has\s+a\s+lead\s+time\s+of|takes|replenishment\s+lead[\s-]time\s+averages?){var_sym}\s*(\d+(?:\.\d+)?)\s*(?:business\s+|working\s+)?(days?|weeks?|months?)?",
            rf"(?:lead[d]?\s*t[i|y]me|deliv[e]?ry\s+t[i|y]me){var_sym}\s*(?:of|is|equals)?\s*(?:constant\s+at|fixed\s+at|estimated\s+at|approximately)?\s*(\d+(?:\.\d+)?)\s*(?:business\s+|working\s+)?(days?|weeks?|months?)?",
        ]
        for pat in lead_time_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                lt_val = float(m.group(1))
                unit = m.group(2).lower() if m.group(2) else "days"
                inputs["lead_time"] = lt_val
                inputs["lead_time_days"] = lt_val if "week" not in unit else lt_val * 7.0
                inputs["lead_time_unit"] = unit
                break

        # -------------------------------------------------------------
        # 4. Service Level & z-Score / z-Value
        # Cycle Service Level (CSL) / in-stock probability / z-statistic
        # -------------------------------------------------------------
        sl_patterns = [
            rf"(?:target\s+cycle\s+serv[i|e]?[s|c]e\s+lev[e|a]l|desired\s+cycle\s+serv[i|e]?[s|c]e\s+lev[e|a]l|cycle\s+serv[i|e]?[s|c]e\s+lev[e|a]l|target\s+serv[i|e]?[s|c]e\s+lev[e|a]l|desired\s+serv[i|e]?[s|c]e\s+lev[e|a]l|serv[i|e]?[s|c]e\s+lev[e|a]l|csl|in[\s-]stock\s+probability|service\s+factor){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(%)?",
            rf"(\d+(?:\.\d+)?)\s*%\s*(?:target\s+cycle\s+serv[i|e]?[s|c]e\s+lev[e|a]l|cycle\s+serv[i|e]?[s|c]e\s+lev[e|a]l|serv[i|e]?[s|c]e\s+lev[e|a]l|csl|in[\s-]stock\s+probability)",
        ]
        for pat in sl_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                inputs["service_level"] = val / 100.0 if val >= 1.0 else val
                break

        # z-score / z-value
        z_patterns = [
            rf"(?:corresponding\s+)?(?:z[\s-]value|z[\s-]score|z\s+statistic){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            rf"\bz\s*[:=]\s*(\d+(?:\.\d+)?)",
        ]
        for pat in z_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                z_val = float(m.group(1))
                inputs["z_score"] = z_val
                inputs["z_value"] = z_val
                break

        # -------------------------------------------------------------
        # 5. Holding Cost: Fixed vs Percentage Holding Cost
        # Distinguish:
        # - fixed holding cost ($/unit/year, Rs./unit/year, €/unit/year, etc.)
        # - percentage holding cost (% of price)
        # -------------------------------------------------------------
        # Percentage holding cost
        holding_pct_patterns = [
            rf"(?:percent[a|e]?ge\s+hold[i]?ng\s+costs?t?|annual\s+hold[i]?ng\s+costs?t?\s+rate|anual\s+hold[i]?ng\s+costs?t?\s+rate|hold[i]?ng\s+costs?t?\s+rate|car?ry?ing\s+costs?t?\s+rate|car?ry?ing\s+rate|hold[i]?ng\s+rate|annual\s+hold[i]?ng\s+costs?t?|anual\s+hold[i]?ng\s+costs?t?|annual\s+car?ry?ing\s+costs?t?|anual\s+car?ry?ing\s+costs?t?|car?ry?ing\s+chag?re|car?ry?ing\s+charge|annual\s+car?ry?ing\s+charge|anual\s+car?ry?ing\s+charge|car?ry?ing\s+costs?t?|hold[i]?ng\s+costs?t?){var_sym}\s*(?:would\s+be|of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:of\s+(?:unit\s+)?(?:costs?t?|price))?\s*{pct_word}",
            rf"(?:hold[i]?ng\s+costs?t?|car?ry?ing\s+costs?t?){var_sym}\s*(?:rate\s+)?(?:would\s+be|of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*{pct_word}",
            rf"(\d+(?:\.\d+)?)\s*{pct_word}\s*(?:of\s+(?:unit\s+)?(?:costs?t?|price))?\s*(?:annual\s+hold[i]?ng\s+costs?t?|anual\s+hold[i]?ng\s+costs?t?|annual\s+hold[i]?ng|car?ry?ing\s+costs?t?|hold[i]?ng\s+costs?t?|percent[a|e]?ge\s+hold[i]?ng\s+costs?t?|costs?t?\s+of\s+capital|car?ry?ing\s+chag?re|car?ry?ing\s+charge)",
            rf"(\d+(?:\.\d+)?)\s*{pct_word}\s*(?:of\s+(?:unit\s+)?(?:costs?t?|price))",
            rf"hold[i]?ng\s+costs?t?\s+(?:rate\s+)?(?:is|equals)?\s*(\d+(?:\.\d+)?)\s*{pct_word}\s*(?:of\s+(?:unit\s+)?(?:costs?t?|price))?",
        ]
        for pat in holding_pct_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                pct = float(m.group(1))
                inputs["holding_cost_rate"] = pct / 100.0 if pct > 1.0 else pct
                inputs["percentage_holding_cost"] = inputs["holding_cost_rate"]
                break

        # Fixed holding cost
        holding_dollar_patterns = [
            rf"fixed\s+hold[i]?ng\s+costs?t?{var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)\b(?!\s*{pct_word})",
            rf"(?<!percentage\s)(?<!rate\s)(?:hold[i]?ng\s+costs?t?|car?ry?ing\s+costs?t?|storage\s+costs?t?|costs?t?\s+to\s+hold|costs?t?\s+to\s+store|car?ry?ing\s+chag?re|car?ry?ing\s+charge){var_sym}\s*(?:of|is|are|equals|would\s+be)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)\b(?!\s*{pct_word})",
            rf"{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:per\s+(?:unit|{unit_nouns})\s+per\s+year|per\s+(?:unit|{unit_nouns})\s+annually|per\s+(?:unit|{unit_nouns})\s+per\s+annum|per\s+(?:unit|{unit_nouns})\s+to\s+hold|per\s+(?:unit|{unit_nouns})\s+to\s+store|fixed\s+hold[i]?ng\s+costs?t?)",
            rf"costs?t?\s*{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:per\s+(?:unit|{unit_nouns})\s+per\s+year|per\s+(?:unit|{unit_nouns})\s+annually|per\s+(?:unit|{unit_nouns})\s+per\s+annum|per\s+(?:unit|{unit_nouns})\s+to\s+(?:hold|store|carry))",
        ]
        for pat in holding_dollar_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["holding_cost"] = float(m.group(1))
                inputs["fixed_holding_cost"] = float(m.group(1))
                break

        # -------------------------------------------------------------
        # 6. Operational Parameters (Working days per year)
        # -------------------------------------------------------------
        work_days_patterns = [
            r"(?:operate|operating|open|works?|working)\s*(\d+(?:\.\d+)?)\s*(?:days?\s+(?:a|per)\s+year|days?\s*\/\s*year|days?\s+annually)",
            r"(\d+(?:\.\d+)?)\s*(?:working|operating|business)?\s*days\s+(?:a|per)\s+year",
            r"(\d+(?:\.\d+)?)\s*days\s+of\s+operation\s+per\s+year",
        ]
        for pat in work_days_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["working_days_per_year"] = float(m.group(1))
                break

        # -------------------------------------------------------------
        # 7. Pricing & Costs (Newsvendor & Quantity Discount)
        # Distinguish:
        # - selling price
        # - unit purchase price / purchase cost
        # - salvage value
        # - underage / shortage cost
        # - overage / excess cost
        # -------------------------------------------------------------
        # Underage / Shortage Cost
        underage_patterns = [
            rf"(?:und[e]?rage\s+costs?t?|costs?t?\s+of\s+und[e]?rage|shortage\s+costs?t?|costs?t?\s+of\s+shortage|lost\s+profit\s+per\s+(?:unit|{unit_nouns})){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:shortage\s+penalty|penalty\s+costs?t?|goodwill\s+penalty|goodwill\s+costs?t?){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
        ]
        for pat in underage_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["cost_of_underage"] = float(m.group(1))
                inputs["penalty_cost"] = float(m.group(1))
                break

        # Overage / Excess Cost
        overage_patterns = [
            rf"(?:ov[e]?rage\s+costs?t?|costs?t?\s+of\s+ov[e]?rage|excess\s+costs?t?|costs?t?\s+of\s+excess|disposal\s+costs?t?|loss\s+per\s+unsold\s+(?:unit|{unit_nouns})){var_sym}\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
        ]
        for pat in overage_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["cost_of_overage"] = float(m.group(1))
                break

        # Unit purchase price / purchase cost
        unit_cost_patterns = [
            rf"(?:unit\s+purchase\s+price|purchase\s+costs?t?|unit\s+costs?t?|item\s+costs?t?|costs?t?\s+per\s+(?:unit|{unit_nouns})|acquisition\s+costs?t?|manufacturing\s+costs?t?|costs?t?\s+to\s+produce|costs?t?\s+to\s+manufacture|wholesale\s+costs?t?|wholesale\s+price|purchase\s+price|baseline\s+unit\s+purchase\s+price|baseline\s+price|base\s+price|regular\s+price|standard\s+price){var_sym}\s*(?:would\s+be|of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:buys?|purchas[e]?s?|purchased|bought)\s+(?:[\w\s]{{0,35}}?\s+)?(?:for|at)\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:each|per)\s+(?:{unit_nouns}|[a-zA-Z-]{{2,25}})\s+costs?t?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?<!order )(?<!orders )(?<!ordering )(?<!ordr )(?<!ordrs )(?<!ordring )\bcosts?t?\s*{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:each|per\s+(?:unit|{unit_nouns})|to\s+manufacture|to\s+produce|to\s+make|to\s+buy|to\s+purchase)?(?!\s*(?:per\s+ord[e]?r|to\s+place))",
            rf"(?:purchas[e]?d?|bought)\s+(?:for|at)\s*{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:each|per\s+(?:unit|{unit_nouns}))?",
            rf"{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:to\s+purchase|unit\s+costs?t?|purchase\s+costs?t?|costs?t?\s+per\s+(?:unit|{unit_nouns})|unit\s+purchase\s+price|per\s+(?:unit|{unit_nouns})\s+purchase|wholesale)",
        ]
        for pat in unit_cost_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["unit_cost"] = float(m.group(1))
                inputs["unit_purchase_price"] = float(m.group(1))
                break

        # Selling price
        price_patterns = [
            rf"\b(?:selling\s+price|retail\s+price|sale\s+price|retail\s+value|customer\s+price|revenue\s+per\s+(?:unit|{unit_nouns}))\b\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:sells?|selling)\s+(?:[\w\s]{{0,30}}?\s+)?(?:for|at)\s+{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:each\s+retail|retail|selling\s+price)",
            rf"(?<!wholesale )(?<!purchase )(?<!baseline )(?<!regular )(?<!standard )\bprice\s*(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
        ]
        for pat in price_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["selling_price"] = float(m.group(1))
                break

        # Salvage value
        salvage_patterns = [
            rf"liquidated\s+(?:to\s+[a-zA-Z\s]{{1,30}}?\s+)?(?:for|at)\s+{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"(?:salv[a|i]?ge\s+val[u]?e|salv[a|i]?ge|liquidation\s+value|mark[\s-]?down\s+price|outlet\s+price|scrap\s+value|residual\s+value)\s*(?:[\w\s]{{0,35}}?\s+)?(?:of|is|equals)?\s*[:=]?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"salv[a|i]?ged?\s*(?:for|is|equals|value|at)?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"liquidated?\s*(?:for|at)?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            rf"{curr_sym}\s*(\d+(?:\.\d+)?)\s*(?:salv[a|i]?ge|salv[a|i]?ge\s+val[u]?e|liquidation|scrap\s+value)",
        ]
        for pat in salvage_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["salvage_value"] = float(m.group(1))
                break

        # Demand Distribution for Newsvendor
        if re.search(r"\buniform\b", text_clean, re.IGNORECASE):
            inputs["distribution_type"] = "uniform"
            unif_m = re.search(r"between\s*(\d+(?:\.\d+)?)\s*and\s*(\d+(?:\.\d+)?)", text_clean, re.IGNORECASE)
            if unif_m:
                inputs["min_demand"] = float(unif_m.group(1))
                inputs["max_demand"] = float(unif_m.group(2))
        elif re.search(r"\bdiscrete\b", text_clean, re.IGNORECASE):
            inputs["distribution_type"] = "discrete"
        elif re.search(r"\bnormal\b", text_clean, re.IGNORECASE):
            inputs["distribution_type"] = "normal"

        # -------------------------------------------------------------
        # 8. Fixed-Period Review Inventory State
        # Distinguish:
        # - demand
        # - review period
        # - lead time
        # - safety stock
        # - inventory position
        # - on-hand inventory
        # - on-order inventory
        # - backorders
        # -------------------------------------------------------------
        # Review Period
        review_patterns = [
            rf"(?:re[v]?e?i[e]?w\s+stock\s+.*?every|re[v]?e?i[e]?wed\s+every|re[v]?e?i[e]?w\s+schedule\s+every|ord[e]?r\s+every|re[v]?e?i[e]?w\s+every)\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)",
            rf"every\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)\s*(?:re[v]?e?i[e]?w|per[i]?od[i]?c\s+re[v]?e?i[e]?w)",
            rf"(?:per[i]?od[i]?c\s+re[v]?e?i[e]?w\s+per[i]?od|re[v]?e?i[e]?w\s+per[i]?od|re[v]?e?i[e]?w\s+int[e]?rval|per[i]?od[i]?c\s+re[v]?e?i[e]?w|re[v]?e?i[e]?w\s+cycle|re[v]?e?i[e]?w\s+frequency)\s*(?:of|is|are|equals|averages)?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)?",
            rf"ord[e]?rs?\s+(?:are\s+)?placed\s+every\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)",
            rf"re[v]?e?i[e]?ws?\s+inve?nt[o|e]ry\s+every\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)",
        ]
        for pat in review_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["review_period"] = float(m.group(1))
                if m.group(2):
                    inputs["time_unit"] = m.group(2).lower()
                break

        # Safety stock buffer directly specified
        ss_patterns = [
            rf"(?:saf[e|t]ty\s+stock\s+buffer|saf[e|t]ty\s+stock|safty\s+stock|saf[e|t]ty\s+cushion|safety\s+cushion|saf[e|t]ty\s+buffer|safety\s+buffer)\s*(?:of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            rf"(\d+(?:\.\d+)?)\s*(?:units|{unit_nouns})?\s*(?:of\s+)?(?:saf[e|t]ty\s+stock|safty\s+stock|safety\s+buffer|safety\s+cushion)",
        ]
        for pat in ss_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["safety_stock"] = float(m.group(1))
                break

        # Inventory Position
        ip_patterns = [
            rf"(?:current\s+inve?nt[o|e]ry\s+position|inve?nt[o|e]ry\s+position|effective\s+inve?nt[o|e]ry)\s*(?:of|is|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
        ]
        for pat in ip_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["inventory_position"] = float(m.group(1))
                break

        # On-hand
        on_hand_m = re.search(
            rf"(?:on[\s-]hand\s+inve?nt[o|e]ry|on[\s-]hand|physical\s+inve?nt[o|e]ry|current\s+inve?nt[o|e]ry|currently\s+in\s+stock|warehouse\s+has|existing\s+on[\s-]hand\s+stock)\s*(?:of|is|are|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            text_clean,
            re.IGNORECASE,
        )
        if on_hand_m:
            inputs["inventory_on_hand"] = float(on_hand_m.group(1))

        # On-order
        on_order_patterns = [
            rf"(?:on[\s-]ord[e]?r\s+inve?nt[o|e]ry|on[\s-]ord[e]?r|pipeline\s+inve?nt[o|e]ry|in[\s-]transit\s+inve?nt[o|e]ry|in[\s-]transit|open\s+ord[e]?rs?)\s*(?:of|is|are|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
            rf"(\d+(?:\.\d+)?)\s*(?:{unit_nouns})?\s*(?:on[\s-]ord[e]?r|in[\s-]transit|pipeline)",
        ]
        for pat in on_order_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["inventory_on_order"] = float(m.group(1))
                break

        # Backorders
        backorder_patterns = [
            rf"(\d+(?:\.\d+)?)\s*(?:backorders?|backordered\s+(?:units?|{unit_nouns})|units\s+on\s+backorder)",
            rf"(?:backorders?|backordered\s+(?:units?|{unit_nouns})|unfulfilled\s+ord[e]?rs?|backlog)\s*(?:of|is|are|equals)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
        ]
        for pat in backorder_patterns:
            m = re.search(pat, text_clean, re.IGNORECASE)
            if m:
                inputs["backorders"] = float(m.group(1))
                break

        # -------------------------------------------------------------
        # 9. Quantity Discount Tiers (Quantity breaks & unit prices)
        # -------------------------------------------------------------
        tiers = []
        # Pattern 1: Bounded ranges, e.g. "0 to 499 units at $25", "500 - 999: $22", "0 to 4999 kg at € 5.00"
        bounded_matches = re.findall(
            rf"(\d+)\s*(?:to|-)\s*(\d+)\s*(?:{unit_nouns})?\s*(?:at|@|costs?t?|for|:)?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            text_clean,
            re.IGNORECASE,
        )
        for bm in bounded_matches:
            tiers.append({"min_qty": float(bm[0]), "max_qty": float(bm[1]), "unit_price": float(bm[2])})

        # Pattern 2: Open-ended top tier, e.g. "1000+ units at $20", "1000 or more units at $20", "over 1000 units at $20"
        open_matches = re.findall(
            rf"(?:over\s+)?(\d+)\s*(?:\+|or\s+more|and\s+above)\s*(?:{unit_nouns})?\s*(?:at|@|costs?t?|for|:)?\s*{curr_sym}\s*(\d+(?:\.\d+)?)",
            text_clean,
            re.IGNORECASE,
        )
        for om in open_matches:
            min_q = float(om[0])
            price = float(om[1])
            if not any(t["min_qty"] == min_q for t in tiers):
                tiers.append({"min_qty": min_q, "max_qty": None, "unit_price": price})

        if tiers:
            inputs["tiers"] = sorted(tiers, key=lambda t: t["min_qty"])
        else:
            # Check for alternative price break patterns: "under 500: $20, 500 or more: $18"
            under_m = re.search(rf"(?:under|below|less\s+than)\s*(\d+)\s*(?:{unit_nouns})?\s*[:=]?\s*(?:at|is)?\s*{curr_sym}\s*(\d+(?:\.\d+)?)", text_clean, re.IGNORECASE)
            over_m = re.search(rf"(\d+)\s*(?:or\s+more|\+)\s*(?:{unit_nouns})?\s*[:=]?\s*(?:at|is)?\s*{curr_sym}\s*(\d+(?:\.\d+)?)", text_clean, re.IGNORECASE)
            if under_m and over_m:
                break_q = float(under_m.group(1))
                price1 = float(under_m.group(2))
                price2 = float(over_m.group(2))
                inputs["tiers"] = [
                    {"min_qty": 0.0, "max_qty": break_q - 1.0, "unit_price": price1},
                    {"min_qty": break_q, "max_qty": None, "unit_price": price2},
                ]

        return inputs


# ---------------------------------------------------------------------------
# Core Supply Chain Inventory Decision Agent
# ---------------------------------------------------------------------------

class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.

    Executes the 7-step architecture:
    1. Customer Question
    2. Identify Models
    3. Extract Inputs
    4. Select Python Tool
    5. Calculate
    6. Verify
    7. Explain
    """

    def __init__(self):
        self.parser = InventoryNLPParser()

    def process_query(
        self,
        customer_question: str,
        manual_model_override: Optional[str] = None,
        manual_input_overrides: Optional[Dict[str, Any]] = None,
    ) -> AgentPipelineResponse:
        """
        Execute full 7-step pipeline on customer's natural-language inventory query.
        """
        # -------------------------------------------------------------
        # Step 1: Customer Question
        # -------------------------------------------------------------
        cleaned_question = customer_question.strip()

        # -------------------------------------------------------------
        # Step 2: Identify Models
        # -------------------------------------------------------------
        if manual_model_override:
            model_key = manual_model_override
            model_name_map = {
                "eoq": "Economic Order Quantity (EOQ)",
                "quantity_discount": "Quantity Discount Model",
                "safety_stock_rop": "Safety Stock and Reorder Point (ROP)",
                "fixed_period_review": "Fixed-Period Review Model",
                "newsvendor": "Newsvendor Model",
            }
            model_name = model_name_map.get(model_key, "Economic Order Quantity (EOQ)")
            confidence = 1.0
            rationale = f"User explicitly selected model: {model_name}."
        else:
            model_name, model_key, confidence, rationale = self.parser.identify_model(cleaned_question)

        # -------------------------------------------------------------
        # Step 3: Extract Inputs
        # -------------------------------------------------------------
        extracted_inputs = self.parser.extract_numbers_with_context(cleaned_question)
        if manual_input_overrides:
            extracted_inputs.update(manual_input_overrides)

        input_notes: List[str] = []
        if "demand_conversion_note" in extracted_inputs:
            input_notes.append(extracted_inputs["demand_conversion_note"])
        for k, v in extracted_inputs.items():
            if k != "demand_conversion_note":
                input_notes.append(f"Extracted {k} = {v}")

        # -------------------------------------------------------------
        # Step 4: Select Python Tool
        # -------------------------------------------------------------
        tool_dispatch = {
            "eoq": ("calculate_eoq", self._run_eoq),
            "quantity_discount": ("calculate_quantity_discount", self._run_quantity_discount),
            "safety_stock_rop": ("calculate_rop", self._run_safety_stock_rop),
            "fixed_period_review": ("calculate_periodic_review", self._run_fixed_period_review),
            "newsvendor": ("calculate_newsvendor", self._run_newsvendor),
        }

        tool_name, tool_runner = tool_dispatch.get(model_key, ("calculate_eoq", self._run_eoq))

        # -------------------------------------------------------------
        # Step 5: Calculate (Deterministic Python Execution)
        # -------------------------------------------------------------
        try:
            raw_result = tool_runner(extracted_inputs)
        except Exception as e:
            # If calculation failed due to genuinely missing inputs
            verification_fail = VerificationResult(
                is_valid=False,
                status="FAILED",
                errors=[f"Validation/Calculation error in {tool_name}: {str(e)}"]
            )
            explanation_fail = ExplanationResult(
                executive_summary="The inventory agent could not complete calculation due to missing or invalid required parameters.",
                mathematical_formula="N/A",
                intermediate_steps=[f"Error encountered: {str(e)}"],
                managerial_insights=["Please ensure all required input values for this model are provided in the question. Missing inputs cannot be invented."]
            )
            return AgentPipelineResponse(
                customer_question=cleaned_question,
                identified_model=model_name,
                model_key=model_key,
                model_confidence=confidence,
                identification_rationale=rationale,
                extracted_inputs=extracted_inputs,
                input_extraction_notes=input_notes,
                selected_tool_name=tool_name,
                raw_calculation_result={"error": str(e)},
                verification=verification_fail,
                explanation=explanation_fail,
            )

        # -------------------------------------------------------------
        # Step 6: Verify
        # -------------------------------------------------------------
        verification = self._verify_calculation(model_key, extracted_inputs, raw_result)

        # -------------------------------------------------------------
        # Step 7: Explain
        # -------------------------------------------------------------
        explanation = self._generate_explanation(model_key, extracted_inputs, raw_result, verification)

        return AgentPipelineResponse(
            customer_question=cleaned_question,
            identified_model=model_name,
            model_key=model_key,
            model_confidence=confidence,
            identification_rationale=rationale,
            extracted_inputs=extracted_inputs,
            input_extraction_notes=input_notes,
            selected_tool_name=tool_name,
            raw_calculation_result=raw_result,
            verification=verification,
            explanation=explanation,
        )

    # -----------------------------------------------------------------------
    # Tool Execution Helpers (Never invent missing inputs)
    # -----------------------------------------------------------------------

    def _run_eoq(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        demand = inputs.get("demand") or inputs.get("annual_demand")
        w_days = inputs.get("working_days_per_year", 365.0)
        if demand is None:
            if inputs.get("monthly_demand") is not None:
                demand = inputs["monthly_demand"] * 12.0
            elif inputs.get("weekly_demand") is not None:
                demand = inputs["weekly_demand"] * 52.0
            elif inputs.get("quarterly_demand") is not None:
                demand = inputs["quarterly_demand"] * 4.0
            elif inputs.get("biweekly_demand") is not None:
                demand = inputs["biweekly_demand"] * 26.0
            elif inputs.get("semiannual_demand") is not None:
                demand = inputs["semiannual_demand"] * 2.0
            elif inputs.get("daily_demand") is not None:
                demand = inputs["daily_demand"] * w_days

        ordering_cost = inputs.get("ordering_cost") or inputs.get("setup_cost")
        holding_cost = inputs.get("holding_cost")
        holding_rate = inputs.get("holding_cost_rate")
        unit_cost = inputs.get("unit_cost") or inputs.get("unit_purchase_price")
        lead_time = inputs.get("lead_time_days") or inputs.get("lead_time")

        res = calculate_eoq(
            demand=demand,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            holding_cost_rate=holding_rate,
            unit_cost=unit_cost,
            lead_time_days=lead_time,
            working_days_per_year=w_days,
        )
        return res.to_dict()

    def _run_quantity_discount(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        demand = inputs.get("demand") or inputs.get("annual_demand")
        w_days = inputs.get("working_days_per_year", 365.0)
        if demand is None:
            if inputs.get("monthly_demand") is not None:
                demand = inputs["monthly_demand"] * 12.0
            elif inputs.get("weekly_demand") is not None:
                demand = inputs["weekly_demand"] * 52.0
            elif inputs.get("quarterly_demand") is not None:
                demand = inputs["quarterly_demand"] * 4.0
            elif inputs.get("biweekly_demand") is not None:
                demand = inputs["biweekly_demand"] * 26.0
            elif inputs.get("semiannual_demand") is not None:
                demand = inputs["semiannual_demand"] * 2.0
            elif inputs.get("daily_demand") is not None:
                demand = inputs["daily_demand"] * w_days

        ordering_cost = inputs.get("ordering_cost") or inputs.get("setup_cost")
        tiers = inputs.get("tiers")
        holding_rate = inputs.get("holding_cost_rate") or inputs.get("percentage_holding_cost")
        fixed_holding = inputs.get("fixed_holding_cost") or inputs.get("holding_cost")

        res = calculate_quantity_discount(
            demand=demand,
            ordering_cost=ordering_cost,
            tiers=tiers,
            holding_cost_rate=holding_rate,
            fixed_holding_cost=fixed_holding,
        )
        return res.to_dict()

    def _run_safety_stock_rop(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        lead_time = inputs.get("lead_time") or inputs.get("lead_time_days")
        lead_time_std = inputs.get("lead_time_std", 0.0)
        demand_mean = inputs.get("daily_demand") or inputs.get("demand_mean")
        w_days = inputs.get("working_days_per_year", 365.0)

        if demand_mean is None:
            if inputs.get("daily_demand") is not None:
                demand_mean = inputs["daily_demand"]
            elif inputs.get("annual_demand") is not None:
                demand_mean = inputs["annual_demand"] / w_days
            elif inputs.get("demand") is not None:
                demand_mean = inputs["demand"] / w_days
            elif inputs.get("monthly_demand") is not None:
                demand_mean = (inputs["monthly_demand"] * 12.0) / w_days
            elif inputs.get("weekly_demand") is not None:
                demand_mean = (inputs["weekly_demand"] * 52.0) / w_days
            elif inputs.get("quarterly_demand") is not None:
                demand_mean = (inputs["quarterly_demand"] * 4.0) / w_days
        elif demand_mean is not None and "daily_demand" not in inputs and inputs.get("annual_demand") == demand_mean and lead_time is not None:
            demand_mean = demand_mean / w_days

        demand_std = inputs.get("demand_std", 0.0)
        service_level = inputs.get("service_level")
        order_qty = inputs.get("order_quantity") or inputs.get("eoq")
        z_score = inputs.get("z_score") or inputs.get("z_value")

        res = calculate_rop(
            lead_time=lead_time,
            demand_mean=demand_mean,
            demand_std=demand_std,
            lead_time_std=lead_time_std,
            service_level=service_level,
            order_quantity=order_qty,
            z_score=z_score,
        )
        return res.to_dict()

    def _run_fixed_period_review(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        review_period = inputs.get("review_period")
        lead_time = inputs.get("lead_time") or inputs.get("lead_time_days")
        demand_mean = inputs.get("daily_demand") or inputs.get("demand_mean")
        w_days = inputs.get("working_days_per_year", 365.0)

        if demand_mean is None:
            if inputs.get("daily_demand") is not None:
                demand_mean = inputs["daily_demand"]
            elif inputs.get("annual_demand") is not None:
                demand_mean = inputs["annual_demand"] / w_days
            elif inputs.get("demand") is not None:
                demand_mean = inputs["demand"] / w_days
            elif inputs.get("monthly_demand") is not None:
                demand_mean = (inputs["monthly_demand"] * 12.0) / w_days
            elif inputs.get("weekly_demand") is not None:
                demand_mean = (inputs["weekly_demand"] * 52.0) / w_days
            elif inputs.get("quarterly_demand") is not None:
                demand_mean = (inputs["quarterly_demand"] * 4.0) / w_days
        elif demand_mean is not None and "daily_demand" not in inputs and inputs.get("annual_demand") == demand_mean and lead_time is not None:
            demand_mean = demand_mean / w_days

        demand_std = inputs.get("demand_std", 0.0)
        service_level = inputs.get("service_level")
        on_hand = inputs.get("inventory_on_hand")
        on_order = inputs.get("inventory_on_order", 0.0)
        backorders = inputs.get("backorders", 0.0)
        inv_pos = inputs.get("inventory_position")
        time_unit = inputs.get("time_unit", "days")
        z_score = inputs.get("z_score") or inputs.get("z_value")

        res = calculate_periodic_review(
            review_period=review_period,
            lead_time=lead_time,
            demand_mean=demand_mean,
            demand_std=demand_std,
            service_level=service_level,
            inventory_position=inv_pos,
            inventory_on_hand=on_hand,
            inventory_on_order=on_order,
            backorders=backorders,
            time_unit=time_unit,
            z_score=z_score,
        )
        return res.to_dict()

    def _run_newsvendor(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        unit_cost = inputs.get("unit_cost") or inputs.get("unit_purchase_price")
        selling_price = inputs.get("selling_price")
        salvage = inputs.get("salvage_value", 0.0)
        penalty = inputs.get("penalty_cost", 0.0)
        demand_mean = inputs.get("demand_mean") or inputs.get("demand")
        demand_std = inputs.get("demand_std")
        dist_type = inputs.get("distribution_type", "normal")

        # If underage/overage cost given directly without selling/purchase prices
        if (unit_cost is None or selling_price is None) and "cost_of_underage" in inputs and "cost_of_overage" in inputs:
            cu = inputs["cost_of_underage"]
            co = inputs["cost_of_overage"]
            unit_cost = co
            selling_price = co + cu
            salvage = 0.0

        res = calculate_newsvendor(
            unit_cost=unit_cost,
            selling_price=selling_price,
            salvage_value=salvage,
            penalty_cost=penalty,
            demand_mean=demand_mean,
            demand_std=demand_std,
            distribution_type=dist_type,
            min_demand=inputs.get("min_demand"),
            max_demand=inputs.get("max_demand"),
            discrete_demands=inputs.get("discrete_demands"),
            discrete_probabilities=inputs.get("discrete_probabilities"),
        )
        return res.to_dict()

    # -----------------------------------------------------------------------
    # Step 6: Verification & Sanity Checks
    # -----------------------------------------------------------------------

    def _verify_calculation(
        self,
        model_key: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any]
    ) -> VerificationResult:
        passed = []
        warnings = []
        errors = []

        if model_key == "eoq":
            eoq = results.get("eoq", 0.0)
            if eoq > 0:
                passed.append(f"EOQ calculated strictly positive: {eoq:.2f} units.")
            else:
                errors.append(f"EOQ must be strictly positive. Got: {eoq}")

            ann_order = results.get("annual_ordering_cost", 0.0)
            ann_hold = results.get("annual_holding_cost", 0.0)
            diff = abs(ann_order - ann_hold)
            if diff < 0.01:
                passed.append("First-order optimality condition verified: Annual Ordering Cost equals Annual Holding Cost.")
            else:
                warnings.append(f"Minor discrepancy in cost equality (|Order - Hold| = ${diff:.4f}).")

            rop = results.get("reorder_point_deterministic")
            if rop is not None and rop > eoq:
                warnings.append(f"Lead time demand ({rop:.1f}) exceeds EOQ ({eoq:.1f}); more than 1 order will be outstanding simultaneously.")

        elif model_key == "quantity_discount":
            opt_q = results.get("optimal_order_quantity", 0.0)
            opt_cost = results.get("optimal_total_cost", 0.0)
            if opt_q > 0 and opt_cost > 0:
                passed.append(f"Optimal quantity ({opt_q:,.2f}) and minimal total cost (${opt_cost:,.2f}) successfully established.")
            else:
                errors.append("Invalid non-positive quantity or total cost.")

            tiers = results.get("tier_evaluations", [])
            for t in tiers:
                if t.get("is_feasible"):
                    passed.append(f"Feasibility verified for Tier {t.get('tier_index')+1} at Q = {t.get('feasible_quantity'):,.0f}.")

        elif model_key == "safety_stock_rop":
            ss = results.get("safety_stock", 0.0)
            rop = results.get("reorder_point", 0.0)
            z = results.get("z_score", 0.0)
            sl = results.get("service_level")

            if sl is not None:
                if sl >= 0.50 and z >= 0:
                    passed.append(f"Service level {sl*100:.1f}% mapped to valid z-score = {z:.3f}.")
                else:
                    warnings.append(f"Service level {sl*100:.1f}% is below 50%, resulting in negative safety stock cushion.")

            if rop >= ss:
                passed.append(f"Logical condition verified: Reorder point ({rop:.2f}) >= Safety stock ({ss:.2f}).")
            else:
                errors.append("Reorder point is unexpectedly smaller than safety stock.")

        elif model_key == "fixed_period_review":
            m_target = results.get("order_up_to_level", 0.0)
            prot_d = results.get("protection_demand_mean", 0.0)
            rec_q = results.get("recommended_order_quantity", 0.0)

            if m_target >= prot_d:
                passed.append(f"Order-up-to level ({m_target:.2f}) adequately covers expected protection demand ({prot_d:.2f}).")
            else:
                errors.append("Order-up-to level fails to cover expected demand during protection period.")

            if rec_q >= 0:
                passed.append(f"Recommended order quantity non-negative: {rec_q:.2f} units.")

        elif model_key == "newsvendor":
            cr = results.get("critical_fractile", 0.0)
            cu = results.get("cost_of_underage", 0.0)
            co = results.get("cost_of_overage", 0.0)
            p_cost = results.get("unit_cost", 0.0)
            p_price = results.get("selling_price", 0.0)

            if p_price > p_cost:
                passed.append(f"Economic viability verified: Unit price (${p_price:.2f}) > Unit cost (${p_cost:.2f}).")
            else:
                errors.append("Selling price must exceed unit cost.")

            if 0.0 < cr < 1.0:
                passed.append(f"Critical Fractile verified in unit probability interval: CR = {cr:.4f}.")
            else:
                errors.append(f"Critical Fractile out of bounds: {cr}")

            opt_q = results.get("optimal_order_quantity", 0.0)
            if opt_q >= 0:
                passed.append(f"Optimal order quantity non-negative: {opt_q:.2f} units.")

        # Incorporate deterministic verification checks from the tool itself
        tool_checks = results.get("verification_checks", [])
        for chk in tool_checks:
            if chk not in passed:
                passed.append(chk)

        status = "FAILED" if errors else ("WARNING" if warnings else "PASSED")
        return VerificationResult(
            is_valid=(len(errors) == 0),
            status=status,
            checks_passed=passed,
            warnings=warnings,
            errors=errors,
        )

    # -----------------------------------------------------------------------
    # Step 7: Explain (Managerial & Technical Narrative Generation)
    # -----------------------------------------------------------------------

    def _generate_explanation(
        self,
        model_key: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        verification: VerificationResult
    ) -> ExplanationResult:

        # Extract deterministic verification elements directly from tool output
        formula = results.get("formula", "")
        formula_sub = results.get("formula_with_substituted_values", "")
        var_defs = results.get("variable_definitions", {})
        assumptions = results.get("assumptions", [])
        tool_verification = results.get("verification_checks", [])
        inter = results.get("intermediate_calculations", {})

        if model_key == "eoq":
            eoq = results["eoq"]
            eoq_r = results["eoq_rounded"]
            d = results["demand"]
            s = results["ordering_cost"]
            h = results["holding_cost"]
            tot_cost = results["total_annual_cost"]
            order_cost = results["annual_ordering_cost"]
            hold_cost = results["annual_holding_cost"]
            orders_yr = results["orders_per_year"]
            cycle_days = results["cycle_time_days"]

            exec_summary = (
                f"**Recommendation**: Order **{eoq_r:,} units** per replenishment order. "
                f"This minimizes the total annual inventory cost to **${tot_cost:,.2f}**, balancing "
                f"annual ordering costs (${order_cost:,.2f}) with annual inventory holding costs (${hold_cost:,.2f})."
            )
            math_formula = formula or rf"EOQ = \sqrt{{\frac{{2 \cdot D \cdot S}}{{H}}}} = \sqrt{{\frac{{2 \cdot {d:,.0f} \cdot {s:,.2f}}}{{{h:,.2f}}}}}"
            intermediate = []
            if inputs.get("demand_conversion_note"):
                intermediate.append(f"0. Demand Normalization: {inputs['demand_conversion_note']}")
            intermediate.extend([
                f"1. Annual Demand (D) = {d:,.0f} units/year",
                f"2. Setup/Ordering Cost (S) = ${s:,.2f} per order",
                f"3. Annual Holding Cost (H) = ${h:,.2f} per unit/year",
                f"4. Intermediate Numerator (2 * D * S) = {inter.get('numerator_2DS', 2.0 * d * s):,.2f}",
                f"5. Variance Ratio (2 * D * S / H) = {inter.get('ratio_2DS_over_H', (2.0 * d * s) / h):,.2f}",
                f"6. Unconstrained EOQ Q* = sqrt(ratio) = {inter.get('sqrt_ratio_Q', eoq):,.2f} units",
                f"7. Annual Ordering Cost = {inter.get('annual_ordering_cost', order_cost):,.2f}",
                f"8. Annual Holding Cost = {inter.get('annual_holding_cost', hold_cost):,.2f}",
            ])
            breakdown = {
                "Annual Ordering Cost": f"${order_cost:,.2f}",
                "Annual Holding Cost": f"${hold_cost:,.2f}",
                "Total Inventory Holding & Ordering Cost": f"${results['total_ordering_holding_cost']:,.2f}",
                "Orders Per Year": f"{orders_yr:.2f} orders",
                "Cycle Time (Order Frequency)": f"{cycle_days:.1f} days",
                "Average Cycle Inventory": f"{results['average_cycle_inventory']:,.1f} units",
            }
            if results.get("annual_purchase_cost"):
                breakdown["Annual Item Purchase Cost"] = f"${results['annual_purchase_cost']:,.2f}"

            insights = [
                "**Economic Trade-Off**: Ordering larger batches lowers annual ordering costs but raises average inventory and warehouse holding costs.",
                "**Flat Cost Curve**: The total cost curve is relatively flat near the EOQ minimum; small deviations (e.g., standard packaging sizes) will not significantly penalize total costs.",
                f"**Reorder Frequency**: The warehouse should expect to place an order approximately every **{cycle_days:.0f} days**."
            ]

        elif model_key == "quantity_discount":
            opt_q = results["optimal_order_quantity"]
            opt_q_r = results["optimal_order_quantity_rounded"]
            opt_tier = results["optimal_tier_index"] + 1
            opt_price = results["optimal_unit_price"]
            opt_cost = results["optimal_total_cost"]
            savings = results["annual_savings_vs_base"]
            pct_savings = results["percentage_savings_vs_base"]

            exec_summary = (
                f"**Recommendation**: Order **{opt_q_r:,} units** in **Tier {opt_tier}** at **${opt_price:,.2f}/unit**. "
                f"This achieves the lowest total annual cost of **${opt_cost:,.2f}**"
            )
            if savings > 0:
                exec_summary += f", saving **${savings:,.2f} ({pct_savings:.2f}%)** annually compared to ordering at the baseline non-discount price."
            else:
                exec_summary += ". Lower discount tiers do not justify the added carrying cost of larger inventory batches."

            math_formula = formula or r"TC(Q) = D \cdot C + \frac{D}{Q} \cdot S + \frac{Q}{2} \cdot H(C)"
            intermediate = []
            if inputs.get("demand_conversion_note"):
                intermediate.append(f"0. Demand Normalization: {inputs['demand_conversion_note']}")
            intermediate.extend([
                f"1. Base Tier Total Cost = ${inter.get('base_tier_cost', 0.0):,.2f}",
                f"2. Evaluated {len(results.get('tier_evaluations', []))} discount pricing tiers balancing purchase, ordering, and holding costs.",
                f"3. Optimal Order Quantity Q* = {opt_q:,.2f} units (Tier {opt_tier})",
                f"4. Minimal Total Annual Cost = ${inter.get('minimal_cost', opt_cost):,.2f}",
                f"5. Annual Savings vs Base Tier = ${inter.get('annual_savings_vs_base', savings):,.2f} ({inter.get('percentage_savings_vs_base', pct_savings):.2f}%)",
            ])
            breakdown = {
                "Optimal Tier": f"Tier {opt_tier}",
                "Optimal Order Quantity": f"{opt_q:,.2f} units",
                "Unit Price": f"${opt_price:,.2f}",
                "Total Annual Cost": f"${opt_cost:,.2f}",
                "Annual Savings vs Base Tier": f"${savings:,.2f} ({pct_savings:.2f}%)",
            }
            insights = [
                "**Bulk Discount Evaluation**: A price break is worthwhile only if the annual purchase cost savings exceed the increased annual holding costs of carrying extra inventory.",
                "**Boundary Condition**: When EOQ falls below a price threshold, moving up to the threshold point is often the most cost-efficient way to secure wholesale discounts.",
            ]

        elif model_key == "safety_stock_rop":
            ss = results["safety_stock"]
            ss_r = results["safety_stock_rounded"]
            rop = results["reorder_point"]
            rop_r = results["reorder_point_rounded"]
            sl = results["service_level"]
            z = results["z_score"]
            mu_l = results["lead_time_demand_mean"]
            sigma_l = results["lead_time_demand_std"]

            sl_label = f"{sl*100:.1f}%" if sl is not None else "Deterministic (100%)"
            exec_summary = (
                f"**Recommendation**: Set the **Reorder Point (ROP) at {rop_r:,} units** with a **Safety Stock cushion of {ss_r:,} units**. "
                f"Whenever inventory drops to {rop_r:,} units, immediately trigger a replenishment order to maintain a **{sl_label} Cycle Service Level**."
            )
            math_formula = formula or r"ROP = \mu_L + SS = (d \cdot L) + z_{\alpha} \cdot \sigma_{LT}"
            intermediate = []
            if inputs.get("demand_conversion_note"):
                intermediate.append(f"0. Demand Normalization: {inputs['demand_conversion_note']}")
            intermediate.extend([
                f"1. Lead Time Uncertainty Model Case: {inter.get('case_type', '')}",
                f"2. Mean Demand during Lead Time (mu_L) = {inter.get('expected_lead_time_demand', mu_l):.2f} units",
                f"3. Lead Time Demand Std Dev (sigma_LT) = {inter.get('sigma_LT', sigma_l):.2f} units",
                f"4. Normal Z-score (Service Level = {sl_label}) = {inter.get('z_score', z):.3f}",
                f"5. Safety Stock Cushion (SS = z * sigma_LT) = {inter.get('safety_stock', ss):.2f} units",
                f"6. Reorder Point (ROP = mu_L + SS) = {inter.get('reorder_point', rop):.2f} units",
            ])
            breakdown = {
                "Cycle Service Level (CSL)": sl_label,
                "Expected Lead Time Demand": f"{mu_l:.2f} units",
                "Safety Stock Buffer": f"{ss_r:,} units",
                "Reorder Point (Trigger Level)": f"{rop_r:,} units",
            }
            if results.get("implied_fill_rate"):
                breakdown["Type 2 Service Level (Fill Rate)"] = f"{results['implied_fill_rate']*100:.2f}%"

            insights = [
                f"**Protection against Demand Surges**: With {ss_r:,} units of safety stock, the company is protected against unexpected lead-time demand spikes.",
                "**Non-Linear Safety Stock Cost**: Increasing service level from 95% to 99% increases safety stock significantly as normal distribution tails taper.",
                "**Continuous Review**: This policy requires continuous or real-time monitoring of inventory positions.",
            ]

        elif model_key == "fixed_period_review":
            m_target = results["order_up_to_level"]
            m_target_r = results["order_up_to_level_rounded"]
            rec_q = results["recommended_order_quantity"]
            rec_q_r = results["recommended_order_quantity_rounded"]
            p = results["review_period"]
            lt = results["lead_time"]
            prot = results["protection_interval"]
            ss = results["safety_stock"]
            ip = results["current_inventory_position"]

            exec_summary = (
                f"**Recommendation**: Establish a **Target Order-Up-To Level (M) of {m_target_r:,} units**. "
                f"At the current review, with an inventory position of {ip:,.0f} units, place an order for **{rec_q_r:,} units**."
            )
            math_formula = formula or r"M = d \cdot (P + L) + z \cdot \sigma_d \sqrt{P + L}, \quad Q = \max(0, M - IP)"
            intermediate = []
            if inputs.get("demand_conversion_note"):
                intermediate.append(f"0. Demand Normalization: {inputs['demand_conversion_note']}")
            intermediate.extend([
                f"1. Protection Interval = Review Period ({p}) + Lead Time ({lt}) = {inter.get('protection_period', prot):.1f} {results['time_unit']}",
                f"2. Expected Demand over Protection Interval = {inter.get('demand_during_protection', results['protection_demand_mean']):.2f} units",
                f"3. Protection Interval Std Dev = {inter.get('std_during_protection', results['protection_demand_std']):.2f} units",
                f"4. Safety Stock Cushion (z = {inter.get('z_score', results['z_score']):.3f}) = {inter.get('safety_stock', ss):.2f} units",
                f"5. Target Order-Up-To Level (M) = {inter.get('target_inventory', m_target):.2f} units",
                f"6. Order Quantity Q = max(0, M - IP) = max(0, {inter.get('target_inventory', m_target):.2f} - {inter.get('inventory_position', ip):.2f}) = {inter.get('recommended_order_quantity', rec_q):.2f} units",
            ])
            breakdown = {
                "Review Interval (P)": f"{p:.1f} {results['time_unit']}",
                "Lead Time (L)": f"{lt:.1f} {results['time_unit']}",
                "Total Protection Horizon": f"{prot:.1f} {results['time_unit']}",
                "Safety Stock Buffer": f"{results['safety_stock_rounded']:,} units",
                "Target Order-Up-To Level (M)": f"{m_target_r:,} units",
                "Current Inventory Position": f"{ip:,.1f} units",
                "Recommended Order Quantity (Q)": f"{rec_q_r:,} units",
                "Average Inventory": f"{results['average_inventory']:.2f} units",
            }
            insights = [
                "**Periodic vs Continuous Review**: Fixed-period review requires larger safety stock than continuous review because uncertainty must cover both the review period AND the lead time (P + L).",
                "**Operational Simplicity**: Well-suited for joint replenishment of multiple items delivered on fixed schedule routes.",
            ]

        elif model_key == "newsvendor":
            opt_q = results["optimal_order_quantity"]
            opt_q_r = results["optimal_order_quantity_rounded"]
            cr = results["critical_fractile"]
            cu = results["cost_of_underage"]
            co = results["cost_of_overage"]
            exp_prof = results["expected_profit"]
            in_stock = results["in_stock_probability"]

            exec_summary = (
                f"**Recommendation**: Order **{opt_q_r:,} units** for this single selling period. "
                f"With a **Critical Fractile of {cr:.4f} ({cr*100:.2f}%)**, this quantity yields an **Expected Profit of ${exp_prof:,.2f}** "
                f"and an in-stock fill probability of **{in_stock*100:.2f}%**."
            )
            math_formula = formula or r"CR = \frac{C_u}{C_u + C_o} = \frac{P - C + g}{(P - C + g) + (C - V)}, \quad Q^* = F^{-1}(CR)"
            intermediate = [
                f"1. Cost of Underage (Cu = P - C + g) = ${inter.get('cost_of_underage', cu):,.2f}",
                f"2. Cost of Overage (Co = C - V) = ${inter.get('cost_of_overage', co):,.2f}",
                f"3. Critical Ratio (CR = Cu / (Cu + Co)) = {inter.get('critical_ratio', cr):.4f}",
                f"4. Optimal Order Quantity Q* = {inter.get('optimal_order_quantity', opt_q):.2f} units",
                f"5. Expected Sales = {inter.get('expected_sales_units', 0.0):.1f} units; Leftover = {inter.get('expected_leftover_units', 0.0):.1f} units",
                f"6. Expected Net Profit = ${inter.get('expected_profit', exp_prof):,.2f}",
            ]
            breakdown = {
                "Underage Cost (Cu)": f"${cu:,.2f}",
                "Overage Cost (Co)": f"${co:,.2f}",
                "Critical Fractile (Target Service Level)": f"{cr*100:.2f}%",
                "Optimal Order Quantity": f"{opt_q_r:,} units",
                "Expected Sales": f"{results['expected_sales_units']:.1f} units",
                "Expected Leftover / Unsold Units": f"{results['expected_leftover_units']:.1f} units",
                "Expected Lost Sales": f"{results['expected_lost_sales_units']:.1f} units",
                "Expected Revenue": f"${results['expected_revenue']:,.2f}",
                "Expected Purchase Cost": f"${results['expected_purchase_cost']:,.2f}",
                "Expected Net Profit": f"${exp_prof:,.2f}",
            }
            insights = [
                "**Asymmetric Risk Management**: When margin (Cu) is high relative to loss on unsold goods (Co), order above the mean demand. When margin is low, order below the mean.",
                "**Perishable & Seasonal Inventory**: This single-period policy is critical for apparel, holiday items, baked goods, and products with steep post-season markdowns.",
            ]

        else:
            exec_summary = "Calculation completed successfully."
            math_formula = ""
            intermediate = []
            breakdown = {}
            insights = []

        return ExplanationResult(
            executive_summary=exec_summary,
            mathematical_formula=math_formula,
            intermediate_steps=intermediate,
            cost_or_risk_breakdown=breakdown,
            managerial_insights=insights,
            formula_with_substituted_values=formula_sub,
            variable_definitions=var_defs,
            assumptions=assumptions,
            verification_checks=tool_verification,
        )

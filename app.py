"""
Supply Chain Analytics — Inventory Decision Agent Web Application
================================================================
Interactive Streamlit UI implementing the 7-step inventory decision architecture:
1. Customer Question
2. Identify Models
3. Extract Inputs
4. Select Python Tool
5. Calculate
6. Verify
7. Explain

Supports 5 inventory models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (ROP)
4. Fixed-Period Review Model
5. Newsvendor Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_safety_stock_and_rop,
    calculate_fixed_period_review,
    calculate_newsvendor,
    calculate_multicriteria_classification,
    MulticriteriaClassificationResult,
)
from inventory_agent import InventoryDecisionAgent, AgentPipelineResponse


# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Analytics | Inventory Decision Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern executive card styling
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .step-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background-color: #DBEAFE;
        color: #1E40AF;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-passed {
        color: #065F46;
        background-color: #D1FAE5;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
    }
    .status-warning {
        color: #92400E;
        background-color: #FEF3C7;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
    }
    .status-failed {
        color: #991B1B;
        background-color: #FEE2E2;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Pre-set Realistic Inventory Questions
# ---------------------------------------------------------------------------

PRESET_QUESTIONS = {
    "1. Economic Order Quantity (EOQ)": (
        "We have an annual demand of 12,000 units for our medical diagnostic kits. "
        "Each order costs $50 to place with the supplier, and our annual holding cost "
        "is $2.50 per unit per year. The supplier has a lead time of 5 days, and we operate 250 days a year. "
        "What is our optimal order quantity and when should we reorder?"
    ),
    "2. Quantity Discount Model": (
        "Our warehouse needs 10,000 components annually with an ordering cost of $60 per order. "
        "Our annual holding cost rate is 20%. The supplier offers the following wholesale discount tiers: "
        "0 to 499 units at $20.00 each; 500 to 999 units at $18.00 each; and 1,000 or more units at $16.00 each. "
        "Which price tier and order quantity minimizes our total annual supply chain cost?"
    ),
    "3. Safety Stock & Reorder Point (ROP)": (
        "Our flagship electric motor experiences average daily demand of 50 units with a standard deviation "
        "of 8 units. Supplier delivery lead time averages 4 days with a lead time standard deviation of 1 day. "
        "Management mandates a 95% cycle service level. "
        "What safety stock cushion and reorder point should we set to prevent stockouts?"
    ),
    "4. Fixed-Period Review Model": (
        "We review stock on a periodic schedule every 14 days. The supplier takes 4 days to deliver. "
        "Daily demand averages 30 units with a standard deviation of 6 units. "
        "We target a 98% cycle service level. Currently, our warehouse has 65 units on hand, "
        "25 units on order from last week, and 0 backorders. "
        "What is our target order-up-to level and how many units should we order right now?"
    ),
    "5. Newsvendor (Single-Period Probabilistic)": (
        "We are ordering custom university jackets for the upcoming winter graduation ceremony. "
        "Each jacket costs $35 to manufacture and sells for $85. "
        "Any unsold jackets after graduation will be liquidated to an outlet for $15. "
        "Estimated graduation demand follows a normal distribution with a mean of 450 units and a standard deviation of 60 units. "
        "How many jackets should we order to maximize expected profit?"
    ),
}


# ---------------------------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("📦 Navigation & Settings")
st.sidebar.caption("🏷️ Version: `v1.4.0` | Core SemVer | 2026-11-07")

st.sidebar.markdown("### Select Preset Example")
selected_preset_label = st.sidebar.selectbox(
    "Choose a sample inventory question:",
    ["Select below or write custom question -->"] + list(PRESET_QUESTIONS.keys()),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Manual Override")
manual_model = st.sidebar.selectbox(
    "Force Model (or Auto-Detect):",
    [
        "Auto-Detect from Question",
        "Economic Order Quantity (EOQ)",
        "Quantity Discount Model",
        "Safety Stock and Reorder Point (ROP)",
        "Fixed-Period Review Model",
        "Newsvendor Model",
    ],
)

model_override_key = None
if manual_model == "Economic Order Quantity (EOQ)":
    model_override_key = "eoq"
elif manual_model == "Quantity Discount Model":
    model_override_key = "quantity_discount"
elif manual_model == "Safety Stock and Reorder Point (ROP)":
    model_override_key = "safety_stock_rop"
elif manual_model == "Fixed-Period Review Model":
    model_override_key = "fixed_period_review"
elif manual_model == "Newsvendor Model":
    model_override_key = "newsvendor"

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"],
    help="Upload an inventory CSV file with either: (1) Item Number, Item Name, UoM, Sales ($), Hits for multicriteria classification, or (2) Product, Category, Units, Unit Cost for portfolio valuation.",
)


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">📦 Supply Chain Analytics: Inventory Decision Agent</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Autonomous 7-Step Decision Pipeline for Classical & Stochastic Inventory Systems '
    "| UNCW BAN 517 Supply Chain Analytics</div>",
    unsafe_allow_html=True,
)

# Step 1: Customer Question Input
st.markdown("### 💬 1. Customer Question")

default_question_text = PRESET_QUESTIONS["1. Economic Order Quantity (EOQ)"]
if selected_preset_label in PRESET_QUESTIONS:
    default_question_text = PRESET_QUESTIONS[selected_preset_label]

user_query = st.text_area(
    "Enter a natural-language inventory question:",
    value=default_question_text,
    height=110,
    help="Type any supply chain scenario or choose an example from the sidebar.",
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run_agent = st.button(
        "🚀 Run Decision Agent", type="primary", use_container_width=True
    )

agent = InventoryDecisionAgent()

# Run the 7-step agent pipeline
pipeline_response: AgentPipelineResponse = agent.process_query(
    customer_question=user_query, manual_model_override=model_override_key
)


# ---------------------------------------------------------------------------
# Plotting Functions for Visual Analytics
# ---------------------------------------------------------------------------


def plot_eoq_curve(res: dict):
    d = res["demand"]
    s = res["ordering_cost"]
    h = res["holding_cost"]
    eoq = res["eoq"]

    q_vals = np.linspace(max(10, eoq * 0.2), eoq * 2.5, 300)
    order_costs = (d / q_vals) * s
    hold_costs = (q_vals / 2.0) * h
    tot_costs = order_costs + hold_costs

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(
        q_vals, tot_costs, label="Total Annual Cost", color="#1E3A8A", linewidth=2.5
    )
    ax.plot(
        q_vals,
        order_costs,
        label="Annual Ordering Cost (D/Q * S)",
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
    )
    ax.plot(
        q_vals,
        hold_costs,
        label="Annual Holding Cost (Q/2 * H)",
        color="#16A34A",
        linestyle="-.",
        linewidth=1.5,
    )

    min_cost = (d / eoq) * s + (eoq / 2.0) * h
    ax.scatter(
        [eoq],
        [min_cost],
        color="#D97706",
        s=100,
        zorder=5,
        label=f"Optimal EOQ ({eoq:.0f} units)",
    )
    ax.axvline(eoq, color="#D97706", linestyle=":", alpha=0.7)

    ax.set_title(
        "EOQ Total Annual Cost Curve & Trade-Off Minimum",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Order Quantity (Q)")
    ax.set_ylabel("Annual Cost ($)")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    return fig


def plot_discount_comparison(res: dict):
    tiers = res["tier_evaluations"]
    labels = [f"Tier {t['tier_index'] + 1}\n(${t['unit_price']:.2f})" for t in tiers]
    purch_costs = [t["annual_purchase_cost"] for t in tiers]
    order_costs = [t["annual_ordering_cost"] for t in tiers]
    hold_costs = [t["annual_holding_cost"] for t in tiers]

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x, purch_costs, width, label="Annual Purchase Cost", color="#3B82F6")
    ax.bar(
        x,
        order_costs,
        width,
        bottom=purch_costs,
        label="Annual Ordering Cost",
        color="#EF4444",
    )
    bottom_hold = [p + o for p, o in zip(purch_costs, order_costs)]
    ax.bar(
        x,
        hold_costs,
        width,
        bottom=bottom_hold,
        label="Annual Holding Cost",
        color="#10B981",
    )

    opt_idx = res["optimal_tier_index"]
    ax.annotate(
        "★ Optimal Tier",
        xy=(opt_idx, res["optimal_total_cost"]),
        xytext=(opt_idx, res["optimal_total_cost"] * 1.05),
        ha="center",
        fontweight="bold",
        color="#B45309",
        arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Annual Cost ($)")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.set_title(
        "Total Annual Cost Breakdown Across Discount Tiers",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_rop_normal(res: dict):
    mu_l = res["lead_time_demand_mean"]
    sigma_l = res["lead_time_demand_std"]
    rop = res["reorder_point"]
    sl = res["service_level"]

    x = np.linspace(max(0, mu_l - 4 * sigma_l), mu_l + 4 * sigma_l, 400)
    y = stats.norm.pdf(x, mu_l, sigma_l)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, y, color="#2563EB", lw=2, label="Lead Time Demand Distribution")

    # Shaded safety stock region
    mask_in_stock = x <= rop
    ax.fill_between(
        x[mask_in_stock],
        y[mask_in_stock],
        color="#93C5FD",
        alpha=0.5,
        label=f"In-Stock Probability ({sl * 100:.1f}%)",
    )

    # Stockout tail
    mask_stockout = x > rop
    ax.fill_between(
        x[mask_stockout],
        y[mask_stockout],
        color="#FCA5A5",
        alpha=0.7,
        label=f"Stockout Risk ({(1 - sl) * 100:.1f}%)",
    )

    ax.axvline(
        mu_l, color="#4B5563", linestyle="--", label=f"Mean Demand (mu = {mu_l:.0f})"
    )
    ax.axvline(
        rop,
        color="#DC2626",
        lw=2,
        label=f"ROP = {rop:.0f} (SS = {res['safety_stock']:.0f})",
    )

    ax.set_title(
        f"Reorder Point & Safety Stock Cushion (CSL = {sl * 100:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Demand during Lead Time (units)")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_newsvendor_curve(res: dict):
    mu = res.get("demand_mean", 100.0)
    sigma = res.get("demand_std", 20.0)
    q_opt = res["optimal_order_quantity"]
    cr = res["critical_fractile"]

    q_range = np.linspace(max(0, mu - 3 * sigma), mu + 3 * sigma, 150)
    profits = []

    p = res["selling_price"]
    c = res["unit_cost"]
    v = res["salvage_value"]
    g = res["penalty_cost"]

    for q in q_range:
        z = (q - mu) / sigma
        phi_z = stats.norm.pdf(z)
        Phi_z = stats.norm.cdf(z)
        loss_z = phi_z - z * (1.0 - Phi_z)
        exp_lost = sigma * loss_z
        exp_sales = mu - exp_lost
        exp_leftover = q - exp_sales
        prof = (p * exp_sales) + (v * exp_leftover) - (c * q) - (g * exp_lost)
        profits.append(prof)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(q_range, profits, color="#059669", lw=2.5, label="Expected Profit ($)")
    ax.scatter(
        [q_opt],
        [res["expected_profit"]],
        color="#DC2626",
        s=100,
        zorder=5,
        label=f"Optimal Q* ({q_opt:.0f} units)",
    )
    ax.axvline(q_opt, color="#DC2626", linestyle=":", alpha=0.7)
    ax.axvline(
        mu, color="#6B7280", linestyle="--", alpha=0.7, label=f"Mean Demand ({mu:.0f})"
    )

    ax.set_title(
        f"Newsvendor Expected Profit Curve (Critical Fractile = {cr * 100:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Order Quantity (Q)")
    ax.set_ylabel("Expected Profit ($)")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Persistent Important Variables Dashboard (Always Visible Below Prompt)
# ---------------------------------------------------------------------------


def render_kpi_card(
    title: str, value: str, subtitle: str = "", card_type: str = "input"
) -> str:
    """Helper to render high-contrast, professional KPI cards for the persistent dashboard."""
    if card_type == "input":
        border_color = "#3B82F6"
        accent_tag = "INPUT"
        tag_bg = "#EFF6FF"
        tag_color = "#1D4ED8"
    elif card_type == "output":
        border_color = "#10B981"
        accent_tag = "DECISION"
        tag_bg = "#ECFDF5"
        tag_color = "#047857"
    elif card_type == "financial":
        border_color = "#F59E0B"
        accent_tag = "FINANCIAL"
        tag_bg = "#FFFBEB"
        tag_color = "#B45309"
    else:
        border_color = "#8B5CF6"
        accent_tag = "SERVICE"
        tag_bg = "#F5F3FF"
        tag_color = "#6D28D9"

    return f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid {border_color}; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.2rem;">
            <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B;">{title}</span>
            <span style="font-size: 0.62rem; font-weight: 700; background-color: {tag_bg}; color: {tag_color}; padding: 0.1rem 0.4rem; border-radius: 4px;">{accent_tag}</span>
        </div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A; line-height: 1.3;">{value}</div>
        <div style="font-size: 0.72rem; color: #64748B; margin-top: 0.15rem;">{subtitle}</div>
    </div>
    """


def render_persistent_variable_dashboard(response: AgentPipelineResponse):
    """
    Renders the persistent dashboard of important input and output variables
    directly below the customer prompt, remaining always visible across all tabs.
    """
    m_key = response.model_key
    res = response.raw_calculation_result
    inp = response.extracted_inputs
    has_calc_error = "error" in res
    v = response.verification

    v_status = v.status
    v_badge_color = (
        "#065F46"
        if v_status == "PASSED"
        else ("#92400E" if v_status == "WARNING" else "#991B1B")
    )
    v_badge_bg = (
        "#D1FAE5"
        if v_status == "PASSED"
        else ("#FEF3C7" if v_status == "WARNING" else "#FEE2E2")
    )

    # Header Ribbon
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border: 1.5px solid #CBD5E1; border-radius: 10px; padding: 0.85rem 1.15rem; margin-top: 0.75rem; margin-bottom: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #1E3A8A;">📌 Important Variables & Decision Dashboard</span>
                    <span style="font-size: 0.8rem; color: #64748B; margin-left: 0.5rem;">(Always visible across all analysis tabs)</span>
                </div>
                <div style="display: flex; gap: 0.4rem; align-items: center;">
                    <span style="font-size: 0.78rem; font-weight: 600; color: #1E40AF; background: #DBEAFE; padding: 0.2rem 0.6rem; border-radius: 9999px;">Model: {response.identified_model}</span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: {v_badge_color}; background: {v_badge_bg}; padding: 0.2rem 0.6rem; border-radius: 9999px;">Audit: {v_status}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Surface Demand Normalization Callout if a conversion was performed
    conv_banner = inp.get("demand_conversion_note") or (
        res.get("inputs_extracted", {}).get("demand_conversion_note")
        if isinstance(res, dict)
        else None
    )
    if conv_banner:
        st.info(f"🔄 **Demand Normalization**: {conv_banner}")

    # -----------------------------------------------------------------------
    # Section A: Key Extracted Input Variables
    # -----------------------------------------------------------------------
    st.markdown(
        "<div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #2563EB; margin-bottom: 0.35rem;'>📥 Key Input Variables (Extracted from Problem)</div>",
        unsafe_allow_html=True,
    )

    if m_key == "eoq":
        d_val = res.get("demand", inp.get("annual_demand", 0.0))
        s_val = res.get("ordering_cost", inp.get("ordering_cost", 0.0))
        h_val = res.get("holding_cost", inp.get("holding_cost", 0.0))
        lt_val = res.get("lead_time_days", inp.get("lead_time_days", 0.0))
        days_val = res.get(
            "operating_days_per_year", inp.get("working_days_per_year", 250.0)
        )

        time_unit = inp.get("demand_time_unit")
        if time_unit and time_unit not in ("year", "unspecified"):
            raw_d = inp.get("raw_demand", inp.get(f"{time_unit}_demand", d_val))
            d_sub = f"Converted from {raw_d:,.0f}/{time_unit}"
        else:
            d_sub = "Annual requirement rate"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(
            render_kpi_card("Annual Demand (D)", f"{d_val:,.0f} units", d_sub, "input"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Ordering Cost (S)",
                f"${s_val:,.2f}",
                "Cost to place one replenishment",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Holding Cost (H)",
                f"${h_val:,.2f}",
                "Carrying cost per unit-year",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Lead Time (L)",
                f"{lt_val:.0f} days" if lt_val > 0 else "0 days",
                "Supplier delivery window",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c5.markdown(
            render_kpi_card(
                "Operating Schedule",
                f"{days_val:.0f} days/yr",
                "Annual operating days",
                "input",
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "quantity_discount":
        d_val = res.get("demand", inp.get("annual_demand", 0.0))
        s_val = res.get("ordering_cost", inp.get("ordering_cost", 0.0))
        h_rate = res.get("holding_cost_rate", inp.get("holding_cost_rate"))
        h_fixed = res.get("holding_cost_fixed", inp.get("holding_cost_fixed"))
        h_disp = (
            f"{h_rate * 100:.1f}% rate"
            if h_rate is not None
            else (f"${h_fixed:,.2f}/unit" if h_fixed else "N/A")
        )
        tiers_list = res.get("tier_evaluations", inp.get("tiers", []))
        n_tiers = len(tiers_list)
        base_p = tiers_list[0].get("unit_price", 0.0) if tiers_list else 0.0

        time_unit = inp.get("demand_time_unit")
        if time_unit and time_unit not in ("year", "unspecified"):
            raw_d = inp.get("raw_demand", inp.get(f"{time_unit}_demand", d_val))
            d_sub = f"Converted from {raw_d:,.0f}/{time_unit}"
        else:
            d_sub = "Annual requirement rate"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(
            render_kpi_card("Annual Demand (D)", f"{d_val:,.0f} units", d_sub, "input"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Ordering Cost (S)",
                f"${s_val:,.2f}",
                "Cost per replenishment order",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Holding Policy", h_disp, "Carrying cost rate / fixed", "input"
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Discount Schedule",
                f"{n_tiers} Tiers",
                "Wholesale volume brackets",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c5.markdown(
            render_kpi_card(
                "Base Tier Price", f"${base_p:,.2f}", "Starting price (Tier 1)", "input"
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "safety_stock_rop":
        d_mean = res.get("demand_mean", inp.get("demand_mean", 0.0))
        d_std = res.get("demand_std", inp.get("demand_std", 0.0))
        lt_mean = res.get(
            "lead_time",
            res.get(
                "lead_time_mean", inp.get("lead_time", inp.get("lead_time_mean", 0.0))
            ),
        )
        lt_std = res.get("lead_time_std", inp.get("lead_time_std", 0.0))
        sl_val = res.get("service_level", inp.get("service_level", 0.95))
        z_val = res.get(
            "z_score", res.get("z_value", inp.get("z_score", inp.get("z_value", 1.645)))
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(
            render_kpi_card(
                "Daily Demand (d̄)",
                f"{d_mean:.1f} units",
                f"Std Dev σd = {d_std:.1f}",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Lead Time (L̄)",
                f"{lt_mean:.1f} days",
                f"Std Dev σL = {lt_std:.1f}",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Target Service (CSL)",
                f"{sl_val * 100:.1f}%",
                f"z-score = {z_val:.2f}",
                "service",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Lead Time Demand (μL)",
                f"{res.get('lead_time_demand_mean', d_mean * lt_mean):.1f} units",
                "Expected demand during LT",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c5.markdown(
            render_kpi_card(
                "LT Demand Std (σLT)",
                f"{res.get('lead_time_demand_std', 0.0):.2f} units",
                "Combined uncertainty",
                "input",
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "fixed_period_review":
        d_mean = res.get("demand_mean", inp.get("demand_mean", 0.0))
        d_std = res.get("demand_std", inp.get("demand_std", 0.0))
        p_val = res.get("review_period", inp.get("review_period", 0.0))
        lt_val = res.get("lead_time", inp.get("lead_time", 0.0))
        ip_val = res.get(
            "current_inventory_position", inp.get("inventory_position", 0.0)
        )
        unit = res.get("time_unit", "days")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(
            render_kpi_card(
                "Daily Demand (d)",
                f"{d_mean:.1f} units",
                f"Std Dev σd = {d_std:.1f}",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Review Interval (P)",
                f"{p_val:.0f} {unit}",
                "Review schedule frequency",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Supplier Lead Time (L)",
                f"{lt_val:.0f} {unit}",
                "Delivery replenishment time",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Current Position (IP)",
                f"{ip_val:.0f} units",
                f"OH:{inp.get('inventory_on_hand', 0):.0f} OO:{inp.get('inventory_on_order', 0):.0f} BO:{inp.get('backorders', 0):.0f}",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c5.markdown(
            render_kpi_card(
                "Target Service Level",
                f"{res.get('service_level', inp.get('service_level', 0.98)) * 100:.1f}%",
                f"z = {res.get('z_value', 2.054):.2f}",
                "service",
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "newsvendor":
        p_val = res.get("selling_price", inp.get("selling_price", 0.0))
        c_val = res.get("unit_cost", inp.get("unit_cost", 0.0))
        v_val = res.get("salvage_value", inp.get("salvage_value", 0.0))
        d_mean = res.get("demand_mean", inp.get("demand_mean", 0.0))
        d_std = res.get("demand_std", inp.get("demand_std", 0.0))
        dist_type = res.get(
            "demand_distribution", inp.get("distribution_type", "normal")
        ).title()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(
            render_kpi_card(
                "Selling Price (P)", f"${p_val:,.2f}", "Retail price per unit", "input"
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Unit Cost (C)",
                f"${c_val:,.2f}",
                "Acquisition/production cost",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Salvage Value (V)",
                f"${v_val:,.2f}",
                "Post-season recovery value",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Demand Mean (μ)",
                f"{d_mean:,.0f} units",
                f"Std Dev σ = {d_std:,.0f}",
                "input",
            ),
            unsafe_allow_html=True,
        )
        c5.markdown(
            render_kpi_card(
                "Distribution Model", dist_type, "Stochastic demand model", "service"
            ),
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Section B: Primary Calculated Decision Variables
    # -----------------------------------------------------------------------
    st.markdown(
        "<div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #059669; margin-top: 0.45rem; margin-bottom: 0.35rem;'>🎯 Primary Decision & Financial Variables (Calculated Results)</div>",
        unsafe_allow_html=True,
    )

    if has_calc_error:
        st.warning(f"⚠️ Calculation Pending: {res['error']}")
        return

    if m_key == "eoq":
        eoq_r = res.get("eoq_rounded", round(res.get("eoq", 0)))
        rop_val = (
            res.get("reorder_point_rounded")
            or res.get("reorder_point_deterministic")
            or res.get("reorder_point")
        )
        rop_disp = f"{round(rop_val):,} units" if rop_val is not None else "N/A"
        rop_sub = (
            "Inventory replenishment trigger"
            if rop_val is not None
            else "Lead time not specified"
        )
        tot_cost = res.get("total_annual_cost", 0.0)
        orders_yr = res.get("orders_per_year", 0.0)
        cycle_days = res.get("cycle_time_days")
        cycle_sub = (
            f"Order cycle: {cycle_days:.1f} days"
            if cycle_days is not None
            else "Cycle days: N/A"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            render_kpi_card(
                "Optimal Order Qty (Q*)",
                f"{eoq_r:,} units",
                "Cost-minimizing EOQ batch",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card("Reorder Point (ROP)", rop_disp, rop_sub, "output"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Total Annual Cost",
                f"${tot_cost:,.2f}",
                f"Hold: ${res.get('annual_holding_cost', 0.0):,.0f} | Order: ${res.get('annual_ordering_cost', 0.0):,.0f}",
                "financial",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Replenishment Rhythm",
                f"{orders_yr:.1f} orders/yr",
                cycle_sub,
                "service",
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "quantity_discount":
        opt_q = res.get(
            "optimal_order_quantity_rounded",
            round(res.get("optimal_order_quantity", 0)),
        )
        opt_tier = res.get("optimal_tier_index", 0) + 1
        opt_price = res.get("optimal_unit_price", 0.0)
        opt_cost = res.get("optimal_total_cost", 0.0)
        savings = res.get("annual_savings_vs_base", 0.0)
        pct_savings = res.get("percentage_savings_vs_base", 0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            render_kpi_card(
                "Optimal Order Qty (Q*)",
                f"{opt_q:,} units",
                "Optimal wholesale order lot",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Winning Tier & Price",
                f"Tier {opt_tier} (${opt_price:.2f})",
                "Min total cost price break",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Total Annual Cost (TC*)",
                f"${opt_cost:,.2f}",
                "Purchase + order + hold",
                "financial",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Annual Cost Savings",
                f"${savings:,.2f}",
                f"{pct_savings:.1f}% savings vs Tier 1",
                "financial",
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "safety_stock_rop":
        rop_r = res.get("reorder_point_rounded", round(res.get("reorder_point", 0)))
        ss_r = res.get("safety_stock_rounded", round(res.get("safety_stock", 0)))
        stockout_prob = res.get(
            "stockout_probability", 1.0 - res.get("service_level", 0.95)
        )
        fill_rate = res.get("fill_rate", 1.0)
        shortage = res.get("expected_shortage_per_cycle", 0.0)
        csl = res.get("service_level", 0.95)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            render_kpi_card(
                "Reorder Point (ROP)",
                f"{rop_r:,} units",
                "Place order when inventory hits ROP",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Safety Stock Cushion (SS)",
                f"{ss_r:,} units",
                "Buffer against demand/LT spikes",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Stockout Risk",
                f"{stockout_prob * 100:.1f}%",
                f"Fill rate: {fill_rate * 100:.2f}%",
                "service",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Expected Shortage / Cycle",
                f"{shortage:.3f} units",
                f"CSL: {csl * 100:.1f}% target",
                "financial",
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "fixed_period_review":
        m_r = res.get(
            "order_up_to_level_rounded", round(res.get("order_up_to_level", 0))
        )
        q_r = res.get(
            "recommended_order_quantity_rounded",
            round(res.get("recommended_order_quantity", 0)),
        )
        ss_r = res.get("safety_stock_rounded", round(res.get("safety_stock", 0)))
        prot = res.get("protection_interval", 0.0)
        unit = res.get("time_unit", "days")
        cur_ip = res.get("current_inventory_position", 0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            render_kpi_card(
                "Order-Up-To Level (M)",
                f"{m_r:,} units",
                "Target replenishment ceiling",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Recommended Order Q",
                f"{q_r:,} units",
                f"Order lot Q = M - IP ({m_r} - {cur_ip:.0f})",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Safety Stock Buffer (SS)",
                f"{ss_r:,} units",
                "Buffer across protection period",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "Protection Interval (P+L)",
                f"{prot:.0f} {unit}",
                f"Review ({res.get('review_period', 0):.0f}) + Lead time ({res.get('lead_time', 0):.0f})",
                "service",
            ),
            unsafe_allow_html=True,
        )

    elif m_key == "newsvendor":
        q_opt_r = res.get(
            "optimal_order_quantity_rounded",
            round(res.get("optimal_order_quantity", 0)),
        )
        cr = res.get("critical_fractile", 0.0)
        exp_prof = res.get("expected_profit", 0.0)
        in_stock = res.get("in_stock_probability", 0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            render_kpi_card(
                "Optimal Order (Q*)",
                f"{q_opt_r:,} units",
                "Single-season target lot",
                "output",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            render_kpi_card(
                "Critical Fractile (CR)",
                f"{cr * 100:.2f}%",
                f"Cu = ${res.get('cost_of_underage', 0.0):,.2f} | Co = ${res.get('cost_of_overage', 0.0):,.2f}",
                "service",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            render_kpi_card(
                "Expected Net Profit",
                f"${exp_prof:,.2f}",
                "Max expected seasonal return",
                "financial",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            render_kpi_card(
                "In-Stock Probability",
                f"{in_stock * 100:.2f}%",
                f"Expected sales: {res.get('expected_sales', 0.0):.1f} units",
                "output",
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# CSV Data Upload & Inventory Portfolio Processing (pandas)
# ---------------------------------------------------------------------------


def process_and_analyze_inventory_csv(file):
    """
    Process an uploaded inventory CSV file using pandas:
    1. Read CSV into a pandas dataframe.
    2. Validate required columns: Product, Category, Units, Unit Cost.
    3. Calculate total Units, average Unit Cost, Product with highest Units,
       Units by Category, Inventory Value column, and total Inventory Value.
    """
    try:
        df = pd.read_csv(file)
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        required_cols = ["Product", "Category", "Units", "Unit Cost"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return (
                None,
                None,
                f"Missing required column(s): {', '.join(missing_cols)}. The uploaded CSV must contain: {', '.join(required_cols)}.",
            )

        # Metadata
        num_rows = len(df)
        num_cols = len(df.columns)
        col_names = list(df.columns)

        # Clean copy for calculations
        df_calc = df.copy()
        df_calc["Units"] = pd.to_numeric(
            df_calc["Units"].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        )
        df_calc["Unit Cost"] = pd.to_numeric(
            df_calc["Unit Cost"]
            .astype(str)
            .str.replace("$", "")
            .str.replace(",", "")
            .str.strip(),
            errors="coerce",
        )

        # 1. Calculate total Units
        total_units = (
            0.0 if pd.isna(df_calc["Units"].sum()) else float(df_calc["Units"].sum())
        )

        # 2. Calculate average Unit Cost
        avg_unit_cost = (
            0.0
            if pd.isna(df_calc["Unit Cost"].mean())
            else float(df_calc["Unit Cost"].mean())
        )

        # 3. Identify Product with highest Units
        if not df_calc["Units"].dropna().empty:
            idx_max = df_calc["Units"].idxmax()
            highest_product = str(df_calc.loc[idx_max, "Product"])
            highest_units = float(df_calc.loc[idx_max, "Units"])
        else:
            highest_product = "N/A"
            highest_units = 0.0

        # 4. Group data by Category and calculate total Units for each Category
        category_units = df_calc.groupby("Category", as_index=False)["Units"].sum()
        category_units = category_units.sort_values(
            by="Units", ascending=False
        ).reset_index(drop=True)

        # 5. Create new calculated column: Inventory Value = Units * Unit Cost
        df_calc["Inventory Value"] = df_calc["Units"] * df_calc["Unit Cost"]

        # 6. Calculate total Inventory Value across all products
        total_inventory_val = (
            0.0
            if pd.isna(df_calc["Inventory Value"].sum())
            else float(df_calc["Inventory Value"].sum())
        )

        metrics = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "column_names": col_names,
            "total_units": total_units,
            "avg_unit_cost": avg_unit_cost,
            "highest_product": highest_product,
            "highest_units": highest_units,
            "category_units": category_units,
            "total_inventory_val": total_inventory_val,
            "raw_dataframe": df,
        }
        return df_calc, metrics, None
    except Exception as e:
        return None, None, f"Error processing CSV: {str(e)}"


def process_and_classify_inventory_csv(
    file_or_df,
    sales_cutoffs: tuple = (0.60, 0.80, 0.90),
    hits_cutoffs: tuple = (0.60, 0.80, 0.90),
    w_sales: float = 0.65,
    w_hits: float = 0.35,
):
    """
    Process an uploaded inventory CSV file using pandas and perform
    multi-criteria inventory classification using Sales ($) and Hits.

    Returns
    -------
    tuple of (classified_df, result_object, error_message)
    """
    try:
        if isinstance(file_or_df, pd.DataFrame):
            df = file_or_df.copy()
        else:
            if hasattr(file_or_df, "seek"):
                file_or_df.seek(0)
            df = pd.read_csv(file_or_df)
            if hasattr(file_or_df, "seek"):
                file_or_df.seek(0)

        df.columns = df.columns.astype(str).str.strip()

        # Check for required classification columns
        cols_lower = [c.lower() for c in df.columns]
        has_sales = any("sales" in c or "revenue" in c for c in cols_lower)
        has_hits = any("hit" in c or "freq" in c for c in cols_lower)

        if not has_sales or not has_hits:
            missing = []
            if not has_sales:
                missing.append("Sales ($)")
            if not has_hits:
                missing.append("Hits")
            return (
                None,
                None,
                f"Missing required column(s) for classification: {', '.join(missing)}. "
                "The inventory file must contain: Item Number, Item Name, UoM, Sales ($), and Hits.",
            )

        result = calculate_multicriteria_classification(
            df=df,
            sales_cutoffs=sales_cutoffs,
            hits_cutoffs=hits_cutoffs,
            w_sales=w_sales,
            w_hits=w_hits,
        )
        return result.classified_dataframe, result, None
    except Exception as e:
        return None, None, f"Error performing inventory classification: {str(e)}"


def render_portfolio_valuation_display(file, preloaded_df=None):
    """Render the existing Portfolio Inventory Valuation and Category Breakdown."""
    st.markdown("## 📊 CSV Inventory Portfolio Valuation")
    st.caption(
        "Automated dataset processing, schema validation, and inventory valuation powered by pandas."
    )

    df_analyzed, metrics, error_msg = process_and_analyze_inventory_csv(file)
    if error_msg:
        st.error(f"❌ {error_msg}")
        return

    # Display: number of rows, number of columns, column names
    st.markdown("### 📋 Dataset Metadata & Dimensions")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Total Rows", f"{metrics['num_rows']:,}")
    c2.metric("Total Columns", f"{metrics['num_cols']}")
    c3.markdown(f"**Column Names:** `{', '.join(metrics['column_names'])}`")

    # Display the uploaded dataframe in the main Streamlit page
    st.markdown("### 📄 Uploaded Raw Dataframe")
    st.dataframe(metrics["raw_dataframe"], use_container_width=True)

    # Display Key Analysis Metrics
    st.markdown("### 🎯 Key Inventory Summary Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        render_kpi_card(
            "Total Units",
            f"{metrics['total_units']:,.0f}",
            "Aggregate inventory volume",
            "input",
        ),
        unsafe_allow_html=True,
    )
    k2.markdown(
        render_kpi_card(
            "Average Unit Cost",
            f"${metrics['avg_unit_cost']:,.2f}",
            "Mean product cost",
            "financial",
        ),
        unsafe_allow_html=True,
    )
    k3.markdown(
        render_kpi_card(
            "Highest Volume Product",
            f"{metrics['highest_product']}",
            f"Volume: {metrics['highest_units']:,.0f} units",
            "output",
        ),
        unsafe_allow_html=True,
    )
    k4.markdown(
        render_kpi_card(
            "Total Inventory Value",
            f"${metrics['total_inventory_val']:,.2f}",
            "Total portfolio capitalization",
            "service",
        ),
        unsafe_allow_html=True,
    )

    # Display Total Units by Category
    st.markdown("### 🏷️ Total Units by Category")
    col_cat1, col_cat2 = st.columns([1, 1])
    with col_cat1:
        st.dataframe(
            metrics["category_units"].style.format({"Units": "{:,.0f}"}),
            use_container_width=True,
        )
    with col_cat2:
        chart_data = metrics["category_units"].set_index("Category")
        st.bar_chart(chart_data["Units"])

    # Display Dataframe including the new Inventory Value column with currency formatting
    st.markdown("### 📦 Detailed Inventory Table (with Calculated Inventory Value)")
    st.caption("Calculated column: Inventory Value = Units × Unit Cost")
    st.dataframe(
        df_analyzed.style.format(
            {"Units": "{:,.0f}", "Unit Cost": "${:,.2f}", "Inventory Value": "${:,.2f}"}
        ),
        use_container_width=True,
    )


def _sync_slider_to_num(slider_key: str, num_key: str) -> None:
    """Callback to synchronize slider adjustments to numeric input."""
    st.session_state[num_key] = st.session_state[slider_key]


def _sync_num_to_slider(num_key: str, slider_key: str) -> None:
    """Callback to synchronize direct numeric entries to slider."""
    st.session_state[slider_key] = st.session_state[num_key]


def _paired_parameter_input(
    label: str,
    default_val: float,
    min_val: float,
    max_val: float,
    step: float,
    key_prefix: str,
    help_text: Optional[str] = None,
) -> float:
    """Render a synchronized slider and direct number input side-by-side with 0.1 step increments."""
    s_key = f"{key_prefix}_slider"
    n_key = f"{key_prefix}_num"
    if s_key not in st.session_state:
        st.session_state[s_key] = float(default_val)
    if n_key not in st.session_state:
        st.session_state[n_key] = float(default_val)

    st.markdown(f"**{label}**")
    col_s, col_n = st.columns([3, 1])
    with col_s:
        st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=s_key,
            on_change=_sync_slider_to_num,
            args=(s_key, n_key),
            help=help_text,
            format="%.1f",
            label_visibility="collapsed",
        )
    with col_n:
        st.number_input(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=n_key,
            on_change=_sync_num_to_slider,
            args=(n_key, s_key),
            help=f"Direct number entry for {label}",
            format="%.1f",
            label_visibility="collapsed",
        )
    return float(st.session_state[s_key])


def render_multicriteria_classification_section(file_or_df, preloaded_df=None):
    """Render the Multi-Criteria Inventory Classification section in Streamlit."""
    st.markdown("## 🏷️ Multicriteria Inventory Classification (Sales & POS Hits)")
    st.caption(
        "BAN 517 — Inventory Analytics Agent Project: Multicriteria inventory stratification combining financial magnitude (Sales) "
        "and operational transaction velocity (POS Hits) using the instructor methodology (65% Sales / 35% POS Hits)."
    )

    # Cutoff tuning expander with direct entry and 0.1 increments
    with st.expander("⚙️ Classification Parameters, Pareto Cutoffs & Weights (Optional)", expanded=False):
        st.markdown("**Criteria Weighting Allocation (Direct Entry & 0.1 Increments):**")
        col_w_ctrl, col_w_card = st.columns([3, 1])
        with col_w_ctrl:
            w_sales_pct = _paired_parameter_input(
                "Sales Weight (%)",
                default_val=65.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                key_prefix="mc_w_sales",
                help_text="Weight applied to Sales Rank Score (Default: 65.0%). Hits Weight is automatically (100% - Sales Weight).",
            )
        with col_w_card:
            w_sales = round(w_sales_pct / 100.0, 3)
            w_hits = round(1.0 - w_sales, 3)
            st.metric("Criteria Weights", f"Sales: {w_sales*100:.1f}%", f"Hits: {w_hits*100:.1f}%")

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown("#### 💰 Sales ($) Pareto Cutoffs (%)")
            a_sales_pct = _paired_parameter_input(
                "Rank A Sales Threshold (%)",
                default_val=60.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                key_prefix="mc_a_sales",
                help_text="Cumulative sales percentage threshold for Rank A (default: 60.0%)",
            )
            b_sales_pct = _paired_parameter_input(
                "Rank B Sales Threshold (%)",
                default_val=80.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                key_prefix="mc_b_sales",
                help_text="Cumulative sales percentage threshold for Rank B (default: 80.0%)",
            )
            c_sales_pct = _paired_parameter_input(
                "Rank C Sales Threshold (%)",
                default_val=90.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                key_prefix="mc_c_sales",
                help_text="Cumulative sales percentage threshold for Rank C (default: 90.0%)",
            )

        with c_p2:
            st.markdown("#### 🎯 Hits (Frequency) Pareto Cutoffs (%)")
            a_hits_pct = _paired_parameter_input(
                "Rank A Hits Threshold (%)",
                default_val=60.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                key_prefix="mc_a_hits",
                help_text="Cumulative hits percentage threshold for Rank A (default: 60.0%)",
            )
            b_hits_pct = _paired_parameter_input(
                "Rank B Hits Threshold (%)",
                default_val=80.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                key_prefix="mc_b_hits",
                help_text="Cumulative hits percentage threshold for Rank B (default: 80.0%)",
            )
            c_hits_pct = _paired_parameter_input(
                "Rank C Hits Threshold (%)",
                default_val=90.0,
                min_val=0.0,
                max_val=100.0,
                step=0.1,
                key_prefix="mc_c_hits",
                help_text="Cumulative hits percentage threshold for Rank C (default: 90.0%)",
            )

        # Cutoff validation
        sales_valid = 0.0 < a_sales_pct < b_sales_pct < c_sales_pct <= 100.0
        hits_valid = 0.0 < a_hits_pct < b_hits_pct < c_hits_pct <= 100.0

        if not sales_valid:
            st.warning(
                f"⚠️ **Sales Cutoffs Out of Order**: Cutoffs must satisfy 0% < Rank A ({a_sales_pct:.1f}%) < Rank B ({b_sales_pct:.1f}%) < Rank C ({c_sales_pct:.1f}%) ≤ 100%. Reverting to baseline (60.0% / 80.0% / 90.0%) for calculation."
            )
            a_sales, b_sales, c_sales = 0.60, 0.80, 0.90
        else:
            a_sales, b_sales, c_sales = a_sales_pct / 100.0, b_sales_pct / 100.0, c_sales_pct / 100.0

        if not hits_valid:
            st.warning(
                f"⚠️ **Hits Cutoffs Out of Order**: Cutoffs must satisfy 0% < Rank A ({a_hits_pct:.1f}%) < Rank B ({b_hits_pct:.1f}%) < Rank C ({c_hits_pct:.1f}%) ≤ 100%. Reverting to baseline (60.0% / 80.0% / 90.0%) for calculation."
            )
            a_hits, b_hits, c_hits = 0.60, 0.80, 0.90
        else:
            a_hits, b_hits, c_hits = a_hits_pct / 100.0, b_hits_pct / 100.0, c_hits_pct / 100.0

        col_rst, _ = st.columns([2, 3])
        with col_rst:
            if st.button("🔄 Reset Parameters to Instructor Defaults (65% Sales / 35% Hits & 60/80/90)", use_container_width=True):
                st.session_state["mc_w_sales_slider"] = 65.0
                st.session_state["mc_w_sales_num"] = 65.0
                st.session_state["mc_a_sales_slider"] = 60.0
                st.session_state["mc_a_sales_num"] = 60.0
                st.session_state["mc_b_sales_slider"] = 80.0
                st.session_state["mc_b_sales_num"] = 80.0
                st.session_state["mc_c_sales_slider"] = 90.0
                st.session_state["mc_c_sales_num"] = 90.0
                st.session_state["mc_a_hits_slider"] = 60.0
                st.session_state["mc_a_hits_num"] = 60.0
                st.session_state["mc_b_hits_slider"] = 80.0
                st.session_state["mc_b_hits_num"] = 80.0
                st.session_state["mc_c_hits_slider"] = 90.0
                st.session_state["mc_c_hits_num"] = 90.0
                st.rerun()

    sales_cutoffs = (a_sales, b_sales, c_sales)
    hits_cutoffs = (a_hits, b_hits, c_hits)

    target_input = preloaded_df if preloaded_df is not None else file_or_df
    df_classified, result, error_msg = process_and_classify_inventory_csv(
        target_input,
        sales_cutoffs=sales_cutoffs,
        hits_cutoffs=hits_cutoffs,
        w_sales=w_sales,
        w_hits=w_hits,
    )

    if error_msg:
        st.error(f"❌ {error_msg}")
        return

    # Metadata & Dimensions
    st.markdown("### 📋 Dataset Metadata & Dimensions")
    m1, m2, m3 = st.columns([1, 1, 2])
    m1.metric("Total Items / SKUs", f"{result.total_items:,}")
    m2.metric("Total Attributes", f"{len(df_classified.columns)}")
    m3.markdown(f"**Detected Columns:** `{', '.join(df_classified.columns)}`")

    # Executive Summary KPIs & Required Classification Summary
    st.markdown("### 📊 Required Classification Summary (Final Categories A, B, C, D)")
    st.caption(
        "Instructor Methodology (BAN 517): Number of items in each final category and summarized total Sales for each category "
        "[Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)]:"
    )
    col_sum_a, col_sum_b, col_sum_c, col_sum_d = st.columns(4)
    with col_sum_a:
        cnt_a = result.class_counts.get("A", 0)
        pct_a = result.class_percentages.get("A", 0.0)
        sales_a = result.class_sales.get("A", 0.0)
        spct_a = result.class_sales_pct.get("A", 0.0)
        st.markdown(
            render_kpi_card(
                "Category A (4.00 - 3.25)",
                f"{cnt_a:,} items ({pct_a:.1f}%)",
                f"Sales: ${sales_a:,.2f} ({spct_a:.1f}%)",
                "output",
            ),
            unsafe_allow_html=True,
        )
    with col_sum_b:
        cnt_b = result.class_counts.get("B", 0)
        pct_b = result.class_percentages.get("B", 0.0)
        sales_b = result.class_sales.get("B", 0.0)
        spct_b = result.class_sales_pct.get("B", 0.0)
        st.markdown(
            render_kpi_card(
                "Category B (3.24 - 2.50)",
                f"{cnt_b:,} items ({pct_b:.1f}%)",
                f"Sales: ${sales_b:,.2f} ({spct_b:.1f}%)",
                "service",
            ),
            unsafe_allow_html=True,
        )
    with col_sum_c:
        cnt_c = result.class_counts.get("C", 0)
        pct_c = result.class_percentages.get("C", 0.0)
        sales_c = result.class_sales.get("C", 0.0)
        spct_c = result.class_sales_pct.get("C", 0.0)
        st.markdown(
            render_kpi_card(
                "Category C (2.49 - 1.75)",
                f"{cnt_c:,} items ({pct_c:.1f}%)",
                f"Sales: ${sales_c:,.2f} ({spct_c:.1f}%)",
                "financial",
            ),
            unsafe_allow_html=True,
        )
    with col_sum_d:
        cnt_d = result.class_counts.get("D", 0)
        pct_d = result.class_percentages.get("D", 0.0)
        sales_d = result.class_sales.get("D", 0.0)
        spct_d = result.class_sales_pct.get("D", 0.0)
        st.markdown(
            render_kpi_card(
                "Category D (< 1.75)",
                f"{cnt_d:,} items ({pct_d:.1f}%)",
                f"Sales: ${sales_d:,.2f} ({spct_d:.1f}%)",
                "input",
            ),
            unsafe_allow_html=True,
        )

    # Portfolio Totals Banner
    st.markdown(
        f"<div style='background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.5rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #334155;'>"
        f"💰 <b>Total Portfolio Sales:</b> ${result.total_sales:,.2f} &nbsp;|&nbsp; "
        f"🎯 <b>Total POS Hits:</b> {result.total_hits:,.0f} &nbsp;|&nbsp; "
        f"📦 <b>Total SKUs:</b> {result.total_items:,} &nbsp;|&nbsp; "
        f"⚖️ <b>Current Weights:</b> {w_sales*100:.1f}% Sales / {w_hits*100:.1f}% POS Hits"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Final Weighted Score Ranges & Increment Table Section
    st.markdown("### 🎯 Final Inventory Classification Allocation Table")
    st.caption(
        f"Combined Weighted Score Range: **[{result.weighted_score_min:.2f}, {result.weighted_score_max:.2f}]** "
        f"| Span: **{result.weighted_score_range:.2f}** | Increments: **4.00-3.25 (A), 3.24-2.50 (B), 2.49-1.75 (C), < 1.75 (D)**"
    )

    c_tab1, c_tab2 = st.columns([3, 2])
    with c_tab1:
        st.markdown("**Classification Categories & Performance Summary:**")
        display_summary_cols = [
            "Final Inventory Class",
            "Combined Weighted Score",
            "SKU Count",
            "SKU %",
            "Total Sales ($)",
            "Sales %",
            "Total Hits",
            "Hits %",
        ]
        sum_view = result.range_summary[display_summary_cols] if all(c in result.range_summary.columns for c in display_summary_cols) else result.range_summary
        st.dataframe(
            sum_view.style.format(
                {
                    "SKU Count": "{:,.0f}",
                    "SKU %": "{:,.1f}%",
                    "Total Sales ($)": "${:,.2f}",
                    "Sales %": "{:,.2f}%",
                    "Total Hits": "{:,.0f}",
                    "Hits %": "{:,.2f}%",
                }
            ),
            use_container_width=True,
        )

    with c_tab2:
        st.markdown("**SKU Distribution across Final Classes:**")
        chart_df = result.range_summary.set_index("Final Inventory Class") if "Final Inventory Class" in result.range_summary.columns else result.range_summary.set_index("Final Rank")
        st.bar_chart(chart_df["SKU Count"])

    # 4x4 Cross-Tabulation Matrix
    st.markdown("### 🔀 4×4 Cross-Tabulation Matrix (Sales Rank vs POS Hits Rank)")
    st.caption(
        "Distribution of inventory items across Sales Rank (rows) and POS Hits Rank (columns):"
    )
    col_mat1, col_mat2 = st.columns([1, 1])

    with col_mat1:
        st.dataframe(
            result.matrix_crosstab.style.set_properties(
                **{"text-align": "center", "font-weight": "bold"}
            ),
            use_container_width=True,
        )

    with col_mat2:
        st.markdown(
            """
            **Instructor Methodology (BAN 517):**
            - **Step 1 - Pareto Thresholds**: A (0–60%), B (61–80%), C (81–90%), D (91–100%).
            - **Step 1 - Numerical Scores**: Class A = 4, Class B = 3, Class C = 2, Class D = 1.
            - **Step 2 - Combined Weighted Score**: $0.65(\\text{Sales Score}) + 0.35(\\text{Hits Score})$.
            - **Step 3 - Final Inventory Class Thresholds**:
              - 🌟 **Class A**: Combined Weighted Score $4.00 - 3.25$
              - 🛡️ **Class B**: Combined Weighted Score $3.24 - 2.50$
              - 📦 **Class C**: Combined Weighted Score $2.49 - 1.75$
              - ⚠️ **Class D**: Combined Weighted Score $< 1.75$
            """
        )

    # Detailed Inventory Table with Calculated Results
    st.markdown(
        "### 📦 Detailed Inventory Classification Table (with Calculated Ranks & Scores)"
    )
    st.caption(
        "Columns: Sales Cumulative Total, Sales Cumulative %, Sales Rank, Sales Rank Score, "
        "Hits Cumulative Total, Hits Cumulative %, Hits Rank, Hits Rank Score, Combined Weighted Score, Final Inventory Class."
    )

    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        class_filter = st.selectbox(
            "Filter Table by Final Rank:",
            ["All Ranks", "Rank A", "Rank B", "Rank C", "Rank D"],
        )

    filtered_df = df_classified.copy()
    if class_filter != "All Ranks":
        target_cls = class_filter.replace("Rank ", "")
        filtered_df = filtered_df[filtered_df["Final Rank"] == target_cls]

    s_col_name = result.sales_column_used
    h_col_name = result.hits_column_used

    # Preferred presentation order of columns
    preferred_cols = [
        "Item Number",
        "Item Name",
        "UoM",
        s_col_name,
        h_col_name,
        "Sales Cumulative Total",
        "Sales Cumulative %",
        "Sales Rank",
        "Sales Rank Score",
        "Hits Cumulative Total",
        "Hits Cumulative %",
        "Hits Rank",
        "Hits Rank Score",
        "Combined Weighted Score",
        "Weighted Score",
        "Final Inventory Class",
        "Final Rank",
    ]
    actual_display_cols = [c for c in preferred_cols if c in filtered_df.columns]
    # Include any remaining columns not explicitly in preferred list
    remaining_cols = [c for c in filtered_df.columns if c not in actual_display_cols]
    final_table_df = filtered_df[actual_display_cols + remaining_cols]

    format_dict = {
        s_col_name: "${:,.2f}",
        h_col_name: "{:,.0f}",
        "Sales Cumulative Total": "${:,.2f}",
        "Sales Cumulative %": "{:,.2f}%",
        "Sales %": "{:,.2f}%",
        "Cum Sales %": "{:,.2f}%",
        "Hits Cumulative Total": "{:,.0f}",
        "Hits Cumulative %": "{:,.2f}%",
        "Hits %": "{:,.2f}%",
        "Cum Hits %": "{:,.2f}%",
        "Combined Weighted Score": "{:,.2f}",
        "Weighted Score": "{:,.2f}",
    }

    st.dataframe(
        final_table_df.style.format(format_dict, na_rep="N/A"),
        use_container_width=True,
    )

    # Download classified CSV
    csv_bytes = df_classified.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Classified Inventory CSV",
        data=csv_bytes,
        file_name="classified_inventory_results.csv",
        mime="text/csv",
        help="Export the complete inventory dataset with all ranks, scores, and weighted classification results.",
    )

    # Verification Checks & Assumptions
    with st.expander(
        "🎓 Multi-Criteria Audit Verification & Governing Assumptions",
        expanded=False,
    ):
        st.markdown("**Automated Audit Checks Passed:**")
        for chk in result.verification_checks:
            st.markdown(f"- ✅ {chk}")
        st.markdown("---")
        st.markdown("**Governing Supply Chain Assumptions:**")
        for asm in result.assumptions:
            st.markdown(f"- 📋 {asm}")


def render_csv_data_analysis_section(file):
    """
    Render the CSV Data Upload and Inventory Analysis section in the main page.
    Automatically detects whether the file corresponds to:
    1. Multi-Criteria Inventory Classification (Item Number, Item Name, UoM, Sales ($), Hits)
    2. Portfolio Inventory Valuation (Product, Category, Units, Unit Cost)
    3. Both (renders tabs for both analyses)
    """
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file)
        if hasattr(file, "seek"):
            file.seek(0)
    except Exception as e:
        st.error(f"❌ Error reading uploaded CSV file: {str(e)}")
        return

    cols_clean = [c.strip().lower() for c in df.columns]
    has_sales = any("sales" in c or "revenue" in c for c in cols_clean)
    has_hits = any("hit" in c or "freq" in c for c in cols_clean)
    is_classification = has_sales and has_hits

    has_units = "units" in cols_clean
    has_cost = any("unit cost" in c for c in cols_clean)
    is_portfolio = has_units and has_cost

    if is_classification and is_portfolio:
        tab_cls, tab_port = st.tabs(
            [
                "🏷️ Multi-Criteria Classification (Sales & Hits)",
                "📊 Portfolio Valuation (Units & Cost)",
            ]
        )
        with tab_cls:
            render_multicriteria_classification_section(file, preloaded_df=df)
        with tab_port:
            render_portfolio_valuation_display(file, preloaded_df=df)
    elif is_classification:
        render_multicriteria_classification_section(file, preloaded_df=df)
    elif is_portfolio:
        render_portfolio_valuation_display(file, preloaded_df=df)
    else:
        st.error(
            "❌ The uploaded CSV file could not be recognized as a supported inventory dataset.\n\n"
            "Supported formats:\n"
            "1. **Inventory Classification**: Must contain `Item Number`, `Item Name`, `UoM`, `Sales ($)`, and `Hits`.\n"
            "2. **Portfolio Valuation**: Must contain `Product`, `Category`, `Units`, and `Unit Cost`."
        )


# ---------------------------------------------------------------------------
# Render Persistent Dashboard & 7 Architectural Steps
# ---------------------------------------------------------------------------

# Render persistent dashboard directly below the prompt
render_persistent_variable_dashboard(pipeline_response)

# Render CSV Data Analysis if a file has been uploaded via the sidebar
if uploaded_file is not None:
    st.markdown("---")
    render_csv_data_analysis_section(uploaded_file)

st.markdown("---")

res = pipeline_response.raw_calculation_result
has_calc_error = "error" in res

tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🎯 2. Identify Model",
        "🔍 3. Extract Inputs",
        "⚙️ 4. Python Tool",
        "📊 5. Calculate & Viz",
        "🛡️ 6. Verify & Audit",
        "💡 7. Executive Decision",
    ]
)

# ---------------------------------------------------------------------------
# Tab 2: Identify Models
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("#### 🎯 Architectural Step 2: Model Identification")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.info(f"**Identified Model**: {pipeline_response.identified_model}")
        st.write(
            f"**Detection Rationale**: {pipeline_response.identification_rationale}"
        )
    with c2:
        conf_pct = int(pipeline_response.model_confidence * 100)
        st.metric("Identification Confidence", f"{conf_pct}%")
        st.progress(conf_pct / 100.0)

    # Student Verification Guide — Item 8: Assumptions
    st.markdown("---")
    st.markdown(
        "##### 🎓 Student Verification Guide — Item 8: Governing Model Assumptions & Prerequisites"
    )
    st.caption(
        "Theoretical boundaries and operational assumptions under which this inventory model is mathematically valid:"
    )
    assumptions_list = (
        res.get("assumptions", []) or pipeline_response.explanation.assumptions
    )
    if assumptions_list:
        for asm in assumptions_list:
            st.markdown(f"- 📋 {asm}")
    else:
        st.info("Model assumptions will display upon model selection.")

# ---------------------------------------------------------------------------
# Tab 3: Extract Inputs
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("#### 🔍 Architectural Step 3: Semantic Input Extraction")
    st.caption(
        "Parameters and theoretical inventory variables parsed from the natural-language question."
    )

    # Student Verification Guide — Item 1: Inputs Extracted
    st.markdown("##### 🎓 Student Verification Guide — Item 1: Inputs Extracted")
    inputs = res.get("inputs_extracted") or pipeline_response.extracted_inputs
    conv_note = inputs.get(
        "demand_conversion_note"
    ) or pipeline_response.extracted_inputs.get("demand_conversion_note")
    if conv_note:
        st.info(f"🔄 **Demand Normalization Applied**: {conv_note}")
    if inputs:
        items = [
            {"Parameter": k, "Extracted Value": str(v)}
            for k, v in inputs.items()
            if k != "tiers"
        ]
        df_inputs = pd.DataFrame(items)
        st.dataframe(df_inputs, use_container_width=True)

        if "tiers" in inputs and inputs["tiers"]:
            st.markdown("**Extracted Discount Tiers Table:**")
            df_tiers = pd.DataFrame(inputs["tiers"])
            st.dataframe(df_tiers, use_container_width=True)
    else:
        st.warning(
            "No parameters were automatically parsed. Defaults or explicit inputs are required."
        )

    if pipeline_response.input_extraction_notes:
        with st.expander("ℹ️ Semantic Extraction Parsing Notes", expanded=False):
            for note in pipeline_response.input_extraction_notes:
                st.markdown(f"- {note}")

    # Student Verification Guide — Item 2: Variable Definitions
    st.markdown("---")
    st.markdown("##### 🎓 Student Verification Guide — Item 2: Variable Definitions")
    st.caption(
        "Formal academic definitions and notation for all parameters in this inventory problem:"
    )
    var_defs = (
        res.get("variable_definitions", {})
        or pipeline_response.explanation.variable_definitions
    )
    if var_defs:
        df_vars = pd.DataFrame(
            [
                {"Variable Symbol": k, "Formal Definition": v}
                for k, v in var_defs.items()
            ]
        )
        st.dataframe(df_vars, use_container_width=True)
    else:
        st.info("Variable definitions will display once the model is executed.")

# ---------------------------------------------------------------------------
# Tab 4: Select Python Tool
# ---------------------------------------------------------------------------
with tab4:
    st.markdown(
        "#### ⚙️ Architectural Step 4: Python Tool Selection & Formula Specification"
    )
    st.success(
        f"Selected Deterministic Tool: `{pipeline_response.tool_module}.{pipeline_response.selected_tool_name}()`"
    )
    st.markdown("""
    All numerical computation is routed strictly through deterministic Python functions
    in `inventory_tools.py` using NumPy and SciPy. LLMs never calculate or reconstruct numerical results independently.
    """)

    # Student Verification Guide — Item 3: Formula
    st.markdown("---")
    st.markdown(
        "##### 🎓 Student Verification Guide — Item 3: Theoretical Governing Formula(s)"
    )
    st.caption(
        "The algebraic mathematical formulation selected and executed by the Python tool:"
    )
    formula_latex = (
        res.get("formula", "") or pipeline_response.explanation.mathematical_formula
    )
    if formula_latex:
        st.latex(formula_latex)
    else:
        st.info("Theoretical formula will display upon model selection.")

# ---------------------------------------------------------------------------
# Tab 5: Calculate & Visualization
# ---------------------------------------------------------------------------
with tab5:
    st.markdown(
        "#### 📊 Architectural Step 5: Deterministic Calculation & Visual Analytics"
    )

    if has_calc_error:
        st.error(f"Calculation Error: {res['error']}")
        st.warning(
            "Please provide all required numerical inputs in the question or select a preset example."
        )
    else:
        # Display Key Metrics
        m_key = pipeline_response.model_key

        if m_key == "eoq":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Optimal EOQ (Q*)", f"{res.get('eoq_rounded', 0):,} units")
            c2.metric("Orders per Year", f"{res.get('orders_per_year', 0.0):.1f}")
            cycle_disp = (
                f"{res['cycle_time_days']:.1f} days"
                if res.get("cycle_time_days") is not None
                else "N/A"
            )
            c3.metric("Cycle Time", cycle_disp)
            c4.metric("Total Annual Cost", f"${res.get('total_annual_cost', 0.0):,.2f}")

            st.pyplot(plot_eoq_curve(res))

        elif m_key == "quantity_discount":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Optimal Order Q", f"{res['optimal_order_quantity_rounded']:,} units"
            )
            c2.metric("Optimal Unit Price", f"${res['optimal_unit_price']:.2f}")
            c3.metric("Total Annual Cost", f"${res['optimal_total_cost']:,.2f}")
            c4.metric("Annual Savings", f"${res['annual_savings_vs_base']:,.2f}")

            st.pyplot(plot_discount_comparison(res))

            st.markdown("##### Detailed Tier Comparison Table:")
            df_eval = pd.DataFrame(res["tier_evaluations"])
            st.dataframe(df_eval, use_container_width=True)

        elif m_key == "safety_stock_rop":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reorder Point (ROP)", f"{res['reorder_point_rounded']:,} units")
            c2.metric("Safety Stock Buffer", f"{res['safety_stock_rounded']:,} units")
            c3.metric("Lead Time Demand", f"{res['lead_time_demand_mean']:.1f} units")
            c4.metric("Cycle Service Level", f"{res['service_level'] * 100:.1f}%")

            st.pyplot(plot_rop_normal(res))

        elif m_key == "fixed_period_review":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Order-Up-To Level (M)", f"{res['order_up_to_level_rounded']:,} units"
            )
            c2.metric(
                "Recommended Order Q",
                f"{res['recommended_order_quantity_rounded']:,} units",
            )
            c3.metric("Safety Stock", f"{res['safety_stock_rounded']:,} units")
            c4.metric(
                "Protection Interval",
                f"{res['protection_interval']:.0f} {res['time_unit']}",
            )

            # Horizontal bar visualization of inventory position vs Target M
            fig, ax = plt.subplots(figsize=(7, 2.5))
            m_val = res["order_up_to_level"]
            ip_val = res["current_inventory_position"]
            q_val = res["recommended_order_quantity"]

            ax.barh(
                ["Inventory"],
                [ip_val],
                color="#3B82F6",
                label=f"Current Inventory Position ({ip_val:.0f})",
            )
            ax.barh(
                ["Inventory"],
                [q_val],
                left=[ip_val],
                color="#10B981",
                label=f"Recommended Order ({q_val:.0f})",
            )
            ax.axvline(
                m_val,
                color="#DC2626",
                linestyle="--",
                lw=2,
                label=f"Target M ({m_val:.0f})",
            )

            ax.set_title(
                "Periodic Review: Inventory Position vs Order-Up-To Level (M)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Units")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, axis="x", linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

        elif m_key == "newsvendor":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Optimal Order (Q*)", f"{res['optimal_order_quantity_rounded']:,} units"
            )
            c2.metric(
                "Critical Fractile (CR)", f"{res['critical_fractile'] * 100:.2f}%"
            )
            c3.metric("Expected Profit", f"${res['expected_profit']:,.2f}")
            c4.metric(
                "In-Stock Probability", f"{res['in_stock_probability'] * 100:.2f}%"
            )

            st.pyplot(plot_newsvendor_curve(res))

        # Student Verification Guide — Item 4: Formula with Substituted Values
        st.markdown("---")
        st.markdown(
            "##### 🎓 Student Verification Guide — Item 4: Formula with Substituted Values"
        )
        st.caption(
            "Theoretical formula with the student's exact numerical values substituted into each term:"
        )
        formula_sub = (
            res.get("formula_with_substituted_values")
            or pipeline_response.explanation.formula_with_substituted_values
        )
        if formula_sub:
            st.latex(formula_sub)
        else:
            st.info(
                "Substituted formula is generated when calculation inputs are complete."
            )

        # Student Verification Guide — Item 5: Intermediate Calculations
        st.markdown(
            "##### 🎓 Student Verification Guide — Item 5: Intermediate Calculations"
        )
        st.caption(
            "Step-by-step intermediate numerical values computed deterministically by the Python tool:"
        )
        inter = res.get("intermediate_calculations", {})
        if inter:
            df_inter = pd.DataFrame(
                [
                    {"Intermediate Step / Variable": k, "Deterministic Value": str(v)}
                    for k, v in inter.items()
                    if not isinstance(v, list)
                ]
            )
            st.dataframe(df_inter, use_container_width=True)
        else:
            st.info("Intermediate calculations will display when inputs are valid.")

        # Student Verification Guide — Item 6: Final Answer
        st.markdown("##### 🎓 Student Verification Guide — Item 6: Final Answer")
        st.caption("Structured final decision metrics computed by the Python tool:")
        final = res.get("final_answer", {})
        if final:
            df_final = pd.DataFrame(
                [
                    {"Output Decision Metric": k, "Calculated Result": str(v)}
                    for k, v in final.items()
                ]
            )
            st.dataframe(df_final, use_container_width=True)
        else:
            st.info("Final answer summary will display when inputs are valid.")

# ---------------------------------------------------------------------------
# Tab 6: Verify & Audit Checks
# ---------------------------------------------------------------------------
with tab6:
    st.markdown("#### 🛡️ Architectural Step 6: Mathematical & Domain Verification")
    v = pipeline_response.verification

    status_class = (
        "status-passed"
        if v.status == "PASSED"
        else ("status-warning" if v.status == "WARNING" else "status-failed")
    )
    st.markdown(
        f"**Verification Audit Status:** <span class='{status_class}'>{v.status}</span>",
        unsafe_allow_html=True,
    )

    # Student Verification Guide — Item 7: Verification Checks
    st.markdown(
        "##### 🎓 Student Verification Guide — Item 7: Automated Verification Checks"
    )
    st.caption(
        "Deterministic boundary, invariant, and sanity checks executed to verify mathematical and domain validity:"
    )

    seen_checks = set()
    all_checks = []
    tool_checks = (
        res.get("verification_checks", [])
        or pipeline_response.explanation.verification_checks
    )
    for chk in tool_checks:
        if chk not in seen_checks:
            seen_checks.add(chk)
            all_checks.append(chk)
    for chk in v.checks_passed:
        if chk not in seen_checks:
            seen_checks.add(chk)
            all_checks.append(chk)

    if all_checks:
        for chk in all_checks:
            st.markdown(f"- ✅ {chk}")
    else:
        st.info("No verification checks recorded.")

    if v.warnings:
        st.markdown("##### ⚠️ Warnings & Edge Conditions:")
        for w in v.warnings:
            st.warning(w)

    if v.errors:
        st.markdown("##### ❌ Verification Violations:")
        for err in v.errors:
            st.error(err)

# ---------------------------------------------------------------------------
# Tab 7: Explain (Managerial Decision)
# ---------------------------------------------------------------------------
with tab7:
    st.markdown("#### 💡 Architectural Step 7: Executive Explanation & Insights")
    exp = pipeline_response.explanation

    # Executive Summary Box
    st.info(exp.executive_summary)

    # Cost / Risk Breakdown
    if exp.cost_or_risk_breakdown:
        st.markdown("##### 💰 Operational & Cost Breakdown:")
        col_b1, col_b2 = st.columns(2)
        half = len(exp.cost_or_risk_breakdown) // 2
        items = list(exp.cost_or_risk_breakdown.items())

        with col_b1:
            for k, val in items[:half]:
                st.markdown(f"**{k}:** `{val}`")
        with col_b2:
            for k, val in items[half:]:
                st.markdown(f"**{k}:** `{val}`")

    # Managerial Insights
    if exp.managerial_insights:
        st.markdown("##### 📈 Strategic & Managerial Supply Chain Insights:")
        for ins in exp.managerial_insights:
            st.markdown(f"- {ins}")

    st.markdown("---")
    st.caption(
        "ℹ️ **Student Verification Note**: Mathematical derivations, variable definitions, "
        "formula substitutions, intermediate calculations, and sanity checks are integrated directly "
        "into the relevant architectural tabs (Steps 2 through 6 above)."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "UNCW Masters Program — BAN 517 Supply Chain Analytics | "
    "Inventory Decision Agent v1.4.0 | "
    "Architecture: 1. Question -> 2. Identify -> 3. Extract -> 4. Tool -> 5. Calculate -> 6. Verify -> 7. Explain"
)

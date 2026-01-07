#!/usr/bin/env python3
"""
TrueYield Marketing Budget Dashboard V2
=======================================
Enhanced interactive dashboard with:
- Strategic scenarios (Futures, Affiliates, No Giveaways)
- Advanced breakeven analysis
- Marketing budget designer
- What-if modeling

Run with: python3 -m streamlit run dashboard_v2.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TrueYield Budget Model V2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff5252; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# BASELINE DATA (From all sources)
# =============================================================================

# From Stripe + GA4 + Google Ads (Aug-Dec 2025)
HISTORICAL = {
    "period": "Aug-Dec 2025",
    "total_revenue": 65000,
    "total_purchasers": 289,
    "first_time_purchasers": 275,
    "aov": 225,
    "revenue_by_channel": {
        "Organic + Direct": {"revenue": 42000, "purchasers": 188, "cac": 0},  # Combined organic & direct (free traffic)
        "Referral/Affiliates": {"revenue": 11000, "purchasers": 65, "cac": 56},
        "Email/SMS": {"revenue": 5800, "purchasers": 20, "cac": 5},
        "Paid Ads (Google)": {"revenue": 14500, "purchasers": 108, "cac": 222},  # Real Google Ads data
    },
    "paid_ads": {
        "spend": 23900,  # Total Google Ads spend
        "revenue": 14500,  # sGTM Purchase value
        "purchases": 107.5,  # sGTM Purchases
        "clicks": 9660,
        "impressions": 83387,
        "ctr": 3.52,
        "cpc_search": 2.45,
        "cpc_cross": 2.52,
        "roas": 0.61,  # 14500 / 23900
        "cpa": 222,  # 23900 / 107.5
        "campaigns": {
            "Search High Intent Keywords": {"cost": 6465, "purchases": 36, "revenue": 3919, "roas": 0.61},
            "Performance Max GER": {"cost": 5551, "purchases": 33.62, "revenue": 4761, "roas": 0.86},
            "Valiant Search High Intent": {"cost": 4048, "purchases": 4.5, "revenue": 516, "roas": 0.13},
            "Traffic Search Brand Europe": {"cost": 2624, "purchases": 26.68, "revenue": 4438, "roas": 1.69},
            "Performance Max ENG": {"cost": 2072, "purchases": 3.7, "revenue": 726, "roas": 0.35},
        },
        "demographics": {
            "top_gender": "Male",
            "top_age": "25-44",
        },
        "devices": {
            "mobile": 57.6,
            "desktop": 41.1,
            "tablet": 1.3,
        },
        "top_keywords": [
            {"keyword": "the prop trading", "cost": 2792, "revenue": 2400, "ctr": 3.79},
            {"keyword": "traders yard", "cost": 1469, "revenue": 221, "ctr": 4.19},
            {"keyword": "Propfirm", "cost": 1057, "revenue": 228, "ctr": 2.58},
            {"keyword": "prop trading firm", "cost": 1000, "revenue": 625, "ctr": 3.0},
        ],
    }
}

# Current State (Jan 2026)
CURRENT = {
    "bank_balance": 1_000_000,
    "monthly_burn": 100_000,
    "team_cost": 80_000,
    "server_infra": 20_000,
    "ad_spend": 12_000,
    "dec_revenue": 25_000,
    "dec_payouts": 50_000,
    "payout_ratio": 2.0,  # 200%
    "payout_ratio_excl_giveaways": 1.0,  # 100%
}

# From Excel Model - Payout Tiers
PAYOUT_TIERS = [
    {"lower_bound": 0, "payout_pct": 0.30},
    {"lower_bound": 5000, "payout_pct": 0.40},
    {"lower_bound": 20000, "payout_pct": 0.50},
    {"lower_bound": 50000, "payout_pct": 0.60},
]

# Product Data (from GA4)
PRODUCTS = {
    "2phase_swing_5k": {"price": 50, "units": 119, "type": "swing"},
    "2phase_swing_50k": {"price": 250, "units": 81, "type": "swing"},
    "2phase_swing_100k": {"price": 400, "units": 71, "type": "swing"},
    "2phase_swing_10k": {"price": 100, "units": 55, "type": "swing"},
    "2phase_swing_25k": {"price": 150, "units": 55, "type": "swing"},
    "tc_standard": {"price": 50, "units": 41, "type": "tc"},
    "tc_advanced": {"price": 100, "units": 31, "type": "tc"},
    "tc_premium": {"price": 200, "units": 15, "type": "tc"},
    "tc_elite": {"price": 300, "units": 12, "type": "tc"},
    "cfd_1phase_5k": {"price": 75, "units": 13, "type": "cfd"},
    "cfd_1phase_10k": {"price": 125, "units": 10, "type": "cfd"},
    "cfd_1phase_25k": {"price": 175, "units": 8, "type": "cfd"},
    "cfd_1phase_50k": {"price": 275, "units": 12, "type": "cfd"},
    "cfd_1phase_100k": {"price": 450, "units": 13, "type": "cfd"},
}

# Marketing Budget Template (from Excel)
MARKETING_CHANNELS = [
    "Google Ads",
    "Meta Ads",
    "TikTok Ads",
    "X Ads",
    "LinkedIn Ads",
    "Influencer Marketing",
    "Affiliate Costs",
    "Giveaway Payouts",
    "Marketing Team",
    "Marketing Tools",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(amount: float, short: bool = True) -> str:
    """Format number as EUR currency."""
    if short:
        if abs(amount) >= 1_000_000:
            return f"€{amount/1_000_000:.2f}M"
        elif abs(amount) >= 1_000:
            return f"€{amount/1_000:.1f}k"
        else:
            return f"€{amount:.0f}"
    else:
        return f"€{amount:,.0f}"


def get_payout_rate(revenue: float) -> float:
    """Get payout rate based on revenue tier."""
    rate = 0.30
    for tier in PAYOUT_TIERS:
        if revenue >= tier["lower_bound"]:
            rate = tier["payout_pct"]
    return rate


def run_monthly_projection(
    months: int,
    starting_revenue: float,
    growth_rate: float,
    payout_ratio: float,
    marketing_ratio: float,
    fixed_costs: float,
    funding: float = 0,
    funding_month: int = 1,
    giveaway_ratio: float = 0,  # Additional giveaway cost as % of revenue
    affiliate_boost: float = 0,  # Additional revenue from affiliate expansion
    futures_revenue: float = 0,  # Additional revenue from futures product
) -> pd.DataFrame:
    """Run detailed monthly projection with all factors."""

    bank = CURRENT["bank_balance"]
    data = []
    revenue = starting_revenue

    # Calculate contribution margin for breakeven
    contribution_margin = 1 - payout_ratio - marketing_ratio - giveaway_ratio
    breakeven_revenue = fixed_costs / contribution_margin if contribution_margin > 0 else float('inf')

    for month in range(1, months + 1):
        # Add funding if this is the funding month
        if month == funding_month:
            bank += funding

        # Calculate additional revenue sources
        affiliate_rev = revenue * affiliate_boost
        futures_rev = futures_revenue * (month / 6) if month <= 6 else futures_revenue  # Ramp up
        total_revenue = revenue + affiliate_rev + futures_rev

        # Calculate costs
        base_payouts = total_revenue * payout_ratio
        giveaway_cost = total_revenue * giveaway_ratio
        total_payouts = base_payouts + giveaway_cost

        marketing = total_revenue * marketing_ratio
        total_costs = fixed_costs + total_payouts + marketing

        # Net cashflow
        net_cf = total_revenue - total_costs
        bank += net_cf

        # Effective payout ratio
        effective_pr = total_payouts / total_revenue if total_revenue > 0 else 0

        # Check if this month hits breakeven (revenue covers all costs)
        is_breakeven = total_revenue >= breakeven_revenue
        is_profitable = net_cf >= 0

        data.append({
            "Month": month,
            "Base Revenue": revenue,
            "Affiliate Revenue": affiliate_rev,
            "Futures Revenue": futures_rev if month <= 6 else futures_revenue,
            "Total Revenue": total_revenue,
            "Base Payouts": base_payouts,
            "Giveaway Cost": giveaway_cost,
            "Total Payouts": total_payouts,
            "Payout Ratio": effective_pr,
            "Marketing": marketing,
            "Fixed Costs": fixed_costs,
            "Total Costs": total_costs,
            "Net Cashflow": net_cf,
            "Bank Balance": bank,
            "Breakeven": is_breakeven,
            "Profitable": is_profitable,
        })

        # Grow base revenue
        revenue *= (1 + growth_rate)

    return pd.DataFrame(data)


def calculate_breakeven(fixed_costs: float, payout_ratio: float, marketing_ratio: float) -> dict:
    """Calculate breakeven with detailed breakdown."""
    contribution_margin = 1 - payout_ratio - marketing_ratio

    if contribution_margin <= 0:
        return {
            "possible": False,
            "revenue": float('inf'),
            "contribution_margin": contribution_margin,
            "message": "Impossible - costs exceed 100% of revenue"
        }

    breakeven_rev = fixed_costs / contribution_margin

    return {
        "possible": True,
        "revenue": breakeven_rev,
        "contribution_margin": contribution_margin,
        "payouts_at_be": breakeven_rev * payout_ratio,
        "marketing_at_be": breakeven_rev * marketing_ratio,
        "message": f"Need {format_currency(breakeven_rev)}/month to break even"
    }


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("📊 TrueYield Budget V2")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📈 Executive Dashboard",
        "👥 Customer Growth Projections",
        "💰 Funding Scenarios",
        "🎯 Strategic Scenarios",
        "📊 Marketing Budget Designer",
        "🎮 Ad Budget Simulator",
        "🔄 Breakeven Playground",
        "📋 Channel Deep Dive",
        "💾 Export Center"
    ]
)

st.sidebar.markdown("---")

# Quick Stats
st.sidebar.markdown("### Current State")
col1, col2 = st.sidebar.columns(2)
col1.metric("Bank", format_currency(CURRENT["bank_balance"]))
col2.metric("Burn", format_currency(CURRENT["monthly_burn"]))

col1, col2 = st.sidebar.columns(2)
col1.metric("Dec Rev", format_currency(CURRENT["dec_revenue"]))
col2.metric("PR", "200%")

# =============================================================================
# PAGE: EXECUTIVE DASHBOARD
# =============================================================================

if page == "📈 Executive Dashboard":
    st.title("TrueYield Executive Dashboard")
    st.markdown("### Financial Overview - January 2026")

    # Hero metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    runway = CURRENT["bank_balance"] / (CURRENT["monthly_burn"] - CURRENT["dec_revenue"])

    col1.metric("Bank Balance", format_currency(CURRENT["bank_balance"]))
    col2.metric("Monthly Burn", format_currency(CURRENT["monthly_burn"]))
    col3.metric("Runway", f"{runway:.0f} months")
    col4.metric("Dec Revenue", format_currency(CURRENT["dec_revenue"]))
    col5.metric("Payout Ratio", "200%", delta="-150% to target", delta_color="inverse")

    st.markdown("---")

    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Targets"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Revenue by Channel (Aug-Dec 2025)")

            channel_data = pd.DataFrame([
                {"Channel": k, "Revenue": v["revenue"], "CAC": v["cac"]}
                for k, v in HISTORICAL["revenue_by_channel"].items()
            ])

            fig = px.pie(
                channel_data, values="Revenue", names="Channel",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.info("💡 **64% of revenue is FREE** (Direct + Organic)")

        with col2:
            st.markdown("#### Cost Structure")

            cost_data = pd.DataFrame([
                {"Category": "Team", "Amount": CURRENT["team_cost"]},
                {"Category": "Server/Infra", "Amount": CURRENT["server_infra"]},
                {"Category": "Ads (Current)", "Amount": CURRENT["ad_spend"]},
                {"Category": "Payouts (Dec)", "Amount": CURRENT["dec_payouts"]},
            ])

            fig = px.bar(
                cost_data, x="Category", y="Amount",
                color="Category",
                text_auto='.2s'
            )
            fig.update_layout(height=350, showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.warning("⚠️ **Payouts = 200% of revenue** - unsustainable")

    with tab2:
        st.markdown("#### Product Sales Distribution (Aug-Dec 2025)")

        product_df = pd.DataFrame([
            {"Product": k, "Units": v["units"], "Price": v["price"],
             "Revenue": v["units"] * v["price"], "Type": v["type"]}
            for k, v in PRODUCTS.items()
        ]).sort_values("Revenue", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                product_df.head(10), x="Revenue", y="Product",
                orientation='h', color="Type",
                color_discrete_map={"swing": "#2E86AB", "tc": "#F18F01", "cfd": "#A23B72"}
            )
            fig.update_layout(height=400, margin=dict(l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Revenue by product type
            type_rev = product_df.groupby("Type")["Revenue"].sum().reset_index()
            fig = px.pie(type_rev, values="Revenue", names="Type",
                        color_discrete_map={"swing": "#2E86AB", "tc": "#F18F01", "cfd": "#A23B72"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Path to Targets")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Revenue Targets")
            st.markdown("""
            | Target | Status |
            |--------|--------|
            | €50k/mo | 🔴 Need 2x |
            | €100k/mo | 🔴 Need 4x |
            | €150k/mo | 🔴 Need 6x |
            """)

        with col2:
            st.markdown("##### Payout Ratio Targets")
            st.markdown("""
            | Target | Status |
            |--------|--------|
            | 100% | 🔴 Current (excl. giveaways) |
            | 50% | 🎯 Goal |
            | 35% | ⭐ FPFX Benchmark |
            """)

        with col3:
            st.markdown("##### Key Milestones")
            st.markdown("""
            | Date | Event |
            |------|-------|
            | Jan 12 | WooCommerce + Futures |
            | Mar '26 | Ingmar Meeting (Florida) |
            | Q4 '26 | Target Breakeven |
            """)

# =============================================================================
# PAGE: CUSTOMER GROWTH PROJECTIONS
# =============================================================================

elif page == "👥 Customer Growth Projections":
    st.title("Customer Growth Projections")
    st.markdown("### More Customers = More Revenue = Path to Breakeven")

    st.info("""
    **The Core Insight:** Fixed costs are locked at €100k/month (lean team, can't reduce).
    The ONLY path to breakeven is **growing revenue through more customers**.
    """)

    # Current state
    AOV = 225

    st.markdown("---")

    # All controls in expander
    with st.expander("⚙️ Scenario Parameters", expanded=True):
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

        with ctrl_col1:
            starting_revenue_cg = st.slider(
                "Starting Monthly Revenue (€)",
                min_value=10000,
                max_value=100000,
                value=25000,
                step=5000,
                format="€%d",
                key="cg_starting_revenue",
                help="Current monthly revenue"
            )
            CURRENT_CUSTOMERS = starting_revenue_cg / AOV

        with ctrl_col2:
            growth_rate = st.slider(
                "Monthly Growth Rate",
                min_value=5,
                max_value=40,
                value=20,
                step=5,
                format="%d%%",
                key="cg_growth_rate",
                help="Expected monthly customer/revenue growth"
            ) / 100  # Convert to decimal

        with ctrl_col3:
            payout_ratio = st.slider(
                "Target Payout Ratio",
                min_value=30,
                max_value=80,
                value=50,
                step=5,
                format="%d%%",
                key="cg_payout_ratio",
                help="Payout ratio target"
            ) / 100  # Convert to decimal

        ctrl_col4, ctrl_col5, ctrl_col6 = st.columns(3)

        with ctrl_col4:
            futures_boost = st.slider(
                "Futures Launch Impact",
                min_value=0,
                max_value=30,
                value=15,
                step=5,
                format="%d%%",
                key="cg_futures",
                help="Extra customers from Futures product"
            ) / 100  # Convert to decimal

        with ctrl_col5:
            affiliate_boost = st.slider(
                "Affiliate Expansion",
                min_value=0,
                max_value=40,
                value=20,
                step=5,
                format="%d%%",
                key="cg_affiliate",
                help="Extra customers from scaled affiliates"
            ) / 100  # Convert to decimal

        with ctrl_col6:
            marketing_pct = st.slider(
                "Marketing Spend %",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                format="%d%%",
                key="cg_marketing",
                help="Marketing as % of revenue"
            ) / 100  # Convert to decimal

    # Display current state metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Starting Customers/Month", f"{CURRENT_CUSTOMERS:.0f}")
    col2.metric("AOV", f"€{AOV}")
    col3.metric("Starting Revenue", format_currency(starting_revenue_cg))
    col4.metric("Fixed Costs", format_currency(100000), help="Cannot be reduced - lean team")

    # Calculate projections
    months = 24
    data = []
    customers = CURRENT_CUSTOMERS
    bank = CURRENT["bank_balance"]

    for month in range(1, months + 1):
        if month > 1:
            customers *= (1 + growth_rate)

        # Apply boosters (ramp up)
        futures_mult = min(month / 6, 1) * futures_boost if futures_boost > 0 else 0
        affiliate_mult = min(month / 3, 1) * affiliate_boost if affiliate_boost > 0 else 0

        total_customers = customers * (1 + futures_mult + affiliate_mult)
        revenue = total_customers * AOV

        # Costs
        payouts = revenue * payout_ratio
        marketing = revenue * marketing_pct
        fixed = 100000
        total_costs = payouts + marketing + fixed

        net = revenue - total_costs
        bank += net

        data.append({
            "Month": month,
            "Customers": int(total_customers),
            "Revenue": revenue,
            "Payouts": payouts,
            "Net P/L": net,
            "Bank Balance": bank,
            "Breakeven": net >= 0
        })

    proj_df = pd.DataFrame(data)

    # Find breakeven month
    at_breakeven = proj_df[proj_df["Breakeven"]]
    breakeven_month = at_breakeven["Month"].min() if len(at_breakeven) > 0 else None

    # Chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=proj_df["Month"], y=proj_df["Customers"],
               name="Customers", marker_color="#2E86AB", opacity=0.7),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=proj_df["Month"], y=proj_df["Revenue"],
                   name="Revenue", line=dict(color="#F18F01", width=3)),
        secondary_y=True
    )

    # Target lines
    fig.add_hline(y=100000, line_dash="dash", line_color="green",
                  annotation_text="€100k target", secondary_y=True)
    fig.add_hline(y=150000, line_dash="dash", line_color="blue",
                  annotation_text="€150k stretch", secondary_y=True)

    if breakeven_month:
        fig.add_vline(x=breakeven_month, line_dash="dot", line_color="green",
                      annotation_text=f"Breakeven M{breakeven_month}")

    fig.update_layout(
        title=f"Customer & Revenue Growth ({growth_rate*100:.0f}%/month)",
        height=400,
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="Revenue (€)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Key metrics
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    m12 = proj_df[proj_df["Month"] == 12].iloc[0]
    m24 = proj_df[proj_df["Month"] == 24].iloc[0]
    min_bank = proj_df["Bank Balance"].min()

    res_col1.metric(
        "Breakeven Month",
        f"Month {breakeven_month}" if breakeven_month else "Not in 24 months",
        delta="✓ Achievable" if breakeven_month and breakeven_month <= 18 else "Needs more growth"
    )
    res_col2.metric("Month 12 Revenue", format_currency(m12["Revenue"]))
    res_col3.metric("Month 12 Customers", f"{m12['Customers']:,}")
    res_col4.metric("Minimum Bank Balance", format_currency(min_bank),
                    delta="Need funding" if min_bank < 0 else "OK")

    # Detailed table
    st.markdown("### Month-by-Month Breakdown")

    display_df = proj_df.copy()
    display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"€{x:,.0f}")
    display_df["Payouts"] = display_df["Payouts"].apply(lambda x: f"€{x:,.0f}")
    display_df["Net P/L"] = display_df["Net P/L"].apply(lambda x: f"€{x:,.0f}")
    display_df["Bank Balance"] = display_df["Bank Balance"].apply(lambda x: f"€{x:,.0f}")
    display_df["Customers"] = display_df["Customers"].apply(lambda x: f"{x:,}")
    display_df["Breakeven"] = display_df["Breakeven"].apply(lambda x: "✅ Yes" if x else "❌ No")

    st.dataframe(
        display_df[["Month", "Customers", "Revenue", "Payouts", "Net P/L", "Bank Balance", "Breakeven"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Breakeven comparison matrix
    st.markdown("### Breakeven Matrix: Growth Rate vs Payout Ratio")

    growth_rates = [0.15, 0.20, 0.25, 0.30]
    payout_ratios = [0.35, 0.45, 0.50, 0.60, 0.75]

    matrix_data = []
    for pr in payout_ratios:
        row = {"Payout Ratio": f"{pr*100:.0f}%"}
        for gr in growth_rates:
            # Quick calculation
            customers = CURRENT_CUSTOMERS
            bank = CURRENT["bank_balance"]
            be_month = None

            for m in range(1, 37):
                customers *= (1 + gr)
                revenue = customers * AOV
                net = revenue * (1 - pr - 0.15) - 100000
                bank += net
                if net >= 0 and be_month is None:
                    be_month = m
                    break

            row[f"{gr*100:.0f}%"] = f"M{be_month}" if be_month else "Never"
        matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    # Key insights
    st.markdown("### Key Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.success(f"""
        **At {growth_rate*100:.0f}% growth + {payout_ratio*100:.0f}% payout:**

        - **Breakeven:** {"Month " + str(breakeven_month) if breakeven_month else "Not achieved"}
        - **Month 12 Customers:** {m12['Customers']:,}
        - **Month 12 Revenue:** {format_currency(m12['Revenue'])}
        - **Year 1 Total Revenue:** {format_currency(proj_df[proj_df['Month'] <= 12]['Revenue'].sum())}
        """)

    with insight_col2:
        # Calculate impact of boosters
        if futures_boost > 0 or affiliate_boost > 0:
            st.info(f"""
            **Growth Boosters Impact:**

            - **Futures (+{futures_boost*100:.0f}% customers):** Adds ~{format_currency(m12['Revenue'] * futures_boost)} revenue/mo by M12
            - **Affiliates (+{affiliate_boost*100:.0f}% customers):** Adds ~{format_currency(m12['Revenue'] * affiliate_boost)} revenue/mo by M12
            - **Combined:** Accelerates breakeven by ~2-4 months
            """)
        else:
            st.warning("""
            **Enable Growth Boosters:**

            - Futures launch can add 15-20% more customers
            - Affiliate expansion can add 20-30% more customers
            - Combined effect: Breakeven 2-4 months sooner
            """)

# =============================================================================
# PAGE: FUNDING SCENARIOS
# =============================================================================

elif page == "💰 Funding Scenarios":
    st.title("Funding Scenarios")
    st.markdown("### Compare Different Funding Levels")

    # Controls
    with st.expander("⚙️ Scenario Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            starting_revenue = st.slider(
                "Starting Monthly Revenue (€)",
                min_value=10000,
                max_value=100000,
                value=25000,
                step=5000,
                format="€%d",
                key="funding_revenue",
                help="Current monthly revenue to start projections from"
            )
        with col2:
            growth_rate = st.slider(
                "Monthly Growth Rate",
                min_value=5,
                max_value=40,
                value=20,
                step=1,
                format="%d%%",
                key="funding_growth",
                help="Expected monthly revenue growth rate"
            ) / 100  # Convert to decimal
        with col3:
            payout_ratio = st.slider(
                "Payout Ratio",
                min_value=30,
                max_value=80,
                value=50,
                step=5,
                format="%d%%",
                key="funding_payout",
                help="Target payout ratio (% of revenue paid out)"
            ) / 100  # Convert to decimal

        col4, col5, col6 = st.columns(3)
        with col4:
            marketing_ratio = st.slider(
                "Marketing Spend %",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                format="%d%%",
                key="funding_marketing",
                help="Marketing spend as % of revenue"
            ) / 100  # Convert to decimal
        with col5:
            fixed_costs = st.number_input(
                "Fixed Costs (€/month)",
                min_value=50000,
                max_value=200000,
                value=100000,
                step=10000,
                key="funding_fixed",
                help="Monthly fixed costs (team, infrastructure)"
            )
        with col6:
            projection_months = st.slider(
                "Projection Period (months)",
                min_value=6,
                max_value=24,
                value=12,
                step=3,
                key="funding_months",
                help="How many months to project"
            )

    # Run all scenarios
    scenarios = {
        "No Funding": 0,
        "€1M": 1_000_000,
        "€2M": 2_000_000,
        "€5M": 5_000_000,
    }

    all_projections = {}
    summary_rows = []

    for name, funding in scenarios.items():
        df = run_monthly_projection(
            months=projection_months,
            starting_revenue=starting_revenue,
            growth_rate=growth_rate,
            payout_ratio=payout_ratio,
            marketing_ratio=marketing_ratio,
            fixed_costs=fixed_costs,
            funding=funding,
        )
        # Rename "Profitable" to "Breakeven" for clarity
        df["Breakeven"] = df["Profitable"]
        all_projections[name] = df

        breakeven_month = df[df["Breakeven"]]["Month"].min() if df["Breakeven"].any() else "Never"

        summary_rows.append({
            "Scenario": name,
            "Final Balance": df["Bank Balance"].iloc[-1],
            "Min Balance": df["Bank Balance"].min(),
            "Year Revenue": df["Total Revenue"].sum(),
            "Breakeven Month": breakeven_month,
        })

    # Summary cards
    cols = st.columns(4)
    for i, row in enumerate(summary_rows):
        with cols[i]:
            st.markdown(f"### {row['Scenario']}")

            balance_color = "green" if row["Final Balance"] > 500000 else ("orange" if row["Final Balance"] > 0 else "red")
            st.markdown(f"**Final Balance:** <span style='color:{balance_color}'>{format_currency(row['Final Balance'])}</span>", unsafe_allow_html=True)
            st.markdown(f"**Min Balance:** {format_currency(row['Min Balance'])}")
            st.markdown(f"**Total Revenue:** {format_currency(row['Year Revenue'])}")
            st.markdown(f"**Breakeven:** Month {row['Breakeven Month']}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bank Balance Over Time")

        fig = go.Figure()
        colors = {"No Funding": "#ff5252", "€1M": "#FFA726", "€2M": "#66BB6A", "€5M": "#42A5F5"}

        for name, df in all_projections.items():
            fig.add_trace(go.Scatter(
                x=df["Month"], y=df["Bank Balance"],
                name=name, line=dict(color=colors[name], width=3),
                mode='lines+markers'
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Bankruptcy")
        fig.add_hline(y=300000, line_dash="dot", line_color="orange", annotation_text="3-month buffer")
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Monthly Revenue Trajectory")

        fig = go.Figure()
        # All scenarios have same revenue trajectory
        df = all_projections["€2M"]

        fig.add_trace(go.Scatter(
            x=df["Month"], y=df["Total Revenue"],
            name="Revenue", line=dict(color="#2E86AB", width=3),
            fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.2)'
        ))

        fig.add_hline(y=100000, line_dash="dash", line_color="green", annotation_text="Target: €100k")
        fig.add_hline(y=150000, line_dash="dash", line_color="blue", annotation_text="Stretch: €150k")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed table with scenario selector
    st.markdown("### Monthly Details")

    selected_scenario = st.selectbox(
        "Select scenario to view details:",
        list(all_projections.keys()),
        index=2,  # Default to €2M
        key="funding_scenario_select"
    )

    df_display = all_projections[selected_scenario].copy()
    for col in ["Total Revenue", "Total Payouts", "Marketing", "Net Cashflow", "Bank Balance"]:
        df_display[col] = df_display[col].apply(lambda x: format_currency(x, short=False))
    df_display["Payout Ratio"] = df_display["Payout Ratio"].apply(lambda x: f"{x*100:.0f}%")
    df_display["Breakeven"] = df_display["Breakeven"].apply(lambda x: "✅ Yes" if x else "❌ No")

    st.dataframe(
        df_display[["Month", "Total Revenue", "Total Payouts", "Payout Ratio", "Marketing", "Net Cashflow", "Bank Balance", "Breakeven"]],
        use_container_width=True, hide_index=True
    )

    # Show current parameters
    st.info(f"""
    **Current Parameters:** Starting Revenue: €{starting_revenue:,} | Growth: {growth_rate*100:.0f}%/mo | Payout: {payout_ratio*100:.0f}% | Marketing: {marketing_ratio*100:.0f}% | Fixed Costs: €{fixed_costs:,}/mo
    """)

# =============================================================================
# PAGE: STRATEGIC SCENARIOS
# =============================================================================

elif page == "🎯 Strategic Scenarios":
    st.title("Strategic Scenarios")
    st.markdown("### What-If Analysis: Business Strategy Changes")

    st.info("""
    Explore how different strategic decisions impact your financials:
    - **Futures Launch** (Jan 12): New product line
    - **Affiliate Expansion**: Scale referral program
    - **No Giveaways**: Reduce promotional payouts
    """)

    # Global controls for all strategic scenarios
    with st.expander("⚙️ Base Scenario Parameters", expanded=True):
        base_col1, base_col2, base_col3, base_col4 = st.columns(4)

        with base_col1:
            ss_starting_revenue = st.slider(
                "Starting Revenue (€)",
                min_value=10000,
                max_value=100000,
                value=25000,
                step=5000,
                format="€%d",
                key="ss_revenue",
                help="Current monthly revenue"
            )
        with base_col2:
            ss_growth_rate = st.slider(
                "Growth Rate",
                min_value=5,
                max_value=40,
                value=20,
                step=5,
                format="%d%%",
                key="ss_growth"
            ) / 100
        with base_col3:
            ss_payout_ratio = st.slider(
                "Payout Ratio",
                min_value=30,
                max_value=80,
                value=50,
                step=5,
                format="%d%%",
                key="ss_payout"
            ) / 100
        with base_col4:
            ss_funding = st.selectbox(
                "Funding Scenario",
                options=[0, 1_000_000, 2_000_000, 5_000_000],
                index=2,
                format_func=lambda x: "No Funding" if x == 0 else f"€{x/1_000_000:.0f}M",
                key="ss_funding"
            )

    # Base assumptions using slider values
    base_params = {
        "growth_rate": ss_growth_rate,
        "payout_ratio": ss_payout_ratio,
        "marketing_ratio": 0.15,
        "fixed_costs": 100000,
        "funding": ss_funding,
    }

    # Scenario selector
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Futures Launch",
        "🤝 Affiliate Expansion",
        "🎁 No Giveaways",
        "🔥 Combined Strategy"
    ])

    with tab1:
        st.markdown("### Futures Product Launch Impact")
        st.markdown("*Launching alongside WooCommerce on January 12, 2026*")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Parameters")
            futures_monthly_rev = st.slider(
                "Futures Monthly Revenue (€)",
                min_value=5000,
                max_value=50000,
                value=15000,
                step=1000,
                format="€%d",
                key="futures_rev",
                help="Expected monthly revenue from futures trading challenges"
            )
            futures_margin = st.slider(
                "Futures Profit Margin",
                min_value=30,
                max_value=80,
                value=60,
                step=5,
                format="%d%%",
                key="futures_margin",
                help="Futures typically have better margins than swing"
            ) / 100

            st.markdown("#### Assumptions")
            st.markdown(f"""
            - 6-month ramp-up period
            - Full revenue: **{format_currency(futures_monthly_rev)}/mo**
            - Better margin: **{futures_margin*100:.0f}%** profit
            - Attracts new customer segment
            """)

        with col2:
            # Run with and without futures
            df_base = run_monthly_projection(
                months=12,
                starting_revenue=ss_starting_revenue,
                **base_params
            )

            df_futures = run_monthly_projection(
                months=12,
                starting_revenue=ss_starting_revenue,
                futures_revenue=futures_monthly_rev,
                **base_params
            )

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Revenue", "Bank Balance"])

            fig.add_trace(go.Scatter(x=df_base["Month"], y=df_base["Total Revenue"],
                                     name="Without Futures", line=dict(color="#888", dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_futures["Month"], y=df_futures["Total Revenue"],
                                     name="With Futures", line=dict(color="#2E86AB", width=3)), row=1, col=1)

            fig.add_trace(go.Scatter(x=df_base["Month"], y=df_base["Bank Balance"],
                                     name="Without Futures", line=dict(color="#888", dash="dash"), showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=df_futures["Month"], y=df_futures["Bank Balance"],
                                     name="With Futures", line=dict(color="#66BB6A", width=3), showlegend=False), row=1, col=2)

            fig.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            # Impact summary
            rev_diff = df_futures["Total Revenue"].sum() - df_base["Total Revenue"].sum()
            balance_diff = df_futures["Bank Balance"].iloc[-1] - df_base["Bank Balance"].iloc[-1]

            col_a, col_b = st.columns(2)
            col_a.metric("Additional Year Revenue", format_currency(rev_diff), delta=f"+{rev_diff/df_base['Total Revenue'].sum()*100:.1f}%")
            col_b.metric("Better Final Balance", format_currency(balance_diff), delta="positive")

    with tab2:
        st.markdown("### Affiliate Program Expansion")
        st.markdown("*Scale referral revenue from current 17% to higher share*")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Parameters")
            affiliate_boost = st.slider(
                "Affiliate Revenue Boost",
                min_value=0,
                max_value=50,
                value=20,
                step=5,
                format="%d%%",
                key="aff_boost",
                help="Additional revenue as % of base from expanded affiliates"
            ) / 100
            affiliate_cost = st.slider(
                "Affiliate Commission Rate",
                min_value=10,
                max_value=30,
                value=20,
                step=5,
                format="%d%%",
                key="aff_cost",
                help="Commission paid to affiliates"
            ) / 100

            st.markdown("#### Current State")
            st.markdown(f"""
            - Referral revenue: **€11k** (17% of total)
            - Estimated CAC: **€56** (vs €222 for ads)
            - High LTV customers
            """)

            st.success(f"""
            **With {affiliate_boost*100:.0f}% boost:**
            - Extra monthly revenue: ~{format_currency(ss_starting_revenue * affiliate_boost)}
            - Commission cost: {format_currency(ss_starting_revenue * affiliate_boost * affiliate_cost)}
            - Net gain: {format_currency(ss_starting_revenue * affiliate_boost * (1 - affiliate_cost))}
            """)

        with col2:
            df_affiliate = run_monthly_projection(
                months=12,
                starting_revenue=ss_starting_revenue,
                affiliate_boost=affiliate_boost,
                marketing_ratio=base_params["marketing_ratio"] + (affiliate_boost * affiliate_cost),  # Include commission
                **{k: v for k, v in base_params.items() if k != "marketing_ratio"}
            )

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=df_affiliate["Month"],
                y=df_affiliate["Base Revenue"],
                name="Base Revenue",
                marker_color="#2E86AB"
            ))
            fig.add_trace(go.Bar(
                x=df_affiliate["Month"],
                y=df_affiliate["Affiliate Revenue"],
                name="Affiliate Revenue",
                marker_color="#F18F01"
            ))

            fig.update_layout(barmode='stack', height=400, title="Revenue Composition")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Eliminate Giveaways")
        st.markdown("*What if we stopped promotional giveaways?*")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Current Giveaway Impact")

            # CEO said giveaways account for difference between 200% and 100% payout ratio
            giveaway_cost_pct = st.slider(
                "Giveaway Cost (% of revenue)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                format="%d%%",
                key="giveaway_pct",
                help="Based on CEO: 200% total - 100% base = 100% giveaways"
            ) / 100

            growth_impact = st.slider(
                "Growth Impact (without giveaways)",
                min_value=-10,
                max_value=0,
                value=-5,
                step=1,
                format="%d%%",
                key="growth_impact",
                help="How much slower might growth be without giveaways?"
            ) / 100

            st.warning(f"""
            **Trade-off Analysis:**
            - Giveaway savings: **{format_currency(ss_starting_revenue * giveaway_cost_pct)}/mo**
            - Potential growth slowdown: **{growth_impact*100:.0f}%**
            - Net decision: Is acquisition cost worth it?
            """)

        with col2:
            # With giveaways (current)
            df_with_giveaways = run_monthly_projection(
                months=12,
                starting_revenue=ss_starting_revenue,
                giveaway_ratio=giveaway_cost_pct,
                **base_params
            )

            # Without giveaways (slower growth but lower costs)
            df_no_giveaways = run_monthly_projection(
                months=12,
                starting_revenue=ss_starting_revenue,
                growth_rate=base_params["growth_rate"] + growth_impact,
                giveaway_ratio=0,
                **{k: v for k, v in base_params.items() if k != "growth_rate"}
            )

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Payout Ratio", "Bank Balance"])

            fig.add_trace(go.Scatter(
                x=df_with_giveaways["Month"], y=df_with_giveaways["Payout Ratio"],
                name="With Giveaways", line=dict(color="#ff5252")
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_no_giveaways["Month"], y=df_no_giveaways["Payout Ratio"],
                name="No Giveaways", line=dict(color="#66BB6A")
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df_with_giveaways["Month"], y=df_with_giveaways["Bank Balance"],
                name="With Giveaways", line=dict(color="#ff5252"), showlegend=False
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=df_no_giveaways["Month"], y=df_no_giveaways["Bank Balance"],
                name="No Giveaways", line=dict(color="#66BB6A"), showlegend=False
            ), row=1, col=2)

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Summary
            savings = df_with_giveaways["Giveaway Cost"].sum()
            st.success(f"**Annual Savings from No Giveaways:** {format_currency(savings)}")

    with tab4:
        st.markdown("### Combined Strategy")
        st.markdown("*Futures + Affiliate Expansion + Reduced Giveaways*")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Combined Parameters")

            comb_futures = st.slider(
                "Futures Revenue (€/mo)",
                min_value=0,
                max_value=50000,
                value=15000,
                step=5000,
                format="€%d",
                key="comb_futures"
            )
            comb_affiliate = st.slider(
                "Affiliate Boost",
                min_value=0,
                max_value=50,
                value=15,
                step=5,
                format="%d%%",
                key="comb_aff"
            ) / 100
            comb_giveaway = st.slider(
                "Giveaway Reduction",
                min_value=0,
                max_value=50,
                value=25,
                step=5,
                format="%d%%",
                key="comb_give"
            ) / 100
            comb_payout = st.slider(
                "Target Payout Ratio",
                min_value=30,
                max_value=70,
                value=45,
                step=5,
                format="%d%%",
                key="comb_pr"
            ) / 100

        with col2:
            # Base case
            df_base = run_monthly_projection(
                months=12,
                starting_revenue=ss_starting_revenue,
                giveaway_ratio=0.50,  # Current
                **base_params
            )

            # Combined strategy
            df_combined = run_monthly_projection(
                months=12,
                starting_revenue=ss_starting_revenue,
                growth_rate=ss_growth_rate + 0.02,  # Slightly higher with better product mix
                payout_ratio=comb_payout,
                marketing_ratio=0.18,  # Higher for affiliate commissions
                fixed_costs=100000,
                funding=ss_funding,
                giveaway_ratio=0.50 - comb_giveaway,
                affiliate_boost=comb_affiliate,
                futures_revenue=comb_futures,
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_base["Month"], y=df_base["Bank Balance"],
                name="Current Strategy", line=dict(color="#888", dash="dash", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_combined["Month"], y=df_combined["Bank Balance"],
                name="Combined Strategy", line=dict(color="#66BB6A", width=4),
                fill='tonexty', fillcolor='rgba(102, 187, 106, 0.2)'
            ))

            fig.update_layout(height=400, title="Bank Balance: Current vs Combined Strategy")
            st.plotly_chart(fig, use_container_width=True)

            # Impact metrics
            col_a, col_b, col_c = st.columns(3)

            rev_impact = df_combined["Total Revenue"].sum() - df_base["Total Revenue"].sum()
            balance_impact = df_combined["Bank Balance"].iloc[-1] - df_base["Bank Balance"].iloc[-1]

            col_a.metric("Revenue Impact", format_currency(rev_impact), delta=f"+{rev_impact/df_base['Total Revenue'].sum()*100:.1f}%")
            col_b.metric("Balance Impact", format_currency(balance_impact))
            col_c.metric("Breakeven",
                        f"Month {df_combined[df_combined['Profitable']]['Month'].min()}" if df_combined["Profitable"].any() else "Never")

# =============================================================================
# PAGE: MARKETING BUDGET DESIGNER
# =============================================================================

elif page == "📊 Marketing Budget Designer":
    st.title("Marketing Budget Designer")
    st.markdown("### Build Your Monthly Marketing Budget")

    # Total budget
    total_budget = st.slider(
        "Total Monthly Marketing Budget",
        10000, 150000, 35000, 5000,
        format="€%d"
    )

    st.markdown("---")

    # Channel allocation with visual editor
    st.markdown("### Channel Allocation")
    st.markdown("*Drag sliders to allocate your budget*")

    # Initialize allocations in session state (using integers 0-100 for percentages)
    if "allocations" not in st.session_state:
        st.session_state.allocations = {
            "Google Ads": 15,
            "Meta Ads": 20,
            "TikTok Ads": 10,
            "X Ads": 5,
            "LinkedIn Ads": 0,
            "Influencer Marketing": 15,
            "Affiliate Costs": 20,
            "Giveaway Payouts": 5,
            "Marketing Team": 5,
            "Marketing Tools": 5,
        }

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        allocations_pct = {}

        # Performance-based channels
        st.markdown("#### 📢 Paid Advertising")
        ad_cols = st.columns(3)

        with ad_cols[0]:
            allocations_pct["Google Ads"] = st.slider(
                "Google Ads", 0, 40, st.session_state.allocations["Google Ads"], 5,
                format="%d%%", key="google"
            )
        with ad_cols[1]:
            allocations_pct["Meta Ads"] = st.slider(
                "Meta Ads", 0, 40, st.session_state.allocations["Meta Ads"], 5,
                format="%d%%", key="meta"
            )
        with ad_cols[2]:
            allocations_pct["TikTok Ads"] = st.slider(
                "TikTok Ads", 0, 30, st.session_state.allocations["TikTok Ads"], 5,
                format="%d%%", key="tiktok"
            )

        ad_cols2 = st.columns(2)
        with ad_cols2[0]:
            allocations_pct["X Ads"] = st.slider(
                "X (Twitter) Ads", 0, 20, st.session_state.allocations["X Ads"], 5,
                format="%d%%", key="x"
            )
        with ad_cols2[1]:
            allocations_pct["LinkedIn Ads"] = st.slider(
                "LinkedIn Ads", 0, 20, st.session_state.allocations["LinkedIn Ads"], 5,
                format="%d%%", key="linkedin"
            )

        st.markdown("#### 🤝 Partnership & Growth")
        growth_cols = st.columns(3)

        with growth_cols[0]:
            allocations_pct["Influencer Marketing"] = st.slider(
                "Influencers", 0, 40, st.session_state.allocations["Influencer Marketing"], 5,
                format="%d%%", key="influencer"
            )
        with growth_cols[1]:
            allocations_pct["Affiliate Costs"] = st.slider(
                "Affiliates", 0, 40, st.session_state.allocations["Affiliate Costs"], 5,
                format="%d%%", key="affiliate"
            )
        with growth_cols[2]:
            allocations_pct["Giveaway Payouts"] = st.slider(
                "Giveaways", 0, 30, st.session_state.allocations["Giveaway Payouts"], 5,
                format="%d%%", key="giveaway"
            )

        st.markdown("#### 🛠️ Operations")
        ops_cols = st.columns(2)

        with ops_cols[0]:
            allocations_pct["Marketing Team"] = st.slider(
                "Marketing Team", 0, 20, st.session_state.allocations["Marketing Team"], 5,
                format="%d%%", key="team"
            )
        with ops_cols[1]:
            allocations_pct["Marketing Tools"] = st.slider(
                "Tools & Software", 0, 15, st.session_state.allocations["Marketing Tools"], 5,
                format="%d%%", key="tools"
            )

        # Convert percentages to decimals for calculations
        allocations = {k: v / 100.0 for k, v in allocations_pct.items()}

    with col2:
        # Summary card
        total_alloc = sum(allocations.values())

        if abs(total_alloc - 1.0) < 0.01:
            st.success(f"✅ **Budget: 100% Allocated**")
        elif total_alloc > 1.0:
            st.error(f"❌ **Over-allocated: {total_alloc*100:.0f}%**")
        else:
            st.warning(f"⚠️ **Under-allocated: {total_alloc*100:.0f}%**")

        st.markdown("---")

        # Budget breakdown
        st.markdown("#### Monthly Breakdown")

        for channel, pct in allocations.items():
            amount = total_budget * pct
            if amount > 0:
                st.markdown(f"**{channel}:** {format_currency(amount)}")

        st.markdown("---")
        st.markdown(f"**TOTAL:** {format_currency(total_budget * total_alloc)}")
        st.markdown(f"**Quarterly:** {format_currency(total_budget * total_alloc * 3)}")
        st.markdown(f"**Annual:** {format_currency(total_budget * total_alloc * 12)}")

    st.markdown("---")

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Budget Distribution")

        budget_df = pd.DataFrame([
            {"Channel": k, "Amount": total_budget * v, "Percentage": v}
            for k, v in allocations.items() if v > 0
        ])

        fig = px.pie(
            budget_df, values="Amount", names="Channel",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Channel Efficiency Comparison")

        efficiency_data = [
            {"Channel": "Affiliates", "CAC": 56, "ARPU": 169, "ROI": 3.0},
            {"Channel": "Email/SMS", "CAC": 5, "ARPU": 290, "ROI": 58.0},
            {"Channel": "Organic", "CAC": 0, "ARPU": 240, "ROI": 999},
            {"Channel": "Paid Ads", "CAC": 222, "ARPU": 112, "ROI": 0.5},
            {"Channel": "Influencer", "CAC": 100, "ARPU": 200, "ROI": 2.0},
        ]

        eff_df = pd.DataFrame(efficiency_data)

        fig = px.bar(
            eff_df[eff_df["ROI"] < 100],  # Exclude organic for scale
            x="Channel", y="ROI",
            color="ROI",
            color_continuous_scale="RdYlGn",
            text_auto='.1f'
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(height=400, title="ROI by Channel")
        st.plotly_chart(fig, use_container_width=True)

    # Monthly budget table
    st.markdown("### 12-Month Marketing Budget")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Create table with growth assumption
    growth_factor = [1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5]

    table_data = []
    for channel, pct in allocations.items():
        if pct > 0:
            row = {"Channel": channel}
            for i, month in enumerate(months):
                row[month] = format_currency(total_budget * pct * growth_factor[i])
            table_data.append(row)

    # Add totals row
    total_row = {"Channel": "**TOTAL**"}
    for i, month in enumerate(months):
        total_row[month] = format_currency(total_budget * total_alloc * growth_factor[i])
    table_data.append(total_row)

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: AD BUDGET SIMULATOR
# =============================================================================

elif page == "🎮 Ad Budget Simulator":
    st.title("Ad Budget Simulator")
    st.markdown("### Play with different ad spend scenarios to see revenue impact")

    st.info("🎮 **Move the sliders to explore how increased Google & Meta ad spend impacts your projected revenue. Not limited to 18-month runway!**")

    # Current state metrics
    st.markdown("---")
    st.markdown("#### 📊 Current State (Baseline)")
    current_cols = st.columns(4)
    with current_cols[0]:
        st.metric("Current Monthly Ads", "€12,000", help="Current combined Google + Meta spend")
    with current_cols[1]:
        st.metric("Current ROAS", "0.61x", "-€0.39 per €1", delta_color="inverse")
    with current_cols[2]:
        st.metric("Current CAC", "€222", help="Customer acquisition cost for paid ads")
    with current_cols[3]:
        st.metric("Monthly Revenue from Ads", "€7,320", help="At 0.61x ROAS")

    st.markdown("---")

    # Interactive sliders
    st.markdown("#### 🎚️ Adjust Your Ad Spend")

    slider_cols = st.columns(4)

    with slider_cols[0]:
        google_ads = st.slider(
            "Monthly Google Ads (€)",
            min_value=5000,
            max_value=50000,
            value=10000,
            step=1000,
            format="€%d",
            help="Adjust monthly Google Ads budget"
        )

    with slider_cols[1]:
        meta_ads = st.slider(
            "Monthly Meta Ads (€)",
            min_value=5000,
            max_value=50000,
            value=10000,
            step=1000,
            format="€%d",
            help="Adjust monthly Meta Ads budget"
        )

    with slider_cols[2]:
        runway_months = st.slider(
            "Campaign Duration (months)",
            min_value=6,
            max_value=36,
            value=12,
            step=3,
            help="How long will you run these campaigns?"
        )

    with slider_cols[3]:
        target_roas = st.slider(
            "Expected ROAS",
            min_value=0.5,
            max_value=4.0,
            value=1.5,
            step=0.1,
            format="%.1fx",
            help="Target return on ad spend after optimization"
        )

    # Calculate projections
    monthly_total = google_ads + meta_ads
    total_investment = monthly_total * runway_months
    projected_revenue = total_investment * target_roas
    net_profit = projected_revenue - total_investment
    profit_percentage = ((projected_revenue / total_investment) - 1) * 100 if total_investment > 0 else 0
    target_cac = 100  # Optimized CAC target
    customers_acquired = int(projected_revenue / target_cac) if target_cac > 0 else 0

    # Comparison with current
    current_monthly = 12000
    percent_increase = ((monthly_total - current_monthly) / current_monthly) * 100 if current_monthly > 0 else 0

    st.markdown("---")

    # Revenue Impact Analysis
    st.markdown("#### 💰 Revenue Impact Analysis")

    impact_cols = st.columns(4)

    with impact_cols[0]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Total Ad Investment</p>
            <h2 style="color: white; margin: 8px 0;">{format_currency(total_investment)}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">{format_currency(monthly_total)}/mo × {runway_months} months</p>
        </div>
        """, unsafe_allow_html=True)

    with impact_cols[1]:
        color = "#10b981" if target_roas >= 1 else "#ef4444"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, {color}99 100%); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Projected Ad Revenue</p>
            <h2 style="color: white; margin: 8px 0;">{format_currency(projected_revenue)}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">At {target_roas:.1f}x ROAS</p>
        </div>
        """, unsafe_allow_html=True)

    with impact_cols[2]:
        profit_color = "#3b82f6" if net_profit >= 0 else "#ef4444"
        profit_sign = "+" if net_profit >= 0 else ""
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {profit_color} 0%, #8b5cf6 100%); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Net Profit from Ads</p>
            <h2 style="color: white; margin: 8px 0;">{profit_sign}{format_currency(abs(net_profit))}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">{profit_sign}{profit_percentage:.0f}% return</p>
        </div>
        """, unsafe_allow_html=True)

    with impact_cols[3]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">New Customers Acquired</p>
            <h2 style="color: white; margin: 8px 0;">~{customers_acquired:,}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">At €{target_cac} target CAC</p>
        </div>
        """, unsafe_allow_html=True)

    # Comparison bar
    st.markdown("---")
    compare_cols = st.columns([1, 0.5, 1, 1])
    with compare_cols[0]:
        st.metric("Current Monthly Ads", "€12,000")
    with compare_cols[1]:
        st.markdown("<h2 style='text-align: center; color: #6b7280;'>→</h2>", unsafe_allow_html=True)
    with compare_cols[2]:
        st.metric("Your Selection", format_currency(monthly_total))
    with compare_cols[3]:
        delta_color = "normal" if percent_increase >= 0 else "inverse"
        st.metric("Change", f"{percent_increase:+.0f}%", delta_color=delta_color)

    st.markdown("---")

    # Revenue vs Spend Chart
    st.markdown("#### 📈 Revenue vs Ad Spend Over Time")

    # Generate monthly data for chart
    months = [f"M{i+1}" for i in range(runway_months)]
    monthly_spend = [monthly_total] * runway_months

    # Revenue grows as ads optimize (starts at 0.61x, grows to target ROAS)
    monthly_revenue = []
    cumulative_revenue = []
    cumulative_spend = []
    running_revenue = 0
    running_spend = 0

    for month in range(runway_months):
        current_roas_improvement = 0.61 + ((target_roas - 0.61) * ((month + 1) / runway_months))
        month_revenue = monthly_total * current_roas_improvement
        monthly_revenue.append(month_revenue)

        running_revenue += month_revenue
        running_spend += monthly_total
        cumulative_revenue.append(running_revenue)
        cumulative_spend.append(running_spend)

    # Create chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Monthly bars
    fig.add_trace(
        go.Bar(name='Monthly Ad Spend', x=months, y=monthly_spend, marker_color='#ef4444', opacity=0.7),
        secondary_y=False
    )
    fig.add_trace(
        go.Bar(name='Monthly Revenue', x=months, y=monthly_revenue, marker_color='#10b981', opacity=0.7),
        secondary_y=False
    )

    # Cumulative line
    fig.add_trace(
        go.Scatter(name='Cumulative Revenue', x=months, y=cumulative_revenue, mode='lines+markers',
                   line=dict(color='#3b82f6', width=3)),
        secondary_y=True
    )

    fig.update_layout(
        barmode='group',
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Monthly (€)", secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="Cumulative Revenue (€)", secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')

    st.plotly_chart(fig, use_container_width=True)

    # Key Insights
    st.markdown("---")
    st.markdown("#### 💡 Key Insights")

    insight_cols = st.columns(3)

    with insight_cols[0]:
        st.warning(f"""
        **⚠️ Current Problem**

        Paid ads have **0.61x ROAS** - we're losing €0.39 for every €1 spent. Must optimize before scaling.
        """)

    with insight_cols[1]:
        st.success(f"""
        **✅ Recommendation**

        Keep paid ads at **40% of marketing** (~€13k/mo). Prioritize affiliates (25%) which have 4x better CAC.
        """)

    with insight_cols[2]:
        st.info(f"""
        **💡 Scale Strategy**

        Once ROAS hits **1.5x+**, increase Google/Meta to 50% of budget. Scale what works, cut what doesn't.
        """)

    # Projection table
    st.markdown("---")
    st.markdown("#### 📊 Monthly Projection Table")

    table_data = []
    running_total = 0
    for i in range(runway_months):
        current_roas = 0.61 + ((target_roas - 0.61) * ((i + 1) / runway_months))
        month_rev = monthly_total * current_roas
        running_total += month_rev
        table_data.append({
            "Month": f"M{i+1}",
            "Google Ads": format_currency(google_ads),
            "Meta Ads": format_currency(meta_ads),
            "Total Spend": format_currency(monthly_total),
            "ROAS": f"{current_roas:.2f}x",
            "Revenue": format_currency(month_rev),
            "Cumulative": format_currency(running_total)
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: BREAKEVEN PLAYGROUND
# =============================================================================

elif page == "🔄 Breakeven Playground":
    st.title("Breakeven Playground")
    st.markdown("### Interactive Breakeven Analysis")

    st.info("🎮 **Drag the sliders to see real-time impact on breakeven point**")

    # Main controls in expander
    with st.expander("⚙️ Scenario Parameters", expanded=True):
        ctrl_row1 = st.columns(4)

        with ctrl_row1[0]:
            be_current_revenue = st.slider(
                "Current Revenue (€/month)",
                min_value=10000,
                max_value=100000,
                value=25000,
                step=5000,
                format="€%d",
                key="be_revenue",
                help="Your current monthly revenue"
            )

        with ctrl_row1[1]:
            be_fixed = st.slider(
                "Fixed Costs (€/month)",
                min_value=50000,
                max_value=200000,
                value=100000,
                step=5000,
                format="€%d",
                key="be_fixed",
                help="Monthly fixed costs (team, infrastructure)"
            )

        with ctrl_row1[2]:
            be_payout = st.slider(
                "Payout Ratio",
                min_value=20,
                max_value=80,
                value=50,
                step=5,
                format="%d%%",
                key="be_payout",
                help="Payout as % of revenue"
            ) / 100

        with ctrl_row1[3]:
            be_marketing = st.slider(
                "Marketing Ratio",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                format="%d%%",
                key="be_marketing",
                help="Marketing spend as % of revenue"
            ) / 100

    # Calculate breakeven
    result = calculate_breakeven(be_fixed, be_payout, be_marketing)

    st.markdown("---")

    # Results display
    col1, col2 = st.columns([1, 2])

    with col1:
        if result["possible"]:
            st.success(f"""
            ## Breakeven Revenue
            # {format_currency(result['revenue'])}/month
            """)

            # Show gap from current revenue
            gap = result['revenue'] - be_current_revenue
            gap_pct = (gap / be_current_revenue) * 100 if be_current_revenue > 0 else 0

            if gap > 0:
                st.warning(f"""
                **Gap to Breakeven:**
                - Current: {format_currency(be_current_revenue)}/mo
                - Need: {format_currency(result['revenue'])}/mo
                - Gap: **{format_currency(gap)}** ({gap_pct:.0f}% more needed)
                """)
            else:
                st.success(f"""
                **Already at Breakeven!**
                - Current: {format_currency(be_current_revenue)}/mo
                - Breakeven: {format_currency(result['revenue'])}/mo
                - Surplus: **{format_currency(-gap)}/mo**
                """)

            st.markdown(f"""
            **Breakdown at Breakeven:**
            - Revenue: {format_currency(result['revenue'])}
            - Payouts ({be_payout*100:.0f}%): {format_currency(result['payouts_at_be'])}
            - Marketing ({be_marketing*100:.0f}%): {format_currency(result['marketing_at_be'])}
            - Fixed Costs: {format_currency(be_fixed)}
            - **Profit: €0**

            **Contribution Margin: {result['contribution_margin']*100:.0f}%**
            """)
        else:
            st.error("""
            ## ❌ Breakeven Impossible

            Payout + Marketing costs exceed 100% of revenue.
            You lose money on every sale!
            """)

    with col2:
        # Visual breakeven chart
        if result["possible"]:
            revenues = np.linspace(0, result["revenue"] * 2, 100)

            costs = be_fixed + (revenues * be_payout) + (revenues * be_marketing)
            profits = revenues - costs

            fig = go.Figure()

            # Revenue line
            fig.add_trace(go.Scatter(
                x=revenues, y=revenues,
                name="Revenue",
                line=dict(color="#2E86AB", width=3)
            ))

            # Total costs line
            fig.add_trace(go.Scatter(
                x=revenues, y=costs,
                name="Total Costs",
                line=dict(color="#C73E1D", width=3)
            ))

            # Profit/Loss area
            fig.add_trace(go.Scatter(
                x=revenues, y=profits,
                name="Profit/Loss",
                line=dict(color="#66BB6A", width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 187, 106, 0.3)'
            ))

            # Breakeven point
            fig.add_vline(x=result["revenue"], line_dash="dash", line_color="green",
                         annotation_text=f"Breakeven: {format_currency(result['revenue'])}")
            fig.add_hline(y=0, line_color="black", line_width=1)

            # Current revenue marker
            fig.add_vline(x=be_current_revenue, line_dash="dot", line_color="orange",
                         annotation_text=f"Current: {format_currency(be_current_revenue)}")

            fig.update_layout(
                height=450,
                title="Breakeven Analysis",
                xaxis_title="Monthly Revenue (€)",
                yaxis_title="Amount (€)",
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Sensitivity analysis
    st.markdown("### Sensitivity Analysis")
    st.markdown("*How does breakeven change with different parameters?*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Payout Ratio Impact")

        payout_range = np.arange(0.30, 0.75, 0.05)
        be_at_payout = []

        for pr in payout_range:
            res = calculate_breakeven(be_fixed, pr, be_marketing)
            be_at_payout.append(res["revenue"] if res["possible"] else None)

        fig = px.line(
            x=[f"{p*100:.0f}%" for p in payout_range],
            y=be_at_payout,
            markers=True
        )
        fig.update_layout(
            xaxis_title="Payout Ratio",
            yaxis_title="Breakeven Revenue",
            height=300
        )
        fig.add_hline(y=100000, line_dash="dash", line_color="green",
                     annotation_text="Target €100k")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Fixed Costs Impact")

        fixed_range = np.arange(60000, 160000, 10000)
        be_at_fixed = []

        for fc in fixed_range:
            res = calculate_breakeven(fc, be_payout, be_marketing)
            be_at_fixed.append(res["revenue"] if res["possible"] else None)

        fig = px.line(
            x=[format_currency(f) for f in fixed_range],
            y=be_at_fixed,
            markers=True
        )
        fig.update_layout(
            xaxis_title="Fixed Costs",
            yaxis_title="Breakeven Revenue",
            height=300
        )
        fig.add_hline(y=100000, line_dash="dash", line_color="green",
                     annotation_text="Target €100k")
        st.plotly_chart(fig, use_container_width=True)

    # Quick scenarios
    st.markdown("### Quick Scenarios")

    scenarios = [
        {"name": "Current State", "payout": 1.0, "marketing": 0.15, "fixed": 100000},
        {"name": "Target (50% PR)", "payout": 0.50, "marketing": 0.15, "fixed": 100000},
        {"name": "FPFX Benchmark (35%)", "payout": 0.35, "marketing": 0.15, "fixed": 100000},
        {"name": "Lean Operations", "payout": 0.45, "marketing": 0.10, "fixed": 80000},
        {"name": "Aggressive Growth", "payout": 0.55, "marketing": 0.25, "fixed": 120000},
    ]

    scenario_results = []
    for s in scenarios:
        res = calculate_breakeven(s["fixed"], s["payout"], s["marketing"])
        scenario_results.append({
            "Scenario": s["name"],
            "Payout Ratio": f"{s['payout']*100:.0f}%",
            "Marketing": f"{s['marketing']*100:.0f}%",
            "Fixed Costs": format_currency(s["fixed"]),
            "Breakeven Revenue": format_currency(res["revenue"]) if res["possible"] else "Impossible",
            "Status": "✅" if res["possible"] and res["revenue"] < 200000 else ("⚠️" if res["possible"] else "❌")
        })

    st.dataframe(pd.DataFrame(scenario_results), use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: CHANNEL DEEP DIVE
# =============================================================================

elif page == "📋 Channel Deep Dive":
    st.title("Channel Deep Dive")
    st.markdown("### Detailed Channel Performance Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "💰 Paid Ads Analysis", "🤝 Organic & Referral"])

    with tab1:
        # Channel comparison
        channel_df = pd.DataFrame([
            {
                "Channel": k,
                "Revenue": v["revenue"],
                "Customers": v["purchasers"],
                "CAC": v["cac"],
                "ARPU": v["revenue"] / v["purchasers"] if v["purchasers"] > 0 else 0,
                "% of Total": v["revenue"] / HISTORICAL["total_revenue"],
            }
            for k, v in HISTORICAL["revenue_by_channel"].items()
        ])

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                channel_df.sort_values("Revenue", ascending=True),
                x="Revenue", y="Channel",
                orientation='h',
                color="CAC",
                color_continuous_scale="RdYlGn_r",
                text_auto='.2s'
            )
            fig.update_layout(height=350, title="Revenue by Channel (colored by CAC)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                channel_df,
                x="CAC", y="ARPU",
                size="Revenue",
                color="Channel",
                text="Channel",
                size_max=50
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=350, title="CAC vs ARPU (bubble size = revenue)")
            st.plotly_chart(fig, use_container_width=True)

        # Data table
        display_df = channel_df.copy()
        display_df["Revenue"] = display_df["Revenue"].apply(lambda x: format_currency(x))
        display_df["CAC"] = display_df["CAC"].apply(lambda x: f"€{x}" if x > 0 else "Free")
        display_df["ARPU"] = display_df["ARPU"].apply(lambda x: f"€{x:.2f}")
        display_df["% of Total"] = display_df["% of Total"].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Paid Ads Performance (Google Ads)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Spend", format_currency(HISTORICAL["paid_ads"]["spend"]))
        col2.metric("Revenue Generated", format_currency(HISTORICAL["paid_ads"]["revenue"]))
        col3.metric("ROAS", f"{HISTORICAL['paid_ads']['roas']:.2f}x")
        col4.metric("Net P&L", format_currency(HISTORICAL["paid_ads"]["revenue"] - HISTORICAL["paid_ads"]["spend"]),
                   delta="Loss", delta_color="inverse")

        st.error(f"""
        ### ⚠️ Critical Issue: Paid Ads Losing Money

        - **Spent:** €{HISTORICAL['paid_ads']['spend']:,}
        - **Generated:** €{HISTORICAL['paid_ads']['revenue']:,}
        - **Net Loss:** €{HISTORICAL['paid_ads']['spend'] - HISTORICAL['paid_ads']['revenue']:,}

        **Every €1 spent on ads returns only €{HISTORICAL['paid_ads']['roas']:.2f}**
        """)

        st.markdown("---")
        st.markdown("### Campaign Performance Breakdown")

        # Campaign performance table
        campaign_data = []
        for name, data in HISTORICAL["paid_ads"]["campaigns"].items():
            campaign_data.append({
                "Campaign": name,
                "Cost": f"€{data['cost']:,}",
                "Purchases": f"{data['purchases']:.1f}",
                "Revenue": f"€{data['revenue']:,}",
                "ROAS": f"{data['roas']:.2f}x",
                "Status": "✅ Profitable" if data['roas'] >= 1.0 else "❌ Losing"
            })

        campaign_df = pd.DataFrame(campaign_data)
        st.dataframe(campaign_df, use_container_width=True, hide_index=True)

        # Best performing campaign insight
        best_campaign = max(HISTORICAL["paid_ads"]["campaigns"].items(), key=lambda x: x[1]["roas"])
        st.success(f"""
        **Best Performer:** {best_campaign[0]}
        - ROAS: **{best_campaign[1]['roas']:.2f}x**
        - Cost: €{best_campaign[1]['cost']:,}
        - Revenue: €{best_campaign[1]['revenue']:,}

        **Recommendation:** Scale this campaign and pause underperformers.
        """)

        # Device & Demographics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Device Breakdown")
            device_data = pd.DataFrame([
                {"Device": "Mobile", "Share": HISTORICAL["paid_ads"]["devices"]["mobile"]},
                {"Device": "Desktop", "Share": HISTORICAL["paid_ads"]["devices"]["desktop"]},
                {"Device": "Tablet", "Share": HISTORICAL["paid_ads"]["devices"]["tablet"]},
            ])
            fig = px.pie(device_data, values="Share", names="Device",
                        color_discrete_sequence=["#3b82f6", "#06b6d4", "#8b5cf6"])
            fig.update_layout(height=250, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Top Demographics")
            st.info(f"""
            **Primary Audience:**
            - Gender: **{HISTORICAL['paid_ads']['demographics']['top_gender']}**
            - Age: **{HISTORICAL['paid_ads']['demographics']['top_age']}**

            **Traffic Metrics:**
            - Clicks: {HISTORICAL['paid_ads']['clicks']:,}
            - Impressions: {HISTORICAL['paid_ads']['impressions']:,}
            - CTR: {HISTORICAL['paid_ads']['ctr']}%
            - CPC: €{HISTORICAL['paid_ads']['cpc_search']:.2f}
            """)

        st.markdown("---")
        st.markdown("### Optimization Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Short-term (This Month)
            1. **Pause underperforming campaigns**
            2. **Review attribution setup** (sGTM tracking?)
            3. **Audit conversion tracking**
            4. **A/B test new creatives**
            """)

        with col2:
            st.markdown("""
            #### Medium-term (Q1 2026)
            1. **Shift budget to affiliates** (€56 CAC vs €222)
            2. **Invest in email marketing** (highest ARPU)
            3. **Build organic content** (free traffic)
            4. **Only scale when ROAS > 1.5x**
            """)

    with tab3:
        st.markdown("### Organic & Referral Channels")

        organic_channels = ["Direct", "Organic", "Referral/Affiliates", "Email/SMS"]
        organic_df = channel_df[channel_df["Channel"].isin(organic_channels)]

        total_organic_rev = organic_df["Revenue"].sum()
        total_organic_cust = organic_df["Customers"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", format_currency(total_organic_rev))
        col2.metric("Total Customers", total_organic_cust)
        col3.metric("% of All Revenue", f"{total_organic_rev/HISTORICAL['total_revenue']*100:.0f}%")

        st.success(f"""
        ### 💰 {total_organic_rev/HISTORICAL['total_revenue']*100:.0f}% of Revenue is FREE or Low-Cost!

        This is your competitive advantage. Focus on scaling these channels.
        """)

        # Growth opportunities
        st.markdown("### Growth Opportunities")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Affiliate Program Expansion
            - Current: €11k revenue (17%)
            - CAC: €56 (vs €222 for ads)
            - **Opportunity:** 2x to €22k/month

            **Actions:**
            - Increase commission rates
            - Recruit more affiliates
            - Create affiliate resources
            - Implement tiered rewards
            """)

        with col2:
            st.markdown("""
            #### Email Marketing
            - Current: €5.8k revenue (9%)
            - Highest ARPU: €290
            - **Opportunity:** 3x to €17k/month

            **Actions:**
            - Build email list aggressively
            - Implement automation sequences
            - Personalized recommendations
            - Win-back campaigns
            """)

# =============================================================================
# PAGE: EXPORT CENTER
# =============================================================================

elif page == "💾 Export Center":
    st.title("Export Center")
    st.markdown("### Download Reports and Data")

    # Generate default projection (18 months to show breakeven)
    df_export = run_monthly_projection(
        months=18,
        starting_revenue=CURRENT["dec_revenue"],
        growth_rate=0.20,
        payout_ratio=0.50,
        marketing_ratio=0.15,
        fixed_costs=100000,
        funding=2_000_000,
    )

    tab1, tab2, tab3 = st.tabs(["📊 Projections", "📋 Summary", "📈 All Data"])

    with tab1:
        st.markdown("### 18-Month Projection (€2M Funding)")
        st.dataframe(df_export, use_container_width=True, hide_index=True)

        csv = df_export.to_csv(index=False)
        st.download_button(
            "📥 Download Projection CSV",
            csv,
            "ty_12month_projection.csv",
            "text/csv"
        )

    with tab2:
        st.markdown("### Executive Summary Data")

        summary = {
            "Metric": [
                "Starting Bank Balance",
                "Additional Funding (Recommended)",
                "Total Starting Capital",
                "Year 1 Projected Revenue",
                "Year 1 Projected Costs",
                "Final Bank Balance (Month 12)",
                "Minimum Bank Balance",
                "Month to Breakeven",
                "Current Monthly Revenue (Dec 2025)",
                "Target Monthly Revenue",
                "Current Payout Ratio",
                "Target Payout Ratio",
                "Paid Ads ROAS",
                "Current CAC (Paid)",
                "Affiliate CAC",
            ],
            "Value": [
                format_currency(CURRENT["bank_balance"]),
                format_currency(2_000_000),
                format_currency(3_000_000),
                format_currency(df_export["Total Revenue"].sum()),
                format_currency(df_export["Total Costs"].sum()),
                format_currency(df_export["Bank Balance"].iloc[-1]),
                format_currency(df_export["Bank Balance"].min()),
                f"Month {df_export[df_export['Breakeven']]['Month'].min()}" if df_export["Breakeven"].any() else "Month 15+ (extend projection)",  # Breakeven month
                format_currency(CURRENT["dec_revenue"]),
                "€100k-150k/month",
                "200%",
                "45-50%",
                "0.61x",
                "€222",
                "€56",
            ]
        }

        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            "📥 Download Summary CSV",
            csv_summary,
            "ty_executive_summary.csv",
            "text/csv"
        )

    with tab3:
        st.markdown("### Channel Performance Data")

        channel_export = pd.DataFrame([
            {
                "Channel": k,
                "Revenue": v["revenue"],
                "Customers": v["purchasers"],
                "CAC": v["cac"],
                "ARPU": v["revenue"] / v["purchasers"] if v["purchasers"] > 0 else 0,
            }
            for k, v in HISTORICAL["revenue_by_channel"].items()
        ])

        st.dataframe(channel_export, use_container_width=True, hide_index=True)

        csv_channel = channel_export.to_csv(index=False)
        st.download_button(
            "📥 Download Channel Data CSV",
            csv_channel,
            "ty_channel_performance.csv",
            "text/csv"
        )

        st.markdown("---")
        st.markdown("### Product Sales Data")

        product_export = pd.DataFrame([
            {"Product": k, "Price": v["price"], "Units Sold": v["units"],
             "Revenue": v["price"] * v["units"], "Type": v["type"]}
            for k, v in PRODUCTS.items()
        ]).sort_values("Revenue", ascending=False)

        st.dataframe(product_export, use_container_width=True, hide_index=True)

        csv_product = product_export.to_csv(index=False)
        st.download_button(
            "📥 Download Product Data CSV",
            csv_product,
            "ty_product_sales.csv",
            "text/csv"
        )

# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
**TrueYield Budget Model V2**

Data Sources:
- Stripe Dashboard
- Google Analytics (GA4)
- Liquidity Plan Oct 2025
- TY Operating Budget Model

*January 2026*
""")

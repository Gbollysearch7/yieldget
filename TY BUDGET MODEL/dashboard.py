#!/usr/bin/env python3
"""
TrueYield Marketing Budget Dashboard
====================================
Interactive dashboard for exploring funding scenarios, marketing budgets,
and revenue projections.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TrueYield Budget Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# BASELINE DATA
# =============================================================================

BASELINE = {
    "total_revenue": 65000,
    "total_purchasers": 289,
    "aov": 225,
    "revenue_by_channel": {
        "Direct": {"revenue": 23000, "purchasers": 109, "color": "#2E86AB"},
        "Organic": {"revenue": 19000, "purchasers": 79, "color": "#A23B72"},
        "Referral/Affiliates": {"revenue": 11000, "purchasers": 65, "color": "#F18F01"},
        "Email/SMS": {"revenue": 5800, "purchasers": 20, "color": "#C73E1D"},
        "Paid Ads": {"revenue": 4800, "purchasers": 43, "color": "#3B1F2B"},
    },
    "google_ads": {
        "total_spend": 23900,
        "conversions": 107.5,
        "conversion_value": 14500,
        "roas": 0.61,
    },
}

CURRENT_STATE = {
    "bank_balance": 1_000_000,
    "monthly_burn": 100_000,
    "team_cost": 80_000,
    "server_infra": 20_000,
    "current_ad_spend": 12_000,
    "dec_revenue": 25_000,
    "dec_payouts": 50_000,
}

PRODUCTS = {
    "2phase_swing_5k": {"price": 50, "units": 119},
    "2phase_swing_50k": {"price": 250, "units": 81},
    "2phase_swing_100k": {"price": 400, "units": 71},
    "2phase_swing_10k": {"price": 100, "units": 55},
    "2phase_swing_25k": {"price": 150, "units": 55},
    "tc_standard": {"price": 50, "units": 41},
    "tc_advanced": {"price": 100, "units": 31},
    "tc_premium": {"price": 200, "units": 15},
    "tc_elite": {"price": 300, "units": 12},
    "cfd_1phase_50k": {"price": 275, "units": 12},
    "cfd_1phase_100k": {"price": 450, "units": 13},
    "cfd_1phase_5k": {"price": 75, "units": 13},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_currency(amount: float) -> str:
    """Format number as EUR currency."""
    if abs(amount) >= 1_000_000:
        return f"€{amount/1_000_000:.2f}M"
    elif abs(amount) >= 1_000:
        return f"€{amount/1_000:.1f}k"
    else:
        return f"€{amount:.0f}"


def run_scenario(
    funding: float,
    starting_revenue: float,
    growth_rate: float,
    payout_ratio: float,
    marketing_ratio: float,
    fixed_costs: float,
    months: int = 12
) -> pd.DataFrame:
    """Run a funding scenario and return monthly data."""
    bank_balance = CURRENT_STATE["bank_balance"] + funding
    data = []
    revenue = starting_revenue

    for month in range(1, months + 1):
        payouts = revenue * payout_ratio
        marketing = revenue * marketing_ratio
        total_costs = fixed_costs + payouts + marketing
        net_cashflow = revenue - total_costs
        bank_balance += net_cashflow

        data.append({
            "Month": month,
            "Revenue": revenue,
            "Payouts": payouts,
            "Payout Ratio": payout_ratio,
            "Marketing": marketing,
            "Fixed Costs": fixed_costs,
            "Total Costs": total_costs,
            "Net Cashflow": net_cashflow,
            "Bank Balance": bank_balance,
            "Cumulative Revenue": sum(d["Revenue"] for d in data) + revenue,
        })

        revenue *= (1 + growth_rate)

    return pd.DataFrame(data)


def calculate_breakeven_revenue(fixed_costs: float, payout_ratio: float) -> float:
    """Calculate revenue needed to break even."""
    if payout_ratio >= 1:
        return float('inf')
    return fixed_costs / (1 - payout_ratio)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("📊 TrueYield Budget Model")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["📈 Executive Summary", "💰 Funding Scenarios", "📊 Channel Analysis",
     "🎯 Marketing Budget", "🔮 What-If Calculator", "📋 Export Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Bank Balance", format_currency(CURRENT_STATE["bank_balance"]))
st.sidebar.metric("Monthly Burn", format_currency(CURRENT_STATE["monthly_burn"]))
st.sidebar.metric("Dec 2025 Revenue", format_currency(CURRENT_STATE["dec_revenue"]))

# =============================================================================
# PAGE: EXECUTIVE SUMMARY
# =============================================================================

if page == "📈 Executive Summary":
    st.title("TrueYield Budget & Funding Model")
    st.markdown("### Executive Summary for Ingmar Meeting")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Bank Balance",
            format_currency(CURRENT_STATE["bank_balance"]),
            delta=None
        )

    with col2:
        runway = CURRENT_STATE["bank_balance"] / (CURRENT_STATE["monthly_burn"] - CURRENT_STATE["dec_revenue"])
        st.metric(
            "Current Runway",
            f"{runway:.1f} months",
            delta="-1/month" if runway < 18 else None,
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "Payout Ratio",
            "200%",
            delta="Target: 50%",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "Paid Ads ROAS",
            "0.61x",
            delta="Losing money",
            delta_color="inverse"
        )

    st.markdown("---")

    # Two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Revenue by Channel (Aug-Dec 2025)")

        channel_data = pd.DataFrame([
            {"Channel": k, "Revenue": v["revenue"], "Purchasers": v["purchasers"]}
            for k, v in BASELINE["revenue_by_channel"].items()
        ])

        fig = px.pie(
            channel_data,
            values="Revenue",
            names="Channel",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **64% of revenue comes from FREE channels** (Direct + Organic)")

    with col2:
        st.markdown("#### Breakeven Analysis")

        payout_ratios = [0.35, 0.45, 0.50, 0.60, 0.75]
        fixed_costs = 100000

        breakeven_data = []
        for ratio in payout_ratios:
            be_rev = calculate_breakeven_revenue(fixed_costs, ratio)
            breakeven_data.append({
                "Payout Ratio": f"{ratio*100:.0f}%",
                "Breakeven Revenue": be_rev if be_rev != float('inf') else 0,
                "Achievable": "Yes" if be_rev < 300000 else "Difficult"
            })

        df_be = pd.DataFrame(breakeven_data)

        fig = px.bar(
            df_be,
            x="Payout Ratio",
            y="Breakeven Revenue",
            color="Achievable",
            color_discrete_map={"Yes": "#2E86AB", "Difficult": "#C73E1D"},
            text_auto='.2s'
        )
        fig.update_layout(height=350, showlegend=True)
        fig.add_hline(y=100000, line_dash="dash", line_color="green",
                      annotation_text="Target: €100k/mo")
        fig.add_hline(y=150000, line_dash="dash", line_color="orange",
                      annotation_text="Stretch: €150k/mo")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Key Recommendations
    st.markdown("### Key Recommendations")

    rec_col1, rec_col2, rec_col3 = st.columns(3)

    with rec_col1:
        st.error("#### 🎯 Priority 1: Payout Ratio")
        st.markdown("""
        - Current: **200%** (impossible to profit)
        - Target: **45-50%**
        - FPFX Benchmark: **35%**
        - New risk management = critical
        """)

    with rec_col2:
        st.warning("#### 📢 Priority 2: Fix Paid Ads")
        st.markdown("""
        - Current ROAS: **0.61x** (losing money)
        - Target ROAS: **1.5x minimum**
        - Fix before scaling
        - Better: shift to organic/referral
        """)

    with rec_col3:
        st.success("#### 💰 Priority 3: Funding")
        st.markdown("""
        - Minimum: **€1M** (survival)
        - Recommended: **€2M** (growth)
        - Aggressive: **€5M** (dominance)
        - Path to €100k/mo: 8 months
        """)

# =============================================================================
# PAGE: FUNDING SCENARIOS
# =============================================================================

elif page == "💰 Funding Scenarios":
    st.title("Funding Scenarios")
    st.markdown("Compare different funding amounts and growth trajectories")

    # Scenario Parameters
    st.sidebar.markdown("### Scenario Parameters")

    growth_rate = st.sidebar.slider(
        "Monthly Growth Rate",
        min_value=0.05,
        max_value=0.35,
        value=0.20,
        step=0.01,
        format="%.0f%%"
    )

    payout_ratio = st.sidebar.slider(
        "Payout Ratio",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        format="%.0f%%"
    )

    marketing_ratio = st.sidebar.slider(
        "Marketing Spend (% of Revenue)",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        format="%.0f%%"
    )

    fixed_costs = st.sidebar.number_input(
        "Monthly Fixed Costs (€)",
        min_value=50000,
        max_value=200000,
        value=100000,
        step=10000
    )

    # Run scenarios
    scenarios = {
        "No Funding": 0,
        "€1M Funding": 1_000_000,
        "€2M Funding": 2_000_000,
        "€5M Funding": 5_000_000,
    }

    all_data = []
    summary_data = []

    for name, funding in scenarios.items():
        df = run_scenario(
            funding=funding,
            starting_revenue=CURRENT_STATE["dec_revenue"],
            growth_rate=growth_rate,
            payout_ratio=payout_ratio,
            marketing_ratio=marketing_ratio,
            fixed_costs=fixed_costs,
        )
        df["Scenario"] = name
        all_data.append(df)

        summary_data.append({
            "Scenario": name,
            "Funding": funding,
            "Final Balance": df["Bank Balance"].iloc[-1],
            "Min Balance": df["Bank Balance"].min(),
            "Total Revenue": df["Revenue"].sum(),
            "Breakeven Month": next(
                (row["Month"] for _, row in df.iterrows() if row["Net Cashflow"] >= 0),
                "Never"
            ),
        })

    combined_df = pd.concat(all_data)
    summary_df = pd.DataFrame(summary_data)

    # Summary Table
    st.markdown("### Scenario Comparison")

    col1, col2, col3, col4 = st.columns(4)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        col = [col1, col2, col3, col4][i]
        with col:
            st.markdown(f"**{row['Scenario']}**")
            st.metric("Final Balance", format_currency(row["Final Balance"]))
            st.metric("Min Balance", format_currency(row["Min Balance"]))
            st.metric("Breakeven", str(row["Breakeven Month"]))

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bank Balance Over Time")
        fig = px.line(
            combined_df,
            x="Month",
            y="Bank Balance",
            color="Scenario",
            markers=True
        )
        fig.update_layout(height=400)
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                      annotation_text="Bankruptcy")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Monthly Revenue Growth")
        fig = px.line(
            combined_df,
            x="Month",
            y="Revenue",
            color="Scenario",
            markers=True
        )
        fig.update_layout(height=400)
        fig.add_hline(y=100000, line_dash="dash", line_color="green",
                      annotation_text="Target: €100k")
        fig.add_hline(y=150000, line_dash="dash", line_color="orange",
                      annotation_text="Stretch: €150k")
        st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    st.markdown("### Detailed Monthly Projection (€2M Scenario)")

    df_2m = combined_df[combined_df["Scenario"] == "€2M Funding"].copy()
    df_2m["Revenue"] = df_2m["Revenue"].apply(lambda x: f"€{x:,.0f}")
    df_2m["Payouts"] = df_2m["Payouts"].apply(lambda x: f"€{x:,.0f}")
    df_2m["Marketing"] = df_2m["Marketing"].apply(lambda x: f"€{x:,.0f}")
    df_2m["Net Cashflow"] = df_2m["Net Cashflow"].apply(lambda x: f"€{x:,.0f}")
    df_2m["Bank Balance"] = df_2m["Bank Balance"].apply(lambda x: f"€{x:,.0f}")

    st.dataframe(
        df_2m[["Month", "Revenue", "Payouts", "Marketing", "Net Cashflow", "Bank Balance"]],
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# PAGE: CHANNEL ANALYSIS
# =============================================================================

elif page == "📊 Channel Analysis":
    st.title("Channel Performance Analysis")
    st.markdown("### Aug - Dec 2025 Performance")

    # Channel data
    channel_df = pd.DataFrame([
        {
            "Channel": k,
            "Revenue": v["revenue"],
            "Purchasers": v["purchasers"],
            "ARPU": v["revenue"] / v["purchasers"] if v["purchasers"] > 0 else 0,
            "% of Revenue": v["revenue"] / BASELINE["total_revenue"] * 100,
        }
        for k, v in BASELINE["revenue_by_channel"].items()
    ])

    # Add CAC estimates
    cac_map = {
        "Direct": 0,
        "Organic": 0,
        "Referral/Affiliates": 56,
        "Email/SMS": 5,
        "Paid Ads": 222,
    }
    channel_df["CAC"] = channel_df["Channel"].map(cac_map)
    channel_df["Efficiency"] = channel_df.apply(
        lambda x: "Excellent" if x["CAC"] < 10 else ("Good" if x["CAC"] < 100 else "Poor"),
        axis=1
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Revenue by Channel")
        fig = px.bar(
            channel_df.sort_values("Revenue", ascending=True),
            x="Revenue",
            y="Channel",
            orientation='h',
            color="Efficiency",
            color_discrete_map={"Excellent": "#2E86AB", "Good": "#F18F01", "Poor": "#C73E1D"},
            text_auto='.2s'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### ARPU by Channel")
        fig = px.bar(
            channel_df.sort_values("ARPU", ascending=True),
            x="ARPU",
            y="Channel",
            orientation='h',
            color="Channel",
            text_auto='€.2f'
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Channel Metrics Table
    st.markdown("#### Channel Metrics")

    display_df = channel_df.copy()
    display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"€{x:,.0f}")
    display_df["ARPU"] = display_df["ARPU"].apply(lambda x: f"€{x:.2f}")
    display_df["CAC"] = display_df["CAC"].apply(lambda x: f"€{x:.0f}" if x > 0 else "Free")
    display_df["% of Revenue"] = display_df["% of Revenue"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Paid Ads Deep Dive
    st.markdown("### Paid Ads Performance (Google Ads)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Spend", format_currency(BASELINE["google_ads"]["total_spend"]))
    with col2:
        st.metric("Conversions", f"{BASELINE['google_ads']['conversions']:.0f}")
    with col3:
        st.metric("Conv. Value", format_currency(BASELINE["google_ads"]["conversion_value"]))
    with col4:
        st.metric("ROAS", f"{BASELINE['google_ads']['roas']:.2f}x",
                  delta="Losing €0.39/€1", delta_color="inverse")

    st.error("""
    **Paid Ads Analysis:**
    - Spent €23.9k → Generated €14.5k = **Lost €9.4k**
    - CAC of €222 is 4x higher than referral (€56)
    - ROAS needs to be >1.0x to break even, >1.5x to be efficient

    **Recommendation:** Pause inefficient campaigns, optimize, or reallocate to referral/email
    """)

# =============================================================================
# PAGE: MARKETING BUDGET
# =============================================================================

elif page == "🎯 Marketing Budget":
    st.title("Marketing Budget Allocation")
    st.markdown("### Recommended Budget Breakdown")

    # Budget selector
    total_budget = st.slider(
        "Total Monthly Marketing Budget",
        min_value=10000,
        max_value=100000,
        value=25000,
        step=5000,
        format="€%d"
    )

    # Default allocation
    default_allocation = {
        "Google Ads": 0.20,
        "Meta Ads": 0.20,
        "TikTok Ads": 0.10,
        "Influencer Marketing": 0.15,
        "Affiliate Payouts": 0.15,
        "Email/SMS Tools": 0.05,
        "Giveaways/Promotions": 0.10,
        "Marketing Tools": 0.05,
    }

    st.markdown("---")
    st.markdown("#### Adjust Channel Allocation")

    # Allow custom allocation
    col1, col2 = st.columns(2)

    allocation = {}
    channels = list(default_allocation.keys())

    for i, channel in enumerate(channels):
        col = col1 if i < 4 else col2
        with col:
            allocation[channel] = st.slider(
                channel,
                min_value=0.0,
                max_value=0.50,
                value=default_allocation[channel],
                step=0.05,
                format="%.0f%%",
                key=channel
            )

    # Check if allocation sums to 100%
    total_alloc = sum(allocation.values())

    if abs(total_alloc - 1.0) > 0.01:
        st.warning(f"⚠️ Allocation totals {total_alloc*100:.0f}% - should be 100%")

    st.markdown("---")

    # Budget breakdown
    budget_df = pd.DataFrame([
        {
            "Channel": k,
            "Allocation": v,
            "Budget": total_budget * v,
        }
        for k, v in allocation.items()
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Budget Distribution")
        fig = px.pie(
            budget_df,
            values="Budget",
            names="Channel",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Budget by Channel")
        fig = px.bar(
            budget_df.sort_values("Budget", ascending=True),
            x="Budget",
            y="Channel",
            orientation='h',
            color="Budget",
            color_continuous_scale="Viridis",
            text_auto='€,.0f'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Budget Table
    st.markdown("#### Monthly Budget Breakdown")

    display_budget = budget_df.copy()
    display_budget["Allocation"] = display_budget["Allocation"].apply(lambda x: f"{x*100:.0f}%")
    display_budget["Budget"] = display_budget["Budget"].apply(lambda x: f"€{x:,.0f}")
    display_budget["Quarterly"] = budget_df["Budget"].apply(lambda x: f"€{x*3:,.0f}")
    display_budget["Annual"] = budget_df["Budget"].apply(lambda x: f"€{x*12:,.0f}")

    st.dataframe(display_budget, use_container_width=True, hide_index=True)

    st.info(f"""
    **Total Monthly Budget: {format_currency(total_budget)}**
    - Quarterly: {format_currency(total_budget * 3)}
    - Annual: {format_currency(total_budget * 12)}
    """)

# =============================================================================
# PAGE: WHAT-IF CALCULATOR
# =============================================================================

elif page == "🔮 What-If Calculator":
    st.title("What-If Calculator")
    st.markdown("### Explore Different Scenarios")

    tab1, tab2, tab3 = st.tabs(["Revenue Target", "Breakeven Analysis", "Custom Scenario"])

    with tab1:
        st.markdown("#### How to reach revenue targets?")

        target_revenue = st.number_input(
            "Target Monthly Revenue (€)",
            min_value=30000,
            max_value=500000,
            value=100000,
            step=10000
        )

        months_to_target = st.slider(
            "Months to Reach Target",
            min_value=3,
            max_value=24,
            value=12
        )

        starting = CURRENT_STATE["dec_revenue"]
        required_growth = (target_revenue / starting) ** (1 / months_to_target) - 1

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Starting Revenue", format_currency(starting))
            st.metric("Target Revenue", format_currency(target_revenue))
            st.metric("Required Monthly Growth", f"{required_growth*100:.1f}%")

        with col2:
            # Projection chart
            months = list(range(1, months_to_target + 1))
            revenues = [starting * (1 + required_growth) ** m for m in months]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=revenues,
                mode='lines+markers',
                name='Projected Revenue',
                line=dict(color='#2E86AB', width=3)
            ))
            fig.add_hline(y=target_revenue, line_dash="dash", line_color="green",
                          annotation_text=f"Target: {format_currency(target_revenue)}")
            fig.update_layout(
                title="Revenue Projection",
                xaxis_title="Month",
                yaxis_title="Revenue (€)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        # Monthly breakdown
        st.markdown("#### Monthly Milestones")
        milestone_df = pd.DataFrame({
            "Month": months,
            "Revenue": [format_currency(r) for r in revenues],
            "Growth from Start": [f"{(r/starting - 1)*100:.0f}%" for r in revenues],
        })
        st.dataframe(milestone_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### Breakeven Calculator")

        col1, col2 = st.columns(2)

        with col1:
            be_fixed_costs = st.number_input(
                "Monthly Fixed Costs (€)",
                min_value=50000,
                max_value=200000,
                value=100000,
                step=5000,
                key="be_fixed"
            )

            be_payout_ratio = st.slider(
                "Payout Ratio",
                min_value=0.30,
                max_value=0.90,
                value=0.50,
                step=0.05,
                key="be_payout"
            )

            be_marketing_ratio = st.slider(
                "Marketing Ratio",
                min_value=0.05,
                max_value=0.30,
                value=0.15,
                step=0.05,
                key="be_mkt"
            )

        with col2:
            # Calculate breakeven
            contribution_margin = 1 - be_payout_ratio - be_marketing_ratio

            if contribution_margin <= 0:
                st.error("❌ Breakeven impossible - payout + marketing >= 100%")
            else:
                breakeven_rev = be_fixed_costs / contribution_margin

                st.success(f"""
                ### Breakeven Revenue: {format_currency(breakeven_rev)}/month

                **Breakdown:**
                - Fixed Costs: {format_currency(be_fixed_costs)}
                - Contribution Margin: {contribution_margin*100:.0f}%
                - Payouts at BE: {format_currency(breakeven_rev * be_payout_ratio)}
                - Marketing at BE: {format_currency(breakeven_rev * be_marketing_ratio)}
                """)

    with tab3:
        st.markdown("#### Custom Scenario Builder")

        col1, col2 = st.columns(2)

        with col1:
            custom_funding = st.number_input(
                "Additional Funding (€)",
                min_value=0,
                max_value=10_000_000,
                value=2_000_000,
                step=500_000
            )

            custom_growth = st.slider(
                "Monthly Growth Rate",
                min_value=0.05,
                max_value=0.40,
                value=0.20,
                step=0.01,
                format="%.0f%%",
                key="custom_growth"
            )

            custom_payout = st.slider(
                "Payout Ratio",
                min_value=0.30,
                max_value=0.80,
                value=0.50,
                step=0.05,
                format="%.0f%%",
                key="custom_payout"
            )

            custom_marketing = st.slider(
                "Marketing Ratio",
                min_value=0.05,
                max_value=0.30,
                value=0.15,
                step=0.01,
                format="%.0f%%",
                key="custom_mkt"
            )

            custom_fixed = st.number_input(
                "Monthly Fixed Costs (€)",
                min_value=50000,
                max_value=200000,
                value=100000,
                step=5000,
                key="custom_fixed"
            )

        with col2:
            if st.button("Run Scenario", type="primary"):
                custom_df = run_scenario(
                    funding=custom_funding,
                    starting_revenue=CURRENT_STATE["dec_revenue"],
                    growth_rate=custom_growth,
                    payout_ratio=custom_payout,
                    marketing_ratio=custom_marketing,
                    fixed_costs=custom_fixed,
                )

                st.metric("Final Bank Balance", format_currency(custom_df["Bank Balance"].iloc[-1]))
                st.metric("Min Bank Balance", format_currency(custom_df["Bank Balance"].min()))
                st.metric("Total Year Revenue", format_currency(custom_df["Revenue"].sum()))

                breakeven_month = next(
                    (row["Month"] for _, row in custom_df.iterrows() if row["Net Cashflow"] >= 0),
                    "Never"
                )
                st.metric("Breakeven Month", str(breakeven_month))

                # Chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Scatter(x=custom_df["Month"], y=custom_df["Bank Balance"],
                               name="Bank Balance", line=dict(color="#2E86AB", width=3)),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(x=custom_df["Month"], y=custom_df["Revenue"],
                               name="Revenue", line=dict(color="#F18F01", width=2)),
                    secondary_y=True
                )

                fig.update_layout(title="Custom Scenario Projection", height=350)
                fig.update_yaxes(title_text="Bank Balance (€)", secondary_y=False)
                fig.update_yaxes(title_text="Revenue (€)", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: EXPORT DATA
# =============================================================================

elif page == "📋 Export Data":
    st.title("Export Data")
    st.markdown("### Download Reports and Data")

    # Generate €2M scenario data
    df_export = run_scenario(
        funding=2_000_000,
        starting_revenue=CURRENT_STATE["dec_revenue"],
        growth_rate=0.20,
        payout_ratio=0.50,
        marketing_ratio=0.15,
        fixed_costs=100_000,
    )

    st.markdown("#### 12-Month Projection (€2M Funding)")
    st.dataframe(df_export, use_container_width=True, hide_index=True)

    # CSV Download
    csv = df_export.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="ty_budget_projection.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Summary Stats
    st.markdown("#### Summary Statistics")

    summary = {
        "Metric": [
            "Starting Bank Balance",
            "Additional Funding",
            "Total Starting Capital",
            "Year 1 Total Revenue",
            "Year 1 Total Costs",
            "Final Bank Balance",
            "Minimum Bank Balance",
            "Average Monthly Burn",
        ],
        "Value": [
            format_currency(CURRENT_STATE["bank_balance"]),
            format_currency(2_000_000),
            format_currency(3_000_000),
            format_currency(df_export["Revenue"].sum()),
            format_currency(df_export["Total Costs"].sum()),
            format_currency(df_export["Bank Balance"].iloc[-1]),
            format_currency(df_export["Bank Balance"].min()),
            format_currency(abs(df_export["Net Cashflow"].mean())),
        ]
    }

    st.table(pd.DataFrame(summary))

    st.markdown("---")
    st.markdown("#### Channel Performance Data")

    channel_export = pd.DataFrame([
        {
            "Channel": k,
            "Revenue (€)": v["revenue"],
            "Purchasers": v["purchasers"],
            "ARPU (€)": round(v["revenue"] / v["purchasers"], 2) if v["purchasers"] > 0 else 0,
            "% of Total": f"{v['revenue'] / BASELINE['total_revenue'] * 100:.1f}%"
        }
        for k, v in BASELINE["revenue_by_channel"].items()
    ])

    st.dataframe(channel_export, use_container_width=True, hide_index=True)

    csv_channels = channel_export.to_csv(index=False)
    st.download_button(
        label="📥 Download Channel Data",
        data=csv_channels,
        file_name="ty_channel_performance.csv",
        mime="text/csv"
    )

# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
**TrueYield Budget Model**
Version 1.0
January 2026

Data Sources:
- Stripe Dashboard
- Google Analytics
- Liquidity Plan Oct 2025
""")

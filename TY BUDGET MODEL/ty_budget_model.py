#!/usr/bin/env python3
"""
TrueYield Marketing Budget & Funding Model
==========================================
Interactive model for forecasting revenue, marketing spend, and funding requirements.
Built for presentation to Ingmar (March 2026 Florida meeting).

All values in EUR (€)
"""

import json
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# BASELINE DATA (Aug-Dec 2025 Actuals)
# =============================================================================

BASELINE = {
    "period": "Aug-Dec 2025 (5 months)",
    "total_revenue": 65000,
    "total_purchasers": 289,
    "first_time_purchasers": 275,
    "repeat_rate": 0.05,  # Only 5% repeat
    "aov": 225,  # Average Order Value

    # Revenue by channel
    "revenue_by_channel": {
        "direct": {"revenue": 23000, "purchasers": 109, "arpu": 1.34},
        "organic": {"revenue": 19000, "purchasers": 79, "arpu": 3.69},
        "referral_affiliates": {"revenue": 11000, "purchasers": 65, "arpu": 2.54},
        "email_sms": {"revenue": 5800, "purchasers": 20, "arpu": 8.14},
        "paid_ads": {"revenue": 4800, "purchasers": 43, "arpu": 1.27},
    },

    # Google Ads performance
    "google_ads": {
        "total_spend": 23900,
        "clicks": 9660,
        "conversions": 107.5,
        "conversion_value": 14500,
        "cpc": 2.47,
        "cpa": 222,
        "roas": 0.61,
    },

    # Monthly marketing spend
    "monthly_ad_spend": 12000,  # ~6k Meta + 6k Google
}

# =============================================================================
# CURRENT FINANCIAL STATE (Jan 2026)
# =============================================================================

CURRENT_STATE = {
    "date": "January 2026",
    "bank_balance": 1_000_000,
    "monthly_burn": 100_000,

    # Cost breakdown
    "team_cost": 80_000,
    "server_infra": 20_000,
    "current_ad_spend": 12_000,

    # December 2025 actuals (from CEO)
    "dec_revenue": 25_000,
    "dec_payouts": 50_000,
    "dec_payout_ratio": 2.0,  # 200%
    "dec_payout_ratio_excl_giveaways": 1.0,  # 100%

    # Targets
    "target_monthly_revenue": 100_000,  # Min target
    "target_monthly_revenue_high": 150_000,  # Stretch target
    "target_payout_ratio": 0.45,  # 45-50%
    "fpfx_benchmark_ratio": 0.35,  # FPFX targets 35%
}

# =============================================================================
# PRODUCT PRICING (Estimated from data)
# =============================================================================

PRODUCTS = {
    "2phase_swing_5k": {"price": 50, "units_sold": 119},
    "2phase_swing_10k": {"price": 100, "units_sold": 55},
    "2phase_swing_25k": {"price": 150, "units_sold": 55},
    "2phase_swing_50k": {"price": 250, "units_sold": 81},
    "2phase_swing_100k": {"price": 400, "units_sold": 71},
    "tc_standard": {"price": 50, "units_sold": 41},
    "tc_advanced": {"price": 100, "units_sold": 31},
    "tc_premium": {"price": 200, "units_sold": 15},
    "tc_elite": {"price": 300, "units_sold": 12},
    "cfd_1phase_5k": {"price": 75, "units_sold": 13},
    "cfd_1phase_10k": {"price": 125, "units_sold": 10},
    "cfd_1phase_25k": {"price": 175, "units_sold": 8},
    "cfd_1phase_50k": {"price": 275, "units_sold": 12},
    "cfd_1phase_100k": {"price": 450, "units_sold": 13},
}

# =============================================================================
# CHANNEL EFFICIENCY METRICS
# =============================================================================

CHANNEL_METRICS = {
    "paid_ads": {
        "current_cac": 222,  # Cost per acquisition
        "current_roas": 0.61,
        "potential_roas_optimized": 1.5,  # With optimization
        "revenue_share": 0.07,
    },
    "organic": {
        "cac": 0,  # Free
        "revenue_share": 0.29,
    },
    "direct": {
        "cac": 0,  # Free (brand awareness)
        "revenue_share": 0.35,
    },
    "referral_affiliates": {
        "cac": 56,  # ~€11k / 65 purchasers (assuming 25% commission)
        "revenue_share": 0.17,
    },
    "email_sms": {
        "cac": 5,  # Very low
        "revenue_share": 0.09,
    },
}


# =============================================================================
# SCENARIO MODELING
# =============================================================================

def calculate_runway(bank_balance: float, monthly_burn: float, monthly_revenue: float) -> float:
    """Calculate months of runway remaining."""
    net_burn = monthly_burn - monthly_revenue
    if net_burn <= 0:
        return float('inf')  # Profitable
    return bank_balance / net_burn


def calculate_breakeven_revenue(fixed_costs: float, payout_ratio: float) -> float:
    """
    Calculate revenue needed to break even.

    Revenue - Payouts - Fixed Costs = 0
    Revenue - (Revenue * payout_ratio) - Fixed Costs = 0
    Revenue * (1 - payout_ratio) = Fixed Costs
    Revenue = Fixed Costs / (1 - payout_ratio)
    """
    return fixed_costs / (1 - payout_ratio)


def project_revenue_growth(
    starting_revenue: float,
    monthly_growth_rate: float,
    months: int
) -> List[float]:
    """Project revenue over time with compound growth."""
    revenues = []
    current = starting_revenue
    for _ in range(months):
        revenues.append(current)
        current *= (1 + monthly_growth_rate)
    return revenues


def calculate_marketing_spend_for_target(
    target_revenue: float,
    current_revenue: float,
    cac: float,
    aov: float,
    organic_ratio: float = 0.64  # 64% of revenue is organic/direct
) -> Dict:
    """
    Calculate marketing spend needed to hit revenue target.

    Assumes organic channels maintain their share while paid scales.
    """
    # Revenue that needs to come from paid channels
    organic_revenue = target_revenue * organic_ratio
    paid_revenue_needed = target_revenue - organic_revenue

    # Current paid contribution
    current_paid_revenue = current_revenue * (1 - organic_ratio)
    additional_paid_revenue = paid_revenue_needed - current_paid_revenue

    if additional_paid_revenue <= 0:
        return {
            "additional_customers_needed": 0,
            "additional_ad_spend": 0,
            "total_monthly_marketing": CURRENT_STATE["current_ad_spend"],
            "projected_roas": float('inf'),
        }

    # Customers needed
    additional_customers = additional_paid_revenue / aov

    # Ad spend needed (using current CAC, can be optimized)
    additional_ad_spend = additional_customers * cac

    return {
        "additional_customers_needed": int(additional_customers),
        "additional_ad_spend": additional_ad_spend,
        "total_monthly_marketing": CURRENT_STATE["current_ad_spend"] + additional_ad_spend,
        "projected_roas": additional_paid_revenue / additional_ad_spend if additional_ad_spend > 0 else 0,
    }


def run_funding_scenario(
    funding_amount: float,
    monthly_fixed_costs: float,
    starting_revenue: float,
    growth_rate: float,
    payout_ratio: float,
    marketing_spend_ratio: float = 0.15
) -> Dict:
    """
    Run a funding scenario to project cash position over 12 months.
    """
    bank_balance = CURRENT_STATE["bank_balance"] + funding_amount
    monthly_data = []

    revenue = starting_revenue

    for month in range(1, 13):
        # Calculate costs
        payouts = revenue * payout_ratio
        marketing = revenue * marketing_spend_ratio
        total_costs = monthly_fixed_costs + payouts + marketing

        # Net cashflow
        net_cashflow = revenue - total_costs
        bank_balance += net_cashflow

        monthly_data.append({
            "month": month,
            "revenue": revenue,
            "payouts": payouts,
            "payout_ratio": payouts / revenue if revenue > 0 else 0,
            "marketing": marketing,
            "fixed_costs": monthly_fixed_costs,
            "total_costs": total_costs,
            "net_cashflow": net_cashflow,
            "bank_balance": bank_balance,
        })

        # Grow revenue for next month
        revenue *= (1 + growth_rate)

    return {
        "funding_amount": funding_amount,
        "final_bank_balance": bank_balance,
        "total_revenue": sum(m["revenue"] for m in monthly_data),
        "total_costs": sum(m["total_costs"] for m in monthly_data),
        "breakeven_month": next(
            (m["month"] for m in monthly_data if m["net_cashflow"] >= 0),
            None
        ),
        "min_bank_balance": min(m["bank_balance"] for m in monthly_data),
        "monthly_data": monthly_data,
    }


# =============================================================================
# MARKETING BUDGET TEMPLATE
# =============================================================================

def generate_marketing_budget_template(
    total_monthly_budget: float,
    channel_allocation: Dict[str, float] = None
) -> Dict:
    """
    Generate a detailed marketing budget breakdown.

    Default allocation based on efficiency analysis:
    - Reduce inefficient paid ads
    - Increase high-ARPU channels (email, referrals)
    """
    if channel_allocation is None:
        # Recommended allocation based on data
        channel_allocation = {
            "google_ads": 0.20,      # Reduced from current
            "meta_ads": 0.20,        # Reduced from current
            "tiktok_ads": 0.10,      # Test channel
            "influencer_marketing": 0.15,
            "affiliate_payouts": 0.15,
            "email_sms_tools": 0.05,
            "giveaways_promotions": 0.10,
            "marketing_tools": 0.05,
        }

    budget = {}
    for channel, ratio in channel_allocation.items():
        budget[channel] = total_monthly_budget * ratio

    return {
        "total_budget": total_monthly_budget,
        "allocation": budget,
        "channel_percentages": channel_allocation,
    }


# =============================================================================
# MAIN ANALYSIS & REPORTING
# =============================================================================

def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_table(headers: List[str], rows: List[List], col_widths: List[int] = None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                      for i in range(len(headers))]

    # Header
    header_row = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * sum(col_widths))

    # Data rows
    for row in rows:
        print("".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def format_currency(amount: float) -> str:
    """Format number as EUR currency."""
    if amount >= 1_000_000:
        return f"€{amount/1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"€{amount/1_000:.1f}k"
    else:
        return f"€{amount:.0f}"


def format_percent(ratio: float) -> str:
    """Format ratio as percentage."""
    return f"{ratio * 100:.1f}%"


def run_full_analysis():
    """Run complete budget analysis and print report."""

    print_separator("TRUEYIELD MARKETING BUDGET & FUNDING MODEL")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("All values in EUR (€)")

    # ---------------------------------------------------------------------
    # SECTION 1: Current State
    # ---------------------------------------------------------------------
    print_separator("1. CURRENT FINANCIAL STATE (January 2026)")

    print(f"""
    Bank Balance:        {format_currency(CURRENT_STATE['bank_balance'])}
    Monthly Burn:        {format_currency(CURRENT_STATE['monthly_burn'])}

    Cost Breakdown:
      - Team:            {format_currency(CURRENT_STATE['team_cost'])}
      - Server/Infra:    {format_currency(CURRENT_STATE['server_infra'])}
      - Ads:             {format_currency(CURRENT_STATE['current_ad_spend'])}

    December 2025 Performance:
      - Revenue:         {format_currency(CURRENT_STATE['dec_revenue'])}
      - Payouts:         {format_currency(CURRENT_STATE['dec_payouts'])}
      - Payout Ratio:    {format_percent(CURRENT_STATE['dec_payout_ratio'])} (incl. giveaways)
      - Payout Ratio:    {format_percent(CURRENT_STATE['dec_payout_ratio_excl_giveaways'])} (excl. giveaways)

    Current Runway:      {calculate_runway(CURRENT_STATE['bank_balance'], CURRENT_STATE['monthly_burn'], CURRENT_STATE['dec_revenue']):.1f} months
    """)

    # ---------------------------------------------------------------------
    # SECTION 2: Breakeven Analysis
    # ---------------------------------------------------------------------
    print_separator("2. BREAKEVEN ANALYSIS")

    fixed_costs = CURRENT_STATE["team_cost"] + CURRENT_STATE["server_infra"]

    breakeven_scenarios = [
        ("Current (200% payout)", 2.0),
        ("Current excl. giveaways (100%)", 1.0),
        ("Target (50% payout)", 0.50),
        ("FPFX Benchmark (35%)", 0.35),
    ]

    print("\n    Revenue needed to break even at different payout ratios:\n")
    headers = ["Scenario", "Payout Ratio", "Breakeven Revenue"]
    rows = []
    for name, ratio in breakeven_scenarios:
        # Adjusted formula: Revenue needs to cover fixed costs + payouts
        # Revenue - (Revenue * ratio) = Fixed Costs
        # Revenue = Fixed Costs / (1 - ratio) -- but this breaks for ratio >= 1
        if ratio >= 1:
            breakeven = "IMPOSSIBLE (ratio >= 100%)"
        else:
            breakeven = format_currency(fixed_costs / (1 - ratio))
        rows.append([name, format_percent(ratio), breakeven])

    print_table(headers, rows, [35, 15, 20])

    print(f"""
    KEY INSIGHT: At 50% payout ratio, you need {format_currency(fixed_costs / 0.5)}/month to break even.
                 At 35% payout ratio, you need {format_currency(fixed_costs / 0.65)}/month to break even.
    """)

    # ---------------------------------------------------------------------
    # SECTION 3: Channel Efficiency
    # ---------------------------------------------------------------------
    print_separator("3. CHANNEL EFFICIENCY ANALYSIS (Aug-Dec 2025)")

    print("\n    Revenue by acquisition channel:\n")
    headers = ["Channel", "Revenue", "Customers", "ARPU", "% of Total", "CAC"]
    rows = []
    for channel, data in BASELINE["revenue_by_channel"].items():
        cac = CHANNEL_METRICS.get(channel, {}).get("cac", "N/A")
        if isinstance(cac, (int, float)):
            cac = format_currency(cac)
        rows.append([
            channel.replace("_", " ").title(),
            format_currency(data["revenue"]),
            data["purchasers"],
            format_currency(data["arpu"]),
            format_percent(data["revenue"] / BASELINE["total_revenue"]),
            cac
        ])

    print_table(headers, rows, [25, 12, 12, 10, 12, 10])

    print("""
    KEY INSIGHTS:
    - Paid ads have NEGATIVE ROI (ROAS 0.61x, CAC €222)
    - Organic + Direct = 64% of revenue (FREE!)
    - Email/SMS has highest ARPU (€8.14) but lowest volume
    - Referrals are efficient (€56 CAC estimated)

    RECOMMENDATION: Shift budget from paid ads to:
    1. Email/SMS marketing (highest ARPU)
    2. Affiliate program expansion
    3. Content/SEO for organic growth
    """)

    # ---------------------------------------------------------------------
    # SECTION 4: Marketing Budget Required for Targets
    # ---------------------------------------------------------------------
    print_separator("4. MARKETING SPEND TO HIT REVENUE TARGETS")

    targets = [50_000, 75_000, 100_000, 150_000]

    print("\n    Marketing spend needed to hit monthly revenue targets:\n")
    print("    (Assuming current CAC of €222 and 64% organic ratio)\n")

    headers = ["Target Revenue", "New Customers", "Add'l Ad Spend", "Total Marketing", "Notes"]
    rows = []

    for target in targets:
        result = calculate_marketing_spend_for_target(
            target_revenue=target,
            current_revenue=CURRENT_STATE["dec_revenue"],
            cac=BASELINE["google_ads"]["cpa"],
            aov=BASELINE["aov"],
        )

        notes = ""
        if result["additional_ad_spend"] > 50000:
            notes = "High spend - optimize CAC first"
        elif target >= 100000:
            notes = "CEO target range"

        rows.append([
            format_currency(target),
            result["additional_customers_needed"],
            format_currency(result["additional_ad_spend"]),
            format_currency(result["total_monthly_marketing"]),
            notes
        ])

    print_table(headers, rows, [18, 15, 18, 18, 25])

    print("""
    IMPORTANT: These numbers assume CURRENT inefficient CAC (€222).

    With optimized ads (target ROAS 1.5x):
    - CAC could drop to ~€150 (30% reduction)
    - Marketing spend would be significantly lower
    """)

    # ---------------------------------------------------------------------
    # SECTION 5: Funding Scenarios
    # ---------------------------------------------------------------------
    print_separator("5. FUNDING SCENARIOS (12-Month Projection)")

    funding_amounts = [0, 1_000_000, 2_000_000, 5_000_000]

    print("\n    Assumptions:")
    print("    - Starting revenue: €25k/month (December 2025)")
    print("    - Monthly growth rate: 20% (aggressive but achievable)")
    print("    - Payout ratio: 50% (target)")
    print("    - Marketing spend: 15% of revenue")
    print("    - Fixed costs: €100k/month\n")

    scenario_results = []
    for funding in funding_amounts:
        result = run_funding_scenario(
            funding_amount=funding,
            monthly_fixed_costs=100_000,
            starting_revenue=25_000,
            growth_rate=0.20,
            payout_ratio=0.50,
            marketing_spend_ratio=0.15,
        )
        scenario_results.append(result)

    headers = ["Funding", "Final Balance", "Min Balance", "Breakeven Mo.", "Year Revenue"]
    rows = []
    for result in scenario_results:
        rows.append([
            format_currency(result["funding_amount"]) if result["funding_amount"] > 0 else "No funding",
            format_currency(result["final_bank_balance"]),
            format_currency(result["min_bank_balance"]),
            f"Month {result['breakeven_month']}" if result["breakeven_month"] else "Never",
            format_currency(result["total_revenue"]),
        ])

    print_table(headers, rows, [15, 18, 18, 15, 15])

    # Detailed monthly for €2M scenario
    print("\n    Detailed Monthly Projection (€2M Funding Scenario):\n")

    scenario_2m = scenario_results[2]  # €2M scenario
    headers = ["Month", "Revenue", "Payouts", "PR%", "Marketing", "Fixed", "Net CF", "Balance"]
    rows = []
    for m in scenario_2m["monthly_data"]:
        rows.append([
            f"M{m['month']}",
            format_currency(m["revenue"]),
            format_currency(m["payouts"]),
            format_percent(m["payout_ratio"]),
            format_currency(m["marketing"]),
            format_currency(m["fixed_costs"]),
            format_currency(m["net_cashflow"]),
            format_currency(m["bank_balance"]),
        ])

    print_table(headers, rows, [8, 12, 12, 8, 12, 10, 12, 14])

    # ---------------------------------------------------------------------
    # SECTION 6: Recommended Marketing Budget Allocation
    # ---------------------------------------------------------------------
    print_separator("6. RECOMMENDED MARKETING BUDGET ALLOCATION")

    # Calculate recommended budget based on €50k revenue target initially
    monthly_budgets = [15_000, 25_000, 35_000, 50_000]

    print("\n    Optimized channel allocation (based on efficiency data):\n")

    for budget in monthly_budgets:
        template = generate_marketing_budget_template(budget)
        print(f"    Monthly Budget: {format_currency(budget)}")
        print("    " + "-" * 50)
        for channel, amount in template["allocation"].items():
            pct = template["channel_percentages"][channel]
            print(f"      {channel.replace('_', ' ').title():25} {format_currency(amount):>10} ({format_percent(pct)})")
        print()

    # ---------------------------------------------------------------------
    # SECTION 7: Key Recommendations
    # ---------------------------------------------------------------------
    print_separator("7. KEY RECOMMENDATIONS FOR INGMAR MEETING")

    print("""
    1. PAYOUT RATIO IS THE BIGGEST LEVER
       - Current: 200% (100% excl. giveaways)
       - Target: 45-50%
       - Every 10% reduction = ~€10k saved at €100k revenue
       - New risk management solution is critical

    2. PAID ADS NEED OPTIMIZATION BEFORE SCALING
       - Current ROAS: 0.61x (losing money)
       - Target ROAS: 1.5x minimum
       - Fix attribution, creative, targeting first
       - Then scale proven campaigns

    3. DOUBLE DOWN ON ORGANIC/REFERRAL
       - 64% of revenue comes from free channels
       - Expand affiliate program
       - Invest in content/SEO
       - Build email list aggressively

    4. FUNDING RECOMMENDATION
       - Minimum: €1M to survive 12 months
       - Recommended: €2M for growth runway
       - Aggressive: €5M for market dominance

    5. REVENUE PATH TO €100k/MONTH
       - Requires 4x growth from current €25k
       - At 20% monthly growth: ~8 months
       - At 15% monthly growth: ~10 months
       - Marketing budget should scale with revenue (15-20%)

    6. BREAKEVEN TARGET
       - At 50% payout ratio: Need €200k/month revenue
       - At 35% payout ratio: Need €154k/month revenue
       - WooCommerce + Futures (Jan 12) could accelerate growth
    """)

    print_separator("END OF REPORT")

    return {
        "baseline": BASELINE,
        "current_state": CURRENT_STATE,
        "channel_metrics": CHANNEL_METRICS,
        "scenario_results": scenario_results,
    }


# =============================================================================
# INTERACTIVE FUNCTIONS
# =============================================================================

def interactive_scenario(
    funding: float = 2_000_000,
    growth_rate: float = 0.20,
    payout_ratio: float = 0.50,
    marketing_ratio: float = 0.15,
    starting_revenue: float = 25_000,
    fixed_costs: float = 100_000,
):
    """
    Run a custom scenario with your own parameters.

    Parameters:
    - funding: Additional funding amount (default €2M)
    - growth_rate: Monthly revenue growth rate (default 20%)
    - payout_ratio: Challenge payout ratio (default 50%)
    - marketing_ratio: Marketing spend as % of revenue (default 15%)
    - starting_revenue: Starting monthly revenue (default €25k)
    - fixed_costs: Monthly fixed costs (default €100k)

    Returns detailed monthly projections.
    """
    result = run_funding_scenario(
        funding_amount=funding,
        monthly_fixed_costs=fixed_costs,
        starting_revenue=starting_revenue,
        growth_rate=growth_rate,
        payout_ratio=payout_ratio,
        marketing_spend_ratio=marketing_ratio,
    )

    print(f"\n{'='*60}")
    print("CUSTOM SCENARIO RESULTS")
    print(f"{'='*60}")
    print(f"""
    Parameters:
      Funding:           {format_currency(funding)}
      Growth Rate:       {format_percent(growth_rate)}/month
      Payout Ratio:      {format_percent(payout_ratio)}
      Marketing Ratio:   {format_percent(marketing_ratio)}
      Starting Revenue:  {format_currency(starting_revenue)}
      Fixed Costs:       {format_currency(fixed_costs)}/month

    Results:
      Final Balance:     {format_currency(result['final_bank_balance'])}
      Min Balance:       {format_currency(result['min_bank_balance'])}
      Breakeven Month:   {result['breakeven_month'] or 'Never'}
      Total Year Revenue: {format_currency(result['total_revenue'])}
    """)

    return result


def what_if_revenue(target_revenue: float, months: int = 12):
    """
    Calculate what growth rate needed to hit target revenue in X months.
    """
    starting = CURRENT_STATE["dec_revenue"]
    # target = starting * (1 + rate)^months
    # rate = (target/starting)^(1/months) - 1
    rate = (target_revenue / starting) ** (1 / months) - 1

    print(f"""
    To reach {format_currency(target_revenue)}/month in {months} months:

    Starting Revenue: {format_currency(starting)}
    Required Growth:  {format_percent(rate)}/month

    Monthly progression:
    """)

    revenues = project_revenue_growth(starting, rate, months)
    for i, rev in enumerate(revenues, 1):
        print(f"    Month {i:2}: {format_currency(rev)}")

    return rate


def export_to_csv(filename: str = "ty_budget_projection.csv"):
    """Export the €2M scenario to CSV for sharing."""
    import csv

    result = run_funding_scenario(
        funding_amount=2_000_000,
        monthly_fixed_costs=100_000,
        starting_revenue=25_000,
        growth_rate=0.20,
        payout_ratio=0.50,
        marketing_spend_ratio=0.15,
    )

    filepath = f"/Users/gbolahan/Documents/marketing budget/TY BUDGET MODEL/{filename}"

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Month", "Revenue", "Payouts", "Payout Ratio",
            "Marketing", "Fixed Costs", "Total Costs",
            "Net Cashflow", "Bank Balance"
        ])

        for m in result["monthly_data"]:
            writer.writerow([
                f"Month {m['month']}",
                f"€{m['revenue']:,.0f}",
                f"€{m['payouts']:,.0f}",
                f"{m['payout_ratio']*100:.1f}%",
                f"€{m['marketing']:,.0f}",
                f"€{m['fixed_costs']:,.0f}",
                f"€{m['total_costs']:,.0f}",
                f"€{m['net_cashflow']:,.0f}",
                f"€{m['bank_balance']:,.0f}",
            ])

    print(f"Exported to: {filepath}")
    return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_full_analysis()

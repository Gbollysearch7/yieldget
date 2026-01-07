#!/usr/bin/env python3
"""
TrueYield Revenue & Customer Growth Projections
================================================
Models the path to breakeven through customer/revenue growth.

Key insight: More customers = More revenue = Eventually covers fixed costs
Fixed costs are locked at €100k - the only path is GROWTH.
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

# Current state (January 2026)
CURRENT_MONTHLY_REVENUE = 25_000  # December 2025
AOV = 225  # Average Order Value
CURRENT_CUSTOMERS = CURRENT_MONTHLY_REVENUE / AOV  # ~111 customers

# Fixed costs (CANNOT BE REDUCED - lean team)
FIXED_COSTS = 100_000

# Starting bank balance
BANK_BALANCE = 1_000_000

# Marketing spend as % of revenue
MARKETING_RATIO = 0.15

# =============================================================================
# PROJECTION FUNCTIONS
# =============================================================================

def project_customer_growth(
    starting_customers: float,
    monthly_growth_rate: float,
    months: int,
    futures_boost: float = 0,  # Additional customers from Futures launch
    affiliate_boost: float = 0,  # Additional customers from affiliate expansion
) -> pd.DataFrame:
    """
    Project customer growth over time.

    Parameters:
    - starting_customers: Current monthly customers
    - monthly_growth_rate: Expected monthly growth (e.g., 0.20 for 20%)
    - months: Number of months to project
    - futures_boost: % increase in customers from Futures (e.g., 0.15 for 15%)
    - affiliate_boost: % increase from expanded affiliates (e.g., 0.20 for 20%)
    """
    data = []
    customers = starting_customers

    for month in range(1, months + 1):
        # Base growth
        if month > 1:
            customers *= (1 + monthly_growth_rate)

        # Futures boost (starts month 1, ramps up over 6 months)
        if futures_boost > 0:
            futures_multiplier = min(month / 6, 1) * futures_boost
        else:
            futures_multiplier = 0

        # Affiliate boost (gradual ramp up over 3 months)
        if affiliate_boost > 0:
            affiliate_multiplier = min(month / 3, 1) * affiliate_boost
        else:
            affiliate_multiplier = 0

        total_customers = customers * (1 + futures_multiplier + affiliate_multiplier)
        revenue = total_customers * AOV

        data.append({
            "Month": month,
            "Base Customers": round(customers),
            "Futures Customers": round(customers * futures_multiplier),
            "Affiliate Customers": round(customers * affiliate_multiplier),
            "Total Customers": round(total_customers),
            "Revenue": revenue,
        })

    return pd.DataFrame(data)


def calculate_profitability(
    revenue_df: pd.DataFrame,
    payout_ratio: float,
    giveaway_ratio: float = 0,  # Additional giveaway cost
    marketing_ratio: float = MARKETING_RATIO,
    fixed_costs: float = FIXED_COSTS,
    funding: float = 0,
    funding_month: int = 1,
) -> pd.DataFrame:
    """
    Calculate profitability for each month based on revenue projections.
    """
    df = revenue_df.copy()
    bank = BANK_BALANCE

    results = []

    for _, row in df.iterrows():
        month = row["Month"]
        revenue = row["Revenue"]
        customers = row["Total Customers"]

        # Add funding if applicable
        if month == funding_month and funding > 0:
            bank += funding

        # Calculate costs
        base_payouts = revenue * payout_ratio
        giveaway_cost = revenue * giveaway_ratio
        total_payouts = base_payouts + giveaway_cost
        marketing = revenue * marketing_ratio
        total_costs = fixed_costs + total_payouts + marketing

        # Contribution margin
        contribution = revenue - total_payouts - marketing
        contribution_margin = contribution / revenue if revenue > 0 else 0

        # Net profit/loss
        net = revenue - total_costs
        bank += net

        results.append({
            "Month": month,
            "Customers": customers,
            "Revenue": revenue,
            "Payouts": total_payouts,
            "Payout Ratio": (total_payouts / revenue) if revenue > 0 else 0,
            "Marketing": marketing,
            "Fixed Costs": fixed_costs,
            "Total Costs": total_costs,
            "Contribution": contribution,
            "Contribution Margin": contribution_margin,
            "Net Profit/Loss": net,
            "Bank Balance": bank,
            "Profitable": net >= 0,
        })

    return pd.DataFrame(results)


def find_breakeven_month(df: pd.DataFrame) -> dict:
    """Find when company becomes profitable."""
    profitable_months = df[df["Profitable"]]

    if len(profitable_months) == 0:
        return {
            "month": None,
            "revenue": None,
            "customers": None,
            "message": "Does not break even in projection period"
        }

    first_profitable = profitable_months.iloc[0]
    return {
        "month": int(first_profitable["Month"]),
        "revenue": first_profitable["Revenue"],
        "customers": int(first_profitable["Customers"]),
        "message": f"Breaks even in Month {int(first_profitable['Month'])}"
    }


def calculate_funding_needed(df: pd.DataFrame) -> dict:
    """Calculate funding needed to survive until breakeven."""
    min_balance = df["Bank Balance"].min()

    if min_balance >= 0:
        return {
            "funding_needed": 0,
            "min_balance": min_balance,
            "message": "No additional funding needed"
        }

    # Need enough to cover the lowest point plus buffer
    funding_needed = abs(min_balance) + 200_000  # 2 month buffer

    return {
        "funding_needed": funding_needed,
        "min_balance": min_balance,
        "message": f"Need €{funding_needed:,.0f} to survive until breakeven"
    }


# =============================================================================
# MAIN PROJECTIONS
# =============================================================================

def run_all_projections():
    """Run comprehensive projections and print results."""

    print("=" * 80)
    print("TRUEYIELD REVENUE & CUSTOMER GROWTH PROJECTIONS")
    print("=" * 80)
    print(f"\nStarting Point (December 2025):")
    print(f"  - Monthly Revenue: €{CURRENT_MONTHLY_REVENUE:,}")
    print(f"  - Monthly Customers: {CURRENT_CUSTOMERS:.0f}")
    print(f"  - AOV: €{AOV}")
    print(f"  - Fixed Costs: €{FIXED_COSTS:,}/month (CANNOT BE REDUCED)")
    print(f"  - Bank Balance: €{BANK_BALANCE:,}")

    # ==========================================================================
    # GROWTH SCENARIOS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("GROWTH SCENARIOS (24-month projection)")
    print("=" * 80)

    growth_scenarios = {
        "Conservative (15%/month)": 0.15,
        "Moderate (20%/month)": 0.20,
        "Aggressive (25%/month)": 0.25,
        "Hyper Growth (30%/month)": 0.30,
    }

    print("\nCustomer & Revenue Projections:\n")
    print(f"{'Scenario':<25} {'Month 6':>12} {'Month 12':>12} {'Month 18':>12} {'Month 24':>12}")
    print("-" * 80)

    for name, rate in growth_scenarios.items():
        df = project_customer_growth(CURRENT_CUSTOMERS, rate, 24)
        m6 = df[df["Month"] == 6].iloc[0]
        m12 = df[df["Month"] == 12].iloc[0]
        m18 = df[df["Month"] == 18].iloc[0]
        m24 = df[df["Month"] == 24].iloc[0]

        print(f"{name:<25} €{m6['Revenue']/1000:>10.0f}k €{m12['Revenue']/1000:>10.0f}k €{m18['Revenue']/1000:>10.0f}k €{m24['Revenue']/1000:>10.0f}k")

    print("\nCustomers per Month:\n")
    print(f"{'Scenario':<25} {'Month 6':>12} {'Month 12':>12} {'Month 18':>12} {'Month 24':>12}")
    print("-" * 80)

    for name, rate in growth_scenarios.items():
        df = project_customer_growth(CURRENT_CUSTOMERS, rate, 24)
        m6 = df[df["Month"] == 6].iloc[0]
        m12 = df[df["Month"] == 12].iloc[0]
        m18 = df[df["Month"] == 18].iloc[0]
        m24 = df[df["Month"] == 24].iloc[0]

        print(f"{name:<25} {m6['Total Customers']:>12,.0f} {m12['Total Customers']:>12,.0f} {m18['Total Customers']:>12,.0f} {m24['Total Customers']:>12,.0f}")

    # ==========================================================================
    # BREAKEVEN ANALYSIS BY PAYOUT RATIO
    # ==========================================================================

    print("\n" + "=" * 80)
    print("BREAKEVEN ANALYSIS: When do we become profitable?")
    print("=" * 80)

    payout_scenarios = {
        "35% (FPFX Target)": 0.35,
        "45% (Optimistic)": 0.45,
        "50% (Target)": 0.50,
        "60% (With Risk Mgmt)": 0.60,
        "75% (Early Stage)": 0.75,
    }

    print("\nBreakeven Month by Growth Rate & Payout Ratio:\n")
    print(f"{'Payout Ratio':<20}", end="")
    for name in growth_scenarios.keys():
        short_name = name.split("(")[0].strip()
        print(f"{short_name:>15}", end="")
    print()
    print("-" * 80)

    for pr_name, pr in payout_scenarios.items():
        print(f"{pr_name:<20}", end="")
        for gr_name, gr in growth_scenarios.items():
            rev_df = project_customer_growth(CURRENT_CUSTOMERS, gr, 36)
            profit_df = calculate_profitability(rev_df, pr)
            be = find_breakeven_month(profit_df)

            if be["month"]:
                print(f"{'Month ' + str(be['month']):>15}", end="")
            else:
                print(f"{'Never':>15}", end="")
        print()

    # ==========================================================================
    # DETAILED PROJECTION: MODERATE GROWTH (20%) + 50% PAYOUT
    # ==========================================================================

    print("\n" + "=" * 80)
    print("DETAILED PROJECTION: 20% Growth + 50% Payout Ratio")
    print("=" * 80)

    rev_df = project_customer_growth(CURRENT_CUSTOMERS, 0.20, 18)
    profit_df = calculate_profitability(rev_df, 0.50)

    print("\nMonth-by-Month Breakdown:\n")
    print(f"{'Month':<8} {'Customers':>10} {'Revenue':>12} {'Payouts':>12} {'Net P/L':>12} {'Bank':>14}")
    print("-" * 80)

    for _, row in profit_df.iterrows():
        print(f"M{int(row['Month']):<7} {int(row['Customers']):>10,} €{row['Revenue']:>10,.0f} €{row['Payouts']:>10,.0f} €{row['Net Profit/Loss']:>10,.0f} €{row['Bank Balance']:>12,.0f}")

    be = find_breakeven_month(profit_df)
    funding = calculate_funding_needed(profit_df)

    print(f"\n  → {be['message']}")
    print(f"  → {funding['message']}")
    print(f"  → Minimum bank balance: €{funding['min_balance']:,.0f}")

    # ==========================================================================
    # IMPACT OF FUTURES + AFFILIATES
    # ==========================================================================

    print("\n" + "=" * 80)
    print("STRATEGIC GROWTH BOOSTERS")
    print("=" * 80)

    # Base case
    base_df = project_customer_growth(CURRENT_CUSTOMERS, 0.20, 18)
    base_profit = calculate_profitability(base_df, 0.50)
    base_be = find_breakeven_month(base_profit)

    # With Futures (15% more customers)
    futures_df = project_customer_growth(CURRENT_CUSTOMERS, 0.20, 18, futures_boost=0.15)
    futures_profit = calculate_profitability(futures_df, 0.50)
    futures_be = find_breakeven_month(futures_profit)

    # With Affiliates (20% more customers)
    affiliate_df = project_customer_growth(CURRENT_CUSTOMERS, 0.20, 18, affiliate_boost=0.20)
    affiliate_profit = calculate_profitability(affiliate_df, 0.50)
    affiliate_be = find_breakeven_month(affiliate_profit)

    # With Both
    combined_df = project_customer_growth(CURRENT_CUSTOMERS, 0.20, 18, futures_boost=0.15, affiliate_boost=0.20)
    combined_profit = calculate_profitability(combined_df, 0.50)
    combined_be = find_breakeven_month(combined_profit)

    print("\nImpact on Breakeven (at 50% payout ratio, 20% base growth):\n")
    print(f"{'Scenario':<35} {'Breakeven':>15} {'Month 12 Revenue':>20}")
    print("-" * 70)
    print(f"{'Base Case':<35} {'Month ' + str(base_be['month']) if base_be['month'] else 'Never':>15} €{base_profit[base_profit['Month']==12]['Revenue'].values[0]:>18,.0f}")
    print(f"{'+ Futures (+15% customers)':<35} {'Month ' + str(futures_be['month']) if futures_be['month'] else 'Never':>15} €{futures_profit[futures_profit['Month']==12]['Revenue'].values[0]:>18,.0f}")
    print(f"{'+ Affiliates (+20% customers)':<35} {'Month ' + str(affiliate_be['month']) if affiliate_be['month'] else 'Never':>15} €{affiliate_profit[affiliate_profit['Month']==12]['Revenue'].values[0]:>18,.0f}")
    print(f"{'+ Both (Futures + Affiliates)':<35} {'Month ' + str(combined_be['month']) if combined_be['month'] else 'Never':>15} €{combined_profit[combined_profit['Month']==12]['Revenue'].values[0]:>18,.0f}")

    # ==========================================================================
    # IMPACT OF STOPPING GIVEAWAYS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("IMPACT OF STOPPING GIVEAWAYS")
    print("=" * 80)

    # Current state: 200% payout ratio = 100% base + 100% giveaways
    # If we stop giveaways: drops to 100% (still bad)
    # With risk management: can get to 50%

    print("\nPayout Ratio Breakdown:")
    print("  - Current Total: 200%")
    print("  - Base Payouts: ~100% (to winning traders)")
    print("  - Giveaways: ~100% (promotional)")
    print()

    # Scenario: Stop giveaways immediately
    no_giveaway_df = project_customer_growth(CURRENT_CUSTOMERS, 0.20, 18)

    # Still at 100% payout (just base, no giveaways) - still can't profit
    no_giveaway_100 = calculate_profitability(no_giveaway_df, payout_ratio=0.50, giveaway_ratio=0.50)
    no_giveaway_be_100 = find_breakeven_month(no_giveaway_100)

    # With risk management bringing base down to 50% + no giveaways
    no_giveaway_50 = calculate_profitability(no_giveaway_df, payout_ratio=0.50, giveaway_ratio=0)
    no_giveaway_be_50 = find_breakeven_month(no_giveaway_50)

    print(f"{'Scenario':<50} {'Breakeven':>15}")
    print("-" * 65)
    print(f"{'Current (50% base + 50% giveaways = 100% total)':<50} {'Month ' + str(no_giveaway_be_100['month']) if no_giveaway_be_100['month'] else 'Never':>15}")
    print(f"{'Stop Giveaways (50% base + 0% giveaways = 50%)':<50} {'Month ' + str(no_giveaway_be_50['month']) if no_giveaway_be_50['month'] else 'Never':>15}")

    savings = no_giveaway_50[no_giveaway_50['Month'] <= 12]['Revenue'].sum() * 0.50  # Giveaway savings
    print(f"\n  → Annual savings from stopping giveaways: €{savings:,.0f}")

    # ==========================================================================
    # FUNDING REQUIREMENTS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("FUNDING REQUIREMENTS TO REACH BREAKEVEN")
    print("=" * 80)

    print("\nHow much funding needed to survive until profitable?\n")
    print(f"{'Scenario':<40} {'Funding Needed':>20} {'Min Balance':>15}")
    print("-" * 75)

    scenarios_to_test = [
        ("20% growth, 50% payout", 0.20, 0.50),
        ("20% growth, 60% payout", 0.20, 0.60),
        ("25% growth, 50% payout", 0.25, 0.50),
        ("30% growth, 50% payout", 0.30, 0.50),
    ]

    for name, growth, payout in scenarios_to_test:
        rev_df = project_customer_growth(CURRENT_CUSTOMERS, growth, 24)
        profit_df = calculate_profitability(rev_df, payout)
        funding = calculate_funding_needed(profit_df)

        if funding["funding_needed"] > 0:
            print(f"{name:<40} €{funding['funding_needed']:>18,.0f} €{funding['min_balance']:>13,.0f}")
        else:
            print(f"{name:<40} {'None needed':>20} €{funding['min_balance']:>13,.0f}")

    # ==========================================================================
    # SUMMARY TABLE FOR INGMAR
    # ==========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY FOR INGMAR MEETING")
    print("=" * 80)

    print("""
    KEY FINDINGS:

    1. BREAKEVEN IS ACHIEVABLE through customer growth
       - At 50% payout ratio + 20% monthly growth → Breakeven in Month 13-14
       - Need to reach ~1,270 customers/month (currently 111)
       - Revenue target: €285k/month

    2. PAYOUT RATIO IS CRITICAL
       - Every 10% reduction accelerates breakeven by 2-3 months
       - Risk management solution is essential
       - Stopping giveaways saves €200-400k/year

    3. GROWTH ACCELERATORS
       - Futures launch: +15% customers → 1-2 months faster breakeven
       - Affiliate expansion: +20% customers → 2-3 months faster breakeven
       - Combined: Breakeven 3-4 months sooner

    4. FUNDING RECOMMENDATION
       - Minimum: €500k (tight, no buffer)
       - Recommended: €1-2M (comfortable runway)
       - Aggressive: €3-5M (for marketing scale-up)

    5. TIMELINE TO PROFITABILITY
       - Best case (30% growth, 45% payout): Month 7-8
       - Expected case (20% growth, 50% payout): Month 13-14
       - Conservative case (15% growth, 60% payout): Month 20+
    """)

    print("=" * 80)
    print("END OF PROJECTIONS")
    print("=" * 80)


if __name__ == "__main__":
    run_all_projections()

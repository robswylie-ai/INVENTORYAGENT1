"""Supply Chain Analytics Inventory Decision Agent Application

Dual-mode entry point:
1. Streamlit Web Dashboard (interactive browser UI):
     streamlit run app.py
2. Rich Terminal CLI (interactive or single-query command line):
     python app.py
     python app.py --question "Annual demand is 10,000, order cost is $50, holding is $4. What is our EOQ?"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from inventory_agent import InventoryDecisionAgent, ModelType


# ---------------------------------------------------------------------------
# CSV Dataset Analytics Engine
# ---------------------------------------------------------------------------

def analyze_inventory_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate and compute summary analytics for an uploaded inventory dataframe.

    Required columns:
    - Product
    - Category
    - Units
    - Unit Cost
    """
    required_cols = ["Product", "Category", "Units", "Unit Cost"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {
            "is_valid": False,
            "missing_columns": missing_cols,
            "num_rows": int(df.shape[0]),
            "num_cols": int(df.shape[1]),
            "column_names": list(df.columns),
            "dataframe": df,
        }

    # Clean numeric columns without in-place mutation of caller's original reference
    df_clean = df.copy()
    df_clean["Units"] = pd.to_numeric(
        df_clean["Units"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
    ).fillna(0)
    df_clean["Unit Cost"] = pd.to_numeric(
        df_clean["Unit Cost"].astype(str).str.replace("$", "").str.replace(",", "").str.strip(), errors="coerce"
    ).fillna(0.0)

    # 1. Total Units
    total_units = float(df_clean["Units"].sum())

    # 2. Average Unit Cost
    avg_unit_cost = float(df_clean["Unit Cost"].mean()) if len(df_clean) > 0 else 0.0

    # 3. Product with highest Units
    if len(df_clean) > 0:
        max_idx = df_clean["Units"].idxmax()
        highest_units_product = str(df_clean.loc[max_idx, "Product"])
        highest_units_count = float(df_clean.loc[max_idx, "Units"])
    else:
        highest_units_product = "N/A"
        highest_units_count = 0.0

    # 4. Group by Category and calculate total Units
    category_summary = (
        df_clean.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
        .sort_values(by="Total Units", ascending=False)
    )

    # 5. Inventory Value = Units * Unit Cost
    df_clean["Inventory Value"] = df_clean["Units"] * df_clean["Unit Cost"]

    # 6. Total Inventory Value across all products
    total_inventory_value = float(df_clean["Inventory Value"].sum())

    return {
        "is_valid": True,
        "missing_columns": [],
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "column_names": list(df.columns),
        "total_units": total_units,
        "average_unit_cost": avg_unit_cost,
        "highest_units_product": highest_units_product,
        "highest_units_count": highest_units_count,
        "category_summary": category_summary,
        "total_inventory_value": total_inventory_value,
        "analyzed_dataframe": df_clean,
    }


# ---------------------------------------------------------------------------
# CLI Mode (Rich Console Interface)
# ---------------------------------------------------------------------------

def run_cli(single_question: str | None = None) -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError:
        print("Rich library not installed. Install with: pip install rich")
        return

    console = Console()
    agent = InventoryDecisionAgent(use_llm_if_available=True)

    sample_questions = [
        "Annual demand is 12,000 units, ordering cost is $50 per order, and annual holding cost is $3 per unit. What is our EOQ and annual cost?",
        "A supplier offers volume discounts: Tier 1 ($20) for 1-499 units, Tier 2 ($18) for 500-999 units, and Tier 3 ($16) for 1000+ units. Annual demand is 5,000 units, ordering cost is $49, and holding cost is 20% of unit price. What quantity should we order?",
        "Daily demand averages 100 units with standard deviation of 15 units. Supplier lead time is 4 days. We target a 95% cycle service level. What safety stock and reorder point should we set?",
        "We review inventory periodically every 14 days with lead time of 7 days. Daily demand is 50 units with std dev of 10. Service level is 95%. Currently we have 300 on hand and 150 on order. What is the order-up-to level and recommended order?",
        "We sell commemorative event t-shirts for $25 that cost $10 to make. Unsold shirts can be salvaged for $3. Demand is normally distributed with mean 800 and std dev 120. How many shirts should we order in this single-period newsvendor model?",
    ]

    console.print(
        Panel.fit(
            "[bold cyan]Supply Chain Analytics - Inventory Decision Agent[/bold cyan]\n"
            "[white]7-Stage Decision Architecture:[/white] "
            "[yellow]Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain[/yellow]\n"
            "[green]Models Supported:[/green] EOQ | Quantity Discount | Safety Stock & ROP | Fixed-Period Review | Newsvendor",
            border_style="cyan",
        )
    )

    def process_and_display(q_text: str) -> None:
        console.print(f"\n[bold green]Processing Question:[/bold green] [italic]{q_text}[/italic]\n")
        response = agent.process_question(q_text)

        # Display Pipeline Trace
        trace_table = Table(title="[bold yellow]7-Stage Pipeline Trace[/bold yellow]", box=box.ROUNDED)
        trace_table.add_column("Step", justify="center", style="cyan", no_wrap=True)
        trace_table.add_column("Stage Name", style="magenta", no_wrap=True)
        trace_table.add_column("Status / Output Details", style="white")

        for step in response.pipeline_trace:
            details_str = ", ".join(f"{k}: {v}" for k, v in list(step.details.items())[:4])
            trace_table.add_row(str(step.step_number), step.step_name, f"{step.description} ({details_str})")

        console.print(trace_table)

        # Display Structured Details Panel (Final Answer, Intermediate Calculations, Assumptions)
        calc_res = response.calculation_result
        if hasattr(calc_res, "assumptions") and calc_res.assumptions:
            assumptions_str = "\n".join(f"  • {a}" for a in calc_res.assumptions)
            console.print(
                Panel(
                    f"[bold yellow]Model Assumptions Identified:[/bold yellow]\n{assumptions_str}",
                    title="[bold magenta]Supply Chain Assumptions[/bold magenta]",
                    border_style="magenta",
                )
            )

        # Display Verification Panel
        ver = response.verification
        ver_status = "[bold green]PASSED[/bold green]" if ver.is_valid else "[bold red]WARNING[/bold red]"
        checks_text = "\n".join(f"  • {c}" for c in ver.checks_passed)
        console.print(
            Panel(
                f"[bold]Status:[/bold] {ver_status}\n[bold]Automated Checks:[/bold]\n{checks_text}",
                title="[bold blue]Step 6: Mathematical Verification[/bold blue]",
                border_style="blue",
            )
        )

        # Display Executive Explanation
        console.print(
            Panel(
                response.executive_explanation,
                title="[bold green]Step 7: Executive Decision Explanation[/bold green]",
                border_style="green",
            )
        )

    if single_question:
        process_and_display(single_question)
        return

    while True:
        console.print("\n[bold]Select an action:[/bold]")
        console.print("  [1-5] Run standard supply chain sample question")
        console.print("  [C]   Enter custom natural-language question")
        console.print("  [Q]   Quit")

        choice = Prompt.ask("Your choice", default="1").strip().upper()
        if choice in ["Q", "QUIT", "EXIT"]:
            console.print("[cyan]Goodbye![/cyan]")
            break
        elif choice in ["1", "2", "3", "4", "5"]:
            idx = int(choice) - 1
            process_and_display(sample_questions[idx])
        elif choice == "C":
            custom_q = Prompt.ask("\nEnter your inventory question")
            if custom_q.strip():
                process_and_display(custom_q.strip())
        else:
            console.print("[red]Invalid choice. Please select 1-5, C, or Q.[/red]")


# ---------------------------------------------------------------------------
# Streamlit Web UI Mode
# ---------------------------------------------------------------------------

def run_streamlit() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        """
    **Autonomous Decision Architecture**:  
    `Customer Question` ➔ `Identify Model` ➔ `Extract Inputs` ➔ `Select Python Tool` ➔ `Calculate` ➔ `Verify` ➔ `Explain`
    """
    )

    agent = InventoryDecisionAgent(use_llm_if_available=True)

    # Preset Questions
    presets = {
        "1. Economic Order Quantity (EOQ)": (
            "Annual demand is 12,000 units, ordering cost is $50 per order, and annual holding cost is $3 per unit. "
            "Lead time is 5 days with 250 working days per year. What is our EOQ, order cycle, and total cost?"
        ),
        "2. All-Units Quantity Discount": (
            "Supplier offers volume discount price breaks: Tier 1 ($20) for 1-499 units, Tier 2 ($18) for 500-999 units, "
            "and Tier 3 ($16) for 1000+ units. Annual demand is 5,000 units, ordering cost is $49, and holding cost is 20% of unit price. "
            "What is our optimal order quantity?"
        ),
        "3. Safety Stock & Reorder Point (Continuous Review)": (
            "Average daily demand is 100 units with standard deviation of 15 units. Supplier lead time is 4 days. "
            "We target a 95% cycle service level. What safety stock and reorder point should we set?"
        ),
        "4. Fixed-Period Review (Order-Up-To Level)": (
            "We review inventory periodically every 14 days with supplier lead time of 7 days. Daily demand is 50 units "
            "with std dev of 10. Service level is 95%. Currently we have 300 on hand and 150 on order. "
            "What is our order-up-to level and recommended order quantity?"
        ),
        "5. Newsvendor (Single-Period Stochastic Demand)": (
            "We sell seasonal commemorative jackets for $120. Cost to produce is $50. Unsold jackets can be salvaged for $20. "
            "Expected demand is normally distributed with mean 500 and std dev 80. How many should we order in this single-period newsvendor model?"
        ),
    }

    st.sidebar.header("📋 Sample Scenarios")
    selected_preset_key = st.sidebar.selectbox("Choose a Benchmark Scenario:", list(presets.keys()))

    # Section called "Data Upload" as the last section in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload an inventory CSV file with Product, Category, Units, and Unit Cost columns",
    )

    custom_input = st.text_area(
        "Enter your inventory question in plain English:",
        value=presets[selected_preset_key],
        height=110,
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        submitted = st.button("🚀 Analyze & Decide", type="primary")

    if submitted or custom_input:
        with st.spinner("Executing 7-Stage Inventory Decision Agent Pipeline..."):
            response = agent.process_question(custom_input)

        st.success(f"Model Identified: **{response.model_name}** | Tool: `{response.selected_tool_name}`")

        # Layout in Tabs
        tab_summary, tab_pipeline, tab_structured, tab_math = st.tabs(
            [
                "📊 Executive Decision & Explanation",
                "🔍 7-Stage Pipeline Trace",
                "📋 Structured Output & Assumptions",
                "🧮 Numerical Details & Verification",
            ]
        )

        with tab_summary:
            st.markdown(response.executive_explanation)

            # Key metric cards
            st.subheader("Key Decision Metrics")
            m_cols = st.columns(4)

            res = response.calculation_result
            if response.model_type == ModelType.EOQ:
                m_cols[0].metric("Optimal Order Qty (Q*)", f"{res.optimal_order_quantity:,.0f} units")
                m_cols[1].metric("Orders per Year", f"{res.orders_per_year:.2f}")
                m_cols[2].metric("Total Inventory Cost", f"${res.total_inventory_cost:,.2f}")
                m_cols[3].metric("Reorder Point", f"{res.reorder_point} units" if res.reorder_point else "N/A")
            elif response.model_type == ModelType.QUANTITY_DISCOUNT:
                m_cols[0].metric("Optimal Order Qty (Q*)", f"{res.optimal_order_quantity:,.0f} units")
                m_cols[1].metric("Optimal Tier", f"Tier {res.optimal_tier_id}")
                m_cols[2].metric("Unit Price", f"${res.optimal_unit_price:.2f}")
                m_cols[3].metric("Total Annual Cost", f"${res.optimal_total_cost:,.2f}")
            elif response.model_type == ModelType.SAFETY_STOCK_ROP:
                m_cols[0].metric("Safety Stock", f"{res.safety_stock:,.1f} units")
                m_cols[1].metric("Reorder Point (ROP)", f"{res.reorder_point:,.1f} units")
                m_cols[2].metric("Cycle Service Level", f"{res.service_level * 100:.1f}%")
                m_cols[3].metric("Lead Time Demand", f"{res.lead_time_demand_mean:,.1f} units")
            elif response.model_type == ModelType.FIXED_PERIOD_REVIEW:
                m_cols[0].metric("Order-Up-To Level (M)", f"{res.order_up_to_level_M:,.1f} units")
                m_cols[1].metric("Safety Stock", f"{res.safety_stock:,.1f} units")
                m_cols[2].metric("Recommended Order (q)", f"{res.recommended_order_quantity:,.0f} units" if res.recommended_order_quantity is not None else "N/A")
                m_cols[3].metric("Protection Interval", f"{res.protection_interval} days")
            elif response.model_type == ModelType.NEWSVENDOR:
                m_cols[0].metric("Optimal Production Q*", f"{res.optimal_order_quantity:,.0f} units")
                m_cols[1].metric("Critical Ratio", f"{res.critical_ratio * 100:.2f}%")
                m_cols[2].metric("Expected Profit", f"${res.expected_profit:,.2f}")
                m_cols[3].metric("Expected Sales", f"{res.expected_sales:,.1f} units")

        with tab_pipeline:
            st.subheader("Execution Trace: Step-by-Step")
            for step in response.pipeline_trace:
                with st.expander(f"Step {step.step_number}: {step.step_name} - {step.description}", expanded=True):
                    st.json(step.details)

        with tab_structured:
            st.subheader("Deterministic Structured Tool Output")
            col_final, col_inter = st.columns(2)
            with col_final:
                st.markdown("#### Final Answer")
                if hasattr(res, "final_answer"):
                    st.json(res.final_answer)
            with col_inter:
                st.markdown("#### Intermediate Calculations")
                if hasattr(res, "intermediate_calculations"):
                    st.json(res.intermediate_calculations)

            st.markdown("#### Supply Chain Model Assumptions")
            if hasattr(res, "assumptions"):
                for a in res.assumptions:
                    st.write(f"- 📌 {a}")

        with tab_math:
            st.subheader("Automated Mathematical Verification")
            v = response.verification
            if v.is_valid:
                st.success("✅ All mathematical bounds, cost convexities, and optimality criteria verified.")
            else:
                st.warning("⚠️ Verification checks raised warnings.")

            for check in v.checks_passed:
                st.write(f"- ✔️ {check}")
            for warning in v.warnings:
                st.write(f"- ⚠️ {warning}")

            if v.sensitivity_analysis:
                st.subheader("Sensitivity Analysis")
                st.json(v.sensitivity_analysis)

    # -----------------------------------------------------------------------
    # Uploaded CSV Data Processing & Analysis
    # -----------------------------------------------------------------------
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            analysis = analyze_inventory_csv(df_raw)

            st.markdown("---")
            st.header("📊 Uploaded Inventory Dataset Analysis")

            if not analysis["is_valid"]:
                st.error(
                    f"⚠️ Missing required columns in uploaded CSV: **{', '.join(analysis['missing_columns'])}**.\n\n"
                    f"Expected columns: **Product, Category, Units, Unit Cost**."
                )
                st.write(f"**Number of Rows:** {analysis['num_rows']} | **Number of Columns:** {analysis['num_cols']}")
                st.write(f"**Column Names Present:** `{', '.join(analysis['column_names'])}`")
                st.subheader("Uploaded Raw Dataframe")
                st.dataframe(df_raw, use_container_width=True)
            else:
                # 4 & 5. Display dataset dimensions and column names
                st.subheader("Dataset Dimensions & Schema")
                d_c1, d_c2, d_c3 = st.columns([1, 1, 3])
                d_c1.metric("Number of Rows", f"{analysis['num_rows']:,}")
                d_c2.metric("Number of Columns", f"{analysis['num_cols']:,}")
                d_c3.write(f"**Column Names:** `{', '.join(analysis['column_names'])}`")

                # Analytical Summary Metrics
                st.subheader("Inventory Metrics Summary")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Total Units", f"{analysis['total_units']:,.0f}")
                kpi2.metric("Average Unit Cost", f"${analysis['average_unit_cost']:,.2f}")
                kpi3.metric(
                    "Product with Highest Units",
                    analysis["highest_units_product"],
                    delta=f"{analysis['highest_units_count']:,.0f} units",
                )
                kpi4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

                # Total Units by Category
                st.subheader("Total Units by Category")
                st.dataframe(
                    analysis["category_summary"],
                    use_container_width=True,
                    column_config={
                        "Total Units": st.column_config.NumberColumn(format="%d"),
                    },
                )

                # Augmented Dataframe with Inventory Value column
                st.subheader("Inventory Data (including Calculated Inventory Value)")
                st.dataframe(
                    analysis["analyzed_dataframe"],
                    use_container_width=True,
                    column_config={
                        "Units": st.column_config.NumberColumn(format="%d"),
                        "Unit Cost": st.column_config.NumberColumn(format="$%.2f"),
                        "Inventory Value": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )
        except Exception as e:
            st.error(f"Error reading or processing uploaded CSV file: {e}")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If launched with 'streamlit run app.py'
    if "streamlit" in sys.modules and any("streamlit" in arg for arg in sys.argv):
        run_streamlit()
    else:
        # Check if user requested Streamlit or CLI
        parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
        parser.add_argument("--question", "-q", type=str, help="Single inventory question to process")
        parser.add_argument("--web", action="store_true", help="Launch Streamlit Web UI")
        args = parser.parse_args()

        if args.web:
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        else:
            run_cli(single_question=args.question)


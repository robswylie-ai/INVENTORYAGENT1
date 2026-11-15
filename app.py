"""
Supply Chain Analytics Inventory Decision Agent Application
============================================================
Dual-Mode Application:
1. Streamlit Interactive Web Application (Default when run via 'streamlit run app.py')
2. Rich Terminal / Command Line Interface (CLI mode when run via 'python app.py --cli')

Implements the 6-stage architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

import sys
import argparse
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from inventory_agent import InventoryDecisionAgent
import inventory_tools as tools


# ============================================================================
# CLI INTERFACE IMPLEMENTATION
# ============================================================================

def run_cli_mode(initial_question: Optional[str] = None):
    """Runs the Inventory Decision Agent in interactive terminal mode."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown

    console = Console()
    agent = InventoryDecisionAgent()

    console.print(
        Panel.fit(
            "[bold cyan]Supply Chain Analytics Inventory Decision Agent[/bold cyan]\n"
            "[yellow]Architecture:[/yellow] Customer Question -> Identify Model -> Extract Inputs -> "
            "Select Python Tool -> Calculate -> Verify -> Explain\n"
            "[dim]Supported Models: 1. EOQ | 2. Quantity Discount | 3. Safety Stock & ROP | 4. Fixed-Period | 5. Newsvendor[/dim]",
            border_style="cyan",
        )
    )

    def process_query(q: str):
        console.print(f"\n[bold green]Processing Query:[/bold green] {q}")
        with console.status("[bold blue]Executing 6-stage agent pipeline...[/bold blue]"):
            trace = agent.answer_question(q)

        # Stage 1: Identify Model
        console.print(
            Panel(
                f"[bold]Model:[/bold] {trace.model_name}\n"
                f"[bold]Confidence:[/bold] {trace.confidence:.0%}\n"
                f"[bold]Reasoning:[/bold] {trace.identification_reasoning}",
                title="[bold yellow]Stage 1: Identify Model[/bold yellow]",
                border_style="yellow",
            )
        )

        # Stage 2: Extract Inputs
        input_table = Table(title="Stage 2: Extracted Numerical Inputs", border_style="magenta")
        input_table.add_column("Parameter", style="cyan")
        input_table.add_column("Extracted Value", style="white")
        for k, v in trace.extracted_inputs.items():
            input_table.add_row(str(k), str(v))
        console.print(input_table)

        # Stage 3: Select Tool
        console.print(
            Panel(
                f"[bold green]Selected Tool Function:[/bold green] [underline]{trace.selected_tool_name}[/underline]",
                title="[bold green]Stage 3: Select Python Tool[/bold green]",
                border_style="green",
            )
        )

        # Stage 4 & 5: Calculate & Verify
        status_color = "green" if trace.verification_passed else "red"
        status_icon = "✅ PASSED" if trace.verification_passed else "⚠️ WARNING"
        verif_text = f"[bold {status_color}]Verification Status: {status_icon}[/bold {status_color}]\n"
        for msg in trace.verification_messages:
            verif_text += f"- {msg}\n"

        console.print(
            Panel(
                verif_text.strip(),
                title="[bold blue]Stage 4 & 5: Calculation & Verification[/bold blue]",
                border_style="blue",
            )
        )

        # Stage 6: Explain
        console.print("\n[bold cyan]Stage 6: Executive Report[/bold cyan]")
        console.print(Markdown(trace.explanation_text))

    if initial_question:
        process_query(initial_question)
        return

    # Interactive loop
    sample_queries = [
        "We sell 10,000 units a year. Placing an order costs $50, and holding cost is $2 per unit per year. What is our optimal batch size?",
        "Annual demand is 5000 units. Ordering cost is $49. For 1-499 units price is $10, 500-999 units is $9.50, and 1000+ units is $9.00. Holding rate is 20%. Which discount tier is best?",
        "Average daily demand is 80 units with standard deviation of 12. Lead time is 5 days. For a 95% service level, what should our safety stock and reorder point be?",
        "We review inventory every 7 days with a 3 day lead time. Average daily demand is 50 with standard deviation 10. For a 95% service level and current stock of 120 units, what is our target level and order amount?",
        "A seasonal jacket costs $50 to make and sells for $120. Unsold jackets clear at $30 at the end of the season. Demand has mean 400 and standard deviation 60. How many jackets should we stock under newsvendor?",
    ]

    console.print("\n[bold]Quick Demo Scenarios (enter 1-5 or type your own question, 'q' to quit):[/bold]")
    for i, sq in enumerate(sample_queries, 1):
        console.print(f"  [dim]{i}.[/dim] {sq[:90]}...")

    while True:
        try:
            user_input = console.input("\n[bold yellow]Ask inventory question > [/bold yellow]").strip()
            if not user_input or user_input.lower() in ("q", "quit", "exit"):
                console.print("[dim]Exiting Inventory Decision Agent. Goodbye![/dim]")
                break
            if user_input.isdigit() and 1 <= int(user_input) <= len(sample_queries):
                user_input = sample_queries[int(user_input) - 1]
            process_query(user_input)
        except (KeyboardInterrupt, EOFError):
            break


# ============================================================================
# STREAMLIT WEB INTERFACE IMPLEMENTATION
# ============================================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        """
        **Interactive Decision Support System for Inventory Management**  
        *Architecture:* `Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain`
        """
    )

    agent = InventoryDecisionAgent()

    # Preset scenarios for quick demo
    preset_scenarios = {
        "Custom Question": "",
        "1. Economic Order Quantity (EOQ)": (
            "We sell 12,000 units annually. Each order costs $60 to place, and unit holding cost is $3 per year. "
            "Our operating calendar has 250 work days. What is our optimal batch size, order frequency, and total inventory cost?"
        ),
        "2. Quantity Discount Model": (
            "Annual demand is 5,000 units and ordering cost is $49. The supplier offers price breaks: "
            "1 to 499 units at $10.00, 500 to 999 units at $9.50, and 1,000 or more units at $9.00. "
            "Our holding cost rate is 20%. Which tier should we order to minimize total annual landed cost?"
        ),
        "3. Safety Stock & Reorder Point (Continuous Review)": (
            "Average daily demand is 80 units with a standard deviation of 12 units. Supplier lead time is 5 days. "
            "We target a 95% cycle service level. What should our safety stock and reorder point (ROP) be?"
        ),
        "4. Fixed-Period Review (Periodic Review / Order-Up-To)": (
            "We review inventory every 7 days with a 3 day lead time. Average daily demand is 50 units with standard deviation of 10. "
            "Target cycle service level is 95%. Current inventory on-hand plus on-order is 120 units. "
            "What is our order-up-to level (M) and recommended order quantity?"
        ),
        "5. Newsvendor Model (Single-Period Stochastic)": (
            "A seasonal ski jacket costs $50 to produce and sells for $130. Unsold jackets at the end of the season are marked down "
            "to a salvage value of $30. Customer demand is normally distributed with a mean of 450 units and standard deviation of 75 units. "
            "How many jackets should we stock under the newsvendor model to maximize expected profit?"
        ),
        "6. Multicriteria Inventory Classification (CSV Upload)": (
            "Conduct multicriteria inventory stratification using two criteria: Sales and POS Hits. "
            "Step 1: Classify Sales and POS Hits separately using Pareto thresholds (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%) and numerical scores (A=4, B=3, C=2, D=1). "
            "Step 2: Calculate Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score). "
            "Step 3: Assign Final Inventory Class: A (4.00-3.25), B (3.24-2.50), C (2.49-1.75), D (< 1.75). "
            "Report the number of items in each final category (A, B, C, D) and summarize the total Sales for each category."
        ),
    }

    # Sidebar
    st.sidebar.header("🎯 Scenario Presets")
    selected_preset = st.sidebar.selectbox("Choose a sample scenario or enter custom query:", list(preset_scenarios.keys()))
    default_text = preset_scenarios[selected_preset]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### Supported Inventory Models
        1. **EOQ (Wilson Formula):** Balances ordering and carrying costs.
        2. **Quantity Discount:** Multi-tier price break landed cost optimization.
        3. **Safety Stock & ROP:** Continuous review $(s, Q)$ with lead-time volatility.
        4. **Fixed-Period Review:** Periodic review $(R, S)$ base stock replenishment.
        5. **Newsvendor Model:** Critical fractile single-period profit optimization.
        6. **Inventory Classification:** Multicriteria ABC/D classification by Sales ($) and Hits via CSV upload.
        """
    )

    # DATA UPLOAD: Last section in the left sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    # DATA PROCESSING & ANALYSIS SECTION (Main Page)
    # Determine data source: uploaded CSV file takes priority; if none uploaded and preset 6 is chosen, use preset sample CSV
    df = None
    data_source_mode = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            data_source_mode = f"Uploaded File: `{uploaded_file.name}`"
        except Exception as e:
            st.error(f"Error reading uploaded CSV file: {str(e)}")
    elif "Inventory Classification" in selected_preset:
        try:
            df = pd.read_csv("sample_inventory_classification.csv")
            data_source_mode = "Preset Scenario Dataset (`sample_inventory_classification.csv`)"
            st.sidebar.info("💡 Preset sample file loaded (`sample_inventory_classification.csv`). You can also upload your own CSV file above.")
        except Exception:
            df = None

    if df is not None:
        st.markdown("---")
        if data_source_mode and "Preset" in data_source_mode:
            st.info(
                f"📊 **Preset Scenario Active:** Currently displaying multicriteria inventory classification for **{data_source_mode}**. "
                f"To classify your own inventory dataset, upload a CSV file in the **📁 Data Upload** section in the left sidebar."
            )
        try:

            num_rows, num_cols = df.shape
            col_names = df.columns.tolist()
            cols_lower = {c.strip().lower(): c for c in col_names}

            # Detect schema capabilities
            has_classification_cols = any(
                c in cols_lower for c in ["sales ($)", "sales", "sales$", "revenue", "sales_dollar"]
            ) and any(
                c in cols_lower for c in ["hits", "hit", "picks", "pick count", "transactions"]
            )

            has_portfolio_cols = all(
                c in df.columns for c in ["Product", "Category", "Units", "Unit Cost"]
            )

            if not has_classification_cols and not has_portfolio_cols:
                st.error(
                    f"⚠️ Uploaded CSV schema not recognized.\n\n"
                    f"**Detected Columns:** {', '.join(col_names)}\n\n"
                    f"**Supported File Schemas:**\n"
                    f"1. **Inventory Classification Schema:** `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`\n"
                    f"2. **Inventory Portfolio Schema:** `Product`, `Category`, `Units`, `Unit Cost`"
                )

            # ================================================================
            # CAPABILITY 1: MULTICRITERIA INVENTORY CLASSIFICATION (A, B, C, D)
            # ================================================================
            if has_classification_cols:
                st.subheader("🏷️ Multicriteria Inventory Stratification (Sales & POS Hits)")
                st.markdown(
                    """
                    **Multicriteria Inventory Stratification Methodology:**
                    - **Step 1 - Classify Sales and POS Hits Separately:**
                      - Pareto thresholds: **A (0-60%)**, **B (61-80%)**, **C (81-90%)**, **D (91-100%)**.
                      - Score conversion: **A = 4, B = 3, C = 2, D = 1**.
                    - **Step 2 - Combined Weighted Score:**
                      - Weights: **65% Sales** and **35% POS Hits**.
                      - Formula: `Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)`.
                    - **Step 3 - Final Inventory Class:**
                      - **A:** 4.00 - 3.25 | **B:** 3.24 - 2.50 | **C:** 2.49 - 1.75 | **D:** < 1.75.
                    """
                )

                # Execute classification via deterministic tools
                class_res = tools.classify_inventory_multicriteria(df)
                enriched_df = class_res.classified_df
                summary_df = class_res.class_summary_df
                cross_tab = class_res.cross_tab_matrix

                # REQUIRED CLASSIFICATION SUMMARY (KPIs)
                st.markdown("#### 📋 Required Classification Summary")
                kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
                with kcol1:
                    st.metric("Total Items", f"{len(enriched_df):,}")
                with kcol2:
                    st.metric("Category A Items", f"{class_res.final_answer['class_A_items']:,}", f"${class_res.final_answer['class_A_total_sales']:,.2f} Sales")
                with kcol3:
                    st.metric("Category B Items", f"{class_res.final_answer['class_B_items']:,}", f"${class_res.final_answer['class_B_total_sales']:,.2f} Sales")
                with kcol4:
                    st.metric("Category C Items", f"{class_res.final_answer['class_C_items']:,}", f"${class_res.final_answer['class_C_total_sales']:,.2f} Sales")
                with kcol5:
                    st.metric("Category D Items", f"{class_res.final_answer['class_D_items']:,}", f"${class_res.final_answer['class_D_total_sales']:,.2f} Sales")

                # Summary Table & 2D Stratification Matrix
                ccol1, ccol2 = st.columns([3, 2])
                with ccol1:
                    st.markdown("##### Stratification Summary Table")
                    display_summary_cols = [
                        "Final Inventory Class",
                        "Number of Items",
                        "Total Sales ($)",
                        "% of Items",
                        "% of Sales",
                        "Combined Score Range",
                        "Avg Weighted Score",
                    ]
                    st.dataframe(
                        summary_df[display_summary_cols].style.format({
                            "Number of Items": "{:,}",
                            "Total Sales ($)": "${:,.2f}",
                            "% of Items": "{:.1f}%",
                            "% of Sales": "{:.1f}%",
                            "Avg Weighted Score": "{:.2f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                with ccol2:
                    st.markdown("##### 2D Stratification Cross-Tabulation Matrix")
                    st.caption("Sales Class (65%) × POS Hits Class (35%) item counts:")
                    st.dataframe(cross_tab, use_container_width=True)

                # ============================================================
                # ADVISABLE INVENTORY MANAGEMENT POLICIES (ALL CLASSES)
                # ============================================================
                st.markdown("#### 🎯 Best Inventory Management Policies (All Classes)")
                st.caption(
                    "Standard managerial action guidelines differentiated by multicriteria material class, "
                    "incorporating cycle counting frequencies, cycle service levels (CSL), and safety-stock policies:"
                )

                if class_res.policy_summary_df is not None:
                    st.dataframe(class_res.policy_summary_df, use_container_width=True, hide_index=True)

                with st.expander("📘 Strategic Policy Guide: Safety Stock, Service Levels & Cycle Counting Impact"):
                    st.markdown(
                        """
                        ##### 1. Cycle Counting Standards
                        - **Class A Items — Every Week:** Class A items generate the bulk of dollar throughput and customer fulfillment impact. Weekly physical auditing is critical to maintain near-100% record accuracy and detect shrink or discrepancy before stockouts occur.
                        - **Class B Items — Once a Month:** Represents intermediate inventory investment; monthly physical verification maintains reliable replenishment data without excessive operational labor.
                        - **Class C Items — Once a Year:** Low monetary value; counting frequently yields negligible return on labor. An annual audit is sufficient.
                        - **Class D Items — Do Not Count:** Stagnant or obsolete material. Routine counting squanders warehouse labor on dead inventory.

                        ---

                        ##### 2. Safety Stock & Cycle Service Level (CSL) Policies
                        - **Class A Items — High CSL: 95%–98%:** Stockouts for Class A items inflict immediate lost sales and damage customer goodwill. Rigorous statistical safety stock ($SS = z \\times \\sigma_{LTD}$) must be maintained with continuous review $(s, Q)$ to protect revenue while avoiding expensive over-buffering.
                        - **Class B Items — Medium CSL: 85%–90%:** Moderate safety stock buffers protect against standard lead-time demand variability while keeping capital carrying costs balanced under periodic review $(R, S)$.
                        - **Class C Items — Low CSL: 70%–80%:** Stockout costs are low and carry minimal business impact. High safety stock is economically wasteful; maintain minimal buffer and order in bulk (large EOQ) using visual two-bin or min-max triggers.
                        - **Class D Items — No Safety Stock (Donate, Destroy, or Discount):** Zero service level commitment. Holding safety stock on inactive material guarantees compounding storage costs, tax exposure, and capital obsolescence. Standard action is **immediate disposition**:
                          - **Donate:** Tax deduction for qualified charitable recipients.
                          - **Destroy:** Certified scrap write-off to free up warehouse rack slots.
                          - **Discount:** Clearance markdowns or salvage liquidation.
                        """
                    )

                # Class Filter and Display Enriched DataFrame
                st.markdown("#### Stratified Inventory Dataset (with Calculated Scores)")
                filter_class = st.radio(
                    "Filter by Final Inventory Class:",
                    ["All Items", "Class A", "Class B", "Class C", "Class D"],
                    horizontal=True,
                )

                if filter_class != "All Items":
                    target_c = filter_class.split(" ")[-1]
                    display_class_df = enriched_df[enriched_df["Final Inventory Class"] == target_c].copy()
                else:
                    display_class_df = enriched_df.copy()

                # Identify sales column for formatting
                disp_sales_col = class_res.inputs_extracted["sales_column"]
                disp_hits_col = class_res.inputs_extracted["hits_column"]

                format_dict = {
                    disp_sales_col: "${:,.2f}",
                    disp_hits_col: "{:,.0f}",
                    "Sales Share (%)": "{:.2f}%",
                    "Cum Sales Share (%)": "{:.2f}%",
                    "Hits Share (%)": "{:.2f}%",
                    "Cum Hits Share (%)": "{:.2f}%",
                    "Combined Weighted Score": "{:.2f}",
                }
                # Add any existing portfolio currency columns if present
                if "Unit Cost" in display_class_df.columns:
                    format_dict["Unit Cost"] = "${:,.2f}"
                if "Inventory Value" in display_class_df.columns:
                    format_dict["Inventory Value"] = "${:,.2f}"

                st.dataframe(
                    display_class_df.style.format(format_dict),
                    use_container_width=True,
                )

            # ================================================================
            # CAPABILITY 2: EXISTING PORTFOLIO ANALYSIS
            # ================================================================
            if has_portfolio_cols:
                st.subheader("📊 Inventory Portfolio Analysis (Units & Value)")

                st.info(
                    f"**Dataset Overview:** {num_rows:,} rows × {num_cols} columns  |  "
                    f"**Columns:** {', '.join(col_names)}"
                )

                # Ensure numeric types using pandas
                df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0)
                df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce").fillna(0.0)

                # 1. Calculate total Units
                total_units = float(df["Units"].sum())

                # 2. Calculate average Unit Cost
                avg_unit_cost = float(df["Unit Cost"].mean())

                # 3. Identify Product with highest Units
                idx_highest = df["Units"].idxmax()
                highest_product_name = str(df.loc[idx_highest, "Product"])
                highest_product_units = float(df.loc[idx_highest, "Units"])

                # 4. Group data by Category and calculate total Units for each Category
                units_by_category = df.groupby("Category")["Units"].sum().reset_index()
                units_by_category.columns = ["Category", "Total Units"]
                units_by_category = units_by_category.sort_values(by="Total Units", ascending=False)

                # 5. Create new calculated column: Inventory Value = Units * Unit Cost
                df["Inventory Value"] = df["Units"] * df["Unit Cost"]

                # 6. Calculate total Inventory Value across all products
                total_inventory_value = float(df["Inventory Value"].sum())

                # RESULTS DISPLAY
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Total Units", f"{total_units:,.0f}")
                with mcol2:
                    st.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
                with mcol3:
                    st.metric("Product with Highest Units", f"{highest_product_name}", f"{highest_product_units:,.0f} units")
                with mcol4:
                    st.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

                # Display category breakdown and full dataframe with calculated column
                col_cat, col_table = st.columns([1, 2])
                with col_cat:
                    st.markdown("#### Total Units by Category")
                    st.dataframe(
                        units_by_category.style.format({"Total Units": "{:,.0f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )

                with col_table:
                    st.markdown("#### Uploaded Dataframe (with Calculated Inventory Value)")
                    formatted_df = df.copy()
                    formatted_display = formatted_df.style.format({
                        "Units": "{:,.0f}",
                        "Unit Cost": "${:,.2f}",
                        "Inventory Value": "${:,.2f}",
                    })
                    st.dataframe(formatted_display, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing uploaded CSV file: {str(e)}")

        st.markdown("---")

    # Input Area
    st.subheader("💬 Ask Your Inventory Question")
    user_query = st.text_area(
        "Enter your supply chain question in plain natural language:",
        value=default_text if default_text else (
            "We sell 10,000 units a year. Placing an order costs $50, and holding cost is $2 per unit per year. "
            "What is our optimal batch size?"
        ),
        height=100,
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_button = st.button("🚀 Run Decision Agent", type="primary", use_container_width=True)

    if run_button or user_query:
        with st.spinner("Executing 6-stage decision pipeline..."):
            trace = agent.answer_question(user_query)

        st.markdown("---")
        st.subheader("🔍 Execution Trace (6-Stage Pipeline)")

        # Stepper visualization columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("1. Identified Model", trace.model_name)
        with col2:
            st.metric("Confidence", f"{trace.confidence:.0%}")
        with col3:
            st.metric("3. Selected Python Tool", trace.selected_tool_name.split(".")[-1])
        with col4:
            status_emoji = "✅ Passed" if trace.verification_passed else "⚠️ Warning"
            st.metric("5. Verification", status_emoji)

        # Tabs for details
        tab_summary, tab_pipeline, tab_charts, tab_sensitivity = st.tabs([
            "📋 Decision & Explanation",
            "⚙️ Pipeline Trace (Stages 1-5)",
            "📊 Visual Analytics",
            "🎛️ What-If Sensitivity",
        ])

        # TAB 1: Executive Explanation
        with tab_summary:
            st.markdown(trace.explanation_text)

        # TAB 2: Deep Pipeline Trace
        with tab_pipeline:
            st.markdown("#### Stage 1: Model Identification")
            st.info(f"**Identified Model:** {trace.model_name}\n\n**Reasoning:** {trace.identification_reasoning}")

            st.markdown("#### Stage 2: Extracted Numerical Inputs")
            input_df = pd.DataFrame(
                [{"Parameter": k, "Extracted Value": str(v)} for k, v in trace.extracted_inputs.items()]
            )
            st.dataframe(input_df, use_container_width=True)

            st.markdown("#### Stage 3: Tool Dispatch")
            st.code(f"Selected Callable: {trace.selected_tool_name}(**extracted_inputs)", language="python")

            st.markdown("#### Stage 4 & 5: Calculation Result & Mathematical Verification")
            if trace.verification_passed:
                st.success("Mathematical and operational domain verification: PASSED")
            else:
                st.warning("Mathematical verification produced alerts:")
            for msg in trace.verification_messages:
                st.write(f"- {msg}")

        # TAB 3: Visual Analytics
        with tab_charts:
            st.markdown("#### Interactive Model Visualization")
            model = trace.identified_model

            if model == "eoq" and trace.calculation_result:
                res: tools.EOQResult = trace.calculation_result
                d = trace.extracted_inputs.get("annual_demand", 10000)
                s = trace.extracted_inputs.get("order_cost", 50)
                h = trace.extracted_inputs.get("holding_cost", 2.0)

                q_opt = res.optimal_order_quantity
                q_range = np.linspace(max(10, q_opt * 0.2), q_opt * 2.2, 100)
                order_costs = (d / q_range) * s
                holding_costs = (q_range / 2.0) * h
                total_costs = order_costs + holding_costs

                chart_df = pd.DataFrame({
                    "Order Quantity (Q)": q_range,
                    "Annual Ordering Cost": order_costs,
                    "Annual Holding Cost": holding_costs,
                    "Total Inventory Cost": total_costs,
                }).set_index("Order Quantity (Q)")

                st.line_chart(chart_df)
                st.caption(f"EOQ trade-off curve. Minimum occurs at Q* = {q_opt:.0f} where holding cost equals ordering cost.")

            elif model == "quantity_discount" and trace.calculation_result:
                res: tools.QuantityDiscountResult = trace.calculation_result
                tier_data = []
                for t in res.tier_evaluations:
                    tier_data.append({
                        "Tier": f"Tier {t.tier_index + 1} (${t.unit_price:.2f})",
                        "Purchasing Cost": t.annual_purchasing_cost,
                        "Ordering Cost": t.annual_ordering_cost,
                        "Holding Cost": t.annual_holding_cost,
                        "Total Cost": t.total_annual_cost,
                    })
                tdf = pd.DataFrame(tier_data).set_index("Tier")
                st.bar_chart(tdf[["Purchasing Cost", "Ordering Cost", "Holding Cost"]])
                st.caption("Breakdown of purchasing, ordering, and holding costs across discount brackets.")

            elif model == "safety_stock_rop" and trace.calculation_result:
                res: tools.SafetyStockROPResult = trace.calculation_result
                mu_ltd = res.expected_lead_time_demand
                sig_ltd = res.std_lead_time_demand
                rop = res.reorder_point

                x = np.linspace(max(0, mu_ltd - 3.5 * sig_ltd), mu_ltd + 3.5 * sig_ltd, 150)
                y = stats.norm.pdf(x, mu_ltd, sig_ltd)

                chart_df = pd.DataFrame({"Lead Time Demand": x, "Probability Density": y}).set_index("Lead Time Demand")
                st.line_chart(chart_df)
                st.caption(
                    f"Lead Time Demand Distribution: Mean = {mu_ltd:.1f}, "
                    f"Safety Stock = {res.safety_stock:.1f}, ROP = {rop:.1f} (CSL = {res.cycle_service_level:.0%})."
                )

            elif model == "fixed_period_review" and trace.calculation_result:
                res: tools.FixedPeriodReviewResult = trace.calculation_result
                st.bar_chart(pd.DataFrame({
                    "Component": ["Current Inventory", "Order Quantity Placed", "Expected Demand (R+L)", "Safety Stock"],
                    "Units": [res.current_inventory, res.recommended_order_quantity, res.expected_demand_protection_period, res.safety_stock],
                }).set_index("Component"))

            elif model == "newsvendor" and trace.calculation_result:
                res: tools.NewsvendorResult = trace.calculation_result
                mu = trace.extracted_inputs.get("demand_mean", 500)
                sigma = trace.extracted_inputs.get("demand_std", 100)
                q_opt = res.optimal_order_quantity

                x = np.linspace(max(0, mu - 3.5 * sigma), mu + 3.5 * sigma, 150)
                y = stats.norm.pdf(x, mu, sigma)
                chart_df = pd.DataFrame({"Demand": x, "Probability Density": y}).set_index("Demand")
                st.line_chart(chart_df)
                st.caption(f"Normal Demand Distribution: Optimal Order Q* = {q_opt:.0f} (Critical Ratio = {res.critical_ratio:.1%}).")

            elif model == "inventory_classification" and trace.calculation_result:
                res: tools.MulticriteriaClassificationResult = trace.calculation_result
                summary_df = res.class_summary_df
                st.bar_chart(summary_df.set_index("Class")[["% of SKUs", "% of Sales", "% of Hits"]])
                st.caption("Multicriteria Distribution: Comparison of SKU Count %, Sales Volume %, and Pick Hits % across Classes A, B, C, D.")

        # TAB 4: What-If Sensitivity
        with tab_sensitivity:
            st.markdown("#### Interactive What-If Parameter Tuning")
            model = trace.identified_model

            if model == "inventory_classification":
                st.info("Multicriteria classification evaluates your uploaded dataset across Pareto sales revenue and warehouse pick velocity criteria. Adjust raw data values in your CSV to evaluate changes.")
            elif model == "eoq":
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    adj_d = st.number_input("Annual Demand (D)", value=float(trace.extracted_inputs.get("annual_demand", 10000.0)), step=500.0)
                with col_s2:
                    adj_s = st.number_input("Order Cost (S)", value=float(trace.extracted_inputs.get("order_cost", 50.0)), step=5.0)
                with col_s3:
                    adj_h = st.number_input("Holding Cost (H)", value=float(trace.extracted_inputs.get("holding_cost", 2.0)), step=0.2)

                adj_res = tools.calculate_eoq(annual_demand=adj_d, order_cost=adj_s, holding_cost=adj_h)
                st.metric("Adjusted EOQ (Q*)", f"{adj_res.optimal_order_quantity:,.0f} units")
                st.metric("Adjusted Total Inventory Cost", f"${adj_res.total_annual_inventory_cost:,.2f}")

            elif model == "safety_stock_rop":
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    adj_sl = st.slider("Target Service Level (CSL)", min_value=0.80, max_value=0.999, value=float(trace.extracted_inputs.get("service_level", 0.95)), step=0.01)
                with col_s2:
                    adj_lt = st.number_input("Lead Time (days)", value=float(trace.extracted_inputs.get("lead_time", 5.0)), step=1.0)

                adj_res = tools.calculate_safety_stock_rop(
                    avg_demand_per_period=float(trace.extracted_inputs.get("avg_demand_per_period", 80.0)),
                    std_demand_per_period=float(trace.extracted_inputs.get("std_demand_per_period", 12.0)),
                    lead_time=adj_lt,
                    service_level=adj_sl,
                )
                st.metric("Adjusted Safety Stock", f"{adj_res.safety_stock:,.0f} units")
                st.metric("Adjusted Reorder Point (ROP)", f"{adj_res.reorder_point:,.0f} units")

            elif model == "newsvendor":
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    adj_p = st.number_input("Selling Price ($)", value=float(trace.extracted_inputs.get("selling_price", 120.0)), step=5.0)
                with col_s2:
                    adj_c = st.number_input("Unit Cost ($)", value=float(trace.extracted_inputs.get("unit_cost", 50.0)), step=5.0)

                adj_res = tools.calculate_newsvendor(
                    unit_cost=adj_c,
                    selling_price=adj_p,
                    demand_mean=float(trace.extracted_inputs.get("demand_mean", 400.0)),
                    demand_std=float(trace.extracted_inputs.get("demand_std", 60.0)),
                    salvage_value=float(trace.extracted_inputs.get("salvage_value", 30.0)),
                )
                st.metric("Adjusted Stocking Q*", f"{adj_res.optimal_order_quantity:,.0f} units")
                st.metric("Adjusted Expected Profit", f"${adj_res.expected_profit:,.2f}")
            else:
                st.info("Direct slider adjustments are enabled for EOQ, Safety Stock & ROP, and Newsvendor.")


# ============================================================================
# ENTRY POINT DISPATCHER
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("--cli", action="store_true", help="Run in interactive command-line mode")
    parser.add_argument("--question", type=str, default=None, help="Execute a single natural language question via CLI")
    args, unknown = parser.parse_known_args()

    # If --cli or --question is passed, or if run without streamlit runtime
    is_streamlit = "streamlit" in sys.modules and ("streamlit.web.cli" in sys.modules or "streamlit.runtime" in sys.modules)

    if args.cli or args.question:
        run_cli_mode(initial_question=args.question)
    elif is_streamlit:
        run_streamlit_app()
    else:
        # If user ran `python app.py` directly, default to CLI mode with instructions
        run_cli_mode()

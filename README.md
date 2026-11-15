# Supply Chain Analytics Inventory Decision Agent

An intelligent, multi-model inventory decision support agent that accepts natural-language questions and automatically solves foundational inventory optimization problems using a rigorous 6-stage pipeline:

$$\text{Customer Question} \longrightarrow \text{Identify Model} \longrightarrow \text{Extract Inputs} \longrightarrow \text{Select Python Tool} \longrightarrow \text{Calculate} \longrightarrow \text{Verify} \longrightarrow \text{Explain}$$

Every calculation tool deterministic Python engine returns **8 Student Verification Sections** so students can check and verify every mathematical step:
1. **Inputs Extracted** (`inputs_extracted`)
2. **Variable Definitions** (`variable_definitions`)
3. **Formula** (`formula`)
4. **Formula with Substituted Values** (`formula_with_substituted_values`)
5. **Intermediate Calculations** (`intermediate_calculations`)
6. **Final Answer** (`final_answer`)
7. **Verification Checks** (`verification_checks`)
8. **Assumptions** (`assumptions`)

All numerical calculation steps originate strictly within the deterministic Python tool. Missing information is never invented.

---

## 1. Supported Inventory Decision Models

1. **Economic Order Quantity (EOQ)**
   - Wilson unconstrained lot sizing:
     $$Q^* = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$
   - Verification checks: Positivity ($Q^* > 0$), First-order condition cost trade-off balance ($TC_{order} \approx TC_{hold}$), and Second-order sufficiency convexity ($\frac{d^2 TC}{dQ^2} = \frac{2DS}{Q^3} > 0$).
2. **Quantity Discount Model**
   - Multi-tier all-units price break evaluation. Computes unconstrained EOQ per bracket, adjusts to feasibility boundaries, and evaluates total annual landed cost.
   - Supports both fixed holding cost ($H$) and percentage holding cost ($H_j = i \times P_j$).
   - Verification checks: Global minimum cost check, bracket eligibility compliance, and economic trade-off justification.
3. **Safety Stock & Reorder Point (SS & ROP - Continuous Review)**
   - Supports all 4 uncertainty combinations:
     - Case 1: Constant demand, Constant lead time ($\sigma_d = 0, \sigma_L = 0 \implies \sigma_{LT}=0, SS=0, ROP = d \cdot L$)
     - Case 2: Variable demand, Constant lead time ($\sigma_d > 0, \sigma_L = 0 \implies \sigma_{LT} = \sqrt{L} \cdot \sigma_d$)
     - Case 3: Constant demand, Variable lead time ($\sigma_d = 0, \sigma_L > 0 \implies \sigma_{LT} = d \cdot \sigma_L$)
     - Case 4: Variable demand, Variable lead time ($\sigma_d > 0, \sigma_L > 0 \implies \sigma_{LT} = \sqrt{L \sigma_d^2 + d^2 \sigma_L^2}$)
   - $SS = z \cdot \sigma_{LT}, \quad ROP = (d \cdot L) + SS$
   - Verification checks: ROP lower bound ($ROP \ge d \cdot L$), non-negativity of safety stock, and variance decomposition integrity.
4. **Fixed-Period Review (Periodic Review / Order-Up-To Level)**
   - Protection period = $R + L$
   - Target inventory (Order-Up-To Level $M$) = Demand during protection period $[d \cdot (R+L)] + \text{Safety Stock } [z \cdot \sigma_d\sqrt{R+L}]$
   - Order quantity = $\max(0, \text{Target Inventory} - \text{Inventory Position})$
   - Verification checks: Protection window coverage, order quantity non-negativity floor at zero, and protection duration check.
5. **Newsvendor Model (Single-Period Stochastic Inventory)**
   - Underage cost: $C_u = P - C + g$
   - Overage cost: $C_o = C - V$
   - Critical ratio: $CR = \frac{C_u}{C_u + C_o}$
   - Optimal order quantity: $Q^* = \mu + z \cdot \sigma \quad (z = \Phi^{-1}(CR))$
   - Expected sales: $\mu - \sigma L(z)$, Expected leftovers: $Q^* - E[\text{Sales}]$, Expected profit: $P \cdot E[\text{Sales}] + V \cdot E[\text{Leftovers}] - C \cdot Q^* - g \cdot E[\text{Lost Sales}]$
   - Verification checks: Critical ratio bounds ($0 < CR < 1$), economic bias vs. mean demand, inventory conservation identity, and operational profitability.

---

## 2. Project Architecture

- **[inventory_tools.py](file:///c:/Users/robsw/OneDrive/BAN517-INV%20MGMT/inventory_tools.py)**: Pure Python computational engine implementing the 5 tools with typed dataclasses, input validators, intermediate trackers, verification checks, and assumption lists.
- **[inventory_agent.py](file:///c:/Users/robsw/OneDrive/BAN517-INV%20MGMT/inventory_agent.py)**: Natural language agent orchestrating the 6-stage pipeline and rendering the student verification report.
- **[app.py](file:///c:/Users/robsw/OneDrive/BAN517-INV%20MGMT/app.py)**: Dual-mode application (Streamlit web app and CLI) rendering interactive charts, what-if sliders, calculation steps, CSV portfolio analysis, and multicriteria inventory classification (Sales & Hits -> A, B, C, D).
- **[sample_inventory.csv](file:///c:/Users/robsw/OneDrive/BAN517-INV%20MGMT/sample_inventory.csv)**: Sample inventory portfolio dataset with columns `Product,Category,Units,Unit Cost`.
- **[sample_inventory_classification.csv](file:///c:/Users/robsw/OneDrive/BAN517-INV%20MGMT/sample_inventory_classification.csv)**: Sample multicriteria inventory classification dataset with columns `Item Number,Item Name,UoM,Sales ($),Hits`.
- **[tests/test_csv_analysis.py](file:///c:/Users/robsw/OneDrive/BAN517-INV%20MGMT/tests/test_csv_analysis.py)**: Automated unit tests for CSV parsing, column validation, portfolio analysis, and multicriteria classification logic.
- **[tests/test_inventory_tools.py](file:///c:/Users/robsw/OneDrive/BAN517-INV%20MGMT/tests/test_inventory_tools.py)**: 20 comprehensive unit tests covering all tools, all uncertainty cases, edge validation, and the 8 verification sections.

---

## 3. Running the Application and Tests

### Run Unit Tests
```bash
python -m pytest tests/test_inventory_tools.py -v
```

### Run Web Dashboard (Streamlit)
```bash
streamlit run app.py
```

### Run via Command-Line Interface (CLI)
```bash
python app.py --question "We sell 10,000 units a year. Placing an order costs $50, and holding cost is $2 per unit per year. What is our optimal batch size?"
```
```bash
python app.py --cli
```

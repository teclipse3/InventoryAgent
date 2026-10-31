# Supply Chain Analytics: Inventory Decision Agent

An autonomous, multi-model inventory decision agent designed for Supply Chain Analytics (UNCW BAN 517). The agent accepts natural-language inventory inquiries, identifies the relevant operational research model, extracts required parameters without inventing missing inputs, executes deterministic numerical computations, verifies mathematical constraints, and provides actionable executive explanations.

---

## 🏛️ 7-Step Decision Architecture

The agent strictly follows this sequential pipeline:

```
[1. Customer Question] 
        ↓
[2. Identify Models]   → Classifies query into 1 of 5 inventory models
        ↓
[3. Extract Inputs]    → Semantic regex & NLP parameter extraction
        ↓
[4. Select Python Tool]→ Matches model to deterministic Python function
        ↓
[5. Calculate]         → High-precision numerical execution (NumPy / SciPy)
        ↓
[6. Verify]            → Sanity checks, boundary assertions & optimality audits
        ↓
[7. Explain]           → Executive summary, mathematical formula & managerial insights
```

---

## 📦 Supported Deterministic Inventory Tools

Every calculation tool returns the **8 Educational Verification Deliverables** enabling students to audit step-by-step arithmetic. These are seamlessly integrated into the corresponding architectural steps within the dashboard:

| Architectural Step | Student Verification Guide Element | Description |
| :--- | :--- | :--- |
| **2. Identify Model** | **Item 8: Assumptions (`assumptions`)** | Operational constraints and theoretical boundaries under which the model holds. |
| **3. Extract Inputs** | **Item 1: Inputs Extracted (`inputs_extracted`)** | Cleanly parsed numerical parameters with units, roles, and discount tiers. |
| **3. Extract Inputs** | **Item 2: Variable Definitions (`variable_definitions`)** | Academic glossary of mathematical symbols ($D$, $S$, $H$, $L$, $\sigma$, etc.). |
| **4. Python Tool** | **Item 3: Formula (`formula`)** | Standard LaTeX formulation for theoretical reference executed by the Python tool. |
| **5. Calculate & Viz** | **Item 4: Formula with Substituted Values (`formula_with_substituted_values`)** | LaTeX formula with the problem's exact numerical values substituted into each term. |
| **5. Calculate & Viz** | **Item 5: Intermediate Calculations (`intermediate_calculations`)** | Deterministic computational steps (numerators, ratios, standard deviations, variances, partial costs). |
| **5. Calculate & Viz** | **Item 6: Final Answer (`final_answer`)** | Primary decision outputs, rounded lot sizes, financial metrics, and visual analytics. |
| **6. Verify and Audit** | **Item 7: Verification Checks (`verification_checks`)** | Domain sanity checks, non-negativity tests, optimality audits, and threshold boundaries. |

All tools:
- **Validate inputs strictly**
- **Execute deterministically in Python** (never construct numerical intermediate calculations independently in LLMs)
- **Never invent missing numerical inputs** (raises `ValueError` if required inputs are missing)

### 1. `calculate_eoq()`
- **Formula**: $Q^* = \sqrt{\frac{2 D S}{H}}$
- **Outputs**: EOQ lot size, annual ordering & holding costs, orders per year, cycle time (days), and deterministic reorder point.

### 2. `calculate_quantity_discount()`
- **Formula**: $TC(Q) = D \cdot C + \frac{D}{Q} S + \frac{Q}{2} H(C)$
- **Holding Cost Support**: Supports both fixed holding cost ($/unit/year) and percentage holding cost rate ($h \cdot C$).
- **Outputs**: Evaluates feasible quantities across all tiers, compares total annual costs, identifies price-break boundary adjustments, and calculates annual dollar & percentage savings vs baseline.

### 3. `calculate_rop()`
- **Supports All 4 Uncertainty Cases**:
  1. **Constant demand and constant lead time**: $\sigma_{LT} = 0, \quad SS = 0, \quad ROP = d \cdot L$
  2. **Variable demand and constant lead time**: $\sigma_{LT} = \sqrt{L} \cdot \sigma_d, \quad SS = z \cdot \sigma_{LT}, \quad ROP = d \cdot L + SS$
  3. **Constant demand and variable lead time**: $\sigma_{LT} = d \cdot \sigma_L, \quad SS = z \cdot \sigma_{LT}, \quad ROP = d \cdot L + SS$
  4. **Variable demand and variable lead time**: $\sigma_{LT} = \sqrt{L \cdot \sigma_d^2 + d^2 \cdot \sigma_L^2}, \quad SS = z \cdot \sigma_{LT}, \quad ROP = d \cdot L + SS$
- **Outputs**: Safety stock cushion, reorder trigger level, $z$-score, stockout risk, expected shortage per cycle, and implied fill rate.

### 4. `calculate_periodic_review()`
- **Formulas**:
  - $\text{Protection period} = \text{review period} + \text{lead time} = P + L$
  - $\text{Target inventory } (M) = \text{demand during protection period} + \text{safety stock} = d(P + L) + z \sigma_d \sqrt{P + L}$
  - $\text{Order quantity } (Q) = \max(0, \text{Target inventory} - \text{inventory position})$
- **Inventory Position**: Supports direct $IP$ or computes $IP = I_{OH} + I_{OO} - BO$.
- **Outputs**: Protection interval, target order-up-to level, current inventory position, recommended order quantity, and average inventory.

### 5. `calculate_newsvendor()`
- **Formulas**:
  - $\text{Underage cost: } C_u = P - C + g$
  - $\text{Overage cost: } C_o = C - V$
  - $\text{Critical ratio (critical fractile): } CR = \frac{C_u}{C_u + C_o} = \frac{P - C + g}{P - V + g}$
  - $\text{Recommended order quantity: } Q^* = F^{-1}(CR)$
- **Outputs**: Underage cost, overage cost, critical ratio, recommended order quantity, expected sales, expected leftover units, expected lost sales, expected revenue, expected cost, and expected net profit. Supports Normal, Uniform, and Discrete demand distributions.

---

## 🔄 Universal Demand Timeframe Normalization

The Natural Language Processing engine automatically detects and mathematically converts demand stated in any time unit into a standardized annual requirement rate ($D$) for deterministic calculation:
- **Monthly Demand**: $\text{Annual Demand } D = \text{Monthly Demand} \times 12$
- **Weekly Demand**: $\text{Annual Demand } D = \text{Weekly Demand} \times 52$
- **Quarterly Demand**: $\text{Annual Demand } D = \text{Quarterly Demand} \times 4$
- **Bi-Weekly Demand**: $\text{Annual Demand } D = \text{Bi-Weekly Demand} \times 26$
- **Semi-Annual Demand**: $\text{Annual Demand } D = \text{Semi-Annual Demand} \times 2$
- **Daily Demand**: $\text{Annual Demand } D = \text{Daily Demand} \times \text{working\_days\_per\_year}$ (defaults to 365 or extracts specified working days)

The normalization step is highlighted as **Step 0** in intermediate calculations, displayed as a conversion banner across the persistent dashboard, and documented with transparent conversion multipliers in Tab 3 (*Semantic Input Extraction*).

---

## 📌 Persistent Variables & Decision Dashboard

Positioned directly below the natural-language question prompt and above the architectural tabs, the application features an always-visible executive dashboard divided into two real-time panels:
1. **Key Input Variables (Extracted)**: Instantly presents the critical governing parameters extracted from the user's scenario (demand rates, setup costs, carrying rates, lead times, service targets, and wholesale discount schedules).
2. **Primary Decision & Financial Variables (Calculated Results)**: Persistently displays the cost-minimizing order lot ($Q^*, M$), reorder point ($ROP$), safety stock buffer ($SS$), total annual supply chain costs, expected profits, and cycle service levels.

This dashboard remains anchored at the top of the screen as users navigate through Steps 2 through 7 (Tabs), guaranteeing immediate situational awareness at every analysis stage.

---

## 📁 Repository Structure

```
Antigravity Project/
├── app.py                      # Interactive Streamlit Web Application & Dashboard
├── inventory_agent.py          # 7-step Agent Pipeline & Advanced NLP Semantic Extraction Engine
├── inventory_tools.py          # 5 Deterministic Python mathematical calculation tools
├── requirements.txt            # Project dependencies
├── tests/
│   └── test_inventory_tools.py # 67 unit & integration tests covering all models, NLP extraction & 8 verification deliverables
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Test Suite
```bash
pytest -v tests/test_inventory_tools.py
```
*(All 67 unit and integration tests verify deterministic mathematical accuracy, strict input validation without inventing inputs, terminology synonym recognition, parameter distinction, parenthetical and Greek notation support [e.g. (d), (σd), (L), (σL), (z)], compound ordering cost phrasing, currency symbols [$ / Rs. / ₹], percentage wording, the 8 educational verification deliverables, demand timeframe normalization, and end-to-end agent pipelines.)*

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interact with preset supply chain scenarios or enter your own custom natural language inventory scenarios.


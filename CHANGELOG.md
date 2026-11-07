# Changelog

All notable changes to the **Inventory Decision Agent** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning (SemVer 2.0.0)](https://semver.org/spec/v2.0.0.html).

---

## [1.5.1] - 2026-11-07

### Fixed
- **EOQ Lead Time NoneType Error in Dashboard**:
  - Resolved `TypeError: '>' not supported between instances of 'NoneType' and 'int'` on `render_persistent_variable_dashboard` triggered when loading EOQ questions without a specified lead time.
  - Implemented null-safe conditional checks for `lead_time_days`: renders `"Not specified"` with subtitle `"Optional delivery window"` when omitted, `"0 days"` for instantaneous delivery, and `"{L:.0f} days"` when explicitly defined.
  - Hardened all dashboard KPI cards and Tab 5 metric displays across all inventory models against `None` values returned by calculation results or unextracted parameters.
  - Expanded automated test suite from 104 to **109 passing tests** (100% pass rate).

---

## [1.5.0] - 2026-11-07

### Added
- **UI Layout Separation & Workspace Views**:
  - Implemented dedicated interface view navigation in the sidebar panel (`st.sidebar.radio`):
    1. `🤖 Inventory Decision Agent`: Focused classical inventory decision agent with prompt input, stress-test loader, persistent dashboard, and the 6 architectural tabs (Steps 2 through 7).
    2. `📊 Multi-Criteria Classification & Portfolio (CSV)`: Dedicated, uncluttered workspace for multi-criteria inventory classification (65/35 Sales & Hits methodology), Pareto curves, scatter matrix, and portfolio valuation.
    3. `📑 Combined View`: Sequential presentation with the Decision Agent on top and the CSV Classification section rendered strictly at the bottom below all 7 tabs.
  - Eliminated the UI layout flaw where CSV tables rendered in the middle of the page between the Customer Question and the tabbed outputs.
- **NLP Robustness & 25 Stress-Test Scenarios**:
  - Enhanced `InventoryNLPParser` with domain resilience against:
    - **Inadvertent typos & misspellings**: e.g., `demmand`, `ordring costt`, `holdng`, `carying chagre`, `leaddtime`, `servise level`, `reveiw`, `safty`, `qunatity discout`, `salvge`.
    - **Diverse international units of measure**: `liters`, `metric tons`, `gallons`, `barrels`, `kilograms`, `meters`, `crates`, `spools`, `cartons`, `vials`, `hectoliters`, `sacks`, `drums`, `rolls`, `bags`.
    - **International currencies**: `€` (Euros), `£` (British Pounds), `$` (USD), `₹` (Indian Rupees), `¥` (Japanese Yen), `CAD $` (Canadian Dollars), `AUD $` (Australian Dollars), `CHF` (Swiss Francs).
  - Integrated interactive `🧪 NLP Robustness & Stress-Test Scenarios` UI expander dynamically displaying 5 model-specific scenarios for whichever inventory model is active. One-click "Load Scenario" button populates the question input without cluttering the Preset Example dropdown.
- **Comprehensive Automated Test Suite**:
  - Created [`tests/test_nlp_stress_robustness.py`](./tests/test_nlp_stress_robustness.py) verifying all 25 stress-test scenarios (5 EOQ, 5 Quantity Discount, 5 ROP, 5 Fixed-Period Review, 5 Newsvendor). Test suite expanded from 79 to **104 passing tests** (100% pass rate).

### Changed
- Preserved preset dropdown with strictly the original 5 sample scenarios and custom query option.
- Updated version tags to `v1.5.0` across sidebar, footer, `Update Prompt.txt`, `README.md`, and `CHANGELOG.md`.

---

## [1.4.0] - 2026-11-07

### Added
- **Instructor Methodology Alignment (BAN 517)**: Adopted the official course methodology as default:
  - **Step 1 - Separate Single-Criterion Classification**: Pareto thresholds ($A: 0\text{--}60\%$, $B: 61\text{--}80\%$, $C: 81\text{--}90\%$, $D: 91\text{--}100\%$) mapped to integer scores ($A = 4, B = 3, C = 2, D = 1$).
  - **Step 2 - Combined Weighted Score**: Configured default weighting to **65% Sales** and **35% POS Hits**:
    $$\text{Combined Weighted Score} = 0.65(\text{Sales Score}) + 0.35(\text{Hits Score})$$
    *(e.g., Sales Class A [4] and Hits Class B [3] $\rightarrow 0.65(4) + 0.35(3) = 3.65 \rightarrow \text{Final Class A}$)*.
  - **Step 3 - Final Inventory Class Thresholds**:
    - Final Inventory Class A: $4.00 - 3.25$
    - Final Inventory Class B: $3.24 - 2.50$
    - Final Inventory Class C: $2.49 - 1.75$
    - Final Inventory Class D: $< 1.75$
  - **Required Classification Summary**: Dedicated executive KPI cards and structured summary table reporting SKU counts and total Sales for all items in each final category (A, B, C, D).
- **Automated Validation Suite**: Expanded unit tests verifying instructor methodology results on `ProjectDataV2_UTF8.csv` (22 Category A, 18 Category B, 23 Category C, 142 Category D; Total Sales: \$82,046,150.00; Category A Sales: \$54,788,937.00 [66.78%]), and added dedicated test for the instructor worked example (total 79 passed).

### Changed
- Default criteria weights updated across `inventory_tools.py`, `app.py`, and `Update Prompt.txt` from $50/50$ to $65/35$.
- Reset parameter action now restores instructor defaults ($65\%$ Sales / $35\%$ Hits & $60/80/90$ cutoffs).
- Updated version tags to `v1.4.0` in sidebar, footer, documentation, and prompt specifications.

---

## [1.3.0] - 2026-11-07

### Added
- **Direct Numeric Parameter Entry**: Added synchronized `st.number_input` widgets paired directly beside sliders for all Classification Parameters, Pareto Cutoffs, and Weights. Users can now directly type exact numeric values into any parameter field.
- **0.1 Increment Sliders**: Configured all parameter and cutoff sliders to move in precise 0.1 increments (`step=0.1`).
- **Real-Time Two-Way Widget Synchronization**: Implemented session-state callback handlers (`_sync_slider_to_num` and `_sync_num_to_slider`) so adjustments on either the slider or the numeric input instantaneously update the other.
- **Threshold Boundary Validation & Reset**: Added automatic boundary validation for Pareto cutoff ordering ($0\% < \text{Rank A} < \text{Rank B} < \text{Rank C} \le 100\%$) with non-blocking user warnings and a single-click "Reset Parameters to Defaults" button.
- **Automated Tests for Synchronization & Decimals**: Added unit tests verifying 0.1-increment decimal cutoff calculations, custom weights, and Streamlit session-state synchronization callbacks (total test suite expanded to 78 passing tests).

### Changed
- Updated `Update Prompt.txt` Section 2 ("Score Calculation Prompt") with detailed mandates for direct numeric input, 0.1 slider increments, paired widget synchronization, and threshold validation.
- Updated version display in Streamlit sidebar and footer to `v1.3.0`.

---

## [1.2.0] - 2026-11-07

### Added
- **Score Calculation Architecture**: Implemented 4-step multi-criteria stratification:
  1. Single-Criterion Cumulative Pareto Ranking with exact cutoffs ($0\text{--}60\% \rightarrow \text{A}$, $>60\text{--}80\% \rightarrow \text{B}$, $>80\text{--}90\% \rightarrow \text{C}$, $>90\% \rightarrow \text{D}$).
  2. Numerical Rank Scoring converting categorical ranks to discrete scores ($\text{Rank A} = 4, \text{Rank B} = 3, \text{Rank C} = 2, \text{Rank D} = 1$).
  3. Composite Weighted Scoring ($\text{Weighted Score} = [w_{\text{sales}} \times \text{Sales Score}] + [w_{\text{hits}} \times \text{Hits Score}]$) with default equal weighting ($0.50 / 0.50$).
  4. 25% Score-Range Increment Final Ranking allocating items into 4 equal segments across the observed score range ($\text{Rank A}$ in top 25%, $\text{Rank D}$ in bottom 25%).
- **Interactive Weighting Slider**: Added Streamlit UI controls in the parameter expander allowing dynamic criteria weight adjustment ($w_{\text{sales}}$ between 0% and 100%, with $w_{\text{hits}} = 100\% - w_{\text{sales}}$).
- **25% Increment Range Allocation Table**: Displaying score intervals, SKU counts, portfolio percentages, aggregate sales, and transaction hit volumes.
- **Enhanced Column Visibility**: Added `Sales Cumulative Total`, `Sales Cumulative %`, `Sales Rank`, `Sales Rank Score`, `Hits Cumulative Total`, `Hits Cumulative %`, `Hits Rank`, `Hits Rank Score`, `Weighted Score`, and `Final Rank` to the interactive Streamlit table and CSV export.
- **Enhanced System Prompts**: Updated [`Update Prompt.txt`](./Update%20Prompt.txt) with formalized execution triggers, default weights, diagnostic failure protocols, and SemVer release mandates.
- **Automated Unit Tests**: Added unit tests verifying 60/80/90 cumulative cutoffs, integer rank scores, score-range increments, and exact record counts on [`ProjectDataV2_UTF8.csv`](./ProjectDataV2_UTF8.csv) (205 records).

### Changed
- Refactored `calculate_multicriteria_classification()` in `inventory_tools.py` to use 60/80/90 cutoffs and 4-tier score range increments.
- Updated `process_and_classify_inventory_csv()` in `app.py` to support criteria weights and updated classification parameters.
- Updated Streamlit sidebar and footer to display active release version `v1.2.0`.

---

## [1.1.0] - 2026-11-01

### Added
- Initial Multi-Criteria Inventory Classification (MCIC) pipeline using Sales ($) and Hits.
- Single-file uploader routing with intelligent schema detection between portfolio valuation and classification datasets.
- 4×4 Cross-Tabulation matrix display of Sales vs. Hits classes.
- CSV export capability for classified inventory items.
- Initial unit tests covering CSV parsing and matrix cross-tabulation.

---

## [1.0.0] - 2026-10-31

### Added
- Autonomous 7-step Inventory Decision Agent pipeline (Question -> Identify -> Extract -> Tool -> Calculate -> Verify -> Explain).
- Five deterministic classical inventory models:
  1. Economic Order Quantity (EOQ)
  2. Quantity Discount Model
  3. Safety Stock and Reorder Point (ROP) with 4 uncertainty cases
  4. Fixed-Period Review Model (Order-Up-To Level)
  5. Newsvendor Model (Single-Period Probabilistic)
- Integrated CSV inventory portfolio valuation analysis (Units, Unit Cost, Category aggregation, Inventory Value).
- Persistent Important Variables Dashboard below customer inquiry input.
- Comprehensive test suite covering all classical models and educational deliverables.

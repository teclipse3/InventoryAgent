# Changelog

All notable changes to the **Inventory Decision Agent** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning (SemVer 2.0.0)](https://semver.org/spec/v2.0.0.html).

---

## [1.6.2] - 2026-11-08

### Fixed
- **Autonomous Holding Cost Computation for Percentage-Based Inventory Queries**:
  - Resolved `Calculation Pending: Must provide either 'holding_cost' directly or both 'holding_cost_rate' and 'unit_cost'. Missing numerical inputs cannot be invented.` when queries specify holding costs as a percentage of unit cost.
  - Implemented autonomous NLP calculation in `InventoryNLPParser.extract_numbers_with_context()`, computing $H = I \times C$ (`inputs["holding_cost"] = holding_cost_rate * unit_cost`) in Step 3 (Semantic Input Extraction) whenever both parameters are identified.
  - Added dedicated combined regex extraction patterns to parse rate and embedded unit cost in single compound clauses:
    - Parenthetical values: `holding cost is 20% of unit cost ($25)` / `cost ($25.00)`
    - Embedded currency descriptors: `holding cost is 20% of the $25 unit cost` / `carrying cost is 20% of $50 purchase price`
    - Relative and appositive clauses: `holding cost is 20% of unit cost, which is $25` / `holding cost is 20% of unit cost of $25`
  - Broadened unit cost synonym recognition across single-price inventory models:
    - Recognized `unit price`, `product cost`, `item cost`, `price per unit`, `cost per unit`, `component cost`, and variable notation `C = $25` / `C = 25`.
    - Added strict negative lookahead guards `(?!\s*(?:%|percent|percentage))` and negative lookbehinds (`order`, `setup`, `receiving`, `transportation`, `holding`) preventing cost tokens from accidentally capturing carrying percentages or setup fees as unit costs.
  - Implemented resilient fallback in `_run_eoq` allowing `unit_cost` to map from `unit_purchase_price`, `unit_price`, or `selling_price` when explicit purchase cost keys are omitted.
  - Updated persistent dashboard KPI card for Holding Cost (H) to display computed holding cost (`$5.00`) alongside contextual derivation subtitle (`20.0% of $25.00 cost`) or `"Awaiting unit cost"` if unit cost is not provided.
  - Surfaced `0b. Holding Cost Computation` in architectural Step 5 mathematical intermediate steps and Step 3 semantic extraction notes.
- **Test Suite Expansion**:
  - Added `TestNLPHoldingCostPercentageCalculations` in `tests/test_nlp_stress_robustness.py` with 14 automated test cases covering diverse international currencies (€, £, $, CHF), parenthetical syntaxes, embedded unit cost phrases, and variable assignments, expanding the test suite from 116 to **130 passing tests** (100% pass rate).

---

## [1.6.1] - 2026-11-07

### Added
- **Top 5 Items by Sales for Each Final Inventory Class**:
  - Implemented `extract_top_items_by_class()` in `inventory_tools.py` extracting the top 5 highest-revenue SKUs for Class A, Class B, Class C, and Class D.
  - Added a dedicated section in the Multi-Criteria Inventory Classification view featuring:
    - A 4-Class Comparison Overview tab (2x2 grid displaying side-by-side Top 5 tables, class totals, and revenue concentration percentages).
    - Dedicated individual tabs for Class A, Class B, Class C, and Class D with detailed SKU KPI metrics (Total Class Revenue, Top 5 Combined Sales, and Revenue Concentration %) and full item attributes.
  - Formatted with native Arrow `column_config` and `hide_index=True`.
- **Clickable Version Change Log Navigation & In-App Viewer**:
  - Transformed the Version identifier in the sidebar (`🏷️ Version: [v1.6.1](...)`) and footer into clickable markdown links navigating directly to `CHANGELOG.md` on GitHub.
  - Added an expandable `"📜 In-App Changelog"` viewer in the sidebar allowing users to read the full formatted release notes directly inside the application without navigating away.
- **Detailed Inventory Classification Product Search & Multi-Filter**:
  - Added real-time product search (`st.text_input`) in the Detailed Inventory Classification Table allowing users to find specific items by SKU / Item Number, Item Name, Product, or Category.
  - Implemented multi-criteria filter combination linking the Final Rank dropdown (`Rank A`, `Rank B`, `Rank C`, `Rank D`) with real-time text query matching.
  - Added interactive filter controls including match count status captions (`Displaying X of Y items`), a "❌ Clear Filters" reset button, and empty state warning alerts.
  - Added dual CSV download options: one-click export for the **Complete Classified CSV** and an export button for **Filtered Results** matching current search queries.

### Fixed
- **UnboundLocalError Resolution for Sales & Hits Column Identifiers in Multi-Criteria View**:
  - Resolved `UnboundLocalError: cannot access local variable 's_col_name' where it is not associated with a value` at line 2310 in `render_multicriteria_classification_section`.
  - Moved `s_col_name = result.sales_column_used` and `h_col_name = result.hits_column_used` initialization to the very top of `render_multicriteria_classification_section` immediately after `process_and_classify_inventory_csv` returns, guaranteeing clean lexical scope availability for both the Top 5 Items analytical section and the Detailed Classification Table.
- **Resilient Module Cache Import Handling on Streamlit Cloud**:
  - Resolved `ImportError: cannot import name 'extract_top_items_by_class' from 'inventory_tools'` occurring on Streamlit Community Cloud during live container hot-reloads.
  - Implemented dynamic module cache purging and resilient self-contained fallback architecture in `app.py`, ensuring instantaneous recovery from stale in-memory module states during git deployments.
- **Refined Methodology Callout & Sequential Step Renumbering**:
  - Renamed `Instructor Methodology (BAN 517)` to `Methodology` across the classification UI and documentation.
  - Renumbered duplicate Step 1 into clean sequential steps (Step 1 - Pareto Thresholds, Step 2 - Numerical Scores, Step 3 - Combined Weighted Score, Step 4 - Final Inventory Class Thresholds).
  - Formatted mathematical formulas and range intervals with clean markdown formatting, eliminating LaTeX/KaTeX browser text-doubling glitches on copy-paste.
- **StreamlitWidgetAlreadyInstantiatedError on Filter Reset & Parameter Defaults**:
  - Resolved `StreamlitWidgetAlreadyInstantiatedError: st.session_state['classification_rank_filter'] cannot be modified after the widget with that key is instantiated` triggered when clicking the "❌ Clear Filters" button in the Detailed Inventory Classification Table.
  - Implemented dedicated `on_click=_clear_classification_filters` callback executing in Streamlit's pre-instantiation lifecycle before widgets are rendered.
  - Refactored the parameter defaults reset button to use `on_click=_reset_instructor_parameters`, guaranteeing zero widget state collisions across all interactive table and parameter controls.
- **Corrected Column Configuration Types for Letter Ranks in Detailed Classification Table**:
  - Resolved `"The value can not be interpreted as a number"` cell tooltip indicators displayed across `Sales Rank`, `Hits Rank`, and `Final Rank` columns.
  - Replaced inappropriate `st.column_config.NumberColumn(format="%d")` with `st.column_config.TextColumn` for all alphabetical classification ranks ("A", "B", "C", "D"), ensuring text values render cleanly without numeric parsing errors.
  - Retained `st.column_config.NumberColumn(format="%d")` for integer scoring columns (`Sales Rank Score`, `Hits Rank Score`) and float columns (`Combined Weighted Score`, `Weighted Score`).
- **Large Dataset Table Virtualization & Styler Limit Resolution**:
  - Resolved `StreamlitAPIException: The dataframe has '731744' cells, but the maximum number of cells allowed to be rendered by Pandas Styler is configured to '262144'` triggered when rendering the Detailed Inventory Classification Table on 20,000 to 30,000 SKU datasets.
  - Configured `pd.set_option("styler.render.max_elements", 5_000_000)` at application module startup to support up to 5,000,000 cells without throwing element threshold exceptions.
  - Migrated both the Detailed Inventory Classification Table and Portfolio Valuation Table to Streamlit's native `column_config` (`st.column_config.NumberColumn`) with Apache Arrow direct serialization.
  - Eliminated server-side HTML table string compilation overhead (~60MB payload reduction), providing instantaneous table load times, smooth virtualized scrolling, and mathematical client-side column sorting.
  - Integrated resilient try/except fallback architecture preserving styled DataFrame rendering in legacy environments.
- **Test Suite Expansion**:
  - Expanded automated test suite from 110 to **116 passing tests** (100% pass rate) with dedicated test coverage for large dataset rendering resilience (>300,000 cells), Styler element configuration, SKU/product keyword search filtering, safe pre-instantiation reset callbacks, version/changelog URL linking, and top 5 items extraction by class.

---

## [1.6.0] - 2026-11-07

### Added
- **Default Sample Demonstration Dataset (20,000 to 30,000 SKUs)**:
  - Implemented `generate_sample_inventory_dataset()` producing a realistic, randomized inventory dataset between 20,000 and 30,000 items with log-normal Pareto revenue and pick velocity distributions.
  - Automatically loads this dataset as the default data when no custom CSV has been uploaded, enabling instant exploration of multi-criteria classification and portfolio valuation.
  - Added prominent demonstration mode banners and sidebar status indicators explicitly denoting that sample data is active.
- **Dynamic Sample Data Reload & Template Export**:
  - Added a "🔄 Reload Sample Data (New Random 20k–30k SKUs)" button in both the sidebar and main page to regenerate a fresh synthetic dataset on demand.
  - Added a "📥 Download Sample Data (CSV)" button allowing visitors to download the active sample dataset directly.
- **CSV Data Schema Requirements & Field Guide**:
  - Integrated an expandable instructions guide detailing required headers, accepted aliases (`Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits` / `Product`, `Category`, `Units`, `Unit Cost`), and automatic currency/comma sanitization.

### Changed
- **Fixed SKU Distribution Visual**:
  - Replaced the default `st.bar_chart` in the Classification Categories tab with a dedicated, customized matplotlib bar chart.
  - Fixed rotated/angled x-axis labels to remain strictly horizontal (`rotation=0`) for effortless readability across all classes (`Class A`, `Class B`, `Class C`, `Class D`).
  - Added direct data callouts on top of every bar showing exact SKU count and percentage (`{count:,.0f} ({pct:.1f}%)`) with color-coded supply chain tier styling.
- **Test Suite Expansion**:
  - Expanded automated test suite from 109 to **110 passing tests** (100% pass rate) with coverage for sample generation and schema compatibility.

---

## [1.5.1] - 2026-11-07

### Fixed
- **Customer Question Input Box Population on Scenario Load**:
  - Resolved issue where clicking "📥 Load Scenario" failed to populate the scenario question into the Customer Question input box (`st.text_area`).
  - Implemented Streamlit `on_click` callback (`_load_stress_scenario`) updating `st.session_state["main_user_query_textarea"]`, `st.session_state["user_query_text"]`, and `st.session_state["preset_example_selectbox"]` before rerun.
  - Reset sidebar preset dropdown to `"Select below or write custom question -->"` upon loading a scenario, preventing widget state cache collisions from overwriting loaded stress questions.
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

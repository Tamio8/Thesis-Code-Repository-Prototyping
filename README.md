# Thesis-Code-Repository-Prototyping

This repository contains the code used for generating clinical decision support prototypes as part of the PhD project *"A Bridge Between Care and Code: An Inquiry into Human-Centred AI in Intensive Care Decision-Making"* by Tamara Orth (Queensland University of Technology, 2026).

The thesis investigates how Artificial Intelligence and Machine Learning (AIML)-based clinical decision support systems can be designed to align with intensive care clinicians' values and needs. It takes a value-sensitive and tension-aware approach, showing how tensions in ICU decision-making signal complexity, potential trade-offs and limitations of AIML support. Insights gained from interviews and prototyping demonstrate that AIML based tools are most supportive when contextually anchored, integrated into existing workflows, and positioned as a tool that complements existing processes and devices. Within data science, this research offers a relational, context-sensitive and tension-aware approach of thinking for problem formulation.

📄 [For further details on the development, design rationale, or intended use of the prototypes, please refer to the full thesis.](https://eprints.qut.edu.au/264873/)

The code includes data filtering and simulated interface logic for three main prototypes:

- **Dynamic Summary**
    
- **Fluid Balance Tracker**
    
- **Similarity-Based Patient Explorer**
    

Each folder contains code related to a specific prototype, including mock interfaces and data processing. The `data-access-big-query/` folder contains supporting scripts for filtering ICU cohorts and exploring the structure of the MIMIC data.  The `prototype-code/` folder contains the code for the prototype visualisation.

### Data Source

All prototypes were developed in Python using Google Colab and Dash, and utilize data extracted from the MIMIC-IV and MIMIC-III databases via Google BigQuery. Access to MIMIC was approved via the PhysioNet credentialing process and conducted in accordance with ethical guidelines.

Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2XW26

Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

### Dependencies
Key libraries:

- `pandas`, `numpy`, `matplotlib`, `plotly`, `dash`, `dash-bootstrap-components`
    
- Data accessed via `google.cloud.bigquery` in Google Colab

### Notes

- This repository is designed for illustrative purposes only. Code was simplified for prototyping and demonstration and is not production-ready.
    
- Filtering choices (e.g., adult, septic, first ICU stay, non-immunocompromised) were made to **reduce data volume and simplify processing**, not for clinical significance.
    
- Code includes simulated interface logic and illustrative data to generate a *good enough* prototype for a deeper discussion with clinical stakeholders

### Citation

If you use or refer to this codebase or parts of this work, please cite the thesis: [https://eprints.qut.edu.au/264873/](https://eprints.qut.edu.au/264873/)

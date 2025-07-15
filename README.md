# Thesis-Code-Repository-Prototyping

This repository contains the code used for generating clinical decision support prototypes as part of the PhD project *"Bridging Human Needs and Machine Potential in Intensive Care:
Conceptual Considerations for a Care-Full Design at the Intersection of Context, Code and Complexity"* by Tamara Orth (Queensland University of Technology, 2025).

The code includes data filtering and simulated interface logic for three main prototypes:

- **Dynamic Summary**
    
- **Fluid Balance Tracker**
    
- **Similarity-Based Patient Explorer**
    

Each folder contains code related to a specific prototype, including mock interfaces and data processing. The `data-access-big-query/` folder contains supporting scripts for filtering ICU cohorts and exploring the structure of the MIMIC data.  The `prototype-code/` folder contains the code for the prototype visualisation.

For further details on the development, design rationale, or intended use of the prototypes, please refer to the full thesis.

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
    
- Code includes simulated interface logic and illustrative data to support clinician feedback collection.

### Citation

If you use or refer to this codebase, please cite the thesis: NEEDS EDIT

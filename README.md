# Energy Fraud Detection — Técnico+ Data Science Project

This project is part of the **Data Science for Engineers** course at **Técnico+ (IST Lisboa)**. The objective is to apply supervised machine learning techniques to detect fraudulent energy consumption using real-world client and invoice data.

---

## Project Structure

```
fraud_electricity_classification/
├── data/
│   ├── client_train.csv
│   ├── invoice_train.csv
│   ├── merged_data.csv
├── images/
├── dslabs/
│   ├── dslabs_functions.py
│   ├── config.py
│   ├── dslabs.mplstyle
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── modeling.ipynb
├── data_profiling.py
├── README.md
```

---

## Dataset

The dataset originates from a Kaggle competition focused on identifying fraud in electricity and gas consumption. This project uses only the labeled training data, merging `client_train.csv` and `invoice_train.csv` into `merged_data.csv`.

### Key Features

- **Client-level**: `client_catg`, `region`, `creation_date`
- **Invoice-level**: `consommation_level_1` to `consommation_level_4`, `old_index`, `new_index`, `months_number`
- **Target**: `target` (1 = Fraudulent, 0 = Not Fraudulent)

---

## Methodology

1. **Data Merging**: Combine client and invoice datasets on `client_id`.
2. **Data Profiling**:
   - Missing values analysis
   - Variable type classification (numeric, symbolic, binary, date)
   - Distribution analysis and outlier detection
3. **Visualization**:
   - Histograms, bar plots using custom `dslabs_functions.py`
   - Styled with `dslabs.mplstyle` for consistency
4. **Feature Engineering** (optional)
5. **Modeling** (to be implemented):
   - Classification using models such as Random Forest and XGBoost
   - Evaluation with metrics like accuracy, F1-score, precision, recall, and ROC-AUC

---

## Dependencies

Install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Ensure the following custom files are included:

- `dslabs/dslabs_functions.py`
- `dslabs/config.py`
- `dslabs.mplstyle`

---

## Learning Objectives

- Apply supervised learning to structured, real-world data  
- Perform effective exploratory data analysis  
- Build custom visualizations using matplotlib and custom styles  
- Prepare features for machine learning models  
- Evaluate and interpret classification results

---

## Notes

- Only the training data is used for this project. The test set is excluded as it lacks labels.  
- Time series analysis is not included in this version.  
- This project is focused on educational outcomes for the Técnico+ course.

---

## Author

**Giovanna Mazzali**  
Data Science for Engineers — Técnico+ (IST Lisboa)

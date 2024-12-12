# csci2470-final-project-fairness

This project implements an adversarial debiasing deep learning model to mitigate bias while maintaining predictive performance. We experiment with both the US Census Income dataset and COMPAS recidivism dataset to demonstrate the effectiveness of our approach.

### Project Structure
```
.
├── data/                      # Data directory
│   ├── adult.csv              # Original Census dataset
│   ├── adult_processed.csv    # Processed Census data
│   ├── compas.csv             # Original COMPAS dataset
│   └── compas_processed.csv   # Processed COMPAS data
├── results/                   # Results and evaluation metrics
├── data_processing.py         # Data preprocessing scripts
├── metrics.py                 # Fairness and performance metrics
├── model.py                   # Model architecture implementation
├── utils.py                   # Utility functions
└── evaluation.ipynb           # Evaluation notebook
```

### Acknowledgments
This project was completed as part of the CSCI2470 Deep Learning class at Brown University.
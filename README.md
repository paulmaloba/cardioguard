# cardioguard

A machine learning web application that predicts the risk of heart disease based on patient health metrics.

##  Features
- Trained k-NN classifier
- Feature scaling using MinMaxScaler
- Interactive Streamlit UI
- Mobile-friendly layout
- Clear risk interpretation (Low / High)

##  Model
- Algorithm: k-Nearest Neighbors
- Target: HeartDisease (binary)
- Evaluation Metric: Accuracy
- Hyperparameter tuning via GridSearchCV

##  Tech Stack
- Python
- pandas, numpy
- scikit-learn
- Streamlit

##  How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

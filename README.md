# task_21
#  Ford Car Price Prediction

A Machine Learning web application built with **Streamlit** that predicts the selling price of a Ford car based on user-provided specifications such as manufacturing year, mileage, transmission, fuel type, engine size, and more.

The application uses a trained Linear Regression model along with feature scaling and one-hot encoding to generate accurate price predictions.

---

##  Features

- Interactive Streamlit web interface
- Predicts Ford car selling price instantly
- User-friendly input forms
- Supports categorical and numerical features
- Automatic feature encoding
- Feature scaling using StandardScaler
- Uses a trained Linear Regression model
- Displays predicted price in British Pounds (£)
- Clean and responsive UI

---

##  Input Features

The application accepts the following inputs:

| Feature | Description |
|---------|-------------|
| Model | Ford car model name |
| Manufacturing Year | Year the car was manufactured |
| Mileage | Total kilometers/miles driven |
| Transmission | Manual, Automatic, Semi-Auto |
| Fuel Type | Petrol, Diesel, Hybrid, etc. |
| Road Tax | Annual road tax |
| MPG | Miles Per Gallon |
| Engine Size | Engine capacity in litres |

---

##  Output

The application predicts:

- **Estimated Selling Price (£)**

Example:

```
Predicted Selling Price:
£15,420.85
```

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

---

##  Project Structure

```
task_21/
│
├── app.py
├── LR_model.pkl
├── scaler.pkl
├── columns.pkl
├── Car.csv
├── requirements.txt
├── README.md
└── screenshots/
```

---

##  Machine Learning Workflow

1. Load trained Linear Regression model.
2. Load StandardScaler.
3. Load encoded training columns.
4. Collect user inputs.
5. Perform One-Hot Encoding.
6. Align columns with training data.
7. Scale numerical features.
8. Predict selling price.
9. Display prediction.

---

##  Installation

Clone the repository

```bash
git clone https://github.com/sanchitasontakke07-exe/task21.git
```

Move into the project folder

```bash
cd task21
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python -m streamlit run app.py
```

The application will open at

```
http://localhost:8501
```

---

## 📷 Screenshots

Add screenshots of your application here.

Example:

```
screenshots/home.png
screenshots/result.png
```

---

##  Required Files

The application requires the following files:

- LR_model.pkl
- scaler.pkl
- columns.pkl
- app.py

---

##  Model Used

- Linear Regression

### Preprocessing

- One-Hot Encoding
- StandardScaler

---

##  Dataset

Ford Car Dataset containing information such as:

- Model
- Year
- Transmission
- Mileage
- Fuel Type
- Tax
- MPG
- Engine Size
- Price

---

##  Future Improvements

- Support multiple ML algorithms
- Improve UI design
- Model comparison
- Upload CSV for batch predictions
- Deploy on Streamlit Cloud
- Prediction confidence score

**Sanchita Sontakke**

GitHub:
https://github.com/sanchitasontakke07-exe


This project is developed for educational and learning purposes.
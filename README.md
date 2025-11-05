# 🧩 Self-Healing Text Classification CLI

This project implements a **Self-Healing Classification System** — a CLI-based workflow that performs **text classification** using a simple machine learning model and a **human-in-the-loop fallback mechanism**.

The system predicts a label (e.g., positive, negative, neutral) for any input text and checks its confidence score.  
If confidence is **below a threshold**, it **asks the user for clarification** — preventing wrong automatic predictions and ensuring accuracy.

---

## 🚀 Features

- **End-to-End CLI Execution**
  - Enter any text and see the prediction with a confidence score.
- **Self-Healing Mechanism**
  - If model confidence is low, it triggers a fallback and asks for user clarification.
- **Confidence Threshold Control**
  - Adjustable threshold for fallback triggering (`--threshold` flag).
- **Structured Logging**
  - Every prediction, fallback, and final decision is logged.
- **Lightweight and Offline-Compatible**
  - Runs even without internet — uses a small rule-based classifier by default.

---

## 🧰 Project Structure

self_healing_classification_project_fixed/
│
├── run.bat # Launcher for Windows
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── data/
│ └── (optional training data can go here)
│
├── logs/
│ └── runtime.log # Logs saved here after execution
│
├── models/
│ └── (placeholder for trained models)
│
└── src/
├── main.py # CLI entry point
├── utils.py # Logger and config utilities
├── nodes.py # Inference, Confidence, and Fallback nodes
└── inference.py # Default classifier implementation


---

## ⚙️ Installation & Setup

### **Step 1: Extract the ZIP**
Unzip the downloaded file `self_healing_classification_project_fixed.zip` anywhere on your system.

### **Step 2: Run on Windows**
Double-click **`run.bat`** — it will:
- Create a Python virtual environment (`venv`)
- Install all dependencies automatically
- Launch the CLI

If you prefer manual setup, run these commands:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.main
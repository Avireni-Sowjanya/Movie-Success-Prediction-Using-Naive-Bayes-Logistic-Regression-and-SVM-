import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# -------------------------
# Global state (shared by buttons)
# -------------------------
dataset = None
processed_X = None
processed_y = None
label_enc = None

X_train = X_test = y_train = y_test = None

trained_models = {}   # store fitted pipelines: {"Naive Bayes": pipeline, ...}
metrics = {}          # store computed metrics per algorithm

# -------------------------
# Helpers
# -------------------------
def safe_insert(text_widget, msg):
    text_widget.config(state="normal")
    text_widget.insert(tk.END, msg + "\n")
    text_widget.see(tk.END)
    text_widget.config(state="disabled")

def clear_output():
    output_text.config(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.config(state="disabled")

# -------------------------
# Button actions
# -------------------------
def upload_dataset():
    global dataset, processed_X, processed_y, label_enc, trained_models, metrics, X_train, X_test, y_train, y_test
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not file_path:
        return

    try:
        dataset = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Load error", f"Could not read CSV:\n{e}")
        return

    # Basic cleaning / preprocessing:
    dataset.dropna(axis=1, how="all", inplace=True)   # drop empty columns
    dataset.dropna(axis=0, how="all", inplace=True)   # drop empty rows
    dataset = dataset.dropna()                         # drop rows with any missing values (simple)
    if dataset.shape[1] < 2:
        messagebox.showerror("Data error", "Dataset must have at least 2 columns (features + target).")
        return

    # Split features / target (last column is target)
    X = dataset.iloc[:, :-1].copy()
    y = dataset.iloc[:, -1].copy()

    # Convert categorical features to numeric via one-hot
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if it's non-numeric
    label_enc = None
    if y.dtype == object or not np.issubdtype(y.dtype, np.number):
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y)
    else:
        y = y.values

    processed_X = X
    processed_y = y

    # Reset previous train/test/models/metrics
    X_train = X_test = y_train = y_test = None
    trained_models.clear()
    metrics.clear()

    # Update UI
    clear_output()
    safe_insert(output_text, f"Dataset loaded: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    safe_insert(output_text, f"Processed feature columns after one-hot: {processed_X.shape[1]}")
    info_label.config(text=f"Loaded dataset: {dataset.shape[0]} rows × {dataset.shape[1]} cols")

    messagebox.showinfo("Upload", "Dataset uploaded and preprocessed successfully.\n(Last column considered target)")

def generate_train_test():
    global X_train, X_test, y_train, y_test, processed_X, processed_y
    if processed_X is None or processed_y is None:
        messagebox.showerror("Error", "Upload & preprocess the dataset first.")
        return

    try:
        strat = processed_y if len(np.unique(processed_y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(processed_X, processed_y,
                                                            test_size=0.3, random_state=42, stratify=strat)
    except Exception as e:
        messagebox.showerror("Error", f"Train/Test split failed:\n{e}")
        return

    # Update UI
    safe_insert(output_text, f"Train/Test split created: Training {len(X_train)} rows | Testing {len(X_test)} rows")
    info_label.config(text=f"Dataset: {dataset.shape[0]} rows × {dataset.shape[1]} cols\n"
                           f"Training: {len(X_train)} rows | Testing: {len(X_test)} rows")
    messagebox.showinfo("Train/Test", "Train & Test sets generated.")

def evaluate_and_display(name, clf):
    """
    Fit the classifier (with scaler if appropriate), evaluate on X_test,
    store model & metrics and append results to the output text.
    """
    global trained_models, metrics, X_train, y_train, X_test, y_test
    if X_train is None:
        messagebox.showerror("Error", "Generate Train/Test Model first.")
        return

    # For SVM and Logistic, scale features. For NB scaling is OK too.
    pipe = make_pipeline(StandardScaler(), clf)
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        messagebox.showerror("Training error", f"{name} training failed:\n{e}")
        return

    # Save trained model
    trained_models[name] = pipe

    # Predict & compute metrics
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    success = int((y_pred == y_test).sum())
    failure = int((y_pred != y_test).sum())

    metrics[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "success": success,
        "failure": failure
    }

    # Append nicely formatted block to output
    out = (
        f"\n{name} Accuracy  : {acc:.2f}\n"
        f"{name} Precision : {prec:.2f}\n"
        f"{name} Recall    : {rec:.2f}\n"
        f"{name} FScore    : {f1:.2f}\n"
        f"{name} Success   : {success}    Failure: {failure}\n"
    )
    safe_insert(output_text, out)

def run_naive_bayes():
    evaluate_and_display("Naive Bayes", GaussianNB())

def run_logistic_regression():
    evaluate_and_display("Logistic Regression", LogisticRegression(max_iter=2000, solver='lbfgs'))

def run_svm():
    evaluate_and_display("Support Vector Machine", SVC())

def predict_movie_success():
    """
    Show per-model classification reports (full) using stored trained models.
    """
    if not trained_models:
        messagebox.showerror("Error", "Train at least one model first.")
        return

    safe_insert(output_text, "\nDetailed classification reports:")
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test)
            rep = classification_report(y_test, y_pred, zero_division=0)
            safe_insert(output_text, f"\n--- {name} ---")
            for line in rep.splitlines():
                safe_insert(output_text, line)
        except Exception as e:
            safe_insert(output_text, f"{name} report error: {e}")

    messagebox.showinfo("Prediction", "Prediction and reports added to the white box.")

def comparison_graph():
    if not metrics:
        messagebox.showerror("Error", "Run algorithms first (press Run Naive Bayes / Logistic / SVM).")
        return

    algorithms = list(metrics.keys())
    success = [metrics[a]["success"] for a in algorithms]
    failure = [metrics[a]["failure"] for a in algorithms]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, success, width, label='Success')
    ax.bar(x + width/2, failure, width, label='Failure')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=20)
    ax.set_ylabel('Number of Predictions')
    ax.set_title('Algorithm Comparison: Success vs Failure')
    ax.legend()
    plt.tight_layout()
    plt.show()

# -------------------------
# GUI layout
# -------------------------
root = tk.Tk()
root.title("Movie Success Prediction")
root.geometry("1100x700")
root.configure(bg="#9ed7f5")  # light blue background

# Title (centered)
title_label = tk.Label(root,
                       text="Movie Success Prediction Using Naive Bayes, Logistic Regression and Support Vector Machine",
                       font=("Arial", 16, "bold"),
                       bg="#9adf7a", fg="#0b4f6c",
                       pady=10)
title_label.pack(fill="x")
title_label.config(anchor="center")

# White main box (output)
main_frame = tk.Frame(root, bg="white", bd=2, relief="groove")
main_frame.pack(padx=20, pady=18, fill="both", expand=True)

# Info label at top of white box
info_label = tk.Label(main_frame, text="No dataset loaded", bg="white", anchor="w", font=("Arial", 11))
info_label.pack(fill="x", padx=8, pady=(6,0))

# Scrollable text widget for results
text_container = tk.Frame(main_frame, bg="white")
text_container.pack(fill="both", expand=True, padx=8, pady=8)

output_text = tk.Text(text_container, wrap="word", state="disabled", font=("Consolas", 10))
scroll = tk.Scrollbar(text_container, command=output_text.yview)
output_text.configure(yscrollcommand=scroll.set)
scroll.pack(side="right", fill="y")
output_text.pack(side="left", fill="both", expand=True)

# Buttons area (2 rows as requested)
btn_frame = tk.Frame(root, bg="#9ed7f5")
btn_frame.pack(pady=(0,16))

# Row 1: Upload, Generate, Naive Bayes, Logistic Regression
row1 = [
    ("Upload & Preprocess Dataset", upload_dataset),
    ("Generate Train & Test Model", generate_train_test),
    ("Run Naive Bayes Algorithm", run_naive_bayes),
    ("Run Logistic Regression Algorithm", run_logistic_regression),
]
for i, (txt, cmd) in enumerate(row1):
    b = tk.Button(btn_frame, text=txt, command=cmd, width=30, height=2, bg="white")
    b.grid(row=0, column=i, padx=6, pady=6)

# Row 2: SVM, Predict, Comparison Graph
row2 = [
    ("Run SVM Algorithm", run_svm),
    ("Predict Movie Success from Test Data", predict_movie_success),
    ("Comparison Graph", comparison_graph),
]
for i, (txt, cmd) in enumerate(row2):
    b = tk.Button(btn_frame, text=txt, command=cmd, width=30, height=2, bg="white")
    b.grid(row=1, column=i, padx=6, pady=6)

# Start with friendly instruction
safe_insert(output_text, "Ready. Please upload a CSV where the LAST column is the target (success/failure).")
safe_insert(output_text, "After upload: press 'Generate Train & Test Model', then run algorithms.")

root.mainloop()

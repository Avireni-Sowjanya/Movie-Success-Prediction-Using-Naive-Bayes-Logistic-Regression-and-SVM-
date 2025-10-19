import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -------------------------
# Global state
# -------------------------
dataset = None
processed_X = None
processed_y = None
label_enc = None
movie_names = None
selected_movie = None
X_train = X_test = y_train = y_test = None
trained_models = {}
metrics = {}

# -------------------------
# Helpers
# -------------------------
def safe_insert(msg):
    output_text.config(state="normal")
    output_text.insert(tk.END, msg + "\n")
    output_text.see(tk.END)
    output_text.config(state="disabled")

def clear_output():
    output_text.config(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.config(state="disabled")

# -------------------------
# Button actions
# -------------------------
def upload_dataset():
    global dataset, processed_X, processed_y, label_enc, trained_models, metrics, movie_names, selected_movie
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not file_path:
        return

    try:
        dataset = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Load error", f"Could not read CSV:\n{e}")
        return

    dataset.dropna(axis=1, how="all", inplace=True)
    dataset.dropna(axis=0, how="all", inplace=True)
    dataset = dataset.dropna()

    if dataset.shape[1] < 3:  # need at least Movie + features + target
        messagebox.showerror("Data error", "Dataset must have at least 3 columns: Movie, features, target.")
        return

    # Assume first column is Movie name
    movie_names = dataset.iloc[:, 0].values
    X = dataset.iloc[:, 1:-1].copy()
    y = dataset.iloc[:, -1].copy()

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if categorical
    label_enc = None
    if y.dtype == object or not np.issubdtype(y.dtype, np.number):
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y)
    else:
        y = y.values

    processed_X = X
    processed_y = y

    trained_models.clear()
    metrics.clear()

    clear_output()
    safe_insert(f"Dataset loaded: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    safe_insert(f"Processed features: {processed_X.shape[1]}")

    # Ask user for movie name immediately
    selected_movie = simpledialog.askstring("Movie Selection", "Enter the movie name to analyze:")
    if not selected_movie:
        messagebox.showwarning("Warning", "No movie selected. Please upload again and enter a movie name.")
        return

    if selected_movie not in dataset.iloc[:, 0].values:
        messagebox.showerror("Error", f"Movie '{selected_movie}' not found in dataset.")
        return

    safe_insert(f"Selected Movie: {selected_movie}")
    info_label.config(text=f"Dataset: {dataset.shape[0]} rows × {dataset.shape[1]} cols | Selected movie: {selected_movie}")

def generate_train_test():
    global X_train, X_test, y_train, y_test, processed_X, processed_y
    if processed_X is None or processed_y is None:
        messagebox.showerror("Error", "Upload dataset first.")
        return

    strat = processed_y if len(np.unique(processed_y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(processed_X, processed_y,
                                                        test_size=0.3, random_state=42, stratify=strat)
    safe_insert(f"Train/Test split created: Training {len(X_train)} rows | Testing {len(X_test)} rows")

def evaluate_and_display(name, clf):
    global trained_models, metrics, selected_movie
    if X_train is None:
        messagebox.showerror("Error", "Generate Train/Test Model first.")
        return
    if not selected_movie:
        messagebox.showerror("Error", "Please upload dataset and select a movie.")
        return

    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe

    # Find row for selected movie
    row = dataset.loc[dataset.iloc[:, 0] == selected_movie].iloc[0, 1:-1]
    row_df = pd.DataFrame([row])
    row_df = pd.get_dummies(row_df)
    row_df = row_df.reindex(columns=processed_X.columns, fill_value=0)

    pred = pipe.predict(row_df)[0]
    if label_enc:
        pred_label = label_enc.inverse_transform([pred])[0]
    else:
        pred_label = pred

    metrics[name] = {"prediction": int(pred), "label": pred_label}

    safe_insert(f"\n{name} → Predicted for '{selected_movie}': {pred_label}")

def run_naive_bayes():
    evaluate_and_display("Naive Bayes", GaussianNB())

def run_logistic_regression():
    evaluate_and_display("Logistic Regression", LogisticRegression(max_iter=2000))

def run_svm():
    evaluate_and_display("Support Vector Machine", SVC())

def comparison_graph():
    if not metrics:
        messagebox.showerror("Error", "Run algorithms first.")
        return

    algos = list(metrics.keys())
    preds = [m["prediction"] for m in metrics.values()]

    plt.figure(figsize=(6, 4))
    plt.bar(algos, preds, color=['green' if p == 1 else 'red' for p in preds])
    plt.ylabel("Prediction (1=Success, 0=Failure)")
    plt.title(f"Comparison Graph for '{selected_movie}'")
    plt.ylim(-0.2, 1.2)
    plt.show()

# -------------------------
# GUI
# -------------------------
root = tk.Tk()
root.title("Movie Success Prediction")
root.geometry("1100x700")
root.configure(bg="#9ed7f5")

title_label = tk.Label(root,
                       text="Movie Success Prediction Using Naive Bayes, Logistic Regression and SVM",
                       font=("Arial", 16, "bold"),
                       bg="#9adf7a", fg="#0b4f6c",
                       pady=10)
title_label.pack(fill="x")
title_label.config(anchor="center")

main_frame = tk.Frame(root, bg="white", bd=2, relief="groove")
main_frame.pack(padx=20, pady=18, fill="both", expand=True)

info_label = tk.Label(main_frame, text="No dataset loaded", bg="white", anchor="w", font=("Arial", 11))
info_label.pack(fill="x", padx=8, pady=(6, 0))

text_container = tk.Frame(main_frame, bg="white")
text_container.pack(fill="both", expand=True, padx=8, pady=8)

output_text = tk.Text(text_container, wrap="word", state="disabled", font=("Consolas", 10))
scroll = tk.Scrollbar(text_container, command=output_text.yview)
output_text.configure(yscrollcommand=scroll.set)
scroll.pack(side="right", fill="y")
output_text.pack(side="left", fill="both", expand=True)

btn_frame = tk.Frame(root, bg="#9ed7f5")
btn_frame.pack(pady=(0, 16))

row1 = [
    ("Upload & Preprocess Dataset", upload_dataset),
    ("Generate Train & Test Model", generate_train_test),
    ("Run Naive Bayes Algorithm", run_naive_bayes),
    ("Run Logistic Regression Algorithm", run_logistic_regression),
]
for i, (txt, cmd) in enumerate(row1):
    tk.Button(btn_frame, text=txt, command=cmd, width=30, height=2, bg="white").grid(row=0, column=i, padx=6, pady=6)

row2 = [
    ("Run SVM Algorithm", run_svm),
    ("Comparison Graph", comparison_graph),
]
for i, (txt, cmd) in enumerate(row2):
    tk.Button(btn_frame, text=txt, command=cmd, width=30, height=2, bg="white").grid(row=1, column=i, padx=6, pady=6)

safe_insert("Ready. Upload CSV (first col = Movie name, last col = Success/Failure target).")

root.mainloop()

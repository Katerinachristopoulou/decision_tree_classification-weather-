# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("muthuj7/weather-dataset")
print("Path to dataset files:", path)

# Read the CSV from the downloaded folder (finds it whatever it's called)
csv_file = next(Path(path).glob("*.csv"))
print("Using file:", csv_file.name)
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
print("Columns found:", df.columns.tolist())

# Time features: weather depends on the hour of day and the season.
# Dates look like "2006-04-01 00:00:00.000 +0200"; drop the UTC offset so
# the hour stays in local time.
dates = pd.to_datetime(df["Formatted Date"].str[:19])
df["Hour"] = dates.dt.hour
df["Month"] = dates.dt.month

# Pressure readings of 0 are sensor gaps, not real values. Mark them as
# missing; the decision tree handles NaN values natively.
df["Pressure (millibars)"] = df["Pressure (millibars)"].replace(0, np.nan)

# Remove unwanted / leakage-prone columns
df = df.drop(columns=[
    "Formatted Date",
    "Daily Summary",
    "Apparent Temperature (C)",
], errors="ignore")

# "Loud Cover" (a typo for Cloud Cover) is all zeros, so it carries no information
df = df.drop(columns=["Loud Cover"], errors="ignore")

# Target: the hourly weather Summary, grouped into 5 general classes.
# Labels like "Breezy and Overcast" mention several conditions, so the
# rarer, more specific condition wins: Rainy > Windy > Foggy > Cloudy > Clear.
TARGET = "Summary"


def group_summary(summary):
    if any(word in summary for word in ["Rain", "Drizzle"]):
        return "Rainy"
    if any(word in summary for word in ["Breezy", "Windy"]):
        return "Windy"
    if "Foggy" in summary:
        return "Foggy"
    if any(word in summary for word in ["Cloudy", "Overcast"]):
        return "Cloudy"
    return "Clear"  # "Clear", "Dry", and their combinations


df[TARGET] = df[TARGET].apply(group_summary)
print("Class counts:", df[TARGET].value_counts().to_dict())
baseline = df[TARGET].value_counts(normalize=True).max()
print(f"Baseline accuracy (always guess the most common class): {baseline:.4f}")

# Handle missing values and encode the categorical feature
df["Precip Type"] = df["Precip Type"].fillna("none")
df = pd.get_dummies(df, columns=["Precip Type"], drop_first=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]
print("Features:", X.columns.tolist())

# Hold out a test set that is not touched during model selection
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# The classes are very imbalanced (most hours are Cloudy), so:
# - class_weight="balanced" makes mistakes on rare classes count for more
# - models are chosen by macro F1, which weighs every class equally,
#   instead of accuracy, which mostly rewards predicting Cloudy

# Use K-fold cross-validation on the training set to choose the tree depth
kf = KFold(n_splits=5, shuffle=True, random_state=42)
depths = range(1, 16)
cv_scores = []
for depth in depths:
    fold_scores = []
    for train_idx, val_idx in kf.split(X_train):
        model = DecisionTreeClassifier(
            max_depth=depth, class_weight="balanced", random_state=42
        )
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        fold_scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
    cv_scores.append(np.mean(fold_scores))
    print(f"max_depth={depth:2d}  CV macro F1={cv_scores[-1]:.4f}")

best_depth = depths[int(np.argmax(cv_scores))]
print("Best max_depth:", best_depth)

# Train the final model on the full training set and evaluate on the test set
final_model = DecisionTreeClassifier(
    max_depth=best_depth, class_weight="balanced", random_state=42
)
final_model.fit(X_train, y_train)
test_preds = final_model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, test_preds):.4f}")
print(f"Test macro F1: {f1_score(y_test, test_preds, average='macro'):.4f}")
print(classification_report(y_test, test_preds, digits=3))

# Grid search over depth, leaf size and pruning strength (ccp_alpha) together
grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=42),
    {
        "max_depth": [5, 8, 12, None],
        "min_samples_leaf": [1, 5, 20],
        "ccp_alpha": [0.0, 0.0001, 0.001],
    },
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
)
grid.fit(X_train, y_train)
print("Grid search best params:", grid.best_params_)
print(f"Grid search CV macro F1: {grid.best_score_:.4f}")
grid_preds = grid.best_estimator_.predict(X_test)
print(f"Grid search test accuracy: {accuracy_score(y_test, grid_preds):.4f}")
print(f"Grid search test macro F1: {f1_score(y_test, grid_preds, average='macro'):.4f}")

# Plot CV macro F1 vs depth
plt.figure(figsize=(8, 5))
plt.plot(list(depths), cv_scores, marker="o")
plt.xlabel("max_depth")
plt.ylabel("Mean CV macro F1")
plt.title("Decision tree depth selection (5-fold CV, macro F1)")
plt.grid(True)
plt.tight_layout()
plt.savefig("cv_accuracy.png")

# Plot the top levels of the final tree (the full tree is too large to read)
plt.figure(figsize=(20, 10))
plot_tree(
    final_model,
    feature_names=X.columns,
    class_names=final_model.classes_,
    filled=True,
    max_depth=3,
    fontsize=8,
)
plt.title(f"Decision tree (max_depth={best_depth}, top 3 levels shown)")
plt.tight_layout()
plt.savefig("decision_tree.png")

# ======================================
# COMPARE TWO PARAMETER SETS
# (uses the same cleaned X and y as above)
# ======================================
param_set_1 = {
    "criterion": "gini",
    "min_samples_leaf": 1,
    "ccp_alpha": 0.0,  # Parameter Set 1: Gini, no pruning
}

param_set_2 = {
    "criterion": "entropy",
    "min_samples_leaf": 5,
    "ccp_alpha": 0.0001,  # Parameter Set 2: Entropy, pruned (value from grid search)
}

param_sets = [("Set1", param_set_1), ("Set2", param_set_2)]


def build_tree(param_set):
    return DecisionTreeClassifier(**param_set, class_weight="balanced", random_state=42)


def score(model, X_tr, y_tr, X_te, y_te):
    """Fit the model and return (accuracy, macro F1) on the test part."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return accuracy_score(y_te, y_pred), f1_score(y_te, y_pred, average="macro")


def summarize(method, name, param, setting, scores):
    accs, f1s = zip(*scores)
    print(f"{method} {setting}  {name}  accuracy={np.mean(accs):.4f}  macro F1={np.mean(f1s):.4f}")
    return {
        "Validation Method": method,
        "Parameter Set": name,
        "Criterion": param["criterion"],
        "k value / split %": setting,
        "Accuracy": np.mean(accs),
        "Accuracy Std Dev": np.std(accs),
        "Macro F1": np.mean(f1s),
        "Macro F1 Std Dev": np.std(f1s),
    }


results = []

# K-fold cross-validation with k = 5 and k = 10
for k in [5, 10]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for name, param in param_sets:
        fold_scores = [
            score(build_tree(param), X.iloc[train_index], y.iloc[train_index],
                  X.iloc[test_index], y.iloc[test_index])
            for train_index, test_index in kf.split(X)
        ]
        results.append(summarize("K-Fold", name, param, k, fold_scores))

# Repeated train/test splits (70/30 and 80/20)
num_train_test_runs = 10

for split_size, label in [(0.3, "70/30"), (0.2, "80/20")]:
    for name, param in param_sets:
        run_scores = []

        for run in range(num_train_test_runs):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=split_size, random_state=run, stratify=y
            )
            run_scores.append(score(build_tree(param), X_tr, y_tr, X_te, y_te))

        results.append(summarize("Train/Test", name, param, label, run_scores))

# Export the comparison to Excel (needs the openpyxl package)
results_df = pd.DataFrame(results)
results_df.to_excel("DecisionTree_Comparison.xlsx", index=False)
print("Excel file created: DecisionTree_Comparison.xlsx")

# Train both trees on the full dataset (for visualization only)
tree_set1 = build_tree(param_set_1).fit(X, y)
tree_set2 = build_tree(param_set_2).fit(X, y)

for tree, title, filename in [
    (tree_set1, "Parameter Set 1 (Gini, No Pruning)", "tree_set1.png"),
    (tree_set2, "Parameter Set 2 (Entropy, Pruned)", "tree_set2.png"),
]:
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=tree.classes_,
        filled=True,
        max_depth=3,  # limit depth for readability
        fontsize=8,
    )
    plt.title(f"Decision Tree – {title}")
    plt.tight_layout()
    plt.savefig(filename)

# All plots are saved as PNG files, so close them instead of opening windows
# (plt.show() would keep the script running until every window is closed)
plt.close("all")
print("Plots saved: cv_accuracy.png, decision_tree.png, tree_set1.png, tree_set2.png")

# ======================================
# EXPORT RESULTS FOR THE README
# ======================================
table = results_df.copy()
for col in ["Accuracy", "Accuracy Std Dev", "Macro F1", "Macro F1 Std Dev"]:
    table[col] = table[col].round(4)

with open("results.md", "w") as f:
    f.write("## Results\n\n")
    f.write(f"Majority-class baseline accuracy: **{baseline:.4f}**\n\n")

    f.write("### Model selection\n\n")
    f.write(f"- Best `max_depth` (5-fold CV): **{best_depth}**\n")
    f.write(f"- Test accuracy / macro F1: **{accuracy_score(y_test, test_preds):.4f}** / "
            f"**{f1_score(y_test, test_preds, average='macro'):.4f}**\n")
    f.write(f"- Best grid search parameters: **{grid.best_params_}**\n")
    f.write(f"- Grid search test accuracy / macro F1: **{accuracy_score(y_test, grid_preds):.4f}** / "
            f"**{f1_score(y_test, grid_preds, average='macro'):.4f}**\n\n")

    f.write("### Parameter set comparison\n\n")
    f.write(table.to_markdown(index=False))  # needs the tabulate package
    f.write("\n\n")

    f.write("### Charts\n\n")
    f.write("![Depth selection](cv_accuracy.png)\n\n")
    f.write("![Decision tree](decision_tree.png)\n\n")
    f.write("![Set 1 tree](tree_set1.png)\n\n")
    f.write("![Set 2 tree](tree_set2.png)\n")

print("Results written to results.md")

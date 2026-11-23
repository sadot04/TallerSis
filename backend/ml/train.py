import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

TARGET = "Accident_severity"
LABELS = ["Slight Injury", "Serious Injury", "Fatal injury"]
SEED = 42
FEATURES = json.loads((Path(__file__).resolve().parents[1] / "features.json").read_text(encoding="utf-8"))
FIELDS = [field["name"] for field in FEATURES]
CATEGORIES = [field["name"] for field in FEATURES if field["type"] == "category"]
NUMERIC = ["Hour", "Number_of_vehicles_involved", "Number_of_casualties"]


def prepare(frame):
    clean = frame[FIELDS].copy()
    for field in CATEGORIES:
        clean[field] = clean[field].astype("string").str.strip().replace("", pd.NA).astype(object)
        clean[field] = clean[field].where(clean[field].notna(), float("nan"))
    hours = clean.pop("Time").astype("string").str.strip()
    parsed = hours.str.extract(r"^(\d{1,2}):[0-5]\d(?::[0-5]\d)?$")[0]
    clean["Hour"] = pd.to_numeric(parsed, errors="coerce").astype(float)
    clean["Hour"] = clean["Hour"].where(clean["Hour"].between(0, 23))
    for field in NUMERIC[1:]:
        values = pd.to_numeric(clean[field], errors="coerce").astype(float)
        minimum = 1 if field == "Number_of_vehicles_involved" else 0
        clean[field] = values.where(values.between(minimum, 1000) & (values % 1 == 0))
    return clean


def read_training(path):
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=100001)
    if len(raw) > 100000:
        raise ValueError("El CSV admite hasta 100 000 filas.")
    missing = set(FIELDS + [TARGET]).difference(raw.columns)
    if missing:
        raise ValueError("Columnas requeridas: " + ", ".join(sorted(missing)))
    y = raw[TARGET].astype("string").str.strip().str.lower().map({label.lower(): label for label in LABELS})
    if y.isna().any():
        raise ValueError("Cada fila debe contener una clase válida en Accident_severity.")
    x = prepare(raw)
    for field in FIELDS:
        source = raw[field].astype("string").str.strip().fillna("")
        column = "Hour" if field == "Time" else field
        if (source.ne("") & x[column].isna()).any():
            raise ValueError(f"Hay un valor inválido en {field}.")
    combined = x.assign(**{TARGET: y}).drop_duplicates()
    y = combined.pop(TARGET)
    if len(combined) < 150 or any(y.value_counts().get(label, 0) < 15 for label in LABELS):
        raise ValueError("Se requieren al menos 150 filas distintas y 15 por clase.")
    return combined.reset_index(drop=True), y.reset_index(drop=True)


def identify(x):
    return pd.util.hash_pandas_object(x, index=False)


def candidates():
    categorical = make_pipeline(SimpleImputer(strategy="constant", fill_value="Sin dato"), OneHotEncoder(handle_unknown="ignore"))
    ordinal = make_pipeline(SimpleImputer(strategy="constant", fill_value="Sin dato"),
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    encoding = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORIES)])
    dense = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", ordinal, CATEGORIES)])
    return {
        "Regresión logística balanceada": make_pipeline(encoding, LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        "Bosque aleatorio balanceado": make_pipeline(encoding, RandomForestClassifier(n_estimators=120, min_samples_leaf=4,
                                                                                      class_weight="balanced", random_state=SEED, n_jobs=2)),
        "Gradient boosting balanceado": make_pipeline(dense, HistGradientBoostingClassifier(categorical_features=list(range(3, 3 + len(CATEGORIES))),
                                                                                           class_weight="balanced", random_state=SEED, max_iter=90)),
    }


def train(source, reference, out):
    x, y = read_training(reference)
    groups = identify(x)
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_ids, test_ids = next(split.split(x, y, groups))
    test_x, test_y = x.iloc[test_ids], y.iloc[test_ids]
    fit_x, fit_y = x.iloc[train_ids], y.iloc[train_ids]
    if source.resolve() != reference.resolve():
        fit_x, fit_y = read_training(source)
        allowed = ~identify(fit_x).isin(groups.iloc[test_ids])
        fit_x, fit_y = fit_x[allowed], fit_y[allowed]
    if any(fit_y.value_counts().get(label, 0) < 12 for label in LABELS):
        raise ValueError("Después de reservar la prueba faltan ejemplos de alguna clase.")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    models = candidates()
    validation = {
        name: float(cross_val_score(model, fit_x, fit_y, groups=identify(fit_x), cv=cv, scoring="f1_macro", error_score="raise").mean())
        for name, model in models.items()
    }
    name = max(validation, key=validation.get)
    estimator = models[name].fit(fit_x, fit_y)
    predicted = estimator.predict(test_x)
    baseline = DummyClassifier(strategy="most_frequent").fit(fit_x, fit_y).predict(test_x)
    metrics = {
        "algorithm": name,
        "accuracy": accuracy_score(test_y, predicted),
        "balancedAccuracy": balanced_accuracy_score(test_y, predicted),
        "macroF1": f1_score(test_y, predicted, average="macro"),
        "baselineMacroF1": f1_score(test_y, baseline, average="macro"),
        "labels": LABELS,
        "perClass": {label: entry for label, entry in classification_report(
            test_y, predicted, labels=LABELS, output_dict=True, zero_division=0).items() if label in LABELS},
        "confusionMatrix": confusion_matrix(test_y, predicted, labels=LABELS).tolist(),
        "validationMacroF1": validation,
        "trainRows": len(fit_x),
        "testRows": len(test_x),
        "testClassCounts": test_y.value_counts().to_dict(),
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "referenceSha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "sklearnVersion": sklearn.__version__,
        "seed": SEED,
    }
    levels = {field: sorted(fit_x[field].dropna().unique().tolist()) for field in CATEGORIES}
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, out / "model.joblib", compress=3)
    (out / "report.json").write_text(json.dumps({"metrics": metrics, "options": levels}, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def predict(path, payload):
    estimator = joblib.load(path)
    row = prepare(pd.DataFrame([payload]))
    predicted = estimator.predict(row)[0]
    probabilities = dict(zip(estimator.classes_, estimator.predict_proba(row)[0]))
    return {"category": predicted, "probabilities": probabilities}


def main():
    cli = argparse.ArgumentParser()
    commands = cli.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    for flag in ["source", "reference", "out"]:
        fit.add_argument("--" + flag, required=True, type=Path)
    check = commands.add_parser("validate")
    check.add_argument("--source", required=True, type=Path)
    predict_cli = commands.add_parser("predict")
    predict_cli.add_argument("--model", required=True, type=Path)
    args = cli.parse_args()
    try:
        if args.command == "train":
            print(json.dumps(train(args.source, args.reference, args.out), ensure_ascii=False))
        elif args.command == "validate":
            x, y = read_training(args.source)
            print(json.dumps({"rows": len(x), "classes": y.value_counts().to_dict()}))
        else:
            import sys
            print(json.dumps(predict(args.model, json.load(sys.stdin)), ensure_ascii=False))
    except (ValueError, pd.errors.ParserError) as error:
        cli.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()

# extract_features.py
# Usage: python extract_features.py
import os, csv
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = os.path.join("data","synthetic")
CSV_PATH = os.path.join(DATA_DIR,"labels.csv")

def map_label(lbl):
    s = str(lbl)
    return "tumor" if s.startswith("tumor") else "no_tumor"

def extract_features_from_mask(mask):
    # mask: uint8 binary 0/255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    x,y,w,h = cv2.boundingRect(c)
    aspect = w / h if h>0 else 0
    extent = area / (w*h) if w*h>0 else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) if len(hull)>0 else 0
    solidity = area / hull_area if hull_area>0 else 0
    circularity = (4*np.pi*area/(perim*perim)) if perim>0 else 0
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    # approx corners
    approx = cv2.approxPolyDP(c, 0.02*perim, True)
    corners = len(approx)
    feats = [area, perim, aspect, extent, solidity, circularity, corners] + hu.tolist()
    return np.array(feats, dtype=float)

def build_dataset():
    df = pd.read_csv(CSV_PATH)
    X, y = [], []
    for idx, row in df.iterrows():
        img_p = os.path.join(DATA_DIR, row['image']) if not os.path.isabs(row['image']) else row['image']
        mask_p = os.path.join(DATA_DIR, row['mask']) if not os.path.isabs(row['mask']) else row['mask']
        lbl = map_label(row['label'])
        if not os.path.exists(mask_p):
            continue
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        # convert to binary 0/255
        _, maskb = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        feats = extract_features_from_mask(maskb)
        if feats is None:
            # treat as no_tumor with zeros
            feats = np.zeros(14)
        X.append(feats)
        y.append(lbl)
    X = np.vstack(X)
    return X, np.array(y)

def main():
    print("Building dataset from", CSV_PATH)
    X, y = build_dataset()
    print("Samples:", X.shape, "Labels:", np.unique(y, return_counts=True))
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print("5-fold CV accuracy: %.4f ± %.4f" % (scores.mean(), scores.std()))
    # final train & report
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print("Train classification report:")
    print(classification_report(y, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    # save feature importances quick CSV
    try:
        fi = clf.feature_importances_
        cols = ['area','perim','aspect','extent','solidity','circularity','corners'] + [f'hu{i+1}' for i in range(len(fi)-7)]
        pd.DataFrame({'feature':cols,'importance':fi}).sort_values('importance',ascending=False).to_csv("results/feature_importances.csv", index=False)
        print("Feature importances saved to results/feature_importances.csv")
    except Exception as e:
        pass

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()

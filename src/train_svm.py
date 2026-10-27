import argparse, os
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.feature_extraction import extract_features

EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
def load_folder(root):
    X,y,paths=[],[],[]
    root=Path(root); classes=sorted(p.name for p in root.iterdir() if p.is_dir())
    for label in classes:
        for p in sorted((root/label).rglob("*")):
            if p.suffix.lower() in EXTS:
                try: X.append(extract_features(p)); y.append(label); paths.append(str(p))
                except Exception as e: print("Skipping",p,e)
    return np.asarray(X),np.asarray(y),paths,classes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True); ap.add_argument("--test-dir")
    ap.add_argument("--output-dir",default="models"); args=ap.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    X,y,_,classes=load_folder(args.data_dir)
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.15,random_state=42,stratify=y)
    model=Pipeline([("scaler",StandardScaler()),("svm",SVC(kernel="rbf",C=10,class_weight="balanced"))])
    model.fit(Xtr,ytr); pred=model.predict(Xv)
    print("Validation accuracy:",round(accuracy_score(yv,pred),4))
    print(classification_report(yv,pred,labels=classes,zero_division=0))
    print("Confusion matrix (class order):",classes); print(confusion_matrix(yv,pred,labels=classes))
    joblib.dump({"model":model,"classes":classes},os.path.join(args.output_dir,"orange_svm.joblib"))
    if args.test_dir:
        Xt,yt,paths,_=load_folder(args.test_dir); pt=model.predict(Xt)
        print("Test accuracy:",round(accuracy_score(yt,pt),4))
        print(classification_report(yt,pt,labels=classes,zero_division=0))
        pd.DataFrame({"image_path":paths,"true_label":yt,"predicted_label":pt}).to_csv(os.path.join(args.output_dir,"svm_test_predictions.csv"),index=False)
if __name__=="__main__": main()

import argparse, joblib
from src.feature_extraction import extract_features
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--image",required=True); a=p.parse_args()
    saved=joblib.load(a.model); model=saved["model"] if isinstance(saved,dict) and "model" in saved else saved
    print("Predicted class:",model.predict(extract_features(a.image).reshape(1,-1))[0])
if __name__=="__main__": main()

import joblib
import xgboost

try:
    print("Loading model...")
    model = joblib.load("model_satisfied_v2.pkl")
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model has feature_importances_: {hasattr(model, 'feature_importances_')}")
    
    print("\nLoading feature names...")
    feature_names = joblib.load("feature_names.pkl")
    print("✅ Feature names loaded successfully!")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names type: {type(feature_names)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
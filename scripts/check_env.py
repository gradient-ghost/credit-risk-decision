#!/usr/bin/env python3
"""Environment Verification Script for Credit Risk Decision System.

Verifies:
1. Python version and virtual environment.
2. Essential ML, API, Dashboard, and Utility packages.
3. GPU acceleration (XGBoost CUDA acceleration and PyTorch CUDA if available).
"""

import sys
import platform

print("=" * 60)
print(" Credit Risk Decision System: Environment Verification")
print("=" * 60)
print(f"Python executable : {sys.executable}")
print(f"Python version    : {platform.python_version()}")
print(f"Platform          : {platform.platform()}")
print("-" * 60)

modules = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("streamlit", "streamlit"),
    ("plotly", "plotly"),
    ("pydantic", "pydantic"),
    ("pytest", "pytest"),
    ("pyyaml", "yaml"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]

all_passed = True

print("\n--- 1. Package Import Checks ---")
for pkg_name, import_name in modules:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [✓] {pkg_name:15s} (v{ver})")
    except ImportError as e:
        print(f"  [✗] {pkg_name:15s} FAILED: {e}")
        all_passed = False

print("\n--- 2. Hardware & GPU Acceleration Checks ---")

# Check PyTorch CUDA (if torch is installed)
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    print(f"  [{'✓' if cuda_avail else '!'}] PyTorch CUDA available: {cuda_avail} ({device_name})")
except ImportError:
    print("  [-] PyTorch not installed (optional, not strictly required for tabular tree models)")

# Check XGBoost GPU device='cuda' support
try:
    import numpy as np
    import xgboost as xgb

    X_dummy = np.random.randn(50, 4).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=50)

    # Test CUDA device training
    clf = xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method="hist", device="cuda")
    clf.fit(X_dummy, y_dummy)
    preds = clf.predict_proba(X_dummy)
    print("  [✓] XGBoost GPU Acceleration (device='cuda'): PASSED")
except Exception as e:
    print(f"  [!] XGBoost GPU test failed or CPU fallback: {e}")

# Check LightGBM
try:
    import lightgbm as lgb
    X_dummy = np.random.randn(50, 4).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=50)
    lgb_clf = lgb.LGBMClassifier(n_estimators=2, max_depth=2, verbose=-1)
    lgb_clf.fit(X_dummy, y_dummy)
    print("  [✓] LightGBM Basic Training: PASSED")
except Exception as e:
    print(f"  [✗] LightGBM test failed: {e}")
    all_passed = False

# Check Config loading
print("\n--- 3. Configuration & Constants Checks ---")
try:
    import yaml
    from pathlib import Path

    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config" / "model_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"  [✓] Loaded model_config.yaml (Project: {cfg.get('project', {}).get('name')})")

    # Add project root to sys.path to test constants import
    sys.path.insert(0, str(root_dir))
    from config import constants
    print(f"  [✓] Loaded config/constants.py (Random seed: {constants.RANDOM_SEED})")
    print(f"  [✓] Post-origination blacklist count: {len(constants.POST_ORIGINATION_COLS)} columns")
except Exception as e:
    print(f"  [✗] Config check failed: {e}")
    all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print(" >>> ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY! <<< ")
    print("=" * 60)
    sys.exit(0)
else:
    print(" >>> SOME CHECKS FAILED - PLEASE REVIEW LOGS ABOVE <<< ")
    print("=" * 60)
    sys.exit(1)

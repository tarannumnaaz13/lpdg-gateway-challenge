import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command):
    print("\n" + "=" * 70)
    print("RUNNING:", " ".join(str(x) for x in command))
    print("=" * 70)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete LPDG gateway prioritisation pipeline."
    )

    parser.add_argument(
        "--data",
        default="data",
        help="Path to the challenge data directory (default: data)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data)

    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    python = sys.executable

    # Step 1: Build weekly gateway features
    run([
        python,
        "src/build_features.py",
        "--data",
        str(data_dir),
    ])

    # Step 2: Train the ML model
    run([
        python,
        "src/train_ml.py",
    ])

    # Step 3: Generate gateway recommendations
    run([
        python,
        "src/predict_ml.py",
    ])

    # Step 4: Copy ML predictions to the required submission filename
    predictions_ml = ROOT / "outputs" / "predictions_ml.csv"
    predictions = ROOT / "predictions.csv"

    if not predictions_ml.exists():
        raise FileNotFoundError(
            f"Expected prediction file was not created: {predictions_ml}"
        )

    shutil.copy2(predictions_ml, predictions)

    # Step 5: Validate the final submission
    run([
        python,
        "validate_submission.py",
        str(predictions),
    ])

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final predictions: {predictions}")
    print("=" * 70)


if __name__ == "__main__":
    main()
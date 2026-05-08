from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
DATA_DIR.mkdir(exist_ok=True)

demo = pd.read_csv(DATA_DIR / "demographics.csv")
lab = pd.read_csv(DATA_DIR / "laboratory.csv")
quest = pd.read_csv(DATA_DIR / "questionnaire.csv")

print("Demographics:", demo.shape)
print("Laboratory:", lab.shape)
print("Questionnaire:", quest.shape)

merged = pd.merge(demo, lab, on="SEQN", how="inner")
merged = pd.merge(merged, quest, on="SEQN", how="inner")

print(f"\nMerged shape: {merged.shape}")
print(merged.head())

merged.to_csv(DATA_DIR / "nhanes_merged.csv", index=False)

print("\nSaved as datasets/nhanes_merged.csv")
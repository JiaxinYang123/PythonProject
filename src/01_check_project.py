import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

required_directories = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "mapping",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "outputs" / "tables",
    PROJECT_ROOT / "outputs" / "model_results",
    PROJECT_ROOT / "outputs" / "logs",
    PROJECT_ROOT / "figures",
    PROJECT_ROOT / "references",
]

print("=" * 60)
print("PROJECT SETUP CHECK")
print("=" * 60)

print(f"Python version: {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")
print(f"Project root: {PROJECT_ROOT}")

print("\nDirectory check:")

all_ready = True

for directory in required_directories:
    relative_path = directory.relative_to(PROJECT_ROOT)

    if directory.exists():
        print(f"[OK] {relative_path}")
    else:
        print(f"[MISSING] {relative_path}")
        all_ready = False

print("\n" + "=" * 60)

if all_ready:
    print("Project structure is ready.")
else:
    print("Some directories are missing.")

print("=" * 60)

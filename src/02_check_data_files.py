from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

required_files = {
    "network data":
        PROJECT_ROOT
        / "data"
        / "raw"
        / "historical-network_week_2020-03-30.json",

    "Protocol Category Mapping":
        PROJECT_ROOT
        / "data"
        / "mapping"
        / "id_to_info.json",

    "token mapping":
        PROJECT_ROOT
        / "data"
        / "mapping"
        / "token_to_protocol.json",
}


def format_size(path):
    size_bytes = path.stat().st_size

    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.3f} GB"

    return f"{size_bytes / (1024 ** 2):.2f} MB"


def is_git_lfs_pointer(path):
    with path.open("rb") as file:
        beginning = file.read(200)

    return b"git-lfs.github.com/spec" in beginning


print("=" * 68)
print("DEXPOSURE DATA FILE CHECK")
print("=" * 68)

all_ready = True

for label, path in required_files.items():

    if not path.exists():
        print(f"[MISSING] {label}")
        print(f"          {path}")
        all_ready = False
        continue

    print(f"[OK] {label}")
    print(f"     File: {path.name}")
    print(f"     Size: {format_size(path)}")

    if is_git_lfs_pointer(path):
        print("     ERROR: This is only a Git LFS pointer.")
        all_ready = False

print("\n" + "=" * 68)

if all_ready:
    print("READY")
else:
    print("NOT READY")

print("=" * 68)
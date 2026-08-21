from pathlib import Path
import subprocess
import sys


project_root = Path(__file__).resolve().parent
source_directory = project_root / "src"

scripts = [
    "01_check_project.py",
    "02_check_data_files.py",
    "03_build_sector_panel.py",
    "04_validate.py",
    "05_descriptive_analysis.py",
    "06_build_forecasting_predictors.py",
    "07_knowledge_discovery_analysis.py",
    "08_select_model.py",
    "09_check.py",
    "10_fit_final_network_model.py",
    "11_analyse_final_network_forecasts.py",
    "12_compare_final_forecasting_models.py",
    "13_robustness_quantile_95.py",
    "14_create_sector_exposure_heatmap.py",
]


for script_name in scripts:
    script_path = source_directory / script_name

    print("\n" + "=" * 72)
    print(f"Running {script_name}")
    print("=" * 72)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
        check=True,
    )


print("\nPipeline completed successfully.")
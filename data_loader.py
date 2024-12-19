import pandas as pd
import os

def load_csv_from_gz(gz_path):
    """
    Loads a CSV file from a gzip archive.
    
    Args:
        gz_path (str): Path to the .csv.gz file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"File not found: {gz_path}")
    return pd.read_csv(gz_path, compression='gzip')

def load_and_preprocess_mimic_demo(base_dir="data/"):
    """
    Loads and preprocesses MIMIC-IV demo data from .csv.gz files.
    
    Args:
        base_dir (str): Base directory where hosp and icu folders are located.
    
    Returns:
        pd.DataFrame: Combined and preprocessed dataset.
    """
    patients = load_csv_from_gz(os.path.join(base_dir, "hosp/patients.csv.gz"))
    admissions = load_csv_from_gz(os.path.join(base_dir, "hosp/admissions.csv.gz"))
    diagnoses = load_csv_from_gz(os.path.join(base_dir, "hosp/diagnoses_icd.csv.gz"))
    labevents = load_csv_from_gz(os.path.join(base_dir, "hosp/labevents.csv.gz"))

    # Merge patients and admissions
    demographics = pd.merge(patients, admissions, on='subject_id', how='inner')
    demographics = demographics[['subject_id', 'anchor_age', 'gender', 'admittime']]
    demographics['gender'] = demographics['gender'].map({'M': 0, 'F': 1})

    # Filter lab events for glucose levels
    glucose_events = labevents[labevents['itemid'] == 50809]
    glucose_events = glucose_events[['subject_id', 'hadm_id', 'valuenum']].rename(columns={'valuenum': 'glucose_level'})

    # Merge with diagnoses
    combined_data = pd.merge(demographics, diagnoses, on=['subject_id'], how='inner')
    combined_data = pd.merge(combined_data, glucose_events, on=['subject_id', 'hadm_id'], how='inner')

    # Simplify ICD codes safely
    def simplify_icd_code(x):
        try:
            return int(str(x)[:3])
        except ValueError:
            return None

    combined_data['icd_code'] = combined_data['icd_code'].apply(simplify_icd_code)
    combined_data = combined_data.dropna(subset=['icd_code'])

    # Final cleanup
    combined_data = combined_data[['subject_id', 'anchor_age', 'gender', 'glucose_level', 'icd_code']]
    return combined_data

# Main script execution for testing
if __name__ == "__main__":
    base_dir = "data/"  # Adjust to your folder path if needed
    data = load_and_preprocess_mimic_demo(base_dir)
    
    print("Sample Combined Data:")
    print(data.head())
    print(f"Dataset Shape: {data.shape}")

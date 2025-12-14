"""
STEP 1: DESCRIPTIVE STATISTICS
Climate-Agriculture Analysis for Ghana (1984-2024)

Selected Crops: Rice, Maize (corn), Cassava (fresh), Cocoa beans
Climate Variables: Temperature, Rainfall, Solar Irradiance



WINDOWS VERSION - Uses relative paths
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("="*70)
print("STEP 1: DESCRIPTIVE STATISTICS")
print("="*70)
print("\n1. Loading Data...")

# NOTE: Place this script in the same folder as your CSV files
# OR modify these paths to point to your data folder

# Get current directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Define file paths (modify these if your files are in a different location)
crop_file = 'cleaned_FAOSTAT_data_en_11-14-2025.csv'
rainfall_file = 'cleaned_POWER_Regional_Monthly_1984_2025__Precipitation__yearly_spatial_average.csv'
temperature_file = 'ghana_temperature_yearly_spatial_average.csv'
solar_file = 'cleaned_POWER_Regional_Monthly_1984_2025__Solar_Irradiance_Downward__yearly_spatial_average.csv'

# Check if files exist
files_to_check = [crop_file, rainfall_file, temperature_file, solar_file]
missing_files = [f for f in files_to_check if not os.path.exists(f)]

if missing_files:
    print("\n❌ ERROR: The following files were not found:")
    for f in missing_files:
        print(f"  - {f}")
    print("\n📁 Please ensure:")
    print("  1. All CSV files are in the same folder as this script, OR")
    print("  2. Update the file paths in lines 31-34 of this script")
    print(f"\n  Current folder: {current_dir}")
    exit(1)

# Load data
try:
    crops_df = pd.read_csv(crop_file)
    rainfall_df = pd.read_csv(rainfall_file)
    temperature_df = pd.read_csv(temperature_file)
    solar_df = pd.read_csv(solar_file)
    print("✓ All data files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit(1)

# ============================================================================
# 2. FILTER AND PREPARE CROP DATA
# ============================================================================

print("\n2. Filtering Selected Crops...")

# Selected crops
selected_crops = ['Rice', 'Maize (corn)', 'Cassava, fresh', 'Cocoa beans']

# Filter for yield data only (Element Code 5412)
yield_data = crops_df[crops_df['Element'] == 'Yield'].copy()

# Filter for selected crops
yield_data = yield_data[yield_data['Item'].isin(selected_crops)].copy()

# Pivot to get crops as columns
crop_yields = yield_data.pivot_table(
    index='Year',
    columns='Item',
    values='Value',
    aggfunc='first'
).reset_index()

# Rename columns for easier handling
crop_yields.columns.name = None
crop_yields = crop_yields.rename(columns={
    'Cassava, fresh': 'Cassava_Yield',
    'Cocoa beans': 'Cocoa_Yield',
    'Maize (corn)': 'Maize_Yield',
    'Rice': 'Rice_Yield'
})

print(f"✓ Filtered data for {len(selected_crops)} crops")
print(f"  Crops: {', '.join(selected_crops)}")

# ============================================================================
# 3. PREPARE CLIMATE DATA
# ============================================================================

print("\n3. Preparing Climate Data...")

# Extract annual values
climate_data = pd.DataFrame({
    'Year': rainfall_df['YEAR'],
    'Rainfall': rainfall_df['ANN'],  # Annual rainfall
    'Temperature': temperature_df['ANN'],  # Annual temperature
    'Solar_Irradiance': solar_df['ANN']  # Annual solar irradiance
})

print("✓ Climate data prepared")

# ============================================================================
# 4. MERGE ALL DATA
# ============================================================================

print("\n4. Merging Crop and Climate Data...")

# Merge crop yields with climate data
full_data = crop_yields.merge(climate_data, on='Year', how='inner')

# Filter to common year range (1985-2024 for 40 years)
full_data = full_data[(full_data['Year'] >= 1985) & (full_data['Year'] <= 2024)]

print(f"✓ Data merged successfully")
print(f"  Year range: {full_data['Year'].min()} - {full_data['Year'].max()}")
print(f"  Total years: {len(full_data)}")

# ============================================================================
# 5. CHECK FOR MISSING VALUES
# ============================================================================

print("\n5. Checking for Missing Values...")
print("="*70)

missing_counts = full_data.isnull().sum()
missing_percentage = (full_data.isnull().sum() / len(full_data)) * 100

missing_df = pd.DataFrame({
    'Variable': missing_counts.index,
    'Missing Count': missing_counts.values,
    'Missing %': missing_percentage.values
})

print(missing_df.to_string(index=False))

if missing_counts.sum() > 0:
    print("\n⚠️  Warning: Missing values detected!")
else:
    print("\n✓ No missing values detected!")

# ============================================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================================

print("\n6. Computing Descriptive Statistics...")
print("="*70)

# Select only numeric columns (exclude Year)
numeric_cols = full_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Year')

# Calculate descriptive statistics
desc_stats = full_data[numeric_cols].describe().T

# Add additional statistics
desc_stats['Variance'] = full_data[numeric_cols].var()
desc_stats['Range'] = desc_stats['max'] - desc_stats['min']
desc_stats['CV (%)'] = (desc_stats['std'] / desc_stats['mean']) * 100  # Coefficient of Variation

# Reorder columns
desc_stats = desc_stats[['count', 'mean', 'std', 'Variance', 'min', '25%', '50%', '75%', 'max', 'Range', 'CV (%)']]

# Round for better display
desc_stats = desc_stats.round(2)

print("\nDESCRIPTIVE STATISTICS SUMMARY")
print("-"*70)
print(desc_stats.to_string())

# ============================================================================
# 7. SUMMARY BY VARIABLE TYPE
# ============================================================================

print("\n\n7. Summary by Variable Type...")
print("="*70)

# Crop Yields Summary
print("\n📊 CROP YIELDS (kg/ha)")
print("-"*70)
crop_cols = ['Cassava_Yield', 'Cocoa_Yield', 'Maize_Yield', 'Rice_Yield']
print(desc_stats.loc[crop_cols].to_string())

# Climate Variables Summary
print("\n\n🌤️  CLIMATE VARIABLES")
print("-"*70)
climate_cols = ['Rainfall', 'Temperature', 'Solar_Irradiance']
print(desc_stats.loc[climate_cols].to_string())

# ============================================================================
# 8. SAMPLE SIZE INFORMATION
# ============================================================================

print("\n\n8. Sample Size Information...")
print("="*70)

print(f"\nTotal observations per variable:")
for col in numeric_cols:
    valid_count = full_data[col].notna().sum()
    print(f"  {col:25s}: {valid_count} years")

print(f"\nTime period: {full_data['Year'].min()} - {full_data['Year'].max()}")
print(f"Total span: {full_data['Year'].max() - full_data['Year'].min() + 1} years")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

print("\n9. Saving Results...")

# Create output folder if it doesn't exist
output_folder = 'outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"✓ Created output folder: {output_folder}")

# Save full merged dataset
output_data_file = os.path.join(output_folder, 'merged_data.csv')
full_data.to_csv(output_data_file, index=False)
print(f"✓ Merged dataset saved: {output_data_file}")

# Save descriptive statistics
output_stats_file = os.path.join(output_folder, 'descriptive_statistics.csv')
desc_stats.to_csv(output_stats_file)
print(f"✓ Descriptive statistics saved: {output_stats_file}")

# ============================================================================
# 10. CREATE SUMMARY REPORT
# ============================================================================

print("\n10. Creating Summary Report...")

# Create a formatted summary report
summary_report = f"""
{'='*70}
DESCRIPTIVE STATISTICS SUMMARY REPORT
Climate-Agriculture Analysis for Ghana (1985-2024)
{'='*70}

DATASET INFORMATION:
-------------------
• Study Period: {full_data['Year'].min()} - {full_data['Year'].max()} ({len(full_data)} years)
• Crops Analyzed: Rice, Maize, Cassava, Cocoa beans
• Climate Variables: Temperature, Rainfall, Solar Irradiance

SAMPLE SIZES:
-------------
{missing_df.to_string(index=False)}

CROP YIELDS SUMMARY (kg/ha):
---------------------------
{desc_stats.loc[crop_cols][['mean', 'std', 'min', 'max']].to_string()}

CLIMATE VARIABLES SUMMARY:
-------------------------
{desc_stats.loc[climate_cols][['mean', 'std', 'min', 'max']].to_string()}

KEY OBSERVATIONS:
-----------------
Crop Yields:
• Cassava has the highest mean yield: {desc_stats.loc['Cassava_Yield', 'mean']:.2f} kg/ha
• Cassava also shows highest variability (CV: {desc_stats.loc['Cassava_Yield', 'CV (%)']:.2f}%)
• Cocoa beans has lowest mean yield: {desc_stats.loc['Cocoa_Yield', 'mean']:.2f} kg/ha

Climate:
• Mean annual temperature: {desc_stats.loc['Temperature', 'mean']:.2f}°C
• Mean annual rainfall: {desc_stats.loc['Rainfall', 'mean']:.2f} mm
• Mean solar irradiance: {desc_stats.loc['Solar_Irradiance', 'mean']:.2f} kWh/m²/day

DATA QUALITY:
-------------
{'✓ No missing values detected!' if missing_counts.sum() == 0 else '⚠️ Missing values present - see table above'}

{'='*70}
Generated: December 2025
{'='*70}
"""

# Save the report
output_report_file = os.path.join(output_folder, 'step1_summary_report.txt')
with open(output_report_file, 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(f"✓ Summary report saved: {output_report_file}")

# Also print the report
print(summary_report)

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print("\n" + "="*70)
print("✅ STEP 1 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files (in 'outputs' folder):")
print("  1. merged_data.csv - Combined crop and climate data")
print("  2. descriptive_statistics.csv - Full statistical summary")
print("  3. step1_summary_report.txt - Formatted summary report")
print("\nNext Step: Proceed to Step 2 (Visual Exploration)")
print("="*70)

# Optional: Display the location of output files
print(f"\n📁 Output files saved in: {os.path.abspath(output_folder)}")
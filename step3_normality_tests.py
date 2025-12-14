"""
STEP 3: TEST NORMALITY OF RAW DATA
Climate-Agriculture Analysis for Ghana (1985-2024)

Uses Shapiro-Wilk test to assess normality of:
- Crop yields (Cassava, Cocoa, Maize, Rice)
- Climate variables (Temperature, Rainfall, Solar Irradiance)

Includes Q-Q plots for visual confirmation

Author: King
Date: December 2025

WINDOWS VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("="*70)
print("STEP 3: TEST NORMALITY OF RAW DATA")
print("="*70)
print("\n1. Loading Merged Data...")

# Load the merged data from Step 1
data_file = os.path.join('outputs', 'merged_data.csv')

if not os.path.exists(data_file):
    print("❌ ERROR: merged_data.csv not found!")
    print("   Please run Step 1 first to generate the merged dataset.")
    exit(1)

df = pd.read_csv(data_file)
print(f"✓ Data loaded: {len(df)} observations from {df['Year'].min()} to {df['Year'].max()}")

# Create output folder for normality tests
normality_folder = os.path.join('outputs', 'normality_tests')
if not os.path.exists(normality_folder):
    os.makedirs(normality_folder)
    print(f"✓ Created normality tests folder: {normality_folder}")

# ============================================================================
# 2. DEFINE VARIABLES
# ============================================================================

print("\n2. Defining Variables to Test...")

# Define crop and climate variables
crop_vars = ['Cassava_Yield', 'Cocoa_Yield', 'Maize_Yield', 'Rice_Yield']
climate_vars = ['Rainfall', 'Temperature', 'Solar_Irradiance']
all_vars = crop_vars + climate_vars

crop_names = ['Cassava', 'Cocoa Beans', 'Maize', 'Rice']
climate_names = ['Rainfall', 'Temperature', 'Solar Irradiance']
all_names = crop_names + climate_names

print(f"✓ Testing normality for {len(all_vars)} variables")

# ============================================================================
# 3. SHAPIRO-WILK TEST
# ============================================================================

print("\n3. Running Shapiro-Wilk Tests...")
print("="*70)

# Store results
normality_results = []

print("\n{:<25s} {:<15s} {:<15s} {:<15s}".format(
    "Variable", "W-Statistic", "P-Value", "Interpretation"))
print("-"*70)

for var, name in zip(all_vars, all_names):
    # Run Shapiro-Wilk test
    w_stat, p_value = stats.shapiro(df[var])
    
    # Interpret result
    if p_value > 0.05:
        interpretation = "Normal ✓"
        is_normal = True
    else:
        interpretation = "Not Normal ✗"
        is_normal = False
    
    # Print result
    print("{:<25s} {:<15.4f} {:<15.4f} {:<15s}".format(
        name, w_stat, p_value, interpretation))
    
    # Store result
    normality_results.append({
        'Variable': name,
        'Variable_Code': var,
        'W_Statistic': w_stat,
        'P_Value': p_value,
        'Normal': is_normal,
        'Interpretation': interpretation
    })

print("-"*70)
print("\nInterpretation Guide:")
print("  • p > 0.05: Data appears normally distributed (fail to reject H₀)")
print("  • p ≤ 0.05: Data significantly deviates from normal distribution")

# ============================================================================
# 4. CREATE Q-Q PLOTS FOR CROP YIELDS
# ============================================================================

print("\n4. Creating Q-Q Plots for Crop Yields...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (var, name) in enumerate(zip(crop_vars, crop_names)):
    ax = axes[i]
    
    # Create Q-Q plot
    stats.probplot(df[var], dist="norm", plot=ax)
    
    # Get result for this variable
    result = [r for r in normality_results if r['Variable'] == name][0]
    
    # Customize plot
    ax.set_title(f'{name} Yield Q-Q Plot\nShapiro-Wilk: W={result["W_Statistic"]:.4f}, p={result["P_Value"]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax.set_ylabel('Sample Quantiles', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add color based on normality
    if result['Normal']:
        ax.get_lines()[0].set_color('green')
        ax.get_lines()[0].set_markerfacecolor('lightgreen')
    else:
        ax.get_lines()[0].set_color('red')
        ax.get_lines()[0].set_markerfacecolor('lightcoral')
    
    # Add text box with interpretation
    textstr = result['Interpretation']
    props = dict(boxstyle='round', facecolor='wheat' if result['Normal'] else 'lightcoral', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

plt.suptitle('Q-Q Plots for Crop Yields - Normality Assessment', 
             fontsize=13, fontweight='bold')
plt.tight_layout()

# Save figure
qq_crops_file = os.path.join(normality_folder, 'qq_plots_crop_yields.png')
plt.savefig(qq_crops_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: qq_plots_crop_yields.png")
plt.close()

# ============================================================================
# 5. CREATE Q-Q PLOTS FOR CLIMATE VARIABLES
# ============================================================================

print("\n5. Creating Q-Q Plots for Climate Variables...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (var, name) in enumerate(zip(climate_vars, climate_names)):
    ax = axes[i]
    
    # Create Q-Q plot
    stats.probplot(df[var], dist="norm", plot=ax)
    
    # Get result for this variable
    result = [r for r in normality_results if r['Variable'] == name][0]
    
    # Customize plot
    ax.set_title(f'{name} Q-Q Plot\nShapiro-Wilk: W={result["W_Statistic"]:.4f}, p={result["P_Value"]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax.set_ylabel('Sample Quantiles', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add color based on normality
    if result['Normal']:
        ax.get_lines()[0].set_color('green')
        ax.get_lines()[0].set_markerfacecolor('lightgreen')
    else:
        ax.get_lines()[0].set_color('red')
        ax.get_lines()[0].set_markerfacecolor('lightcoral')
    
    # Add text box with interpretation
    textstr = result['Interpretation']
    props = dict(boxstyle='round', facecolor='wheat' if result['Normal'] else 'lightcoral', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

plt.suptitle('Q-Q Plots for Climate Variables - Normality Assessment', 
             fontsize=13, fontweight='bold')
plt.tight_layout()

# Save figure
qq_climate_file = os.path.join(normality_folder, 'qq_plots_climate_variables.png')
plt.savefig(qq_climate_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: qq_plots_climate_variables.png")
plt.close()

# ============================================================================
# 6. CREATE COMBINED Q-Q PLOT
# ============================================================================

print("\n6. Creating Combined Q-Q Plot...")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

# Hide the last two subplots (we only have 7 variables)
axes[7].axis('off')
axes[8].axis('off')

colors_normal = ['green', 'lightgreen']
colors_abnormal = ['red', 'lightcoral']

for i, (var, name) in enumerate(zip(all_vars, all_names)):
    ax = axes[i]
    
    # Create Q-Q plot
    stats.probplot(df[var], dist="norm", plot=ax)
    
    # Get result
    result = [r for r in normality_results if r['Variable'] == name][0]
    
    # Customize
    ax.set_title(f'{name}\nW={result["W_Statistic"]:.4f}, p={result["P_Value"]:.4f}',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=8)
    ax.set_ylabel('Sample Quantiles', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Color based on normality
    if result['Normal']:
        ax.get_lines()[0].set_color(colors_normal[0])
        ax.get_lines()[0].set_markerfacecolor(colors_normal[1])
        ax.patch.set_facecolor('lightgreen')
        ax.patch.set_alpha(0.1)
    else:
        ax.get_lines()[0].set_color(colors_abnormal[0])
        ax.get_lines()[0].set_markerfacecolor(colors_abnormal[1])
        ax.patch.set_facecolor('lightcoral')
        ax.patch.set_alpha(0.1)

plt.suptitle('Complete Normality Assessment: All Variables (Shapiro-Wilk Test)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
qq_combined_file = os.path.join(normality_folder, 'qq_plots_combined.png')
plt.savefig(qq_combined_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: qq_plots_combined.png")
plt.close()

# ============================================================================
# 7. SAVE RESULTS TO CSV
# ============================================================================

print("\n7. Saving Test Results...")

# Convert results to DataFrame
results_df = pd.DataFrame(normality_results)
results_df = results_df[['Variable', 'W_Statistic', 'P_Value', 'Normal', 'Interpretation']]

# Save to CSV
results_file = os.path.join(normality_folder, 'shapiro_wilk_results.csv')
results_df.to_csv(results_file, index=False)
print(f"✓ Saved: shapiro_wilk_results.csv")

# ============================================================================
# 8. CREATE SUMMARY REPORT
# ============================================================================

print("\n8. Creating Summary Report...")

# Count normal vs non-normal
normal_count = sum([r['Normal'] for r in normality_results])
total_count = len(normality_results)

# Create report
report_lines = []
report_lines.append("="*70)
report_lines.append("NORMALITY TEST SUMMARY REPORT")
report_lines.append("Shapiro-Wilk Test Results")
report_lines.append("="*70)
report_lines.append("")
report_lines.append(f"Dataset: Ghana Climate-Agriculture Data (1985-2023)")
report_lines.append(f"Sample Size: {len(df)} observations")
report_lines.append(f"Variables Tested: {total_count}")
report_lines.append("")
report_lines.append("OVERALL SUMMARY:")
report_lines.append("-"*70)
report_lines.append(f"Normal distributions: {normal_count}/{total_count} ({normal_count/total_count*100:.1f}%)")
report_lines.append(f"Non-normal distributions: {total_count-normal_count}/{total_count} ({(total_count-normal_count)/total_count*100:.1f}%)")
report_lines.append("")
report_lines.append("DETAILED RESULTS:")
report_lines.append("-"*70)
report_lines.append(f"{'Variable':<25s} {'W-Statistic':<15s} {'P-Value':<15s} {'Result':<15s}")
report_lines.append("-"*70)

for result in normality_results:
    report_lines.append(f"{result['Variable']:<25s} {result['W_Statistic']:<15.4f} {result['P_Value']:<15.4f} {result['Interpretation']:<15s}")

report_lines.append("")
report_lines.append("CROP YIELDS:")
report_lines.append("-"*70)
crop_results = [r for r in normality_results if r['Variable'] in crop_names]
for result in crop_results:
    status = "passes" if result['Normal'] else "fails"
    report_lines.append(f"• {result['Variable']}: {status} normality test (p={result['P_Value']:.4f})")

report_lines.append("")
report_lines.append("CLIMATE VARIABLES:")
report_lines.append("-"*70)
climate_results = [r for r in normality_results if r['Variable'] in climate_names]
for result in climate_results:
    status = "passes" if result['Normal'] else "fails"
    report_lines.append(f"• {result['Variable']}: {status} normality test (p={result['P_Value']:.4f})")

report_lines.append("")
report_lines.append("INTERPRETATION:")
report_lines.append("-"*70)
report_lines.append("The Shapiro-Wilk test assesses whether data follows a normal distribution.")
report_lines.append("• H₀ (null hypothesis): Data is normally distributed")
report_lines.append("• H₁ (alternative): Data is not normally distributed")
report_lines.append("• Significance level: α = 0.05")
report_lines.append("")
report_lines.append("Decision Rule:")
report_lines.append("• If p > 0.05: Fail to reject H₀ (data appears normal)")
report_lines.append("• If p ≤ 0.05: Reject H₀ (data deviates from normality)")
report_lines.append("")
report_lines.append("IMPORTANT NOTE:")
report_lines.append("-"*70)
report_lines.append("If raw data is not normal, this is acceptable. In regression analysis,")
report_lines.append("what matters most is the normality of RESIDUALS (which will be tested in")
report_lines.append("Step 5 after fitting the regression model). Non-normal raw data can still")
report_lines.append("produce valid regression results if residual assumptions are met.")
report_lines.append("")
report_lines.append("="*70)
report_lines.append("Generated: December 2025")
report_lines.append("="*70)

# Save report
report_file = os.path.join(normality_folder, 'normality_test_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Saved: normality_test_report.txt")

# Print report
print("\n" + '\n'.join(report_lines))

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print("\n" + "="*70)
print("✅ STEP 3 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files (in 'outputs/normality_tests' folder):")
print("  1. qq_plots_crop_yields.png - Q-Q plots for crop yields")
print("  2. qq_plots_climate_variables.png - Q-Q plots for climate variables")
print("  3. qq_plots_combined.png - All Q-Q plots in one figure")
print("  4. shapiro_wilk_results.csv - Test results in table format")
print("  5. normality_test_report.txt - Comprehensive summary report")
print("\nKey Findings:")
print(f"  • {normal_count} out of {total_count} variables are normally distributed")
print(f"  • {total_count-normal_count} variables deviate from normality")
print("\nReminder: Non-normal raw data is acceptable - we'll check residuals in Step 5")
print("\nNext Step: Proceed to Step 4 (Regression Analysis)")
print("="*70)

print(f"\n📁 Results saved in: {os.path.abspath(normality_folder)}")
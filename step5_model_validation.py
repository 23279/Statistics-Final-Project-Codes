"""
STEP 5: MODEL VALIDATION (CHECK RESIDUALS)
Climate-Agriculture Analysis for Ghana (1985-2024)

Validates regression models by checking residual assumptions:
- Step 5a: Test Normality of Residuals (Shapiro-Wilk + Q-Q plots)
- Step 5b: Check Homoscedasticity (Residuals vs Fitted plots)
- Step 5c: Check Independence (Durbin-Watson test)
- Step 5d: Check for Outliers (Cook's Distance)
- Step 5e: Model Validation Summary



WINDOWS VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import OLSInfluence
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 1. LOAD DATA AND REFIT MODELS
# ============================================================================

print("="*70)
print("STEP 5: MODEL VALIDATION (CHECK RESIDUALS)")
print("="*70)
print("\n1. Loading Data and Refitting Models...")

# Load the merged data
data_file = os.path.join('outputs', 'merged_data.csv')

if not os.path.exists(data_file):
    print("❌ ERROR: merged_data.csv not found!")
    print("   Please run Step 1 first to generate the merged dataset.")
    exit(1)

df = pd.read_csv(data_file)
print(f"✓ Data loaded: {len(df)} observations")

# Create output folder for residual diagnostics
residuals_folder = os.path.join('outputs', 'residual_diagnostics')
if not os.path.exists(residuals_folder):
    os.makedirs(residuals_folder)
    print(f"✓ Created residuals folder: {residuals_folder}")

# Define variables
crops = ['Cassava_Yield', 'Cocoa_Yield', 'Maize_Yield', 'Rice_Yield']
crop_names = ['Cassava', 'Cocoa Beans', 'Maize', 'Rice']
predictors = ['Rainfall', 'Temperature', 'Solar_Irradiance']

print(f"\n✓ Validating {len(crops)} regression models")

# Refit all models and store residuals
models_data = []

for crop, crop_name in zip(crops, crop_names):
    # Prepare data
    X = df[predictors]
    y = df[crop]
    X_with_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    # Extract residuals and fitted values
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Store data
    models_data.append({
        'Crop': crop_name,
        'Crop_Code': crop,
        'Model': model,
        'Residuals': residuals,
        'Fitted_Values': fitted_values,
        'N_Obs': len(residuals)
    })

print(f"✓ All models refitted successfully")

# ============================================================================
# STEP 5a: TEST NORMALITY OF RESIDUALS
# ============================================================================

print("\n" + "="*70)
print("STEP 5a: TEST NORMALITY OF RESIDUALS")
print("="*70)

print("\nRunning Shapiro-Wilk tests on residuals...")

normality_results = []

print("\n{:<15s} {:<15s} {:<15s} {:<20s}".format(
    "Crop", "W-Statistic", "P-Value", "Interpretation"))
print("-"*70)

for model_data in models_data:
    crop_name = model_data['Crop']
    residuals = model_data['Residuals']
    
    # Shapiro-Wilk test
    w_stat, p_value = stats.shapiro(residuals)
    
    # Interpretation
    if p_value > 0.05:
        interpretation = "Normal ✓"
        is_normal = True
    else:
        interpretation = "Not Normal ✗"
        is_normal = False
    
    print("{:<15s} {:<15.4f} {:<15.4f} {:<20s}".format(
        crop_name, w_stat, p_value, interpretation))
    
    normality_results.append({
        'Crop': crop_name,
        'W_Statistic': w_stat,
        'P_Value': p_value,
        'Normal': is_normal
    })

print("-"*70)
print("\nInterpretation:")
print("  p > 0.05: Residuals appear normally distributed ✓")
print("  p ≤ 0.05: Residuals deviate from normality")

# Create Q-Q plots for all crops
print("\nCreating Q-Q plots for residuals...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (model_data, norm_result) in enumerate(zip(models_data, normality_results)):
    ax = axes[i]
    residuals = model_data['Residuals']
    crop_name = model_data['Crop']
    
    # Create Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax)
    
    # Customize
    color = 'green' if norm_result['Normal'] else 'red'
    ax.get_lines()[0].set_color(color)
    ax.get_lines()[0].set_markerfacecolor('light' + color if color == 'green' else 'lightcoral')
    
    ax.set_title(f'{crop_name} Residuals Q-Q Plot\nW={norm_result["W_Statistic"]:.4f}, p={norm_result["P_Value"]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=9)
    ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation box
    textstr = "Normal ✓" if norm_result['Normal'] else "Not Normal ✗"
    props = dict(boxstyle='round', facecolor='lightgreen' if norm_result['Normal'] else 'lightcoral', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

plt.suptitle('Q-Q Plots: Residual Normality Assessment', fontsize=13, fontweight='bold')
plt.tight_layout()

qq_file = os.path.join(residuals_folder, 'qq_plots_residuals.png')
plt.savefig(qq_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: qq_plots_residuals.png")
plt.close()

# ============================================================================
# STEP 5b: CHECK HOMOSCEDASTICITY
# ============================================================================

print("\n" + "="*70)
print("STEP 5b: CHECK HOMOSCEDASTICITY (CONSTANT VARIANCE)")
print("="*70)

print("\nCreating Residuals vs Fitted Values plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

homoscedasticity_results = []

for i, model_data in enumerate(models_data):
    ax = axes[i]
    residuals = model_data['Residuals']
    fitted_values = model_data['Fitted_Values']
    crop_name = model_data['Crop']
    
    # Scatter plot
    ax.scatter(fitted_values, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    
    # Add lowess smoothing line to show pattern
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(residuals, fitted_values, frac=0.3)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2, label='Trend')
    
    # Formatting
    ax.set_xlabel('Fitted Values', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(f'{crop_name}: Residuals vs Fitted Values', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    # Visual assessment
    residual_std = residuals.std()
    upper_band = 2 * residual_std
    lower_band = -2 * residual_std
    
    # Check if variance appears constant
    # Simple check: compare variance in first half vs second half
    mid_point = fitted_values.median()
    var_low = residuals[fitted_values <= mid_point].var()
    var_high = residuals[fitted_values > mid_point].var()
    variance_ratio = max(var_low, var_high) / min(var_low, var_high)
    
    if variance_ratio < 2.0:
        homoscedastic = True
        assessment = "Constant variance ✓"
    else:
        homoscedastic = False
        assessment = "Variance issue ⚠"
    
    homoscedasticity_results.append({
        'Crop': crop_name,
        'Variance_Ratio': variance_ratio,
        'Homoscedastic': homoscedastic,
        'Assessment': assessment
    })
    
    # Add text box
    textstr = f'{assessment}\nVar ratio: {variance_ratio:.2f}'
    props = dict(boxstyle='round', facecolor='lightgreen' if homoscedastic else 'lightyellow', alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

plt.suptitle('Homoscedasticity Assessment: Residuals vs Fitted Values', fontsize=13, fontweight='bold')
plt.tight_layout()

homo_file = os.path.join(residuals_folder, 'residuals_vs_fitted.png')
plt.savefig(homo_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: residuals_vs_fitted.png")
plt.close()

print("\nHomoscedasticity Assessment:")
print("-"*70)
for result in homoscedasticity_results:
    print(f"{result['Crop']:<15s}: {result['Assessment']} (variance ratio: {result['Variance_Ratio']:.2f})")
print("-"*70)
print("Note: Variance ratio < 2.0 indicates acceptable homoscedasticity")

# ============================================================================
# STEP 5c: CHECK INDEPENDENCE (DURBIN-WATSON TEST)
# ============================================================================

print("\n" + "="*70)
print("STEP 5c: CHECK INDEPENDENCE (AUTOCORRELATION)")
print("="*70)

print("\nRunning Durbin-Watson tests...")

independence_results = []

print("\n{:<15s} {:<15s} {:<30s}".format("Crop", "DW Statistic", "Interpretation"))
print("-"*70)

for model_data in models_data:
    crop_name = model_data['Crop']
    residuals = model_data['Residuals']
    
    # Durbin-Watson test
    dw_stat = durbin_watson(residuals)
    
    # Interpretation
    if 1.5 <= dw_stat <= 2.5:
        interpretation = "No autocorrelation ✓"
        autocorrelated = False
    elif dw_stat < 1.5:
        interpretation = "Positive autocorrelation ⚠"
        autocorrelated = True
    else:
        interpretation = "Negative autocorrelation ⚠"
        autocorrelated = True
    
    print("{:<15s} {:<15.4f} {:<30s}".format(crop_name, dw_stat, interpretation))
    
    independence_results.append({
        'Crop': crop_name,
        'DW_Statistic': dw_stat,
        'Autocorrelated': autocorrelated,
        'Interpretation': interpretation
    })

print("-"*70)
print("Interpretation Guide:")
print("  DW ≈ 2.0: No autocorrelation (ideal)")
print("  DW < 1.5: Positive autocorrelation (successive residuals are similar)")
print("  DW > 2.5: Negative autocorrelation (successive residuals alternate)")
print("  Acceptable range: 1.5 - 2.5")

# Create Residuals vs Year plots
print("\nCreating Residuals vs Year plots...")

# Add Year to models_data for plotting
for model_data in models_data:
    model_data['Year'] = df['Year'].values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (model_data, indep_result) in enumerate(zip(models_data, independence_results)):
    ax = axes[i]
    residuals = model_data['Residuals']
    years = model_data['Year']
    crop_name = model_data['Crop']
    
    # Plot residuals over time
    ax.plot(years, residuals, marker='o', linestyle='-', alpha=0.6, markersize=5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(f'{crop_name}: Residuals Over Time\nDW={indep_result["DW_Statistic"]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation box
    textstr = indep_result['Interpretation']
    color = 'lightgreen' if not indep_result['Autocorrelated'] else 'lightyellow'
    props = dict(boxstyle='round', facecolor=color, alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

plt.suptitle('Independence Assessment: Residuals Over Time', fontsize=13, fontweight='bold')
plt.tight_layout()

indep_file = os.path.join(residuals_folder, 'residuals_vs_year.png')
plt.savefig(indep_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: residuals_vs_year.png")
plt.close()

# ============================================================================
# STEP 5d: CHECK FOR OUTLIERS (COOK'S DISTANCE)
# ============================================================================

print("\n" + "="*70)
print("STEP 5d: CHECK FOR OUTLIERS AND INFLUENTIAL POINTS")
print("="*70)

print("\nCalculating Cook's Distance...")

outlier_results = []

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, model_data in enumerate(zip(models_data)):
    model_data = model_data[0]  # Unpack tuple
    ax = axes[i]
    model = model_data['Model']
    crop_name = model_data['Crop']
    
    # Calculate Cook's Distance
    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]
    
    # Find influential points
    n_obs = len(cooks_d)
    threshold = 4 / n_obs  # Common threshold
    influential_points = np.where(cooks_d > threshold)[0]
    high_influence = np.where(cooks_d > 1.0)[0]
    
    # Plot Cook's Distance
    ax.stem(range(n_obs), cooks_d, linefmt='grey', markerfmt='o', basefmt=' ')
    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='High influence (1.0)')
    
    # Highlight influential points
    if len(influential_points) > 0:
        ax.plot(influential_points, cooks_d[influential_points], 'ro', markersize=8, 
                label=f'{len(influential_points)} influential points')
    
    # Formatting
    ax.set_xlabel('Observation Index', fontsize=10)
    ax.set_ylabel("Cook's Distance", fontsize=10)
    ax.set_title(f'{crop_name}: Cook\'s Distance', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Store results
    outlier_results.append({
        'Crop': crop_name,
        'N_Influential': len(influential_points),
        'N_High_Influence': len(high_influence),
        'Max_Cooks_D': cooks_d.max(),
        'Influential_Indices': influential_points.tolist(),
        'Years': df['Year'].iloc[influential_points].tolist() if len(influential_points) > 0 else []
    })

plt.suptitle("Outlier Detection: Cook's Distance", fontsize=13, fontweight='bold')
plt.tight_layout()

cooks_file = os.path.join(residuals_folder, 'cooks_distance.png')
plt.savefig(cooks_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: cooks_distance.png")
plt.close()

print("\nOutlier Assessment:")
print("-"*70)
for result in outlier_results:
    print(f"\n{result['Crop']}:")
    print(f"  Influential points: {result['N_Influential']}")
    print(f"  High influence points (D>1.0): {result['N_High_Influence']}")
    print(f"  Maximum Cook's D: {result['Max_Cooks_D']:.4f}")
    if result['Years']:
        print(f"  Influential years: {result['Years']}")
print("-"*70)

# ============================================================================
# STEP 5e: MODEL VALIDATION SUMMARY
# ============================================================================

print("\n" + "="*70)
print("STEP 5e: MODEL VALIDATION SUMMARY")
print("="*70)

# Create summary table
summary_data = []

for i in range(len(crops)):
    crop_name = crop_names[i]
    
    summary_data.append({
        'Crop': crop_name,
        'Normality': '✓' if normality_results[i]['Normal'] else '✗',
        'Normality_P': normality_results[i]['P_Value'],
        'Homoscedasticity': '✓' if homoscedasticity_results[i]['Homoscedastic'] else '⚠',
        'Independence': '✓' if not independence_results[i]['Autocorrelated'] else '⚠',
        'DW_Stat': independence_results[i]['DW_Statistic'],
        'Outliers': outlier_results[i]['N_High_Influence'],
        'Valid': 'Yes' if (normality_results[i]['Normal'] and 
                          homoscedasticity_results[i]['Homoscedastic'] and 
                          not independence_results[i]['Autocorrelated'] and
                          outlier_results[i]['N_High_Influence'] == 0) else 'With notes'
    })

summary_df = pd.DataFrame(summary_data)

print("\n" + "="*70)
print("VALIDATION SUMMARY TABLE")
print("="*70)
print("\n{:<15s} {:<10s} {:<15s} {:<10s} {:<15s}".format(
    "Crop", "Normality", "Homosced.", "Independence", "Outliers"))
print("-"*70)
for _, row in summary_df.iterrows():
    print("{:<15s} {:<10s} {:<15s} {:<10s} {:<15s}".format(
        row['Crop'],
        f"{row['Normality']} (p={row['Normality_P']:.3f})",
        row['Homoscedasticity'],
        f"{row['Independence']} (DW={row['DW_Stat']:.2f})",
        f"{row['Outliers']} high" if row['Outliers'] > 0 else "None"
    ))
print("-"*70)

# Overall assessment
valid_count = sum([1 for r in summary_data if r['Valid'] == 'Yes'])
print(f"\nOverall: {valid_count}/{len(crops)} models meet all assumptions perfectly")
if valid_count < len(crops):
    print("Note: Models with minor violations can still provide valid inference")
    print("      Violations noted should be acknowledged in limitations")

# Save summary
summary_file = os.path.join(residuals_folder, 'validation_summary.csv')
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Saved: validation_summary.csv")

# ============================================================================
# 6. CREATE COMPREHENSIVE REPORT
# ============================================================================

print("\n6. Creating Validation Report...")

report_lines = []
report_lines.append("="*70)
report_lines.append("MODEL VALIDATION REPORT")
report_lines.append("Residual Diagnostics for Regression Models")
report_lines.append("="*70)
report_lines.append("")
report_lines.append(f"Dataset: Ghana Climate-Agriculture Data (1985-2023)")
report_lines.append(f"Models Validated: {len(crops)}")
report_lines.append("")

report_lines.append("VALIDATION RESULTS BY CROP:")
report_lines.append("-"*70)

for i, crop_name in enumerate(crop_names):
    report_lines.append(f"\n{crop_name}:")
    report_lines.append(f"  Normality: {normality_results[i]['Normal']} (W={normality_results[i]['W_Statistic']:.4f}, p={normality_results[i]['P_Value']:.4f})")
    report_lines.append(f"  Homoscedasticity: {homoscedasticity_results[i]['Homoscedastic']} (variance ratio={homoscedasticity_results[i]['Variance_Ratio']:.2f})")
    report_lines.append(f"  Independence: {not independence_results[i]['Autocorrelated']} (DW={independence_results[i]['DW_Statistic']:.4f})")
    report_lines.append(f"  Outliers: {outlier_results[i]['N_High_Influence']} high-influence points")
    report_lines.append(f"  Overall: {summary_data[i]['Valid']}")

report_lines.append("")
report_lines.append("\nKEY FINDINGS:")
report_lines.append("-"*70)

# Count how many pass each test
normal_count = sum([r['Normal'] for r in normality_results])
homo_count = sum([r['Homoscedastic'] for r in homoscedasticity_results])
indep_count = sum([not r['Autocorrelated'] for r in independence_results])
no_outliers_count = sum([1 for r in outlier_results if r['N_High_Influence'] == 0])

report_lines.append(f"• Normality: {normal_count}/{len(crops)} models have normal residuals")
report_lines.append(f"• Homoscedasticity: {homo_count}/{len(crops)} models show constant variance")
report_lines.append(f"• Independence: {indep_count}/{len(crops)} models show no autocorrelation")
report_lines.append(f"• No high-influence outliers: {no_outliers_count}/{len(crops)} models")

report_lines.append("")
report_lines.append("\nCONCLUSION:")
report_lines.append("-"*70)
if valid_count == len(crops):
    report_lines.append("All models meet regression assumptions. Results are valid and reliable.")
else:
    report_lines.append(f"{valid_count}/{len(crops)} models fully meet all assumptions.")
    report_lines.append("Models with minor assumption violations can still provide valid inference,")
    report_lines.append("but results should be interpreted with appropriate caution.")
    report_lines.append("\nViolations to note:")
    for i, data in enumerate(summary_data):
        if data['Valid'] != 'Yes':
            violations = []
            if not normality_results[i]['Normal']:
                violations.append("non-normal residuals")
            if not homoscedasticity_results[i]['Homoscedastic']:
                violations.append("heteroscedasticity")
            if independence_results[i]['Autocorrelated']:
                violations.append("autocorrelation")
            if outlier_results[i]['N_High_Influence'] > 0:
                violations.append(f"{outlier_results[i]['N_High_Influence']} influential outlier(s)")
            report_lines.append(f"  • {data['Crop']}: {', '.join(violations)}")

report_lines.append("")
report_lines.append("="*70)
report_lines.append("Generated: December 2025")
report_lines.append("="*70)

# Save report
report_file = os.path.join(residuals_folder, 'validation_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Saved: validation_report.txt")

# Print report
print("\n" + '\n'.join(report_lines))

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print("\n" + "="*70)
print("✅ STEP 5 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files (in 'outputs/residual_diagnostics' folder):")
print("  1. qq_plots_residuals.png - Q-Q plots for normality check")
print("  2. residuals_vs_fitted.png - Homoscedasticity assessment")
print("  3. residuals_vs_year.png - Independence/autocorrelation check")
print("  4. cooks_distance.png - Outlier and influence detection")
print("  5. validation_summary.csv - Summary table of all tests")
print("  6. validation_report.txt - Comprehensive validation report")
print("\nValidation Results:")
print(f"  • {valid_count}/{len(crops)} models meet all assumptions")
print(f"  • {normal_count}/{len(crops)} models have normal residuals")
print(f"  • {homo_count}/{len(crops)} models show homoscedasticity")
print(f"  • {indep_count}/{len(crops)} models show independence")
print("\nYour regression models are validated and ready for interpretation!")
print("="*70)

print(f"\n📁 Results saved in: {os.path.abspath(residuals_folder)}")
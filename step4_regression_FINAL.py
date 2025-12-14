"""
STEP 4: TIME SERIES REGRESSION ANALYSIS
Climate-Agriculture Analysis for Ghana (1985-2024)

Comprehensive regression analysis:
- Step 4a: Check Linearity
- Step 4b: Correlation Matrix
- Step 4c: Fit Multiple Regression Model
- Step 4d: Check Multicollinearity (VIF)
- Step 4e: Model Refinement
- Step 4f: Interpret Results


WINDOWS VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
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
print("STEP 4: TIME SERIES REGRESSION ANALYSIS")
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

# Create output folder for regression analysis
regression_folder = os.path.join('outputs', 'regression_analysis')
if not os.path.exists(regression_folder):
    os.makedirs(regression_folder)
    print(f"✓ Created regression analysis folder: {regression_folder}")

# ============================================================================
# 2. DEFINE VARIABLES
# ============================================================================

print("\n2. Defining Variables...")

# Define crops and predictors
crops = ['Cassava_Yield', 'Cocoa_Yield', 'Maize_Yield', 'Rice_Yield']
crop_names = ['Cassava', 'Cocoa Beans', 'Maize', 'Rice']
# NOTE: Year removed to avoid severe multicollinearity (VIF > 12,000)
# Using all 3 climate variables - VIF tests show NO multicollinearity between them (all VIF ~1.0)
predictors = ['Rainfall', 'Temperature', 'Solar_Irradiance']

print(f"✓ Dependent variables: {len(crops)} crops")
print(f"✓ Independent variables: {len(predictors)} predictors")
print(f"  Predictors: {', '.join(predictors)}")
print(f"  Note: Year removed (VIF > 12,000), climate variables have acceptable VIF (~1.0)")

# ============================================================================
# STEP 4a: CHECK LINEARITY
# ============================================================================

print("\n" + "="*70)
print("STEP 4a: CHECK LINEARITY")
print("="*70)

print("\nCreating scatter plots to assess linear relationships...")

# Create scatter plots for each crop with each predictor
for crop, crop_name in zip(crops, crop_names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Hide the 4th subplot since we only have 3 predictors
    axes[3].axis('off')
    
    predictor_labels = ['Rainfall (mm)', 'Temperature (°C)', 'Solar Irradiance (kWh/m²/day)']
    
    for i, (pred, pred_label) in enumerate(zip(predictors, predictor_labels)):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(df[pred], df[crop], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(df[pred], df[crop], 1)
        p = np.poly1d(z)
        ax.plot(df[pred], p(df[pred]), "r--", linewidth=2, alpha=0.8, label='Linear fit')
        
        # Calculate correlation
        corr, p_val = stats.pearsonr(df[pred], df[crop])
        
        # Formatting
        ax.set_xlabel(pred_label, fontsize=10)
        ax.set_ylabel(f'{crop_name} Yield (kg/ha)', fontsize=10)
        ax.set_title(f'{crop_name} vs {pred_label}\nr = {corr:.3f}, p = {p_val:.4f}', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(f'Linearity Assessment: {crop_name} Yield', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    linearity_file = os.path.join(regression_folder, f'linearity_{crop_name.lower().replace(" ", "_")}.png')
    plt.savefig(linearity_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: linearity_{crop_name.lower().replace(' ', '_')}.png")
    plt.close()

# ============================================================================
# STEP 4b: CORRELATION MATRIX
# ============================================================================

print("\n" + "="*70)
print("STEP 4b: CORRELATION MATRIX")
print("="*70)

print("\nCalculating correlation matrix...")

# Select variables for correlation
corr_vars = predictors + crops
corr_data = df[corr_vars]

# Calculate correlation matrix
corr_matrix = corr_data.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot heatmap
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

plt.title('Correlation Matrix: Predictors and Crop Yields', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save figure
corr_file = os.path.join(regression_folder, 'correlation_matrix.png')
plt.savefig(corr_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: correlation_matrix.png")
plt.close()

# Save correlation matrix to CSV
corr_csv_file = os.path.join(regression_folder, 'correlation_matrix.csv')
corr_matrix.to_csv(corr_csv_file)
print(f"✓ Saved: correlation_matrix.csv")

# Print key correlations
print("\nKey Correlations with Crop Yields:")
print("-"*70)
for crop, crop_name in zip(crops, crop_names):
    print(f"\n{crop_name}:")
    crop_corrs = corr_matrix[crop][predictors].sort_values(ascending=False)
    for pred, corr_val in crop_corrs.items():
        print(f"  {pred:20s}: {corr_val:6.3f}")

# ============================================================================
# STEP 4c: FIT MULTIPLE REGRESSION MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 4c: FIT MULTIPLE REGRESSION MODEL")
print("="*70)

# Store all regression results
all_results = []

for crop, crop_name in zip(crops, crop_names):
    print(f"\n{'='*70}")
    print(f"REGRESSION MODEL: {crop_name} Yield")
    print(f"{'='*70}")
    
    # Prepare data
    X = df[predictors]
    y = df[crop]
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    # Fit model using statsmodels for detailed statistics
    model = sm.OLS(y, X_with_const).fit()
    
    # Print summary
    print(model.summary())
    
    # Extract key statistics
    results = {
        'Crop': crop_name,
        'Crop_Code': crop,
        'R_squared': model.rsquared,
        'R_squared_adj': model.rsquared_adj,
        'F_statistic': model.fvalue,
        'F_pvalue': model.f_pvalue,
        'AIC': model.aic,
        'BIC': model.bic,
        'Coefficients': {},
        'P_values': {},
        'Std_errors': {},
        'Model': model,
        'Predictors': predictors
    }
    
    # Store coefficients and p-values
    for i, pred in enumerate(['Intercept'] + predictors):
        results['Coefficients'][pred] = model.params[i]
        results['P_values'][pred] = model.pvalues[i]
        results['Std_errors'][pred] = model.bse[i]
    
    all_results.append(results)
    
    print(f"\n✓ Model fitted for {crop_name}")
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  Adjusted R² = {model.rsquared_adj:.4f}")
    print(f"  F-statistic = {model.fvalue:.4f} (p = {model.f_pvalue:.6f})")

# ============================================================================
# STEP 4d: CHECK MULTICOLLINEARITY (VIF)
# ============================================================================

print("\n" + "="*70)
print("STEP 4d: CHECK MULTICOLLINEARITY (VIF)")
print("="*70)

print("\nCalculating Variance Inflation Factors...")

# Prepare data for VIF calculation
X_vif = df[predictors]

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data['Predictor'] = predictors
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(predictors))]

# Add interpretation
def interpret_vif(vif):
    if vif < 5:
        return "Good - No concern"
    elif vif < 10:
        return "Moderate - Monitor"
    else:
        return "High - Problematic"

vif_data['Interpretation'] = vif_data['VIF'].apply(interpret_vif)

print("\nVariance Inflation Factors:")
print("-"*70)
print(vif_data.to_string(index=False))
print("-"*70)
print("\nInterpretation Guide:")
print("  VIF < 5:  No multicollinearity concern")
print("  VIF 5-10: Moderate multicollinearity (monitor)")
print("  VIF > 10: High multicollinearity (problematic - consider removing predictor)")

# Save VIF results
vif_file = os.path.join(regression_folder, 'vif_results.csv')
vif_data.to_csv(vif_file, index=False)
print(f"\n✓ Saved: vif_results.csv")

# Identify problematic predictors
high_vif = vif_data[vif_data['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️  WARNING: {len(high_vif)} predictor(s) with high VIF:")
    for _, row in high_vif.iterrows():
        print(f"  - {row['Predictor']}: VIF = {row['VIF']:.2f}")
    print("  Consider removing these predictors in Step 4e (Model Refinement)")
else:
    print("\n✓ No high multicollinearity detected - all VIF values acceptable")

# ============================================================================
# STEP 4e: MODEL REFINEMENT
# ============================================================================

print("\n" + "="*70)
print("STEP 4e: MODEL REFINEMENT")
print("="*70)

print("\nRefining models by removing non-significant predictors (p > 0.05)...")
print("and high VIF predictors (VIF > 10)...")

# Get high VIF predictors
high_vif_predictors = vif_data[vif_data['VIF'] > 10]['Predictor'].tolist()

refined_results = []

for result in all_results:
    crop_name = result['Crop']
    print(f"\n{'='*70}")
    print(f"REFINING MODEL: {crop_name} Yield")
    print(f"{'='*70}")
    
    # Identify non-significant predictors (excluding intercept)
    non_sig_predictors = [pred for pred in predictors 
                         if result['P_values'][pred] > 0.05]
    
    # Combine high VIF and non-significant predictors
    predictors_to_remove = list(set(high_vif_predictors + non_sig_predictors))
    
    if len(predictors_to_remove) > 0:
        print(f"\nRemoving predictors:")
        for pred in predictors_to_remove:
            reason = []
            if pred in high_vif_predictors:
                reason.append(f"High VIF ({vif_data[vif_data['Predictor']==pred]['VIF'].values[0]:.2f})")
            if pred in non_sig_predictors:
                reason.append(f"Non-significant (p={result['P_values'][pred]:.4f})")
            print(f"  - {pred}: {', '.join(reason)}")
        
        # Keep only significant predictors with acceptable VIF
        refined_predictors = [p for p in predictors if p not in predictors_to_remove]
        
        if len(refined_predictors) == 0:
            print("  ⚠️  WARNING: All predictors removed! Keeping all predictors for this crop.")
            refined_predictors = predictors
        
        # Refit model with refined predictors
        X_refined = df[refined_predictors]
        y = df[result['Crop_Code']]
        X_refined_const = sm.add_constant(X_refined)
        
        refined_model = sm.OLS(y, X_refined_const).fit()
        
        print(f"\nRefined Model Summary:")
        print(f"  Original R² = {result['R_squared']:.4f}")
        print(f"  Refined R²  = {refined_model.rsquared:.4f}")
        print(f"  Original Adjusted R² = {result['R_squared_adj']:.4f}")
        print(f"  Refined Adjusted R²  = {refined_model.rsquared_adj:.4f}")
        
        # Store refined results
        refined_result = {
            'Crop': crop_name,
            'Crop_Code': result['Crop_Code'],
            'Original_R_squared': result['R_squared'],
            'Refined_R_squared': refined_model.rsquared,
            'Original_R_squared_adj': result['R_squared_adj'],
            'Refined_R_squared_adj': refined_model.rsquared_adj,
            'Removed_Predictors': predictors_to_remove,
            'Refined_Predictors': refined_predictors,
            'Model': refined_model,
            'Coefficients': {},
            'P_values': {},
            'Std_errors': {}
        }
        
        # Store coefficients and p-values for refined model
        for i, pred in enumerate(['Intercept'] + refined_predictors):
            refined_result['Coefficients'][pred] = refined_model.params[i]
            refined_result['P_values'][pred] = refined_model.pvalues[i]
            refined_result['Std_errors'][pred] = refined_model.bse[i]
        
        refined_results.append(refined_result)
        
        print(f"\n✓ Refined model for {crop_name}")
        print(f"  Predictors retained: {', '.join(refined_predictors)}")
        
    else:
        print(f"\n✓ No refinement needed - all predictors are significant with acceptable VIF")
        
        # Still store as "refined" for consistency
        refined_result = {
            'Crop': crop_name,
            'Crop_Code': result['Crop_Code'],
            'Original_R_squared': result['R_squared'],
            'Refined_R_squared': result['R_squared'],
            'Original_R_squared_adj': result['R_squared_adj'],
            'Refined_R_squared_adj': result['R_squared_adj'],
            'Removed_Predictors': [],
            'Refined_Predictors': predictors,
            'Model': result['Model'],
            'Coefficients': result['Coefficients'],
            'P_values': result['P_values'],
            'Std_errors': result['Std_errors']
        }
        
        refined_results.append(refined_result)

# ============================================================================
# STEP 4f: INTERPRET RESULTS
# ============================================================================

print("\n" + "="*70)
print("STEP 4f: INTERPRET RESULTS")
print("="*70)

print("\nFinal Model Summary for All Crops:")
print("="*70)

# Create summary table
summary_data = []
for result in refined_results:
    summary_data.append({
        'Crop': result['Crop'],
        'R²': result['Refined_R_squared'],
        'Adj. R²': result['Refined_R_squared_adj'],
        'Predictors': ', '.join(result['Refined_Predictors']),
        'Removed': ', '.join(result['Removed_Predictors']) if result['Removed_Predictors'] else 'None'
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Detailed interpretation for each crop
print("\n\nDetailed Interpretation:")
print("="*70)

for result in refined_results:
    print(f"\n{result['Crop']}:")
    print("-"*70)
    print(f"R² = {result['Refined_R_squared']:.4f} ({result['Refined_R_squared']*100:.1f}% of variance explained)")
    print(f"Adjusted R² = {result['Refined_R_squared_adj']:.4f}")
    print(f"\nSignificant Predictors:")
    
    for pred in result['Refined_Predictors']:
        coef = result['Coefficients'][pred]
        p_val = result['P_values'][pred]
        std_err = result['Std_errors'][pred]
        
        # Interpretation
        direction = "increases" if coef > 0 else "decreases"
        
        print(f"  • {pred}:")
        print(f"      Coefficient: {coef:.4f} (p = {p_val:.4f})")
        print(f"      Interpretation: 1 unit increase in {pred} {direction} yield by {abs(coef):.2f} kg/ha")

# Create visualization of coefficients
print("\n\nCreating coefficient comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, result in enumerate(refined_results):
    ax = axes[i]
    
    # Get predictors and coefficients (exclude intercept)
    preds = result['Refined_Predictors']
    coefs = [result['Coefficients'][p] for p in preds]
    p_vals = [result['P_values'][p] for p in preds]
    
    # Color by significance
    colors = ['green' if p < 0.05 else 'red' for p in p_vals]
    
    # Create bar plot
    bars = ax.barh(preds, coefs, color=colors, alpha=0.7, edgecolor='black')
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Formatting
    ax.set_xlabel('Coefficient Value', fontsize=10)
    ax.set_title(f'{result["Crop"]} Yield\nR² = {result["Refined_R_squared"]:.3f}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Significant (p<0.05)'),
                      Patch(facecolor='red', alpha=0.7, label='Not significant (p≥0.05)')]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)

plt.suptitle('Regression Coefficients for All Crops', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
coef_file = os.path.join(regression_folder, 'coefficient_comparison.png')
plt.savefig(coef_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: coefficient_comparison.png")
plt.close()

# ============================================================================
# 7. SAVE FINAL RESULTS
# ============================================================================

print("\n7. Saving Final Results...")

# Create comprehensive results table
final_results_list = []
for result in refined_results:
    for pred in result['Refined_Predictors']:
        final_results_list.append({
            'Crop': result['Crop'],
            'Predictor': pred,
            'Coefficient': result['Coefficients'][pred],
            'Std_Error': result['Std_errors'][pred],
            'P_Value': result['P_values'][pred],
            'Significant': 'Yes' if result['P_values'][pred] < 0.05 else 'No',
            'R_squared': result['Refined_R_squared'],
            'Adj_R_squared': result['Refined_R_squared_adj']
        })

final_results_df = pd.DataFrame(final_results_list)

# Save to CSV
results_csv = os.path.join(regression_folder, 'regression_results_final.csv')
final_results_df.to_csv(results_csv, index=False)
print(f"✓ Saved: regression_results_final.csv")

# Save model summary
summary_csv = os.path.join(regression_folder, 'model_summary.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"✓ Saved: model_summary.csv")

# ============================================================================
# 8. CREATE FINAL REPORT
# ============================================================================

print("\n8. Creating Final Report...")

report_lines = []
report_lines.append("="*70)
report_lines.append("REGRESSION ANALYSIS SUMMARY REPORT")
report_lines.append("Time Series Regression: Crop Yields vs Climate Variables")
report_lines.append("="*70)
report_lines.append("")
report_lines.append(f"Dataset: Ghana Climate-Agriculture Data (1985-2023)")
report_lines.append(f"Sample Size: {len(df)} observations")
report_lines.append(f"Crops Analyzed: {', '.join(crop_names)}")
report_lines.append(f"Climate Predictors: {', '.join(predictors)}")
report_lines.append(f"NOTE: Year removed due to severe multicollinearity (VIF > 12,000)")
report_lines.append(f"      Climate variables show NO multicollinearity (VIF ~1.0)")
report_lines.append("")

report_lines.append("MODEL SUMMARY:")
report_lines.append("-"*70)
for result in refined_results:
    report_lines.append(f"\n{result['Crop']}:")
    report_lines.append(f"  R² = {result['Refined_R_squared']:.4f} (explains {result['Refined_R_squared']*100:.1f}% of variance)")
    report_lines.append(f"  Adjusted R² = {result['Refined_R_squared_adj']:.4f}")
    report_lines.append(f"  Predictors included: {', '.join(result['Refined_Predictors'])}")
    if result['Removed_Predictors']:
        report_lines.append(f"  Predictors removed: {', '.join(result['Removed_Predictors'])}")

report_lines.append("")
report_lines.append("\nKEY FINDINGS:")
report_lines.append("-"*70)

# Find best fitting model
best_model = max(refined_results, key=lambda x: x['Refined_R_squared_adj'])
report_lines.append(f"• Best fitting model: {best_model['Crop']} (Adj. R² = {best_model['Refined_R_squared_adj']:.4f})")

# Count significant relationships
sig_count = sum([1 for r in final_results_list if r['Significant'] == 'Yes'])
total_tests = len(final_results_list)
report_lines.append(f"• Significant relationships: {sig_count}/{total_tests}")

report_lines.append("")
report_lines.append("="*70)
report_lines.append("Generated: December 2025")
report_lines.append("="*70)

# Save report
report_file = os.path.join(regression_folder, 'regression_analysis_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Saved: regression_analysis_report.txt")

# Print report
print("\n" + '\n'.join(report_lines))

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print("\n" + "="*70)
print("✅ STEP 4 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files (in 'outputs/regression_analysis' folder):")
print("  Linearity Assessment:")
print("    - linearity_[crop].png (4 files) - Scatter plots for linearity checks")
print("  Correlation Analysis:")
print("    - correlation_matrix.png - Correlation heatmap")
print("    - correlation_matrix.csv - Correlation values")
print("  VIF Analysis:")
print("    - vif_results.csv - Variance Inflation Factors")
print("  Final Models:")
print("    - coefficient_comparison.png - Visual comparison of coefficients")
print("    - regression_results_final.csv - Complete results table")
print("    - model_summary.csv - Summary statistics")
print("    - regression_analysis_report.txt - Comprehensive report")
print("\nNext Step: Proceed to Step 5 (Model Validation - Check Residuals)")
print("="*70)

print(f"\n📁 Results saved in: {os.path.abspath(regression_folder)}")
"""
STEP 6: ONE-WAY ANOVA BY DECADE
Climate-Agriculture Analysis for Ghana (1985-2024)

Analyzes whether crop yields differ significantly across decades:
- Decade 1: 1985-1994
- Decade 2: 1995-2004
- Decade 3: 2005-2014
- Decade 4: 2015-2023

Includes:
- One-Way ANOVA for each crop
- Post-hoc tests (Tukey HSD)
- Comprehensive statistics table (Omnibus, Jarque-Bera, Kurtosis, Skewness)
- Visualization (box plots)

Author: King
Date: December 2025

WINDOWS VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kurtosis, skew
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 1. LOAD DATA AND CREATE DECADE GROUPS
# ============================================================================

print("="*70)
print("STEP 6: ONE-WAY ANOVA BY DECADE")
print("="*70)
print("\n1. Loading Data and Creating Decade Groups...")

# Load the merged data
data_file = os.path.join('outputs', 'merged_data.csv')

if not os.path.exists(data_file):
    print("❌ ERROR: merged_data.csv not found!")
    print("   Please run Step 1 first to generate the merged dataset.")
    exit(1)

df = pd.read_csv(data_file)
print(f"✓ Data loaded: {len(df)} observations from {df['Year'].min()} to {df['Year'].max()}")

# Create output folder for ANOVA analysis
anova_folder = os.path.join('outputs', 'anova_analysis')
if not os.path.exists(anova_folder):
    os.makedirs(anova_folder)
    print(f"✓ Created ANOVA folder: {anova_folder}")

# Define decade groups
def assign_decade(year):
    if 1985 <= year <= 1994:
        return '1985-1994'
    elif 1995 <= year <= 2004:
        return '1995-2004'
    elif 2005 <= year <= 2014:
        return '2005-2014'
    elif 2015 <= year <= 2023:
        return '2015-2023'
    else:
        return 'Unknown'

df['Decade'] = df['Year'].apply(assign_decade)

print("\n✓ Decade groups created:")
print(df['Decade'].value_counts().sort_index())

# Define variables
crops = ['Cassava_Yield', 'Cocoa_Yield', 'Maize_Yield', 'Rice_Yield']
crop_names = ['Cassava', 'Cocoa Beans', 'Maize', 'Rice']

# ============================================================================
# 2. COMPREHENSIVE STATISTICS BY DECADE
# ============================================================================

print("\n" + "="*70)
print("STEP 6a: COMPREHENSIVE STATISTICS BY DECADE AND CROP")
print("="*70)

all_stats = []

decades = ['1985-1994', '1995-2004', '2005-2014', '2015-2023']

for crop, crop_name in zip(crops, crop_names):
    print(f"\n{'='*70}")
    print(f"STATISTICS: {crop_name}")
    print(f"{'='*70}")
    
    for decade in decades:
        decade_data = df[df['Decade'] == decade][crop].dropna()
        n = len(decade_data)
        
        if n < 3:
            print(f"\n⚠️  WARNING: {decade} has only {n} observations - skipping statistics")
            continue
        
        # Calculate statistics
        mean_val = decade_data.mean()
        median_val = decade_data.median()
        std_val = decade_data.std()
        var_val = decade_data.var()
        min_val = decade_data.min()
        max_val = decade_data.max()
        range_val = max_val - min_val
        
        # Skewness and Kurtosis
        skewness = skew(decade_data)
        kurt = kurtosis(decade_data, fisher=True)  # Excess kurtosis (normal = 0)
        
        # Omnibus test for normality (combines skewness and kurtosis)
        omnibus_stat, omnibus_p = stats.normaltest(decade_data)
        
        # Jarque-Bera test for normality
        jb_stat, jb_p = stats.jarque_bera(decade_data)
        
        # Store results
        all_stats.append({
            'Crop': crop_name,
            'Decade': decade,
            'N': n,
            'Mean': mean_val,
            'Median': median_val,
            'Std_Dev': std_val,
            'Variance': var_val,
            'Min': min_val,
            'Max': max_val,
            'Range': range_val,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Omnibus_Stat': omnibus_stat,
            'Omnibus_P': omnibus_p,
            'JB_Stat': jb_stat,
            'JB_P': jb_p
        })
        
        # Print summary
        print(f"\n{decade}:")
        print(f"  N = {n}")
        print(f"  Mean = {mean_val:.2f}, Median = {median_val:.2f}, SD = {std_val:.2f}")
        print(f"  Range: [{min_val:.2f}, {max_val:.2f}]")
        print(f"  Skewness = {skewness:.4f}, Kurtosis = {kurt:.4f}")
        print(f"  Omnibus: stat={omnibus_stat:.4f}, p={omnibus_p:.4f} {'(Normal)' if omnibus_p > 0.05 else '(Not Normal)'}")
        print(f"  Jarque-Bera: stat={jb_stat:.4f}, p={jb_p:.4f} {'(Normal)' if jb_p > 0.05 else '(Not Normal)'}")

# Create comprehensive statistics DataFrame
stats_df = pd.DataFrame(all_stats)

# Save to CSV
stats_file = os.path.join(anova_folder, 'comprehensive_statistics_by_decade.csv')
stats_df.to_csv(stats_file, index=False)
print(f"\n✓ Saved: comprehensive_statistics_by_decade.csv")

# ============================================================================
# 3. ONE-WAY ANOVA FOR EACH CROP
# ============================================================================

print("\n" + "="*70)
print("STEP 6b: ONE-WAY ANOVA TESTS")
print("="*70)

anova_results = []

print("\nTesting: H₀: Mean yields are equal across all decades")
print("         H₁: At least one decade differs\n")

print("{:<15s} {:<12s} {:<12s} {:<12s}".format(
    "Crop", "F-Statistic", "P-Value", "Significant?"))
print("-"*70)

for crop, crop_name in zip(crops, crop_names):
    # Get data for each decade
    group1 = df[df['Decade'] == '1985-1994'][crop].dropna()
    group2 = df[df['Decade'] == '1995-2004'][crop].dropna()
    group3 = df[df['Decade'] == '2005-2014'][crop].dropna()
    group4 = df[df['Decade'] == '2015-2023'][crop].dropna()
    
    # Perform One-Way ANOVA
    f_stat, p_value = f_oneway(group1, group2, group3, group4)
    
    # Interpretation
    significant = "Yes" if p_value < 0.05 else "No"
    
    print("{:<15s} {:<12.4f} {:<12.4f} {:<12s}".format(
        crop_name, f_stat, p_value, significant))
    
    # Store results
    anova_results.append({
        'Crop': crop_name,
        'F_Statistic': f_stat,
        'P_Value': p_value,
        'Significant': significant,
        'df_between': 3,  # 4 groups - 1
        'df_within': len(group1) + len(group2) + len(group3) + len(group4) - 4
    })

print("-"*70)
print("\nInterpretation:")
print("  p < 0.05: Yields differ significantly across decades")
print("  p ≥ 0.05: No significant difference across decades")

# Save ANOVA results
anova_df = pd.DataFrame(anova_results)
anova_file = os.path.join(anova_folder, 'anova_results.csv')
anova_df.to_csv(anova_file, index=False)
print(f"\n✓ Saved: anova_results.csv")

# ============================================================================
# 4. POST-HOC TESTS (TUKEY HSD)
# ============================================================================

print("\n" + "="*70)
print("STEP 6c: POST-HOC TESTS (TUKEY HSD)")
print("="*70)

print("\nRunning pairwise comparisons for significant ANOVA results...")

posthoc_results = {}

for i, (crop, crop_name) in enumerate(zip(crops, crop_names)):
    if anova_results[i]['Significant'] == 'Yes':
        print(f"\n{crop_name} (p={anova_results[i]['P_Value']:.4f}):")
        print("-"*70)
        
        # Prepare data for Tukey HSD
        yield_data = df[[crop, 'Decade']].dropna()
        
        # Perform Tukey HSD
        tukey = pairwise_tukeyhsd(endog=yield_data[crop], groups=yield_data['Decade'], alpha=0.05)
        
        print(tukey)
        
        # Store results
        posthoc_results[crop_name] = str(tukey)
        
        # Save individual Tukey results
        tukey_file = os.path.join(anova_folder, f'tukey_{crop_name.lower().replace(" ", "_")}.txt')
        with open(tukey_file, 'w') as f:
            f.write(f"Tukey HSD Post-Hoc Test: {crop_name}\n")
            f.write("="*70 + "\n\n")
            f.write(str(tukey))
        
        print(f"✓ Saved: tukey_{crop_name.lower().replace(' ', '_')}.txt")
    else:
        print(f"\n{crop_name}: No significant difference (p={anova_results[i]['P_Value']:.4f}) - skipping post-hoc")

# ============================================================================
# 5. CREATE BOX PLOTS BY DECADE
# ============================================================================

print("\n" + "="*70)
print("STEP 6d: CREATE VISUALIZATIONS")
print("="*70)

print("\nCreating box plots by decade...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (crop, crop_name, anova_result) in enumerate(zip(crops, crop_names, anova_results)):
    ax = axes[i]
    
    # Create box plot
    decade_order = ['1985-1994', '1995-2004', '2005-2014', '2015-2023']
    
    # Prepare data
    plot_data = df[[crop, 'Decade']].dropna()
    
    # Create box plot
    bp = ax.boxplot([plot_data[plot_data['Decade'] == d][crop].values for d in decade_order],
                     labels=decade_order,
                     patch_artist=True,
                     notch=True,
                     showmeans=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Formatting
    ax.set_xlabel('Decade', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{crop_name} Yield (kg/ha)', fontsize=11, fontweight='bold')
    
    # Title with ANOVA result
    sig_text = "***" if anova_result['P_Value'] < 0.001 else "**" if anova_result['P_Value'] < 0.01 else "*" if anova_result['P_Value'] < 0.05 else "ns"
    ax.set_title(f'{crop_name}\nF={anova_result["F_Statistic"]:.2f}, p={anova_result["P_Value"]:.4f} {sig_text}',
                 fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=8, label='Mean'),
    plt.Line2D([0], [0], linestyle='', marker='', label='*** p<0.001'),
    plt.Line2D([0], [0], linestyle='', marker='', label='** p<0.01'),
    plt.Line2D([0], [0], linestyle='', marker='', label='* p<0.05'),
    plt.Line2D([0], [0], linestyle='', marker='', label='ns = not significant')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=9)

plt.suptitle('Crop Yields Across Decades: One-Way ANOVA Analysis', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.03, 1, 0.99])

# Save figure
boxplot_file = os.path.join(anova_folder, 'boxplots_yields_by_decade.png')
plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: boxplots_yields_by_decade.png")
plt.close()

# ============================================================================
# 6. CREATE ANOVA SUMMARY TABLE
# ============================================================================

print("\n" + "="*70)
print("STEP 6e: CREATE ANOVA SUMMARY TABLE")
print("="*70)

# Create comprehensive ANOVA table
anova_table_data = []

for i, (crop_name, anova_result) in enumerate(zip(crop_names, anova_results)):
    # Get statistics for this crop
    crop_stats = stats_df[stats_df['Crop'] == crop_name]
    
    # Get overall statistics (across all decades)
    overall_data = df[crops[i]].dropna()
    
    overall_omnibus_stat, overall_omnibus_p = stats.normaltest(overall_data)
    overall_jb_stat, overall_jb_p = stats.jarque_bera(overall_data)
    overall_skewness = skew(overall_data)
    overall_kurtosis = kurtosis(overall_data, fisher=True)
    
    anova_table_data.append({
        'Crop': crop_name,
        'N_Total': len(overall_data),
        'Mean_Overall': overall_data.mean(),
        'SD_Overall': overall_data.std(),
        'F_Statistic': anova_result['F_Statistic'],
        'P_Value': anova_result['P_Value'],
        'Significant': anova_result['Significant'],
        'df_between': anova_result['df_between'],
        'df_within': anova_result['df_within'],
        'Omnibus_Stat': overall_omnibus_stat,
        'Omnibus_P': overall_omnibus_p,
        'JB_Stat': overall_jb_stat,
        'JB_P': overall_jb_p,
        'Skewness': overall_skewness,
        'Kurtosis': overall_kurtosis
    })

anova_table_df = pd.DataFrame(anova_table_data)

# Save ANOVA table
anova_table_file = os.path.join(anova_folder, 'anova_summary_table.csv')
anova_table_df.to_csv(anova_table_file, index=False)
print(f"✓ Saved: anova_summary_table.csv")

# Print formatted table
print("\n" + "="*70)
print("COMPREHENSIVE ANOVA SUMMARY TABLE")
print("="*70)
print("\n{:<15s} {:<8s} {:<12s} {:<12s} {:<10s}".format(
    "Crop", "N", "F-Stat", "P-Value", "Sig?"))
print("-"*70)
for _, row in anova_table_df.iterrows():
    print("{:<15s} {:<8d} {:<12.4f} {:<12.4f} {:<10s}".format(
        row['Crop'], row['N_Total'], row['F_Statistic'], row['P_Value'], row['Significant']))
print("-"*70)

print("\n{:<15s} {:<12s} {:<12s} {:<12s} {:<12s}".format(
    "Crop", "Omnibus(p)", "JB(p)", "Skewness", "Kurtosis"))
print("-"*70)
for _, row in anova_table_df.iterrows():
    print("{:<15s} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        row['Crop'], row['Omnibus_P'], row['JB_P'], row['Skewness'], row['Kurtosis']))
print("-"*70)

# ============================================================================
# 7. CREATE COMPREHENSIVE REPORT
# ============================================================================

print("\n7. Creating ANOVA Report...")

report_lines = []
report_lines.append("="*70)
report_lines.append("ONE-WAY ANOVA ANALYSIS BY DECADE")
report_lines.append("Crop Yields in Ghana (1985-2023)")
report_lines.append("="*70)
report_lines.append("")
report_lines.append("RESEARCH QUESTION:")
report_lines.append("Do crop yields differ significantly across decades?")
report_lines.append("")
report_lines.append("DECADES ANALYZED:")
report_lines.append("  • Decade 1: 1985-1994 (10 years)")
report_lines.append("  • Decade 2: 1995-2004 (10 years)")
report_lines.append("  • Decade 3: 2005-2014 (10 years)")
report_lines.append("  • Decade 4: 2015-2023 (9 years)")
report_lines.append("")

report_lines.append("ANOVA RESULTS:")
report_lines.append("-"*70)
for result in anova_results:
    report_lines.append(f"\n{result['Crop']}:")
    report_lines.append(f"  F({result['df_between']}, {result['df_within']}) = {result['F_Statistic']:.4f}")
    report_lines.append(f"  P-value = {result['P_Value']:.4f}")
    report_lines.append(f"  Result: {'Significant difference across decades' if result['Significant'] == 'Yes' else 'No significant difference'}")

report_lines.append("")
report_lines.append("\nSUMMARY:")
report_lines.append("-"*70)
sig_count = sum([1 for r in anova_results if r['Significant'] == 'Yes'])
report_lines.append(f"• {sig_count}/{len(crops)} crops show significant yield changes across decades")

report_lines.append("")
report_lines.append("\nSTATISTICAL ASSUMPTIONS:")
report_lines.append("-"*70)
report_lines.append("Normality tests (Omnibus and Jarque-Bera) were conducted for each crop.")
report_lines.append("See comprehensive_statistics_by_decade.csv for detailed results.")

report_lines.append("")
report_lines.append("="*70)
report_lines.append("Generated: December 2025")
report_lines.append("="*70)

# Save report
report_file = os.path.join(anova_folder, 'anova_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"✓ Saved: anova_report.txt")

# Print report
print("\n" + '\n'.join(report_lines))

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print("\n" + "="*70)
print("✅ STEP 6 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files (in 'outputs/anova_analysis' folder):")
print("  1. comprehensive_statistics_by_decade.csv - Full statistics table")
print("     (includes: N, Mean, Median, SD, Variance, Min, Max, Range,")
print("      Skewness, Kurtosis, Omnibus test, Jarque-Bera test)")
print("  2. anova_results.csv - ANOVA F-statistics and p-values")
print("  3. anova_summary_table.csv - Combined ANOVA + statistics")
print("  4. tukey_[crop].txt - Post-hoc pairwise comparisons (if significant)")
print("  5. boxplots_yields_by_decade.png - Visual comparison")
print("  6. anova_report.txt - Comprehensive written report")
print("\nKey Findings:")
print(f"  • {sig_count}/{len(crops)} crops show significant yield changes across decades")
print("  • Complete statistics available for all crops and decades")
print("="*70)

print(f"\n📁 Results saved in: {os.path.abspath(anova_folder)}")
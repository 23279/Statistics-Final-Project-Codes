"""
STEP 2: VISUAL EXPLORATION
Climate-Agriculture Analysis for Ghana (1985-2024)

Creates comprehensive visualizations:
- Histograms for all variables
- Time series plots for crop yields
- Time series plots for climate variables
- Box plots by decade



WINDOWS VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("="*70)
print("STEP 2: VISUAL EXPLORATION")
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

# Create visualizations output folder
viz_folder = os.path.join('outputs', 'visualizations')
if not os.path.exists(viz_folder):
    os.makedirs(viz_folder)
    print(f"✓ Created visualization folder: {viz_folder}")

# ============================================================================
# 2. HISTOGRAMS FOR ALL VARIABLES
# ============================================================================

print("\n2. Creating Histograms...")

# Define crop and climate variables
crop_vars = ['Cassava_Yield', 'Cocoa_Yield', 'Maize_Yield', 'Rice_Yield']
climate_vars = ['Rainfall', 'Temperature', 'Solar_Irradiance']

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Plot crop yield histograms
for i, crop in enumerate(crop_vars):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    ax.hist(df[crop], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Yield (kg/ha)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{crop.replace("_", " ")} Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean line
    mean_val = df[crop].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
    ax.legend()

# Plot climate variable histograms
climate_titles = ['Rainfall (mm)', 'Temperature (°C)', 'Solar Irradiance (kWh/m²/day)']
for i, (var, title) in enumerate(zip(climate_vars, climate_titles)):
    ax = fig.add_subplot(gs[2, i])
    ax.hist(df[var], bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel(title, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{var.replace("_", " ")} Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean line
    mean_val = df[var].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.legend()

plt.suptitle('Distribution of Crop Yields and Climate Variables (1985-2023)', 
             fontsize=14, fontweight='bold', y=0.995)

# Save figure
hist_file = os.path.join(viz_folder, 'histograms_all_variables.png')
plt.savefig(hist_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: histograms_all_variables.png")
plt.close()

# ============================================================================
# 3. TIME SERIES PLOTS - CROP YIELDS
# ============================================================================

print("\n3. Creating Time Series Plots for Crop Yields...")

# Individual time series for each crop
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
crop_names = ['Cassava', 'Cocoa Beans', 'Maize', 'Rice']

for i, (crop, color, name) in enumerate(zip(crop_vars, colors, crop_names)):
    ax = axes[i]
    
    # Plot data
    ax.plot(df['Year'], df[crop], marker='o', linewidth=2, color=color, 
            markersize=4, markerfacecolor='white', markeredgewidth=1.5)
    
    # Add trend line
    z = np.polyfit(df['Year'], df[crop], 1)
    p = np.poly1d(z)
    ax.plot(df['Year'], p(df['Year']), "--", color='red', linewidth=2, 
            alpha=0.8, label=f'Trend')
    
    # Formatting
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Yield (kg/ha)', fontsize=11)
    ax.set_title(f'{name} Yield Over Time (1985-2023)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add statistics text box
    mean_yield = df[crop].mean()
    min_yield = df[crop].min()
    max_yield = df[crop].max()
    textstr = f'Mean: {mean_yield:.0f}\nMin: {min_yield:.0f}\nMax: {max_yield:.0f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Crop Yield Trends Over 38 Years', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
ts_crops_file = os.path.join(viz_folder, 'timeseries_crop_yields.png')
plt.savefig(ts_crops_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: timeseries_crop_yields.png")
plt.close()

# ============================================================================
# 4. TIME SERIES PLOTS - CLIMATE VARIABLES
# ============================================================================

print("\n4. Creating Time Series Plots for Climate Variables...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

climate_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
climate_labels = ['Rainfall (mm)', 'Temperature (°C)', 'Solar Irradiance (kWh/m²/day)']

for i, (var, color, label) in enumerate(zip(climate_vars, climate_colors, climate_labels)):
    ax = axes[i]
    
    # Plot data
    ax.plot(df['Year'], df[var], marker='o', linewidth=2, color=color,
            markersize=4, markerfacecolor='white', markeredgewidth=1.5)
    
    # Add trend line
    z = np.polyfit(df['Year'], df[var], 1)
    p = np.poly1d(z)
    ax.plot(df['Year'], p(df['Year']), "--", color='red', linewidth=2,
            alpha=0.8, label='Trend')
    
    # Formatting
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{var.replace("_", " ")} Over Time (1985-2023)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add statistics
    mean_val = df[var].mean()
    min_val = df[var].min()
    max_val = df[var].max()
    trend_slope = z[0]
    textstr = f'Mean: {mean_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nTrend: {trend_slope:.4f}/year'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Climate Variables Trends Over 38 Years', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
ts_climate_file = os.path.join(viz_folder, 'timeseries_climate_variables.png')
plt.savefig(ts_climate_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: timeseries_climate_variables.png")
plt.close()

# ============================================================================
# 5. BOX PLOTS BY DECADE
# ============================================================================

print("\n5. Creating Box Plots by Decade...")

# Create decade categories
df['Decade'] = pd.cut(df['Year'], 
                      bins=[1984, 1994, 2004, 2014, 2024],
                      labels=['1985-1994', '1995-2004', '2005-2014', '2015-2023'])

# Create box plots for crop yields by decade
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (crop, name) in enumerate(zip(crop_vars, crop_names)):
    ax = axes[i]
    
    # Create box plot
    df.boxplot(column=crop, by='Decade', ax=ax, patch_artist=True,
               boxprops=dict(facecolor='lightblue', edgecolor='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Decade', fontsize=11)
    ax.set_ylabel('Yield (kg/ha)', fontsize=11)
    ax.set_title(f'{name} Yield by Decade', fontsize=13, fontweight='bold')
    ax.get_figure().suptitle('')  # Remove automatic title
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.suptitle('Crop Yields Variation Across Decades', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
box_file = os.path.join(viz_folder, 'boxplots_yields_by_decade.png')
plt.savefig(box_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: boxplots_yields_by_decade.png")
plt.close()

# ============================================================================
# 6. COMBINED OVERVIEW PLOT
# ============================================================================

print("\n6. Creating Combined Overview Plot...")

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# All crop yields on one plot
ax1 = fig.add_subplot(gs[0, :2])
for crop, color, name in zip(crop_vars, colors, crop_names):
    ax1.plot(df['Year'], df[crop], marker='o', linewidth=2, color=color,
             label=name, markersize=3, alpha=0.8)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Yield (kg/ha)', fontsize=11)
ax1.set_title('All Crop Yields Over Time', fontsize=13, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# All climate variables (normalized) on one plot
ax2 = fig.add_subplot(gs[1, :2])
for var, color, label in zip(climate_vars, climate_colors, climate_labels):
    # Normalize to 0-1 scale for comparison
    normalized = (df[var] - df[var].min()) / (df[var].max() - df[var].min())
    ax2.plot(df['Year'], normalized, marker='o', linewidth=2, color=color,
             label=label.split('(')[0].strip(), markersize=3, alpha=0.8)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Normalized Value (0-1)', fontsize=11)
ax2.set_title('Climate Variables Over Time (Normalized)', fontsize=13, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Summary statistics table
ax3 = fig.add_subplot(gs[:, 2])
ax3.axis('off')

# Create summary statistics
summary_data = []
summary_data.append(['Variable', 'Mean', 'Std Dev', 'Range'])
summary_data.append(['---', '---', '---', '---'])

for crop, name in zip(crop_vars, crop_names):
    mean_val = df[crop].mean()
    std_val = df[crop].std()
    range_val = df[crop].max() - df[crop].min()
    summary_data.append([name, f'{mean_val:.0f}', f'{std_val:.0f}', f'{range_val:.0f}'])

summary_data.append(['', '', '', ''])
for var, label in zip(climate_vars, ['Rainfall', 'Temperature', 'Solar Irr.']):
    mean_val = df[var].mean()
    std_val = df[var].std()
    range_val = df[var].max() - df[var].min()
    summary_data.append([label, f'{mean_val:.2f}', f'{std_val:.2f}', f'{range_val:.2f}'])

# Create table
table = ax3.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.35, 0.25, 0.25, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax3.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Overview: Crop Yields and Climate Variables (1985-2023)', 
             fontsize=14, fontweight='bold')

# Save figure
overview_file = os.path.join(viz_folder, 'overview_combined.png')
plt.savefig(overview_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: overview_combined.png")
plt.close()

# ============================================================================
# 7. SUMMARY OF VISUAL OBSERVATIONS
# ============================================================================

print("\n7. Generating Visual Observations Summary...")

# Calculate trends
observations = []
observations.append("="*70)
observations.append("VISUAL EXPLORATION SUMMARY")
observations.append("="*70)
observations.append("")

observations.append("CROP YIELD TRENDS:")
observations.append("-" * 70)
for crop, name in zip(crop_vars, crop_names):
    z = np.polyfit(df['Year'], df[crop], 1)
    slope = z[0]
    direction = "increasing" if slope > 0 else "decreasing"
    observations.append(f"• {name}: {direction} trend ({slope:.2f} kg/ha per year)")

observations.append("")
observations.append("CLIMATE VARIABLE TRENDS:")
observations.append("-" * 70)
for var, label in zip(climate_vars, climate_labels):
    z = np.polyfit(df['Year'], df[var], 1)
    slope = z[0]
    direction = "increasing" if slope > 0 else "decreasing"
    unit = label.split('(')[1].strip(')')
    observations.append(f"• {var.replace('_', ' ')}: {direction} trend ({slope:.4f} {unit} per year)")

observations.append("")
observations.append("VARIABILITY:")
observations.append("-" * 70)
for crop, name in zip(crop_vars, crop_names):
    cv = (df[crop].std() / df[crop].mean()) * 100
    observations.append(f"• {name}: CV = {cv:.1f}% variability")

observations.append("")
observations.append("KEY VISUAL OBSERVATIONS:")
observations.append("-" * 70)

# Find crop with strongest trend
crop_trends = {name: abs(np.polyfit(df['Year'], df[crop], 1)[0]) 
               for crop, name in zip(crop_vars, crop_names)}
strongest_trend = max(crop_trends, key=crop_trends.get)
observations.append(f"• {strongest_trend} shows the strongest trend over time")

# Find most variable crop
crop_cvs = {name: (df[crop].std() / df[crop].mean()) * 100 
            for crop, name in zip(crop_vars, crop_names)}
most_variable = max(crop_cvs, key=crop_cvs.get)
observations.append(f"• {most_variable} exhibits highest year-to-year variability")

observations.append("")
observations.append("="*70)
observations.append("Generated: December 2025")
observations.append("="*70)

# Save observations
obs_file = os.path.join(viz_folder, 'visual_observations.txt')
with open(obs_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(observations))

print(f"✓ Saved: visual_observations.txt")

# Print observations
print("\n" + '\n'.join(observations))

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print("\n" + "="*70)
print("✅ STEP 2 COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Visualizations (in 'outputs/visualizations' folder):")
print("  1. histograms_all_variables.png - Distribution of all variables")
print("  2. timeseries_crop_yields.png - Individual crop yield trends")
print("  3. timeseries_climate_variables.png - Climate variable trends")
print("  4. boxplots_yields_by_decade.png - Yield variation by decade")
print("  5. overview_combined.png - Combined summary visualization")
print("  6. visual_observations.txt - Summary of visual findings")
print("\nNext Step: Proceed to Step 3 (Test Normality of Raw Data)")
print("="*70)

print(f"\n📁 Visualizations saved in: {os.path.abspath(viz_folder)}")
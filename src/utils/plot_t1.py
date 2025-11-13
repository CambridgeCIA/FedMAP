import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

FILENAME = "results/example_metrics_test_tier1.csv"
OUTPUT_DIR = "results/performance_plots"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

LINEWIDTH = 1.8
MARKERSIZE = 4
FONT_TITLE = 13
FONT_LABEL = 11
DPI = 300

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    try:
        df = pd.read_csv(FILENAME)
    except FileNotFoundError:
        print(f"Error: Could not find file at '{FILENAME}'")
        print("Please check the path and try again.")
        return

    df['Client_ID'] = df['Client_ID'].astype(int)
    df['Round'] = df['Round'].astype(int)

    print(f"Loaded {len(df)} records")
    print(f"Clients : {sorted(df['Client_ID'].unique())}")
    print(f"Rounds  : {df['Round'].min()} to {df['Round'].max()}")


    client_order = sorted(df['Client_ID'].unique())
    final_df = df[df['Round'] == df['Round'].max()].copy()
    final_auc_map = dict(zip(final_df['Client_ID'], final_df['ROC_AUC'].round(4)))
    legend_labels = {cid: f"Client {cid}" for cid in client_order}


    metrics = ['Accuracy', 'Balanced_Accuracy', 'ROC_AUC', 'Loss']
    metric_titles = {
        'Accuracy': 'Accuracy',
        'Balanced_Accuracy': 'Balanced Accuracy',
        'ROC_AUC': 'ROC AUC',
        'Loss': 'Loss'
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        handles, labels = [], []
        for client in client_order:
            cdata = df[df['Client_ID'] == client].sort_values('Round')
            line, = ax.plot(
                cdata['Round'], cdata[metric],
                linewidth=LINEWIDTH,
                marker='o',
                markersize=MARKERSIZE,
                alpha=0.85
            )
            handles.append(line)
            labels.append(legend_labels[client])

        # Add global average line (except for Loss)
        if metric != 'Loss':
            avg = df.groupby('Round')[metric].mean()
            avg_line, = ax.plot(
                avg.index, avg.values,
                'k--', linewidth=2.8, label='Global Average', alpha=0.95
            )
            handles.insert(0, avg_line)
            labels.insert(0, 'Global Average')

        # --- Style and Save Plot ---
        ax.set_title(f'{metric_titles[metric]} over Communication Rounds',
                     fontsize=FONT_TITLE, fontweight='bold', pad=20)
        ax.set_xlabel('Communication Round', fontsize=FONT_LABEL)
        ax.set_ylabel(metric_titles[metric], fontsize=FONT_LABEL)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(df['Round'].min() - 0.5, df['Round'].max() + 0.5)

        # Place legend outside the plot
        ax.legend(handles, labels,
                  loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=9, frameon=True, fancybox=True, shadow=False)

        plt.tight_layout()
        plt.subplots_adjust(right=0.78)  # Adjust for outside legend
        
        safe_name = f"{metric}_over_rounds_tier1.png"
        plt.savefig(os.path.join(OUTPUT_DIR, safe_name),
                    dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved: {safe_name}")


if __name__ == "__main__":
    main()
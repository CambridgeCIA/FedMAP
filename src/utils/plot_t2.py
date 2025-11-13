import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Generate performance plots for a specific tier."
    )
    
    parser.add_argument(
        '--tier', 
        type=int, 
        required=True, 
        choices=[2, 3], 
        help='The tier to process (2 or 3)'
    )
    
    args = parser.parse_args()
    tier = args.tier


    filename = f'results/example_metrics_test_tier{tier}.csv'
    print(f"Attempting to load data from: {filename}")

   
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Data file not found at: {filename}", file=sys.stderr)
        print("Please make sure the file is in the same directory as the script.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}", file=sys.stderr)
        sys.exit(1)


    if 'Client_ID' in df.columns:
        df['Client_ID'] = df['Client_ID'].astype(str)
    else:
        print("Error: 'Client_ID' column not found in the file.", file=sys.stderr)
        sys.exit(1)

    sns.set_theme(style="whitegrid")
    
    df_sorted_ba = df.sort_values('Balanced_Accuracy', ascending=False)
    
    plt.figure(figsize=(14, 7)) # Set figure size
    
    ba_plot = sns.barplot(
        data=df_sorted_ba,
        x='Client_ID',
        y='Balanced_Accuracy'
    )
    
    plt.title(f'Balanced Accuracy by Client (Tier {tier})', fontsize=16)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Balanced Accuracy', fontsize=12)
    

    plt.ylim(0.5, 1.0) 

    if len(df['Client_ID']) > 10:
        plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()
    
    plot1_filename = f'results/performance_plots/balanced_accuracy_tier{tier}.png'
    plt.savefig(plot1_filename)
    print(f"Saved plot: {plot1_filename}")
    plt.close()
    
    df_sorted_roc = df.sort_values('ROC_AUC', ascending=False)
    
    plt.figure(figsize=(14, 7)) # Set figure size
    
    roc_plot = sns.barplot(
        data=df_sorted_roc,
        x='Client_ID',
        y='ROC_AUC',
        palette='magma' # Use a different color palette
    )
    
    plt.title(f'ROC_AUC by Client (Tier {tier})', fontsize=16)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('ROC_AUC', fontsize=12)
    

    plt.ylim(0.5, 1.0) 
    
    if len(df['Client_ID']) > 10:
        plt.xticks(rotation=45, ha='right')
        
    plt.tight_layout()
    
    # Save the figure
    plot2_filename = f'results/performance_plots/roc_auc_tier{tier}.png'
    plt.savefig(plot2_filename)
    print(f"Saved plot: {plot2_filename}")
    plt.close()

if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import numpy as np

sns.set_style("whitegrid")


class TrainingAnalyzer:
    """Analyzes and visualizes training metrics from YOLO results."""
    
    def __init__(self, results_csv_path: str):
        """
        Initialize analyzer with results CSV.
        
        Args:
            results_csv_path: Path to YOLO results.csv file
        """
        self.results_path = Path(results_csv_path)
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_csv_path}")
        
        self.df = pd.read_csv(results_csv_path)
        self.df.columns = self.df.columns.str.strip()
        
    def plot_loss_curves(self, save_path: Optional[str] = None):
        """Plot all loss components over epochs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')
        
        # Box loss
        axes[0, 0].plot(self.df['epoch'], self.df['train/box_loss'], 
                       label='Train', linewidth=2, color='#2E86AB')
        axes[0, 0].plot(self.df['epoch'], self.df['val/box_loss'], 
                       label='Validation', linewidth=2, color='#A23B72')
        axes[0, 0].set_title('Box Loss (Localization)', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Class loss
        axes[0, 1].plot(self.df['epoch'], self.df['train/cls_loss'], 
                       label='Train', linewidth=2, color='#2E86AB')
        axes[0, 1].plot(self.df['epoch'], self.df['val/cls_loss'], 
                       label='Validation', linewidth=2, color='#A23B72')
        axes[0, 1].set_title('Classification Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # DFL loss
        axes[1, 0].plot(self.df['epoch'], self.df['train/dfl_loss'], 
                       label='Train', linewidth=2, color='#2E86AB')
        axes[1, 0].plot(self.df['epoch'], self.df['val/dfl_loss'], 
                       label='Validation', linewidth=2, color='#A23B72')
        axes[1, 0].set_title('DFL Loss (Distribution Focal)', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined loss comparison
        train_total = (self.df['train/box_loss'] + 
                      self.df['train/cls_loss'] + 
                      self.df['train/dfl_loss'])
        val_total = (self.df['val/box_loss'] + 
                    self.df['val/cls_loss'] + 
                    self.df['val/dfl_loss'])
        
        axes[1, 1].plot(self.df['epoch'], train_total, 
                       label='Train Total', linewidth=2, color='#2E86AB')
        axes[1, 1].plot(self.df['epoch'], val_total, 
                       label='Val Total', linewidth=2, color='#A23B72')
        axes[1, 1].set_title('Total Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Loss curves saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_evolution(self, save_path: Optional[str] = None):
        """Plot precision, recall, mAP metrics over epochs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Metrics Evolution', fontsize=16, fontweight='bold')
        
        # Precision
        axes[0, 0].plot(self.df['epoch'], self.df['metrics/precision(B)'], 
                       linewidth=2, color='#06A77D', marker='o', markersize=3)
        axes[0, 0].set_title('Precision', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[0, 1].plot(self.df['epoch'], self.df['metrics/recall(B)'], 
                       linewidth=2, color='#F77F00', marker='o', markersize=3)
        axes[0, 1].set_title('Recall', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim([0, 1.05])
        axes[0, 1].grid(True, alpha=0.3)
        
        # mAP@50
        axes[1, 0].plot(self.df['epoch'], self.df['metrics/mAP50(B)'], 
                       linewidth=2, color='#D62828', marker='o', markersize=3)
        axes[1, 0].set_title('mAP@50', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP@50')
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].grid(True, alpha=0.3)
        
        # mAP@50-95
        axes[1, 1].plot(self.df['epoch'], self.df['metrics/mAP50-95(B)'], 
                       linewidth=2, color='#003049', marker='o', markersize=3)
        axes[1, 1].set_title('mAP@50-95', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP@50-95')
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics evolution saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_learning_rate(self, save_path: Optional[str] = None):
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all LR groups
        lr_cols = [col for col in self.df.columns if col.startswith('lr/')]
        
        for col in lr_cols:
            label = col.replace('lr/', '').replace('pg', 'param_group_')
            ax.plot(self.df['epoch'], self.df[col], 
                   label=label, linewidth=2, marker='o', markersize=3)
        
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Learning rate plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_overfitting_analysis(self, save_path: Optional[str] = None):
        """Analyze train vs validation gap to detect overfitting."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Overfitting Analysis', fontsize=16, fontweight='bold')
        
        # Loss gap
        box_gap = self.df['val/box_loss'] - self.df['train/box_loss']
        cls_gap = self.df['val/cls_loss'] - self.df['train/cls_loss']
        dfl_gap = self.df['val/dfl_loss'] - self.df['train/dfl_loss']
        
        axes[0].plot(self.df['epoch'], box_gap, label='Box Loss Gap', linewidth=2)
        axes[0].plot(self.df['epoch'], cls_gap, label='Class Loss Gap', linewidth=2)
        axes[0].plot(self.df['epoch'], dfl_gap, label='DFL Loss Gap', linewidth=2)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_title('Train-Val Loss Gap (Overfitting Indicator)', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation - Train Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Moving average of total gap
        total_gap = box_gap + cls_gap + dfl_gap
        window = min(5, len(total_gap) // 3)
        if window > 0:
            ma_gap = pd.Series(total_gap).rolling(window=window).mean()
            axes[1].plot(self.df['epoch'], total_gap, 
                        label='Total Gap', alpha=0.5, linewidth=1)
            axes[1].plot(self.df['epoch'], ma_gap, 
                        label=f'MA({window})', linewidth=2, color='red')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].fill_between(self.df['epoch'], 0, total_gap, 
                                where=(total_gap > 0), alpha=0.3, color='red',
                                label='Overfitting Region')
            axes[1].set_title('Total Loss Gap Trend', fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Total Gap')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Overfitting analysis saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_training_time(self, save_path: Optional[str] = None):
        """Plot training time analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Training Time Analysis', fontsize=16, fontweight='bold')
        
        # Time per epoch
        time_per_epoch = self.df['time'].diff()
        time_per_epoch.iloc[0] = self.df['time'].iloc[0]
        
        axes[0].bar(self.df['epoch'], time_per_epoch, 
                color='#4A90E2', alpha=0.7, edgecolor='black')
        axes[0].axhline(y=time_per_epoch.mean(), 
                    color='red', linestyle='--', linewidth=2,
                    label=f'Average: {time_per_epoch.mean():.2f} sec')
        axes[0].set_title('Time per Epoch', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Cumulative time
        axes[1].plot(self.df['epoch'], self.df['time'] / 60, 
                    linewidth=2, color='#E94B3C', marker='o', markersize=4)
        axes[1].fill_between(self.df['epoch'], 0, self.df['time'] / 60, 
                            alpha=0.3, color='#E94B3C')
        axes[1].set_title('Cumulative Training Time', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Total Time (minutes)')
        axes[1].grid(True, alpha=0.3)
        
        total_time_sec = self.df['time'].iloc[-1]
        total_time_min = total_time_sec / 60
        axes[1].text(0.98, 0.02, f'Total: {total_time_min:.2f} min ({total_time_sec:.1f} sec)',
                    transform=axes[1].transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training time plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_efficiency_metrics(self, save_path: Optional[str] = None):
        """Plot training efficiency: metric improvement vs time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # mAP50 vs Time
        axes[0, 0].plot(self.df['time'] / 60, self.df['metrics/mAP50(B)'], 
                    linewidth=2, color='#D62828', marker='o', markersize=4)
        axes[0, 0].set_title('mAP@50 vs Training Time', fontweight='bold')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('mAP@50')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])
        
        # Precision vs Time
        axes[0, 1].plot(self.df['time'] / 60, self.df['metrics/precision(B)'], 
                    linewidth=2, color='#06A77D', marker='o', markersize=4)
        axes[0, 1].set_title('Precision vs Training Time', fontweight='bold')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])
        
        # Recall vs Time
        axes[1, 0].plot(self.df['time'] / 60, self.df['metrics/recall(B)'], 
                    linewidth=2, color='#F77F00', marker='o', markersize=4)
        axes[1, 0].set_title('Recall vs Training Time', fontweight='bold')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.05])
        
        # Loss reduction rate
        total_loss = (self.df['train/box_loss'] + 
                    self.df['train/cls_loss'] + 
                    self.df['train/dfl_loss'])
        axes[1, 1].plot(self.df['time'] / 60, total_loss, 
                    linewidth=2, color='#003049', marker='o', markersize=4)
        axes[1, 1].set_title('Total Loss vs Training Time', fontweight='bold')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Total Training Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Efficiency metrics saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_summary_report(self) -> dict:
        last_epoch = self.df.iloc[-1]
        best_map50_idx = self.df['metrics/mAP50(B)'].idxmax()
        best_epoch = self.df.iloc[best_map50_idx]
        
        # Calculate time statistics (time is in seconds)
        time_per_epoch = self.df['time'].diff()
        time_per_epoch.iloc[0] = self.df['time'].iloc[0]
        
        summary = {
            'total_epochs': len(self.df),
            'training_time': {
                'total_seconds': float(last_epoch['time']),
                'total_minutes': float(last_epoch['time'] / 60),
                'avg_seconds_per_epoch': float(time_per_epoch.mean()),
                'fastest_epoch_seconds': float(time_per_epoch.min()),
                'slowest_epoch_seconds': float(time_per_epoch.max()),
            },
            'final_metrics': {
                'precision': float(last_epoch['metrics/precision(B)']),
                'recall': float(last_epoch['metrics/recall(B)']),
                'mAP50': float(last_epoch['metrics/mAP50(B)']),
                'mAP50_95': float(last_epoch['metrics/mAP50-95(B)']),
            },
            'best_epoch': {
                'epoch': int(best_epoch['epoch']),
                'mAP50': float(best_epoch['metrics/mAP50(B)']),
                'precision': float(best_epoch['metrics/precision(B)']),
                'recall': float(best_epoch['metrics/recall(B)']),
            },
            'final_losses': {
                'box_loss': float(last_epoch['train/box_loss']),
                'cls_loss': float(last_epoch['train/cls_loss']),
                'dfl_loss': float(last_epoch['train/dfl_loss']),
            }
        }
        
        return summary
    
    def create_all_plots(self, output_dir: str):
        """Generate all analysis plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*50)
        print("  Generating Training Analysis Plots")
        print("="*50)
        
        self.plot_loss_curves(str(output_path / 'loss_curves.png'))
        self.plot_metrics_evolution(str(output_path / 'metrics_evolution.png'))
        self.plot_learning_rate(str(output_path / 'learning_rate.png'))
        self.plot_overfitting_analysis(str(output_path / 'overfitting_analysis.png'))
        self.plot_training_time(str(output_path / 'training_time.png'))
        self.plot_efficiency_metrics(str(output_path / 'efficiency_metrics.png'))
        
        summary = self.generate_summary_report()
        
        print("\n" + "="*50)
        print("  Training Summary")
        print("="*50)
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"\nTraining Time:")
        print(f"  Total: {summary['training_time']['total_minutes']:.2f} min ({summary['training_time']['total_seconds']:.1f} sec)")
        print(f"  Average per epoch: {summary['training_time']['avg_seconds_per_epoch']:.2f} sec")
        print(f"  Range: {summary['training_time']['fastest_epoch_seconds']:.2f} - {summary['training_time']['slowest_epoch_seconds']:.2f} sec")
        print(f"\nFinal Metrics:")
        print(f"  Precision: {summary['final_metrics']['precision']:.4f}")
        print(f"  Recall: {summary['final_metrics']['recall']:.4f}")
        print(f"  mAP@50: {summary['final_metrics']['mAP50']:.4f}")
        print(f"  mAP@50-95: {summary['final_metrics']['mAP50_95']:.4f}")
        print(f"\nBest Epoch: {summary['best_epoch']['epoch']}")
        print(f"  mAP@50: {summary['best_epoch']['mAP50']:.4f}")
        print("="*50 + "\n")
        
        return summary


def analyze_training(results_csv: str, output_dir: str = 'analysis'):
    """
    Convenience function to analyze training results.
    
    Args:
        results_csv: Path to YOLO results.csv
        output_dir: Directory to save plots
    """
    analyzer = TrainingAnalyzer(results_csv)
    return analyzer.create_all_plots(output_dir)
 
 

"""
Performance analysis module for spatial matching and OCR efficiency.
Generates visualizations and metrics to evaluate system performance.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


class PerformanceAnalyzer:
    """Analyzes and visualizes spatial matching and OCR performance."""
    
    def __init__(self, output_dir: str = 'outputs/analysis'):
        """
        Initialize the performance analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def analyze_spatial_matching(
        self,
        results: List[Dict],
        save_prefix: str = 'spatial_matching'
    ) -> Dict:
        """
        Analyze spatial matching efficiency and generate visualizations.
        
        Args:
            results: List of detection results from predict.py
            save_prefix: Prefix for saved plot files
            
        Returns:
            Dictionary with spatial matching metrics
        """
        if not results:
            print("No results to analyze")
            return {}
        
        df = pd.DataFrame(results)
        
        # Calculate metrics
        total_people = len(df)
        matched_plates = df['plate_matched'].sum()
        match_rate = (matched_plates / total_people * 100) if total_people > 0 else 0
        
        # Plates detected vs matched
        plates_with_text = len(df[df['license_plate'].notna() & 
                                   (df['license_plate'] != 'NO_PLATE_MATCHED')])
        
        # Violations with plates
        violations = df[df['has_helmet'] == False]
        violations_with_plates = violations['plate_matched'].sum()
        violation_match_rate = (violations_with_plates / len(violations) * 100) if len(violations) > 0 else 0
        
        metrics = {
            'total_people': total_people,
            'matched_plates': int(matched_plates),
            'match_rate': round(match_rate, 2),
            'plates_with_text': plates_with_text,
            'violations': len(violations),
            'violations_with_plates': int(violations_with_plates),
            'violation_match_rate': round(violation_match_rate, 2)
        }
        
        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Spatial Matching Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Match Rate Overview
        ax1 = axes[0, 0]
        categories = ['Total People', 'Matched Plates', 'Unmatched']
        values = [total_people, matched_plates, total_people - matched_plates]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'Plate Matching Overview\nMatch Rate: {match_rate:.1f}%', fontsize=13, fontweight='bold')
        for i, v in enumerate(values):
            ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. Match Rate by Helmet Status
        ax2 = axes[0, 1]
        helmet_match = df.groupby('has_helmet')['plate_matched'].agg(['sum', 'count'])
        helmet_match['rate'] = (helmet_match['sum'] / helmet_match['count'] * 100)
        
        labels = ['With Helmet', 'Without Helmet (Violation)']
        match_rates = [
            helmet_match.loc[True, 'rate'] if True in helmet_match.index else 0,
            helmet_match.loc[False, 'rate'] if False in helmet_match.index else 0
        ]
        colors_helmet = ['#2ecc71', '#e74c3c']
        bars = ax2.bar(labels, match_rates, color=colors_helmet, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Match Rate (%)', fontsize=12)
        ax2.set_title('Plate Match Rate by Helmet Status', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 110)
        for bar, rate in zip(bars, match_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Detection Confidence Distribution
        ax3 = axes[1, 0]
        matched_df = df[df['plate_matched'] == True]
        unmatched_df = df[df['plate_matched'] == False]
        
        if len(matched_df) > 0:
            ax3.hist(matched_df['detection_confidence'], bins=15, alpha=0.6, 
                    label='Matched', color='#2ecc71', edgecolor='black')
        if len(unmatched_df) > 0:
            ax3.hist(unmatched_df['detection_confidence'], bins=15, alpha=0.6,
                    label='Unmatched', color='#e74c3c', edgecolor='black')
        
        ax3.set_xlabel('Detection Confidence', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Person Detection Confidence Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total People Detected', f"{total_people}"],
            ['Plates Matched', f"{matched_plates} ({match_rate:.1f}%)"],
            ['Plates with Text', f"{plates_with_text}"],
            ['', ''],
            ['Violations (No Helmet)', f"{len(violations)}"],
            ['Violations with Plate', f"{violations_with_plates} ({violation_match_rate:.1f}%)"],
            ['', ''],
            ['Avg Detection Confidence', f"{df['detection_confidence'].mean():.3f}"]
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(len(summary_data)):
            if i == 0:
                table[(i, 0)].set_facecolor('#3498db')
                table[(i, 1)].set_facecolor('#3498db')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            elif summary_data[i][0] == '':
                table[(i, 0)].set_facecolor('#ecf0f1')
                table[(i, 1)].set_facecolor('#ecf0f1')
            else:
                table[(i, 0)].set_facecolor('#f8f9fa')
                table[(i, 1)].set_facecolor('#ffffff')
        
        ax4.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{save_prefix}_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSpatial matching analysis saved: {output_path}")
        plt.close()
        
        return metrics
    
    def analyze_ocr_performance(
        self,
        results: List[Dict],
        save_prefix: str = 'ocr_performance'
    ) -> Dict:
        """
        Analyze OCR performance and generate visualizations.
        
        Args:
            results: List of detection results from predict.py
            save_prefix: Prefix for saved plot files
            
        Returns:
            Dictionary with OCR performance metrics
        """
        if not results:
            print("No results to analyze")
            return {}
        
        df = pd.DataFrame(results)
        
        matched_df = df[df['plate_matched'] == True].copy()
        
        if len(matched_df) == 0:
            print("No matched plates to analyze OCR performance")
            return {}
        
        # Calculate OCR metrics
        total_matched = len(matched_df)
        
        successful_reads = len(matched_df[
            (matched_df['license_plate'] != 'NO_TEXT_DETECTED') &
            (matched_df['license_plate'] != 'PLATE_TOO_SMALL') &
            (matched_df['license_plate'] != 'INVALID_COORDINATES')
        ])
        
        ocr_success_rate = (successful_reads / total_matched * 100) if total_matched > 0 else 0
        
        avg_confidence = matched_df['plate_confidence'].mean()
        
        metrics = {
            'total_matched_plates': total_matched,
            'successful_ocr_reads': successful_reads,
            'ocr_success_rate': round(ocr_success_rate, 2),
            'avg_plate_confidence': round(avg_confidence, 3),
            'failed_reads': total_matched - successful_reads
        }
        
        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('OCR Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. OCR Success Rate
        ax1 = axes[0, 0]
        categories = ['Successful Reads', 'Failed Reads']
        values = [successful_reads, total_matched - successful_reads]
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors,
                                            autopct='%1.1f%%', startangle=90,
                                            explode=(0.05, 0))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        ax1.set_title(f'OCR Success Rate\n{ocr_success_rate:.1f}% Success', 
                     fontsize=13, fontweight='bold')
        
        # 2. Confidence Distribution
        ax2 = axes[0, 1]
        successful_df = matched_df[
            (matched_df['license_plate'] != 'NO_TEXT_DETECTED') &
            (matched_df['license_plate'] != 'PLATE_TOO_SMALL')
        ]
        
        if len(successful_df) > 0:
            ax2.hist(successful_df['plate_confidence'], bins=20, color='#3498db',
                    alpha=0.7, edgecolor='black')
            ax2.axvline(avg_confidence, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {avg_confidence:.3f}')
            ax2.set_xlabel('OCR Confidence', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('OCR Confidence Distribution', fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Confidence by Result Type
        ax3 = axes[1, 0]
        
        result_types = []
        confidences = []
        
        for _, row in matched_df.iterrows():
            plate_text = row['license_plate']
            if plate_text == 'NO_TEXT_DETECTED':
                result_types.append('No Text')
            elif plate_text == 'PLATE_TOO_SMALL':
                result_types.append('Too Small')
            elif plate_text == 'INVALID_COORDINATES':
                result_types.append('Invalid')
            else:
                result_types.append('Success')
            confidences.append(row['plate_confidence'])
        
        result_df = pd.DataFrame({'Type': result_types, 'Confidence': confidences})
        
        if len(result_df) > 0:
            sns.boxplot(data=result_df, x='Type', y='Confidence', ax=ax3, palette='Set2')
            ax3.set_xlabel('Result Type', fontsize=12)
            ax3.set_ylabel('OCR Confidence', fontsize=12)
            ax3.set_title('Confidence by OCR Result Type', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Matched Plates', f"{total_matched}"],
            ['Successful OCR Reads', f"{successful_reads}"],
            ['Failed Reads', f"{total_matched - successful_reads}"],
            ['', ''],
            ['OCR Success Rate', f"{ocr_success_rate:.1f}%"],
            ['Avg Confidence (All)', f"{avg_confidence:.3f}"],
            ['Avg Confidence (Success)', f"{successful_df['plate_confidence'].mean():.3f}" if len(successful_df) > 0 else "N/A"],
            ['', ''],
            ['Min Confidence', f"{matched_df['plate_confidence'].min():.3f}"],
            ['Max Confidence', f"{matched_df['plate_confidence'].max():.3f}"]
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)
        
        for i in range(len(summary_data)):
            if i == 0:
                table[(i, 0)].set_facecolor('#3498db')
                table[(i, 1)].set_facecolor('#3498db')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            elif summary_data[i][0] == '':
                table[(i, 0)].set_facecolor('#ecf0f1')
                table[(i, 1)].set_facecolor('#ecf0f1')
            else:
                table[(i, 0)].set_facecolor('#f8f9fa')
                table[(i, 1)].set_facecolor('#ffffff')
        
        ax4.set_title('OCR Statistics', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{save_prefix}_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"OCR performance analysis saved: {output_path}")
        plt.close()
        
        return metrics
    
    def generate_combined_report(
        self,
        results: List[Dict],
        save_name: str = 'performance_report'
    ) -> Dict:
        """
        Generate a combined performance report with both analyses.
        
        Args:
            results: List of detection results
            save_name: Name for the report file
            
        Returns:
            Dictionary with all metrics
        """
        print("\n" + "="*70)
        print("  PERFORMANCE ANALYSIS REPORT")
        print("="*70)
        
        spatial_metrics = self.analyze_spatial_matching(results, save_name + '_spatial')
        ocr_metrics = self.analyze_ocr_performance(results, save_name + '_ocr')
        
        combined_metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(set([r['image_file'] for r in results])),
            'spatial_matching': spatial_metrics,
            'ocr_performance': ocr_metrics
        }
        
        report_path = os.path.join(self.output_dir, f'{save_name}_metrics.json')
        with open(report_path, 'w') as f:
            json.dump(combined_metrics, f, indent=2)
        
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print("="*70)
        print(f"\nSpatial Matching:")
        print(f"  - Match Rate: {spatial_metrics.get('match_rate', 0):.1f}%")
        print(f"  - Violations with Plates: {spatial_metrics.get('violations_with_plates', 0)}/{spatial_metrics.get('violations', 0)}")
        
        print(f"\nOCR Performance:")
        print(f"  - Success Rate: {ocr_metrics.get('ocr_success_rate', 0):.1f}%")
        print(f"  - Avg Confidence: {ocr_metrics.get('avg_plate_confidence', 0):.3f}")
        
        print(f"\nMetrics saved: {report_path}")
        print("="*70 + "\n")
        
        return combined_metrics


def main():
    """CLI for performance analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze spatial matching and OCR performance'
    )
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to CSV or JSON results file from predict.py'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/analysis',
        help='Output directory for analysis plots'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='performance',
        help='Name prefix for output files'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return
    
    if args.results.endswith('.csv'):
        df = pd.read_csv(args.results)
        results = df.to_dict('records')
    elif args.results.endswith('.json'):
        with open(args.results, 'r') as f:
            results = json.load(f)
    else:
        print("Error: Results file must be CSV or JSON")
        return
    
    analyzer = PerformanceAnalyzer(output_dir=args.output)
    analyzer.generate_combined_report(results, save_name=args.name)


if __name__ == '__main__':
    main()

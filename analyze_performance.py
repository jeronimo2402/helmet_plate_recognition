"""
CLI script to analyze spatial matching and OCR performance.
Generates visualizations and metrics from prediction results.

Usage:
    python analyze_performance.py --results outputs/reports/results.csv
    python analyze_performance.py --results outputs/reports/results.json --output outputs/custom_analysis
"""

import argparse
import os
import sys
import pandas as pd
import json
from src.utils import PerformanceAnalyzer


def main():
    """Main function for performance analysis CLI."""
    parser = argparse.ArgumentParser(
        description='Analyze spatial matching and OCR performance from prediction results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Analyze results from CSV file
            python analyze_performance.py --results outputs/reports/results.csv
            
            # Analyze with custom output directory
            python analyze_performance.py --results outputs/reports/results.json --output my_analysis
            
            # Specify custom name for output files
            python analyze_performance.py --results outputs/reports/results.csv --name test_run_1
        """
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
        help='Output directory for analysis plots (default: outputs/analysis)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='performance',
        help='Name prefix for output files (default: performance)'
    )
    parser.add_argument(
        '--spatial-only',
        action='store_true',
        help='Only generate spatial matching analysis'
    )
    parser.add_argument(
        '--ocr-only',
        action='store_true',
        help='Only generate OCR performance analysis'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)
    
    # Load results
    print(f"\nLoading results from: {args.results}")
    
    try:
        if args.results.endswith('.csv'):
            df = pd.read_csv(args.results)
            results = df.to_dict('records')
            print(f"Loaded {len(results)} records from CSV")
        elif args.results.endswith('.json'):
            with open(args.results, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} records from JSON")
        else:
            print("Error: Results file must be CSV or JSON format")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    if not results:
        print("Error: No results found in file")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(output_dir=args.output)
    
    # Generate analysis
    if args.spatial_only:
        print("\nGenerating spatial matching analysis only...")
        metrics = analyzer.analyze_spatial_matching(results, save_prefix=args.name)
    elif args.ocr_only:
        print("\nGenerating OCR performance analysis only...")
        metrics = analyzer.analyze_ocr_performance(results, save_prefix=args.name)
    else:
        print("\nGenerating complete performance analysis...")
        metrics = analyzer.generate_combined_report(results, save_name=args.name)
    
    print("\nAnalysis complete!")
    print(f"Output directory: {args.output}")


if __name__ == '__main__':
    main()

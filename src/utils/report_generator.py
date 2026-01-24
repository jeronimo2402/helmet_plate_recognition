import pandas as pd
from datetime import datetime
from typing import List, Dict
import os


class ReportGenerator:
    
    def __init__(self, output_folder_path: str = 'outputs/reports'):
        self.report_output_folder = output_folder_path
        os.makedirs(self.report_output_folder, exist_ok=True)
    
    def create_report(
        self, 
        list_of_detections: List[Dict], 
        file_format: str = 'csv'
    ) -> str:
        
        data_table = pd.DataFrame(list_of_detections)
        
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if file_format == 'csv':
            report_filename = f'{self.report_output_folder}/report_{current_timestamp}.csv'
            data_table.to_csv(report_filename, index=False)
        
        elif file_format == 'json':
            report_filename = f'{self.report_output_folder}/report_{current_timestamp}.json'
            data_table.to_json(report_filename, orient='records', indent=2)
        
        elif file_format == 'excel':
            report_filename = f'{self.report_output_folder}/report_{current_timestamp}.xlsx'
            data_table.to_excel(report_filename, index=False, engine='openpyxl')
        
        else:
            raise ValueError(f"Unsupported format: {file_format}. Use 'csv', 'json', or 'excel'")
        
        return report_filename
    
    def print_summary(self, list_of_detections: List[Dict]) -> None:
        
        total_people_detected = len(list_of_detections)
        people_with_helmets = sum(1 for d in list_of_detections if d.get('has_helmet', False))
        people_without_helmets = total_people_detected - people_with_helmets
        plates_detected = sum(1 for d in list_of_detections if d.get('license_plate') != 'NO_PLATE_DETECTED')
        
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total people detected: {total_people_detected}")
        print(f"People WITH helmets: {people_with_helmets}")
        print(f"People WITHOUT helmets: {people_without_helmets}")
        print(f"License plates detected: {plates_detected}")
        print("="*50 + "\n")

# combine_csv_data.py
# Combines multiple months of CSV data into one file, adding DayOfWeek and Month features.
# Removes Volume column and ensures output date format is '%m/%d/%y' (e.g., 07/25/25).
# Usage: python combine_csv_data.py
# For BESS + Solar Park: Add solar_kwh column if input CSV includes PV data.

import csv
from datetime import datetime
import io

# --- Configuration ---
INPUT_FILENAME = '/Users/chavdarbilyanski/energyscale/src/ml/data/combine/combined_output.csv'
OUTPUT_FILENAME = '/Users/chavdarbilyanski/energyscale/src/ml/data/combine/combined_output_with_features.csv'
# ---------------------

try:
    with open(INPUT_FILENAME, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILENAME, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile, delimiter=';')
        writer = csv.writer(outfile, delimiter=';')

        # 1. Read the header row
        header = next(reader)
        
        # Find Volume index if present (assumed before Price (EUR))
        volume_idx = None
        if 'Volume' in header:
            volume_idx = header.index('Volume')
        
        # Remove Volume from header if found
        if volume_idx is not None:
            header.pop(volume_idx)
        
        # Insert DayOfWeek and Month after Date (index 1 and 2)
        header.insert(1, 'DayOfWeek')
        header.insert(2, 'Month')
        
        # Write updated header
        writer.writerow(header)

        # 2. Process each data row
        for row in reader:
            if not row:  # Skip empty rows
                continue

            date_str = row[0]

            try:
                # 3. Parse the date string (handle both %m/%d/%y and %m/%d/%Y for robustness)
                try:
                    date_obj = datetime.strptime(date_str, '%m/%d/%y')
                except ValueError:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')  # Fallback for four-digit year
                
                # 4. Get the day of the week (Monday=0, Sunday=6) and month
                day_of_week = date_obj.weekday()
                month_of_year = date_obj.month
                
                # 5. Reformat date to '%m/%d/%y' for output consistency
                formatted_date = date_obj.strftime('%m/%d/%y')  # e.g., 01/01/25
                
                # 6. Update row: reformat date, insert new features
                row[0] = formatted_date
                row.insert(1, day_of_week)
                row.insert(2, month_of_year)
                
                # Remove Volume data if index was found (adjust for insertions)
                if volume_idx is not None:
                    # After insertions, original Volume index shifts +2 (after Date + DayOfWeek + Month)
                    adjusted_volume_idx = volume_idx + 2
                    row.pop(adjusted_volume_idx)
                
                writer.writerow(row)

            except ValueError:
                # Handle potential errors in date format for a specific row
                print(f"Warning: Skipping row due to date parse error: {row}")

    print(f"Successfully processed the file.")
    print(f"Output saved to: {OUTPUT_FILENAME}")

except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
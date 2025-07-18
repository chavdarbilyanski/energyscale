import csv
from datetime import datetime
import io

# --- Configuration ---
INPUT_FILENAME = '/Users/chavdarbilyanski/powerbidder/src/ml/data/combine/JanFeb2025.csv'
OUTPUT_FILENAME = '/Users/chavdarbilyanski/powerbidder/src/ml/data/combine/JanFeb2025_output_with_features.csv'
# ---------------------


try:
    with open(INPUT_FILENAME, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILENAME, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile, delimiter=';')
        writer = csv.writer(outfile, delimiter=';')

        # 1. Read and write the header row with new column names
        header = next(reader)
        header.insert(1, 'DayOfWeek')
        header.insert(2, 'Month')
        # header.extend(['DayOfWeek', 'Month'])
        writer.writerow(header)

        # 2. Process each data row
        for row in reader:
            if not row:  # Skip empty rows
                continue

            date_str = row[0]

            try:
                # 3. Parse the date string (assuming d/m/YY format)
                date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                
                # 4. Get the day of the week (Monday=0, Sunday=6) and month
                day_of_week = date_obj.weekday()
                month_of_year = date_obj.month
                
                # 5. Insert the new data into the row at the correct positions
                row.insert(1, day_of_week)
                row.insert(2, month_of_year)

                writer.writerow(row)

            except ValueError:
                # This handles potential errors in the date format for a specific row
                print(f"Warning: Skipping row due to date parse error: {row}")

    print(f"Successfully processed the file.")
    print(f"Output saved to: {OUTPUT_FILENAME}")

except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
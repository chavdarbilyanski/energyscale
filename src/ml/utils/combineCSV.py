import pandas as pd
import glob
import os

#Column names
date_name = 'Date'
hour_name = 'Hour'
volumne_name = 'Volume'
price_name = 'Price (EUR)'

def combine_csv_files(input_path, output_file):
    """
    Combines all CSV files in the specified directory into a single CSV file.
    
    Parameters:
    input_path (str): Directory containing CSV files
    output_file (str): Path for the output combined CSV file
    """
    try:
        # Get all CSV files in the specified directory
        all_files = glob.glob(os.path.join(input_path, "*.csv"))
        
        if not all_files:
            print("No CSV files found in the specified directory!")
            return
        
        # Create an empty list to store DataFrames
        df_list = []
        
        # Read each CSV file and append to the list
        for file in all_files:
            df = pd.read_csv(file, header=None, sep = ';', names=[date_name, hour_name, volumne_name, price_name])
            df_list.append(df)
            print(f"Processed: {file}")

        # Combine all DataFrames
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv(output_file, sep=';', index=False)
        print(f"\nSuccessfully combined {len(all_files)} CSV files into {output_file}")
        print(f"Total rows in combined file: {len(combined_df)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Specify your input directory and output file path
    input_directory = "/Users/chavdarbilyanski/powerbidder/combine"  # Replace with your directory path
    output_file = "/Users/chavdarbilyanski/powerbidder/combine/combined_output.csv"  # Replace with your output file path
    
    combine_csv_files(input_directory, output_file)
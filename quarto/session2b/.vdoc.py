# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

import os
import pandas as pd
from great_tables import GT

# Load the dataset
cancer_data = pd.read_csv(os.path.join('example_data', 'Cancer_Data.csv'))

# Display the first few rows of the dataset
cancer_data.head()


#
#
#
#
#
#
#
#

# Display the shape of the dataset
print("Dataset Shape:", cancer_data.shape)




#
#
#
#
#
#
#

# Display column names, data types, and non-null counts
cancer_data.info()



#
#
#
#
#
#
#

# Display column names
print("Column Names:", cancer_data.columns.tolist())



#
#
#
#
#
#
#
# Generate summary statistics for numeric columns
cancer_data.describe()

#
#
#
#
#
#
#

# Count occurrences of each unique value in the 'diagnosis' column
diagnosis_counts = cancer_data['diagnosis'].value_counts()
print("Diagnosis Counts:\n", diagnosis_counts)



#
#
#
#
#

# Group by 'diagnosis' and get summary statistics for each group
grouped_summary = cancer_data.groupby('diagnosis').mean()
print(grouped_summary)


#Group by 'diagnosis' and get summary statistics for only one variable
grouped_radius_mean = cancer_data.groupby('diagnosis')['radius_mean'].mean()
print(grouped_radius_mean)


#
#
#
#
#
#

# Rename specific columns for readability

new_columns={
    'radius_mean': 'Radius Mean',
    'texture_mean': 'Texture Mean',
    'perimeter_mean': 'Perimeter Mean'
}

cancer_data = cancer_data.rename(columns=new_columns)

# Display the new column names to verify the changes
print("\nUpdated Column Names:", cancer_data.columns.tolist())


#
#
#
#
#
# Count missing values in each column
missing_values = cancer_data.isnull().sum()
print("Missing Values per Column:")
print(missing_values)




#
#
#
#
#
#
# Drop the 'Unnamed: 32' column if it contains no data
cancer_data = cancer_data.drop(columns=['Unnamed: 32'])

# Verify the column has been dropped
print("\nColumns after dropping 'Unnamed: 32':", cancer_data.columns.tolist())




#
#
#
#
#
#

# Select the 'diagnosis' column - diagnosis_column will be a series
diagnosis_column = cancer_data['diagnosis']
print("Diagnosis Column:\n", diagnosis_column.head())


#
#
#
#
#
# Select multiple columns: 'diagnosis', 'radius_mean', and 'area_mean' - selected_columns will be a pandas DataFrame

selected_columns = cancer_data[['diagnosis', 'Radius Mean', 'area_mean']]
print("Selected Columns:\n", selected_columns.head())



#
#
#
#
#
#
#
#
#

# Select rows by labels (assuming integer index here) and specific columns
selected_rows_labels = cancer_data.loc[0:4, ['diagnosis', 'Radius Mean', 'area_mean']]
print("Selected Rows with loc:\n", selected_rows_labels)



#
#
#
#
#
#

# Select rows by integer position and specific columns
selected_rows_position = cancer_data.iloc[0:5, [1, 2, 3]]  # Select first 5 rows and columns at position 1, 2, 3
print("Selected Rows with iloc:\n", selected_rows_position)




#
#
#
#
#

# Filter rows where 'diagnosis' is "M" (Malignant)
malignant_cases = cancer_data[cancer_data['diagnosis'] == 'M']
print("Malignant Cases:\n", malignant_cases.head(20))




#
#
#
#
#
#
#

# Filter for Malignant cases with radius_mean > 15
large_malignant_cases = cancer_data[(cancer_data['diagnosis'] == 'M') & (cancer_data['Radius Mean'] > 15)]
print("Large Malignant Cases (Radius Mean > 15):\n", large_malignant_cases.head())



#
#
#
#
#
#
#
#

# Add a new column 'area_ratio' by dividing 'area_worst' by 'area_mean'
cancer_data['area_ratio'] = cancer_data['area_worst'] / cancer_data['area_mean']
print("New Column 'area_ratio':\n", cancer_data[['area_worst', 'area_mean', 'area_ratio']].head())




#
#
#
#
#
#

# Access and print the original value at index 0 and column 'radius_mean'
original_value = cancer_data.at[0, 'Radius Mean']
print("Original Radius Mean at index 0:", original_value)


# Change the value at index 0 and column 'radius_mean' to 18.5
cancer_data.at[0, 'Radius Mean'] = 18.5


# Verify the updated value
updated_value = cancer_data.at[0, 'Radius Mean']
print("Updated Radius Mean at index 0:", updated_value)


#
#
#
#
#
#

# Sort by 'diagnosis' first, then by 'area_mean' within each diagnosis group
sorted_by_diagnosis_area = cancer_data.sort_values(by=['diagnosis', 'area_mean'], ascending=[True, True])
print("Data sorted by Diagnosis and Area Mean:\n", sorted_by_diagnosis_area[['diagnosis', 'area_mean', 'Radius Mean']].head())



#
#
#
#
#
#

# Move 'area_ratio' to the end of the DataFrame
columns_reordered = [col for col in cancer_data.columns if col != 'area_ratio'] + ['area_ratio']
cancer_data_with_area_ratio_last = cancer_data[columns_reordered]

# Display the reordered columns
print("Data with 'area_ratio' at the end:\n", cancer_data_with_area_ratio_last.head())




#
#
#
#
#
#
#
#
#
#
#
#

# Define a custom function to categorize tumors by area_mean
def categorize_tumor(size):
    if size < 500:
        return 'Small'
    elif 500 <= size < 1000:
        return 'Medium'
    else:
        return 'Large'

# Apply the function to the 'area_mean' column and create a new column 'tumor_size_category'
cancer_data['tumor_size_category'] = cancer_data['area_mean'].apply(categorize_tumor)

# Display the new column to verify the transformation
print("Tumor Size Categories:\n", cancer_data[['area_mean', 'tumor_size_category']].head())



#
#
#
#
#
#
# Apply a lambda function to classify 'diagnosis' into numerical codes
cancer_data['diagnosis_code'] = cancer_data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Display the new column to verify the transformation
print("Diagnosis Codes:\n", cancer_data[['diagnosis', 'diagnosis_code']].head())


#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Apply a lambda function with multiple conditions to create a 'risk_level' column
cancer_data['risk_level'] = cancer_data.apply(
    lambda row: 'High Risk' if row['diagnosis'] == 'M' and row['area_mean'] > 1000 
    else ('Moderate Risk' if row['diagnosis'] == 'M' else 'Low Risk'), axis=1
)

# Display the new column to verify the transformation
print("Risk Levels:\n", cancer_data[['diagnosis', 'area_mean', 'risk_level']].head())

#Axis=1 tells the function to apply it to the rows. axis=0 (default) applies function to the columns


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from great_tables import GT
import numpy as np

from great_tables import style, loc, html, md

# Get counts for categorical columns

grouped_summary = (
    cancer_data.select_dtypes(include=["object", "string"])
    .groupby("diagnosis")
    .value_counts()
)
print(grouped_summary)

## because the return object from the .value_counts() method is a pandas Series, we need to use .reset_index() to make it into a pandas dataframe before using GT to make a table. 

GT(grouped_summary.reset_index())

#
#
#
#
#
# Generate summary statistics for numeric columns
# Select only numeric columns for aggregation
numeric_columns = cancer_data.select_dtypes(include="number")


# Group by "diagnosis" and calculate mean and std for numeric columns
# Custom aggregation function to combine mean and std
def mean_sd(series):
    mean = series.mean()
    std = series.std()
    return f"{mean:.2f} ({std:.2f})"


# Apply the custom function to each numeric column
grouped_data = numeric_columns.groupby(cancer_data["diagnosis"]).agg(mean_sd)
## UPDATE: if we want to make this into a nice looking table, we can use the function GT from great tables package
mapping_dict = {
    col: col.replace("_mean", "").replace("_se", "").replace("_worst", "")
    for col in cancer_data.columns
    if any(suffix in col for suffix in ["_mean", "_se", "_worst"])
}


(
    GT(
        cancer_data.drop(columns=["id"]).describe().reset_index(),
        rowname_col="index",
    )
    .fmt_number(
        columns=cancer_data.describe().columns.tolist(),
        decimals=2,
        drop_trailing_zeros=True,
    )
    .tab_header(title="Cancer Data Summary Statistics")
    .tab_stubhead(label="Stat")
    .tab_spanner(
        label="Mean",
        columns=[col for col in cancer_data.columns if col.endswith("mean")],
    )
    .tab_spanner(
        label="Standard Error",
        columns=[col for col in cancer_data.columns if col.endswith("se")],
    )
    .tab_spanner(
        label="Worst",
        columns=[col for col in cancer_data.columns if col.endswith("worst")],
    )
    .tab_style(
        style=style.text(color="#84a1f0"),
        locations=loc.spanner_labels(ids=["Mean", "Standard Error", "Worst"]),
    )
    .cols_label(cases=mapping_dict)
    .opt_row_striping()
    .opt_all_caps(locations=["stub", "column_labels"])
)


#
#
#
#

# Export to CSV
'''

df.to_csv('/path/to/directory/example.csv', index=False)  # index=False excludes the row indices


'''
#Export to xlsx

'''

df.to_excel('/path/to/directory/example.xlsx', index=False)

'''

#
#
#

import pandas as pd

# Importing CSV file
csv_file = pd.read_csv('NaturalScience_CALIFORNIAONLY.csv') 

# Data that has been filtered out for only California, should be around ~83k
filtered = csv_file[csv_file['STATE'] == 'CA'] 

#Replacing the Julian time to Gregorian time
filtered['DISCOVERY_DATE'] = pd.to_datetime(filtered['DISCOVERY_DATE'], origin="julian", unit='D')


# Filtering out relevant data
selected_columns = filtered[['SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME', 'FIRE_CODE', 
                               'FIRE_NAME', 'MTBS_ID', 'MTBS_FIRE_NAME', 'FIRE_YEAR', 'DISCOVERY_DATE', 
                               'DISCOVERY_DOY', 'DISCOVERY_TIME', 'CONT_DATE', 'CONT_DOY', 'CONT_TIME', 
                               'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE', 'LONGITUDE', 'COUNTY', 'STATE']]
selected_columns2 = filtered[['DISCOVERY_DATE','LATITUDE', 'LONGITUDE', 'FIRE_YEAR']]

selected_columns.to_csv('parsed_data.csv', index=False)
selected_columns2.to_csv('edmund_data2.csv', index=False)
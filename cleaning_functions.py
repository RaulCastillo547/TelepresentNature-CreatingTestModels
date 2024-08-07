import datetime as dt

import numpy as np
import pandas as pd

# Generates clean csv files for each of the deer; interpolation is used only on NULL values
def basic_clean_deer(url_dest='CSVFiles/CleanCSV/Deer/'):    
    # Get Deer Data
    raw_deer_data = pd.read_csv(r'CSVFiles/RawCSV/EuroDeer_ Roe deer in Italy 2005-2008.csv')
    raw_deer_data['timestamp'] = pd.to_datetime(raw_deer_data['timestamp'])
    raw_deer_data.set_index('timestamp', inplace=True)

    # Drop empty columns and rows without latitude, longitude, or altitude
    raw_deer_data.drop(columns=['comments', 'manually-marked-outlier', 'study-name', 'individual-taxon-canonical-name', 'sensor-type', 'individual-local-identifier', 'event-id', 'visible', 'tag-tech-spec', 'tag-voltage'], inplace=True)

    # Rename columns
    raw_deer_data.rename({'location-long': 'longitude', 'location-lat': 'latitude', 'height-above-ellipsoid': 'altitude'}, axis=1, inplace=True)

    # Generate csv file for each deer
    for deer_tag in list(raw_deer_data['tag-local-identifier'].unique()):
        one_deer_data = raw_deer_data[raw_deer_data['tag-local-identifier'] == deer_tag]
        one_deer_data.loc[:, ['longitude', 'latitude', 'altitude']] = one_deer_data[['longitude', 'latitude', 'altitude']].interpolate('time', limit_direction='both')
        one_deer_data.drop(columns=['tag-local-identifier'], inplace=True)
        one_deer_data[['external-temperature', 'longitude', 'latitude', 'altitude']].to_csv(f"{url_dest}{deer_tag}.csv", index=True)

# Generates clean csv files for each of the deer; interpolation is used extensively to create consistent 4-hour rows
def interpolation_clean_deer(url_dest='CSVFiles/CleanCSV/Deer/'):    
    # Get Deer Data
    raw_deer_data = pd.read_csv(r'CSVFiles/RawCSV/EuroDeer_ Roe deer in Italy 2005-2008.csv')
    raw_deer_data['timestamp'] = pd.to_datetime(raw_deer_data['timestamp'])
    raw_deer_data.set_index('timestamp', inplace=True)

    # Drop unneeded columns
    raw_deer_data.drop(columns=['comments', 'manually-marked-outlier', 'study-name', 'individual-taxon-canonical-name', 'sensor-type', 'individual-local-identifier', 'event-id', 'visible', 'tag-tech-spec', 'tag-voltage'], inplace=True)

    # Rename columns
    raw_deer_data.rename({'location-long': 'longitude', 'location-lat': 'latitude', 'height-above-ellipsoid': 'altitude'}, axis=1, inplace=True)

    # Generate csv file for each deer
    for deer_tag in list(raw_deer_data['tag-local-identifier'].unique()):
        # Separate out one deer    
        one_deer_data = raw_deer_data[raw_deer_data['tag-local-identifier'] == deer_tag].copy(deep=True)
        one_deer_data.drop(columns=['tag-local-identifier'], inplace=True)
        one_deer_data.loc[:, 'to_keep'] = False

        # Create a pd.Series from day 1 to last day + 4 hours with 4 hour intervals
        start = min(one_deer_data.index)
        end = max(one_deer_data.index)

        start = dt.datetime(year=start.year, month=start.month, day=start.day)
        end = dt.datetime(year=end.year, month=end.month, day=end.day) + dt.timedelta(days=1)

        new_index = pd.DatetimeIndex(np.arange(start, end, dt.timedelta(hours=4))).drop(labels=one_deer_data.index, errors='ignore')

        # Create New DataFrame
        add_on = pd.DataFrame(index=new_index, columns=one_deer_data.columns)
        add_on.loc[:, 'to_keep'] = True

        one_deer_data = pd.concat(objs=[one_deer_data, add_on])
        
        one_deer_data.sort_index(inplace=True)

        one_deer_data.infer_objects(copy=False)
        one_deer_data.interpolate('time', limit_direction='both', inplace=True)

        # Create DataFrame
        one_deer_data = one_deer_data[one_deer_data['to_keep'] == True]
        one_deer_data.drop(columns=['to_keep'], inplace=True)
        one_deer_data[['external-temperature', 'longitude', 'latitude', 'altitude']].to_csv(f"{url_dest}{deer_tag}_interpolated.csv", index=True, index_label='timestamp')

# Generates clean csv files for each of the moose; interpolation is used only on NULL values
def basic_clean_moose(url_dest='CSVFiles/CleanCSV/Moose/'):
    # Retrieve Data
    raw_moose_data = pd.read_csv(r'CSVFiles/RawCSV/ABoVE_ Boutin Alberta Moose.csv')
    raw_moose_data['timestamp'] = pd.DatetimeIndex(raw_moose_data['timestamp'])

    # Remove meta-data and individual string data
    raw_moose_data.drop(columns=['event-id', 'visible', 'sensor-type', 'individual-taxon-canonical-name', 'individual-local-identifier', 'study-name', 'gps:dop', 'gps:fix-type', 'tag-voltage'], inplace=True)

    raw_moose_data.rename({'location-long': 'longitude', 'location-lat': 'latitude', 'height-above-ellipsoid': 'altitude'}, axis=1, inplace=True)

    # Generate CSV file for each moose with tracking data that spans greater than 900 days
    for moose_tag in raw_moose_data['tag-local-identifier'].unique():
        # Get the moose
        specific_moose = raw_moose_data[raw_moose_data['tag-local-identifier'] == moose_tag].set_index('timestamp')
        specific_moose.drop(columns=['tag-local-identifier'], inplace=True)

        start = min(specific_moose.index)
        end = max(specific_moose.index)

        # Because there are many mooses, we can select for the mooses with the longest timelines (i.e., greater than 900 days)
        if ((end - start) < dt.timedelta(days=900)):
            continue

        # Fill in any missing gaps with interpolation and create CSV
        specific_moose.interpolate('time', limit_direction = 'both', inplace=True)
        specific_moose[['external-temperature', 'longitude', 'latitude', 'altitude']].to_csv(f'{url_dest}{moose_tag}.csv', index=True, index_label='timestamp')

# Generates clean csv files for each of the moose; interpolation is used extensively to create consistent 3-hour rows
def interpolation_clean_moose(url_dest='CSVFiles/CleanCSV/Moose/'):
    # Retrieve Data
    raw_moose_data = pd.read_csv(r'CSVFiles/RawCSV/ABoVE_ Boutin Alberta Moose.csv')
    raw_moose_data['timestamp'] = pd.DatetimeIndex(raw_moose_data['timestamp'])

    # Remove meta-data and individual string data
    raw_moose_data.drop(columns=['event-id', 'visible', 'sensor-type', 'individual-taxon-canonical-name', 'individual-local-identifier', 'study-name', 'gps:dop', 'gps:fix-type', 'tag-voltage'], inplace=True)

    raw_moose_data.rename({'location-long': 'longitude', 'location-lat': 'latitude', 'height-above-ellipsoid': 'altitude'}, axis=1, inplace=True)
    
    # Generate CSV file for each moose with tracking data that spans greater than 900 days
    for moose_tag in raw_moose_data['tag-local-identifier'].unique():
        # Get the moose with lifespans greater than 900 day
        specific_moose = raw_moose_data[raw_moose_data['tag-local-identifier'] == moose_tag].set_index('timestamp')
        specific_moose.drop(columns=['tag-local-identifier'], inplace=True)
        specific_moose['to_keep'] = False

        start = min(specific_moose.index)
        end = max(specific_moose.index)

        if ((end - start) < dt.timedelta(days=900)):
            continue

        # Develop Interpolated Dataset
        start = dt.datetime(year=start.year, month=start.month, day=start.day)
        end = dt.datetime(year=end.year, month=end.month, day=end.day) + dt.timedelta(days=1)

        new_index = pd.DatetimeIndex(np.arange(start, end, dt.timedelta(hours=3))).drop(labels=specific_moose.index, errors='ignore')

        # Create New DataFrame
        add_on = pd.DataFrame(index=new_index, columns=specific_moose.columns)
        add_on.loc[:, 'to_keep'] = True

        specific_moose = pd.concat(objs=[specific_moose, add_on])

        specific_moose.sort_index(inplace=True)

        specific_moose.infer_objects(copy=False)
        specific_moose.interpolate('time', limit_direction='Both', inplace=True)

        # Create DataFrame
        specific_moose = specific_moose[specific_moose['to_keep'] == True]

        specific_moose.drop(columns=['to_keep'], inplace=True)
        specific_moose[['external-temperature', 'longitude', 'latitude', 'altitude']].to_csv(f'{url_dest}{moose_tag}_interpolated.csv', index=True, index_label='timestamp')

if __name__ == '__main__':
    basic_clean_deer()
    interpolation_clean_deer()
    
    basic_clean_moose()
    interpolation_clean_moose()

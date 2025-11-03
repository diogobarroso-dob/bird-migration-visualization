import pandas as pd
import os

# =============================================================================
# DATA LOADING AND INITIAL SUBSET
# =============================================================================

file_path = os.path.join('data', 'Bird_Migration_Data_with_Origin.csv')

try:
    df_original = pd.read_csv(file_path, sep=',')
except FileNotFoundError:
    print(f"ERROR: File not found at {file_path}.")
    raise

features = ['Bird_ID', 'Species', 'Region', 'Habitat', 
'Weather_Condition', 'Migration_Reason', 'Start_Latitude', 'Start_Longitude', 
'End_Latitude', 'End_Longitude', 'Flight_Distance_km', 'Flight_Duration_hours', 
'Average_Speed_kmph', 'Max_Altitude_m', 'Min_Altitude_m', 'Temperature_C', 
'Wind_Speed_kmph', 'Migration_Start_Month', 'Migration_End_Month', 
'Rest_Stops', 'Migrated_in_Flock', 'Flock_Size', 
'Migration_Interrupted', 'Interrupted_Reason', 'Migration_Success', 'Observation_Counts', 'Origin']

df = df_original[features].copy()
print(f"Initial DataFrame loaded with {len(df)} records.")
print("\n--- Initial Data Head ---")
print(df.head())
print("\n" + "="*50 + "\n")

# =============================================================================
# DATA QUALITY CHECKS AND CONVERSION
# =============================================================================

# Data Type Check and Conversion
print("--- Data Type Check and Conversion ---")

NUMERIC_COLS = ['Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude', 
                'Flight_Distance_km', 'Flight_Duration_hours', 'Average_Speed_kmph', 
                'Max_Altitude_m', 'Min_Altitude_m']
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Region'] = df['Region'].astype('category')
df['Species'] = df['Species'].astype('category')

print("Data Types after conversion:")
print(df.dtypes)
print("-" * 40)

## B. Summary Statistics and Outlier Detection
print("--- Summary Statistics and Outlier Detection ---")
print(df[['Flight_Distance_km', 'Flight_Duration_hours', 'Average_Speed_kmph', 'Max_Altitude_m']].describe())
print("-" * 40)

## C. Missing Data Check
print("--- Missing Data Check ---")
print("Total NaN values per column:")
print(df.isnull().sum())
print("-" * 40)

df_missing_coords = df[df[['Start_Latitude', 'End_Latitude', 'Start_Longitude', 'End_Longitude']].isnull().any(axis=1)]
print(f"Records with missing coordinates: {len(df_missing_coords)}")

initial_rows = len(df)
df.dropna(subset=['Start_Latitude', 'End_Latitude', 'Start_Longitude', 'End_Longitude'], inplace=True)
print(f"Dropped {initial_rows - len(df)} rows due to missing coordinates. {len(df)} records remaining.")
print("-" * 40)

#Categorical Consistency Check
print("--- Categorical Consistency (Species/Region) ---")
# Standardize Species and Region columns
df['Species'] = df['Species'].str.strip().str.lower()
df['Region'] = df['Region'].str.strip().str.lower()

print(f"Unique Species after standardization: {df['Species'].nunique()}")
print(f"Unique Regions after standardization: {df['Region'].nunique()}")
print("\n" + "="*50 + "\n")


# =============================================================================
# GEOSPATIAL AND FLIGHT DYNAMICS PLAUSIBILITY CHECKS
# =============================================================================

#Unrealistic Coordinate Check
print("--- Unrealistic Coordinate Check ---")
geo_filter = (
    (df['Start_Latitude'].abs() > 90) | (df['End_Latitude'].abs() > 90) |
    (df['Start_Longitude'].abs() > 180) | (df['End_Longitude'].abs() > 180)
)
df_impossible_coords = df[geo_filter].copy()
df = df[~geo_filter].copy()
print(f"Dropped {len(df_impossible_coords)} records with impossible Lat/Lon values.")
print("-" * 40)


#Zero/Near-Zero Migration Distance Check
print("--- Zero/Near-Zero Migration Distance Check ---")
DISTANCE_THRESHOLD_KM = 1 
zero_distance_filter = (df['Flight_Distance_km'] <= DISTANCE_THRESHOLD_KM) | \
                       (df['Flight_Duration_hours'] <= 0) | \
                       (df['Average_Speed_kmph'] == 0)

df_zero_distance = df[zero_distance_filter].copy()
df = df[~zero_distance_filter].copy()
print(f"Dropped {len(df_zero_distance)} records with zero/near-zero distance, duration, or speed.")
print("-" * 40)


## C. Extreme Speed Check
print("--- Extreme Speed Check ---")
MAX_SPEED_KMH = 300
extreme_speed_filter = df['Average_Speed_kmph'] > MAX_SPEED_KMH

df_extreme_speed = df[extreme_speed_filter].copy()
df = df[~extreme_speed_filter].copy()
print(f"Dropped {len(df_extreme_speed)} records with extreme speeds (> {MAX_SPEED_KMH} kmph).")
print("-" * 40)


## D. Ocean-to-Ocean Migration Check
print("--- Ocean-to-Ocean Migration Check ---")
# Central Atlantic Example boundaries: 20-50 Lat and -50 to -20 Lon
atlantic_ocean_start = (
    (df['Start_Latitude'] > 20) & (df['Start_Latitude'] < 50) &
    (df['Start_Longitude'] > -50) & (df['Start_Longitude'] < -20)
)
atlantic_ocean_end = (
    (df['End_Latitude'] > 20) & (df['End_Latitude'] < 50) &
    (df['End_Longitude'] > -50) & (df['End_Longitude'] < -20)
)

ocean_birds_filter = atlantic_ocean_start & atlantic_ocean_end
ocean_birds = df[ocean_birds_filter].copy()
df = df[~ocean_birds_filter].copy()

print(f"Dropped {len(ocean_birds)} records flagged as ocean-to-ocean migrations.")
# Check if there are any ocean birds remaining 
if not ocean_birds.empty:
    print(f"Example Ocean Bird (ID: {ocean_birds['Bird_ID'].iloc[0]}):")
    print(ocean_birds[['Bird_ID', 'Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude']].head(1))
print("\n" + "="*50 + "\n")


# =============================================================================
# FINALIZED DATASET
# =============================================================================
def save_cleaned_data(df_input, output_filename='cleaned_migration_data.csv', output_dir='data'):
    """
    Finalized the DataFrame (resets index) and saves it.
    """
    df_final = df_input.reset_index(drop=True)
    
    output_path = os.path.join(output_dir, output_filename)
    
    
    df_final.to_csv(output_path, index=False)
    
    print(f"FINAL CLEANING REPORT: {len(df_final)} records saved.")
    print(f"Saved cleaned data to: {output_path}")
    
    return df_final


df_cleaned_result = save_cleaned_data(df)


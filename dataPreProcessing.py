import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

data = pd.read_csv('alertas_plagas.csv')

# Clean csv format
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(';', '', regex=False)

# Separation of the period by year and season

data['year'] = data['period'].astype(str).str[:4]
data['season'] = data['period'].astype(str).str[-1]

#Remove original colum
data.drop(columns=['period'], inplace=True)

# Handling empty variables

data['pest_name'] = data['pest_name'].fillna('Pest_free')
data['species'] = data['species'].fillna('Unknown')

# Standardization by categories

data['climate_risk'] = (
    (data['avg_temperature'] > 28) &
    (data['total_precipitation'] > 100)
).astype(int)

data['category_temp'] = pd.cut(
    data['avg_temperature'],
    bins=[0, 20, 28, 50],
    labels=['low', 'medium', 'high']
)

data['category_rain'] = pd.cut(
    data['total_precipitation'],
    bins=[0,50,150,10000],
    labels=['low', 'medium', 'high']
)

labels_col = [
    'department','municipality','crop','year','season', 'pest_name', 'species','category_temp',
    'category_rain','pest_alert'
]

le = LabelEncoder()

for col in labels_col:
    data[col] = le.fit_transform(data[col].astype(str))

# Standardization

scaler = MinMaxScaler()

numeric_cols = [
    'yield_t_ha','relative_yield', 'avg_temperature', 'total_precipitation'
]

data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

data.to_csv('processed_alerts.csv', index=False)
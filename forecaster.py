from datetime import timedelta
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import TimeSeriesSplit
import autosklearn
from autosklearn import regression
import pywt
import geopandas as gpd
import libpysal as lps

class Forecaster:

    def __init__(self, config):
        self.save_base = config['save_base']
        self.debug_mode = config['debug_mode']
        self.continent = config['continent']
        self.multiple = config['multiple']
        self.max_window_size = config['max_window_size']
        self.horizon = config['horizon']
        self.test_days = config['test_days']
        self.validation_days = config['validation_days']
        self.shift = config['shift']
        self.target_column = config['target_column']
        self.end_train_date = config['end_train_date']
        self.data_path = config['data_path']
        self.rso = config['rso']
        self.rso_steps = config['rso_steps']
        self.wavelets = config['wavelets']
        self.column_group = config['column_group']
        self.spatial_lags = config['spatial_lags']
        self.extended = config['extended']
        self.extended_data_path = config['extended_data_path']
        self.end_extended_date = config['end_extended_date']
        self.data = None
        self.countries = None
        
    def create_forecasts(self, X_test):
        if self.rso:
            if self.wavelets:
                self.rso_steps = int(self.rso_steps / 2)
            pd.options.mode.chained_assignment = None  # default = 'warn', suppress warnings
            for country in self.countries:
                series = X_test.loc[X_test.country == country]
                for i in range(self.rso_steps):
                    pred = self.automl.predict(series)
                    for j in range(len(series.columns) - 2):
                        series[j] = series[j + 1]
                    series[j + 1] = pred
                series = series.drop(columns = ['country'])
                if self.wavelets:
                    series = from_wavelets(series)
                    np.savetxt(self.save_base + '_' + country + '.csv', series, delimiter = ',')
                else:
                    series.to_csv(self.save_base + '_' + country + '.csv', header = False, index = False)
        else:
            for country in self.countries:
                pred = self.automl.predict(X_test.loc[X_test.country == country])
                if self.wavelets:
                    pred = from_wavelets(pred)
                np.savetxt(self.save_base + '_' + country + '.csv', pred, delimiter = ',')
                
    def create_splits(self, X, y, testing=True):
        # Add countries
        country_col = []
        for i in range(int(len(X) / len(self.countries))):
            for country in self.countries:
                country_col.append(country)
        X['country'] = country_col
        X['country'] = X.country.astype('category')
        
        if testing:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=self.test_days * len(self.countries), shuffle = False)
        else:
            X_train = X
            X_test = None
            y_train = y
            y_test = None
        # Create train / validation split
        self.train_size = (len(X_train) - self.validation_days * len(self.countries)) / len(X_train)
        X_validation = X_train[len(X_train) - self.validation_days * len(self.countries):]
        y_validation = y_train[len(X_train) - self.validation_days * len(self.countries):]
        return X_train, X_test, X_validation, y_validation, y_train, y_test
            
    def fit(self, X_train, y_train): 
        if self.debug_mode:
            self.automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task = 180,
            per_run_time_limit = 30,
            tmp_folder = self.save_base + '_tmp',
            output_folder = self.save_base + '_out',
            memory_limit = 10240,
            n_jobs = 2,
            metric=autosklearn.metrics.mean_squared_error,
            resampling_strategy_arguments={'train_size': self.train_size, 'shuffle': False})
        else:
            self.automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=10800,
            tmp_folder = self.save_base + '_tmp',
            output_folder = self.save_base + '_out',
            memory_limit=10240,
            n_jobs=8,
            metric=autosklearn.metrics.mean_squared_error,
            resampling_strategy_arguments={'train_size': self.train_size, 'shuffle': False})
        
        self.automl.fit(X_train, y_train)

    def preprocess(self, second_part = False):     
        # Load data
        if second_part:
            df = pd.read_csv(self.extended_data_path)
        else:
            df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df.date, format = '%Y-%m-%d')
                
        # Scale time series
        if self.target_column == 'deaths' or self.target_column == 'cases':
            df['scaled_ts'] = scale_column(df, self.target_column, second_part)
        else:
            print('Error: target_column should be either "cases" or "deaths".')

        if self.spatial_lags:
            spatial_column, countries = compute_spatial_lags(df, 'scaled_ts')

        # Limit days
        if second_part:
            df = df.loc[df.date <= self.end_extended_date + timedelta(days = self.test_days)]
        elif self.extended:
            df = df
        else:
            df = df.loc[df.date <= self.end_train_date + timedelta(days = self.test_days)]
                
        # Limit countries
        if self.continent == 'world':
            pass
        else:
            df = df.loc[df.continentExp == self.continent]
        if self.extended:
            cf = pd.read_csv(self.extended_data_path)
            self.countries = cf.country.unique()
            df = df.loc[df.country.isin(self.countries)]
        else:
            self.countries = df.country.unique()
        
        if self.spatial_lags:
            self.countries = countries
            df = df.loc[df.country.isin(countries)]
            df['spatial_ts'] = spatial_column
            
        # Shift time series
        df['shifted_ts'] = df.scaled_ts.shift(self.shift * len(self.countries))          
 
        # Preprocess mobility data
        if self.multiple:
            if self.column_group == 'Apple':
                df['driving'] = df.driving - 100
                df['walking'] = df.walking - 100
                mobility_columns = ['walking', 'driving']
            elif self.column_group == 'Google':
                mobility_columns = ['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline',
                      'parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline',
                      'workplaces_percent_change_from_baseline','residential_percent_change_from_baseline']
            else:
                df['driving'] = df.driving - 100
                df['walking'] = df.walking - 100
                mobility_columns = ['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline',
                      'parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline',
                      'workplaces_percent_change_from_baseline','residential_percent_change_from_baseline', 'walking', 'driving']
            # Shift columns
            for col in mobility_columns:
                df[col] = df[col].shift(self.shift * len(self.countries))
            # Select columns
            multiple_columns = ['shifted_ts']
            for col in mobility_columns:
                multiple_columns.append(col)
        else:
            multiple_columns = None
        
        # Remove NaN-rows created by shifting
        df = df.dropna().reset_index()

        # Split data into instances
        X, y = create_instances(df, self.max_window_size, self.horizon, len(self.countries), 'shifted_ts', multiple_columns, self.spatial_lags)
        
        return X, y
        
    def print_run_summary(self):
        with open(self.save_base + '_sprint.txt', 'w') as f:
            f.write(self.automl.sprint_statistics())
        with open(self.save_base + '_models.txt', 'w') as f:
            f.write(self.automl.show_models())
    
    def refit(self, X_train, y_train):
        self.automl.refit(X_train, y_train)
            
    def validate(self, X_validation, y_validation):
        validation_forecast = self.automl.predict(X_validation)
        with open(self.save_base + '_val_mse.csv', 'w') as f:
            f.write(str(sklearn.metrics.mean_squared_error(y_validation, validation_forecast)))     
    
    def wavelet_transform(self, X, y):
        cA_X = X.apply(to_wavelets, axis = 1, result_type = 'expand')
        cA_y = y.apply(to_wavelets, axis = 1, result_type = 'expand')
        return cA_X, cA_y
    
def create_instances(df, max_window_size, horizon, num_countries, target_column, multiple_columns, spatial_lags):
    X = []
    y = []
    for i in range(len(df)):
        X_indices = []
        y_indices = []
        if i + (max_window_size + horizon - 1) * num_countries < len(df):
            for X_step in range(max_window_size):
                X_indices.append(i + X_step * num_countries)
            if multiple_columns is not None:
                X.append(df.loc[X_indices, multiple_columns].values)
            elif spatial_lags:
                X.append(df.loc[X_indices, 'spatial_ts'].values)
            else:
                X.append(df.loc[X_indices, target_column].values)
            for y_step in range(1, horizon + 1):
                y_indices.append(i + (X_step + y_step) * num_countries)
            y.append(df.loc[y_indices, target_column].values)

    if multiple_columns is not None:
        X = np.array(X)
        X = X.reshape(X.shape[0], -1)
	
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    
    return X, y

def compute_spatial_lags(df, column):
    countries = df.country.unique()
    gdf = gpd.read_file('countries/TM_WORLD_BORDERS-0.3.dbf')
    gdf = gdf.replace('Czech Republic', 'Czechia')
    gdf = gdf.replace('Korea, Republic of', 'South Korea')
    gdf = gdf.replace('Viet Nam', 'Vietnam')
    gdf = gdf.replace("Cote d\'Ivoire", 'Cote dIvoire')
    gdf = gdf.replace('Falkland Islands (Malvinas)', 'Falkland Islands')
    gdf = gdf.replace('Guinea-Bissau', 'Guinea Bissau')
    gdf = gdf.replace('Holy See (Vatican City)', 'Holy See')
    gdf = gdf.replace('Iran (Islamic Republic of)', 'Iran')
    gdf = gdf.replace('Libyan Arab Jamahiriya', 'Libya')
    gdf = gdf.replace('Republic of Moldova', 'Moldova')
    gdf = gdf.replace('The former Yugoslav Republic of Macedonia', 'North Macedonia')
    gdf = gdf.replace('Saint Martin', 'Sint Maarten')
    gdf = gdf.replace('Syrian Arab Republic', 'Syria')
    gdf = gdf.replace('Timor-Leste', 'Timor Leste')
    gdf = gdf.replace('Turks and Caicos Islands', 'Turks and Caicos islands')
    gdf = gdf.replace('Burma', 'Myanmar')
    gdf = gdf.replace('Swaziland', 'Eswatini')
    gdf = gdf.replace("Lao People's Democratic Republic", 'Laos')
    gdf = gdf[gdf.NAME.isin(countries)].sort_values('NAME').reset_index(drop=True)
    countries = gdf.NAME.unique()
    df = df.loc[df.country.isin(countries)]

    wq =  lps.weights.Queen.from_dataframe(gdf)
    wq.transform = 'r'

    sp_column = []
    for i in range(int(len(df)/len(countries))):
        gdf[column] = df[column].values[i*len(countries):(i+1)*len(countries)]
        y = gdf[column]
        ylag = lps.weights.lag_spatial(wq, y)
        sp_column.append(ylag)
    sp_column = [item for sublist in sp_column for item in sublist]
    return sp_column, countries

def scale_column(df, column, second_part):
    if second_part:
        scaled_column = df[column] / (df.popData2020 * 1e-6)
    else:
        scaled_column = df[column] / (df.popData2019 * 1e-6)
    return scaled_column    
    
def to_wavelets(X):
    (cA, cD) = pywt.dwt(X, 'haar')
    return cA
    
def from_wavelets(cA, cD = None):
    X = pywt.idwt(cA, cD, wavelet = 'haar')
    return X
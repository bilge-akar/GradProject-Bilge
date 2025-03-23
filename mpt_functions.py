
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def plot_columns(df, x_col, y_col, title="Scatter Plot", xlabel=None, ylabel=None):

    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')

    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.title(title)
    plt.grid(True)
    plt.show()


def get_the_data_and_convert_datetime(path):

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def drop_a_column(df, column_name):

    newdf = df.drop(column_name, axis='columns')
    return newdf


def plot_box(df, column, title="Box Plot"):

    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[column])
    
    plt.title(title)
    plt.ylabel(column)
    plt.grid(True)
    plt.show()

def plot_violin(df, column_name):
    sns.violinplot(x=column_name, data=df)



def filter_a_day(specific_date,df):

    
    df_filtered = df[df['timestamp'].dt.date == pd.to_datetime(specific_date).date()]

    df_filtered = df_filtered.reset_index(drop=True)

    return df_filtered


def aggregate_as_a_minute(df_day1):
    df_day1 = df_day1.set_index('timestamp')

    # Resample data by minute and compute mean
    df_minute_avg_day1 = df_day1.resample('T').mean().reset_index()

    return df_minute_avg_day1


def aggregate_with_sliding_window(df, window_size, slide):
    df = df.set_index('timestamp')

    # Resample data by minute and compute mean
    df_minute_avg = df.resample('T').mean()

    # Compute rolling mean with specified window size and step
    df_sliding_avg = df_minute_avg.rolling(window=window_size).mean()[::slide].dropna().reset_index()

    return df_sliding_avg


def aggregate_with_sliding_window_rowwise(df, window_size, slide):
    df = df.reset_index(drop=True)  # Satır indexlerini sıfırdan başlat
    
    rolling_means = [
        df.iloc[i:i+window_size].mean()
        for i in range(0, len(df) - window_size + 1, slide)
    ]

    df_sliding_avg = pd.DataFrame(rolling_means)
    
    return df_sliding_avg

def filter_rows_between_the_given_timestamps(df, start, end, varible_name="timestamp"):

    new_df = df.loc[(df[varible_name] >= start) & (df[varible_name] <= end)]
    return new_df


def change_the_values_by_applying_a_time_filter(df, start_date, end_date, feature, new_value):

    df_new = df.copy()

    df_new.loc[(df_new["timestamp"] <= end_date) & (df_new["timestamp"] >= start_date), feature] = new_value
    return df_new


def apply_ttest(df, column_names, variable):

    summary_stats = df.groupby('condition')[column_names].agg(['mean', 'median', 'std'])
    print(summary_stats[variable])

    condition_0 = df[df['condition'] == 0][variable]
    condition_1 = df[df['condition'] == 1][variable]

    print()

    t_stat, p_value = stats.ttest_ind(condition_0, condition_1)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant difference: {p_value < 0.05}")



def plot_columns_failure_comparison(df, x_col, y_col, color_col, title="Scatter Plot", xlabel=None, ylabel=None):
    colors = {0: 'red', 1: 'blue', 2: 'black'}
    
    plt.figure(figsize=(15, 8))
    
    # Plot each category as a separate scatter plot
    for category in df[color_col].unique():
        subset = df[df[color_col] == category]
        plt.scatter(subset[x_col], subset[y_col], 
                   color=colors.get(category, 'gray'),
                   label=f'Category {category}',
                   alpha=0.7,  # Add some transparency
                   s=30)  # Control point size
    
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def counter_for_maintenance(df, start_of_interval, finish_date):

    start_of_interval = pd.to_datetime(start_of_interval)
    finish_date = pd.to_datetime(finish_date)
    

    df_new = df.copy()
    

    
    counter = 0
    

    for idx, row in df_new.iterrows():
        time = row["timestamp"]
        
        if time >= finish_date:
            break
        elif time < start_of_interval:
            pass
        else:
            counter += 1
            df_new.at[idx, "counter"] = counter
    
    return df_new


def scale_columns(df, columns):
    std_scaler = StandardScaler()
    
    df_scaled = df.copy()
    df_scaled[columns] = std_scaler.fit_transform(df[columns])  # Scale only numerical columns
    
    return df_scaled




def apply_kmeans_clustering(df, number_of_clusters, target_variable):

    X = df.select_dtypes(include=[np.number]).drop(columns=[target_variable], errors="ignore")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    return df 


def check_cluster_distribution(df, condition_column, cluster_column):

    distribution = pd.crosstab(df[cluster_column], df[condition_column])
    print("\nCluster Distribution by Condition:\n", distribution)

    return distribution


def apply_smote(df, target_column, seed):

    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors="ignore")
    y = df[target_column]

    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    print("Before : ", df[target_column].value_counts())
    print("After : ", y_resampled.value_counts())

    return df_resampled


def apply_random_forest_and_get_results(df, target, seed=10):

    target = df[target]

    X = df.drop("condition", axis='columns')


    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return model, accuracy





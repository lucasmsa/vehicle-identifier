import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(1, f'{os.getcwd()}/src/config')
from vehicle_detection_constants import RESULTS_PATH, STATISTICS_PATH

class ResultsAnalyzer:
    def __init__(self, filter_type):
        self.filter_type = filter_type
        self.results_path = f"{RESULTS_PATH}/{filter_type.lower()}.csv"
        self.dataframe = pd.read_csv(self.results_path)
        self.intensity_column_name = f'{self.filter_type.lower()}_intensities'
        self.match_percentages_column = self.dataframe['match_percentages']
        self.unique_intensities = sorted(self.dataframe[self.intensity_column_name].unique())
        
    def generate_error_statistics(self):
        errored_intensities = self.dataframe.loc[self.match_percentages_column.astype(str).str.startswith("ERROR"),\
                                            self.intensity_column_name]
        
        sorted_errored_intensity_values = errored_intensities.value_counts().sort_index(ascending=True)
        
        filtered_sorted_errored_intensity = []
        
        if(len(sorted_errored_intensity_values) < len(self.unique_intensities)):
            set_sorted_errored_intensity = set(sorted_errored_intensity_values.index)
            for intensity in self.unique_intensities:
                if intensity not in set_sorted_errored_intensity:
                    filtered_sorted_errored_intensity.append(0)
                else: 
                    filtered_sorted_errored_intensity.append(sorted_errored_intensity_values[intensity])
                    
            sorted_errored_intensity_values = pd.Series(filtered_sorted_errored_intensity)
        
        errored_dataframe = pd.DataFrame({f'{self.filter_type.capitalize()} Intensities': self.unique_intensities, 'Errored': sorted_errored_intensity_values})
        errored_bar_plot = errored_dataframe.plot.bar(x=f'{self.filter_type.capitalize()} Intensities', y='Errored', rot=0)
        plt.show(block=True)
        
        errored_bar_plot.figure.savefig(f"{STATISTICS_PATH}/errors/{self.filter_type.lower()}_errors.png")

    def generate_average_statistics(self):
        filtered_dataframe = self.dataframe.copy()
        filtered_dataframe.loc[self.match_percentages_column.astype(str).str.startswith("ERROR"), 'match_percentages'] = 0
        filtered_dataframe['match_percentages'] = filtered_dataframe['match_percentages'].astype(int)
        avg_df = filtered_dataframe.groupby([self.intensity_column_name])['match_percentages'].mean().round(1)
        sorted_average_percentage_value = avg_df.values
        average_dataframe = pd.DataFrame({f'{self.filter_type.capitalize()} Intensities': self.unique_intensities, 'Mean accuracy': sorted_average_percentage_value})
        average_line_plot = average_dataframe.plot.line(x=f'{self.filter_type.capitalize()} Intensities', y='Mean accuracy', rot=0)
        plt.show(block=True)
        
        average_line_plot.figure.savefig(f"{STATISTICS_PATH}/{self.filter_type.lower()}_average.png")
    
results_analyzer = ResultsAnalyzer('RESOLUTION')
results_analyzer.generate_error_statistics()
results_analyzer.generate_average_statistics()
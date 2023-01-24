'''
Input for this functionality is a pandas data series which contains the progression of the total consumption (e.g. water consumption or electricity consumption 
of a certain device or device group). The indices of this series are given by the timestamps of the respective values that visualize the consumption progression.
The other input is a pandas timedelta object that depicts the minimum time window length.
'''

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.signal import savgol_filter, find_peaks, peak_widths
import math

def create_daily_five_min_groups(data_series):
    grouped_by_day = data_series.groupby(by=lambda x: x.floor('d')) 
    # Groups the data series by day

    grouped_five_min = []
    for group in grouped_by_day:
        grouped_five_min.append(group[1].groupby(pd.Grouper(freq='5T')))
    
    return grouped_five_min

def compute_five_min_consumptions(grouped_five_min):
    consumption_series_list_five_min = []
    for day in grouped_five_min:
        five_min_data = []
        for group in day:
            try:
                five_min_data.append(list(group[1])[-1]-list(group[1])[0])
            except:
                five_min_data.append(0)
        consumption_series_list_five_min.append(pd.Series(data=five_min_data, index=[group[0] for group in day]))

    for i, series in reversed(list(enumerate(consumption_series_list_five_min))):
        if len(series) != 288:
            del consumption_series_list_five_min[i]
    # The series with less than 288 entries in the list consumption_series_list_five_min got deleted.

    return consumption_series_list_five_min


def window_determination(data_series):
    grouped_five_min = create_daily_five_min_groups(data_series)
    # Each object in the list grouped_five_min contains the data from data_series from a single day grouped into slots of five minutes.
    
    consumption_series_list_five_min =  compute_five_min_consumptions(grouped_five_min)                                    
    # Each object in the list consumption_series_list_five_min contains a pandas series in which for a single day the consumption within the 5-minutes time slots is stored.
    # I.e. eech pandas series in this list has 24*(60/5)=24*12=288 entries if the data is clean. The series with less entries get ignored.

    
    index = [series.index[0].date() for series in consumption_series_list_five_min] # This just a list which contains the days.

    aux_dict = defaultdict(list)
    for series in consumption_series_list_five_min:
        for timestamp in list(series.index):
            aux_dict[timestamp.time()].append(series.loc[timestamp])
    df = pd.DataFrame(data=aux_dict, index=index)
    # The pandas dataframe df has the days as indices and the 288 5min time slots in a day as columns. The entries are the consumption from the respective day in the 
    # respective time slot.

    scaled_mean_array = 10000*df.mean(axis=0)
    # We compute the mean along the rows (i.e. days) and scale up by 10000.

    smooth_scaled_mean_array = savgol_filter(scaled_mean_array, 20, 2)
    smoothed_mean_list = smooth_scaled_mean_array.tolist()
    # As a next step we smoothen the arrays and convert them to lists.

    prominence = float(max(smooth_scaled_mean_array))/float(25) # This is kind of random here!?
    peaks_mean, _ = find_peaks(smoothed_mean_list, distance=20, prominence=prominence)
    # Compute the peaks and bases of the peaks.

    # Now we compute the third, two third and full height of the peaks as well as the x-coordinates of the points where the curve attains the respective heights
    results_third = peak_widths(smoothed_mean_list, peaks_mean, rel_height=0.33)
    results_two_third = peak_widths(smoothed_mean_list, peaks_mean, rel_height=0.66)
    results_full = peak_widths(smoothed_mean_list, peaks_mean, rel_height=1)
    x_coord_third = list(results_third[2])+list(results_third[3])
    x_coord_two_third = list(results_two_third[2])+list(results_two_third[3])
    x_coord_full = list(results_full[2])+list(results_full[3])

    # We use these coordinates now to divide the time line into windows (we add the start and the end of the day to the list, i.e. 0 and 287).

    window_boundaries_points = sorted(list(set([0]+x_coord_third+x_coord_two_third+x_coord_full+[287])))

    # Now we check if boundary point are too close together (i.e. closer than 0.5 hour):

    points_to_delete_indices = []
    for i, x in enumerate(window_boundaries_points[:-1]):
        if window_boundaries_points[i+1]-x < 6:
            points_to_delete_indices.append(i+1)
    points_to_delete_indices.reverse()    
    for j in points_to_delete_indices:
        del window_boundaries_points[j]

    # Now we check if boundary points are too far away (i.e. the gap between the points is larger than 2 hours)

    
    new_window_boundary_points = []
    for i, x in enumerate(window_boundaries_points[:-1]):
        if window_boundaries_points[i+1]-x >= 24:
            aux = window_boundaries_points.copy()
            number_of_add_slots = math.floor(float(window_boundaries_points[i+1]-x)/float(12))
            aux[i+1:i+1]=list(np.linspace(x,window_boundaries_points[i+1], num=number_of_add_slots+1))[1:-1]
            new_window_boundary_points += aux
    
    window_boundaries_points = sorted(list(set(new_window_boundary_points)))    

    # Then we convert the boundary points to time objects.

    window_boundaries_times = [(pd.Timestamp.now().floor('d')+i*pd.Timedelta('5T')).time() for i in window_boundaries_points]

    # As a last step we replace the timesampt from 23:55 to 23:59:59:999000

    window_boundaries_times[-1] = (pd.Timestamp.now().floor('d')-pd.Timedelta('1ms')).time()

    return window_boundaries_times


'''The input is a list of pd.Timestamp.time() objects and the input is a pandas data series indexed over pd.Timestamps with values given by certain consumption values. 
The first entry of the list is 00:00:00 and the last entry is 24:00:00. The output is given by a list with the following properties: Each entry consists of a collection
of pandas data series which corresponds to the consumption data of one day divided with respect to the list of times given as input.'''
    
def window_division(list_of_timestamps, data_series):
    daily_groups=data_series.groupby(by=lambda x: x.floor('d'))
    grouped_wrt_timestamps = []
    for group in daily_groups:
        data_of_one_day = group[1]
        divided_data_for_one_day = []
        for i, time in enumerate(list_of_timestamps[:-1]):
            divided_data_for_one_day.append(data_of_one_day[group[1].index[0].floor('d')+pd.Timedelta(str(time)):
                                                            group[1].index[0].floor('d')+pd.Timedelta(str(list_of_timestamps[i+1]))])
        grouped_wrt_timestamps.append(divided_data_for_one_day)
    return grouped_wrt_timestamps
    
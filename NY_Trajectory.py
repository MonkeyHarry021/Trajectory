import numpy as np
import pandas as pd
'''
data preprocessing

input:
    data_O: the origin out data , a dataframe
    data_D: the origin in data , a dataframe
'''
def data_preprocess(data_O, data_D):
    data_O.insert(0, 'TimeStamp', pd.to_datetime(data_O[' pickup_datetime']))
    data_O.insert(0, 'hour', data_O['TimeStamp'].apply(lambda x:x.hour))
    data_O.insert(0, 'day', data_O['TimeStamp'].apply(lambda x:x.day))
    data_O.insert(0, 'mapped_time', (data_O['day']-data_O['day'].min())*24+data_O['hour'])

    data_D.insert(0, 'TimeStamp', pd.to_datetime(data_D[' dropoff_datetime']))
    data_D.insert(0, 'hour', data_D['TimeStamp'].apply(lambda x:x.hour))
    data_D.insert(0, 'day', data_D['TimeStamp'].apply(lambda x:x.day))
    data_D.insert(0, 'mapped_time', (data_D['day']-data_D['day'].min())*24+data_D['hour'])
    return data_O, data_D

'''
compute the attractiveness of the home location in next 24 hours
'''
def compute_attractiveness(data_D, start_locaiton, depart_time):
    att = []
    data_D_in_next_24 = data_D[(data_D['mapped_time'] > depart_time) & (data_D['mapped_time'] <= depart_time+24)]
    data_D_in_next_24 = data_D_in_next_24[data_D_in_next_24['bct2010_D'] == start_locaiton]

    total_inflow = data_D_in_next_24.shape[0]
    data_D_in_next_24_groupby_mapped_time = list(data_D_in_next_24.groupby('mapped_time', as_index=False))

    for time in data_D_in_next_24_groupby_mapped_time:
        inflow = time[1].shape[0]
        att.append(inflow/total_inflow)
        
    return att

'''
compute the time of return

input:
    att: the attractiveness of the home location in next 24 hours, a 1-d array whose length is 24
    depart_time: the time of departure, an integer

output:
    return_time: the time of return, an integer
'''

def compute_return_time(att, depart_time):    
    att_multinomial = np.random.multinomial(1, att/np.sum(att))
    return_time = depart_time + int(np.where(att_multinomial == 1)[0]) + 1
    if return_time > 167:
        return_time = 167

    return return_time

'''
compute the time of leave current location

input:
    data_O: the origin out data , a dataframe
    location: the current location of the random walker, an integer
    time: the current time of the random walker, an integer

output:
    leave_time: the time of leave current location, an integer
'''
def compute_leave_time(data_O, location, time):
    leave_list = []
    data_O_leave = data_O[(data_O['mapped_time'] > time) & (data_O['mapped_time'] <= time+24)]
    data_O_leave = data_O_leave[data_O_leave['bct2010_O'] == location]
    data_O_leave_gourpby_mapped_time = list(data_O_leave.groupby('mapped_time', as_index=False))
    for hour in data_O_leave_gourpby_mapped_time:
        outflow = hour[1].shape[0]
        leave_list.append(outflow)
    leave_multinomial = np.random.multinomial(1, leave_list/np.sum(leave_list))
    leave_time = time + int(np.where(leave_multinomial == 1)[0]) + 1

    return leave_time

'''
find the location id of next step

input:
    data_O: the origin out data , a dataframe
    location: the current location of the random walker, an integer
    time: the current time of the random walker, an integer

output:
    next_location: the location id of next step, an integer
'''
def find_next_location(data_O, location, time):
    data_O_next_location = data_O[(data_O['mapped_time'] == time) & (data_O['bct2010_O'] == location)]
    counts_for_each_id = data_O_next_location['bct2010_D'].value_counts()
    counts_array = np.array(counts_for_each_id.values)
    id_array = np.array(counts_for_each_id.index)

    next_location_multinomial = np.random.multinomial(1, counts_array/np.sum(counts_array))
    next_location = id_array[np.where(next_location_multinomial == 1)[0]][0]

    return next_location


'''
generate the trajectory of the random walker from start_tome to end_time (0~167)

input:
    data_O: the origin out data , a dataframe
    data_D: the origin in data , a dataframe
    start_location: the start location of the random walker, an integer
    start_day: the start day of the random walker, an integer
    end_time: the end time of the random walker, an integer

output:
    trajectory: the trajectory of the random walker, a list that contains the location id of the random walker
    time_seq: the time sequence of the random walker, a list that contains the time of the random walker
'''
def generate_trajectory(data_O, data_D, start_location, start_day, end_time, verbose):
    trajectory = []
    time_seq = []

    #seed condition
    #compute depart time
    depart_list = []
    data_O_start = data_O[(data_O['bct2010_O'] == start_location) & (data_O['day'] == start_day)]
    data_O_start_groupby_hour = list(data_O_start.groupby('hour', as_index=False))
    for hour in data_O_start_groupby_hour:
        outflow = hour[1].shape[0]
        depart_list.append(outflow)
    depart_multinomial = np.random.multinomial(1, depart_list/np.sum(depart_list))
    depart_time = np.where(depart_multinomial == 1)[0][0]
    #fill trajectory and time_seq according to the seed condition
    for step in range(depart_time):
        trajectory.append(start_location)
        if len(time_seq) == 0: time_seq.append(0)
        else: time_seq.append(time_seq[-1]+1)

    if verbose == True:
        print("**********seed condition**********")
        print('start_location:', start_location)
        print('depart_time: ', depart_time)

    
    while(time_seq[-1] < end_time):
        #find the attractiveness of the home location in next 24 hours
        att = compute_attractiveness(data_D, start_location, depart_time)
        #find the time of return
        return_time = compute_return_time(att, depart_time)
        #find the next location
        next_location = find_next_location(data_O, trajectory[-1], time_seq[-1])
        #find the time of leave current location
        leave_time = compute_leave_time(data_O, next_location, time_seq[-1])
        if verbose == True:
            print("---------depart from home again---------")
            print("next_locaiton: ", next_location)
            print("leave_time: ", leave_time)
            print("return_time: ", return_time)

        #if leave time is smaller than return time, fill trajectory and time_seq according to the leave time
        while(leave_time < return_time):
            #fill trajectory and time_seq according to the leave time
            for step in range(time_seq[-1]+1, leave_time):
                trajectory.append(next_location)
                time_seq.append(time_seq[-1]+1)
            #up to leave time, find the next location
            next_location = find_next_location(data_O, trajectory[-1], time_seq[-1])
            leave_time = compute_leave_time(data_O, next_location, time_seq[-1])
            if verbose == True:
                print("----------continue to random walk----------")
                print("next_locaiton: ", next_location)
                print("leave_time: ", leave_time)

       #once leave time is larger than return time, fill trajectory and time_seq according to the return time
        for step in range(time_seq[-1]+1, return_time):
            trajectory.append(next_location)
            time_seq.append(time_seq[-1]+1)
        #return to home
        trajectory.append(start_location)
        time_seq.append(time_seq[-1]+1)
        if verbose == True:
            print("----------it's time return to home---------")
            print("the last location:", trajectory[-1])
            print("the last time:", time_seq[-1])

        if time_seq[-1] >= end_time: break
        #compute depart time from home
        depart_time = compute_leave_time(data_O, start_location, time_seq[-1])
        #if depart time is smaller than end time, fill trajectory and time_seq according to the depart time
        if depart_time < end_time:
            for step in range(time_seq[-1]+1, depart_time):
                trajectory.append(start_location)
                time_seq.append(time_seq[-1]+1)
        #if depart time is larger than end time, fill trajectory and time_seq according to the depart time
        else:
            for step in range(time_seq[-1]+1, end_time):
                trajectory.append(start_location)
                time_seq.append(time_seq[-1]+1)
        if verbose == True:
            print("----------after back to home----------")
            print("new depart time:", depart_time)

    return trajectory, time_seq


if __name__ == "__main__":

    data_O = pd.read_csv(r"D:\SUPD\Project\Container\Container4NYC\data\nyc\week1_res_O_filter.csv")
    data_D = pd.read_csv(r"D:\SUPD\Project\Container\Container4NYC\data\nyc\week1_res_D_filter.csv")
        
    data_O, data_D = data_preprocess(data_O, data_D)
    trajectory, time_seq = generate_trajectory(data_O, data_D, 1006800, 21, 167, verbose=True)

    print(trajectory)
    print(len(trajectory))
    print(time_seq)
    print(len(time_seq))
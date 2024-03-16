import numpy as np
import fastf1 as ff1
import csv
import parsing
import pandas as pd

#  Globals
races = ['albert park', 'bahrain', 'catalunya', 'istanbul', 'monaco', 'villeneuve',
         'silverstone', 'hockenheimring', 'hungaroring', 'spa', 'monza', 'marina bay',
         'shanghai', 'interlagos', 'suzuka', 'rodriguez', 'ricard', 'zandvoort',
         'las vegas', 'americas', 'red bull ring', 'sochi', 'baku', 'algarve',
         'mugello', 'jeddah', 'miami', 'lusail']

average_temps = {'Australian Grand Prix': 22.925582944230484, 'Bahrain Grand Prix': 25.146710892239195, 'Chinese '
                                                                                                        'Grand Prix':
    19.388174915635545, 'Azerbaijan Grand Prix': 22.178385671902202, 'Spanish Grand Prix': 24.573009873941526,
                 'Monaco Grand Prix': 23.49934558655908, 'Canadian Grand Prix': 22.29626218761439, 'French Grand '
                                                                                                   'Prix':
                     26.7298842643514, 'Austrian Grand Prix': 24.987384554857012, 'British Grand Prix':
                     22.792472592173684, 'German Grand Prix': 23.95266221873365, 'Hungarian Grand Prix':
                     25.79263132416941, 'Belgian Grand Prix': 17.661480024250768, 'Italian Grand Prix':
                     26.221360393352246, 'Singapore Grand Prix': 28.9768605063958, 'Russian Grand Prix':
                     23.68815525288639, 'Japanese Grand Prix': 23.472606076821183, 'United States Grand Prix':
                     26.16060413511571, 'Mexican Grand Prix': 21.663109354413702, 'Brazilian Grand Prix':
                     22.393196946564885, 'Abu Dhabi Grand Prix': 26.745110314340618, 'Pre-Season Test 1':
                     21.29140625, 'Pre-Season Test 2': 21.29140625, 'Styrian Grand Prix': 23.757436440677964,
                 '70th Anniversary Grand Prix': 25.287068965517243, 'Tuscan Grand Prix': 30.169886363636362,
                 'Eifel Grand Prix': 9.499999999999998, 'Portuguese Grand Prix': 19.949003031716416, 'Emilia Romagna '
                                                                                                     'Grand Prix':
                     13.850295033676652, 'Turkish Grand Prix': 14.460580580121889, 'Sakhir Grand Prix':
                     20.947244094488195, 'Pre-Season Test': 18.82286964193809, 'Dutch Grand Prix':
                     20.342119252716074, 'Mexico City Grand Prix': 23.68893433695102, 'São Paulo Grand Prix':
                     22.202743168294386, 'Qatar Grand Prix': 28.93901233035419, 'Saudi Arabian Grand Prix':
                     26.776116268299933, 'Pre-Season Track Session': 18.25320197044335, 'Miami Grand Prix':
                     28.867351652727997, 'Pre-Season Testing': 21.473509933774835, 'Las Vegas Grand Prix': 18.05}

years = [2018, 2019, 2020, 2021, 2022, 2023]


def parse_races_avgtemp():
    result = {}
    for year in years:
        calendar = ff1.get_event_schedule(year)['EventName']
        for race in calendar:
            try:
                if race not in result.keys():
                        result[race] = []
                session = ff1.get_session(year, race, 'R')
                session.load()
                avg_temp = np.average(np.array(session.weather_data['AirTemp']))
                result[race].append(avg_temp)
            except:
                print(race, "in ", year, "is not found!")
    for race in result.keys():
        new_avg = np.average(np.array(result[race]))
        result[race] = new_avg
    print(result)


def main():
    # parse_races_avgtemp()
    session = ff1.get_session(2018, "albert park", 'R')



if __name__ == "__main__":
    main()

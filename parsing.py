import csv
import pandas as pd
import numpy as np
import fastf1 as ff1

# Globals
podium = {'1', '2', '3'}
podium_int = {1, 2, 3}

years = [2018, 2019, 2020, 2021, 2022, 2023]

races = ['albert park', 'bahrain', 'catalunya', 'istanbul', 'monaco', 'villeneuve',
         'silverstone', 'hockenheimring', 'hungaroring', 'spa', 'monza', 'marina bay',
         'shanghai', 'interlagos', 'suzuka', 'rodriguez', 'ricard', 'zandvoort',
         'las vegas', 'americas', 'red bull ring', 'sochi', 'baku', 'algarve',
         'mugello', 'jeddah', 'miami', 'lusail']

circ_to_num = {'albert park': 1,
               'australian grand prix': 1,
               'bahrain': 2,
               'bahrain grand prix': 2,
               'catalunya': 3,
               'spanish grand prix': 3,
               'istanbul': 4,
               'turkish grand prix': 4,
               'monaco': 5,
               'monaco grand prix': 5,
               'villeneuve': 6,
               'guillime villeneuve': 6,
               'canadian grand prix': 6,
               'silverstone': 7,
               'british grand prix': 7,
               'hockenheimring': 8,
               'german grand prix': 8,
               'hungaroring': 9,
               'hungarian grand prix': 9,
               'spa': 10,
               'belgian grand prix': 10,
               'monza': 11,
               'italian grand prix': 11,
               'marina bay': 12,
               'singapore grand prix': 12,
               'shanghai': 13,
               'chinese grand prix': 13,
               'interlagos': 14,
               'brazilian grand prix': 14,
               'suzuka': 15,
               'japanese grand prix': 15,
               'rodriguez': 16,
               'mexican grand prix': 16,
               'ricard': 17,
               'french grand prix': 17,
               'zandvoort': 18,
               'dutch grand prix': 18,
               'las vegas': 19,
               'las vegas grand prix': 19,
               'cota': 20,
               'circuit of the americas': 20,
               'united states grand prix': 20,
               'red bull ring': 21,
               'austrian grand prix': 21,
               'styrian grand prix': 21,
               'sochi': 22,
               'russian grand prix': 22,
               'baku': 23,
               'zzerbaijan grand prix': 23,
               'algarve': 24,
               'portugese grand prix': 24,
               'mugello': 25,
               'tuscan grand prix': 25,
               'jeddah': 26,
               'saudi arabia grand prix': 26,
               'miami': 27,
               'miami grand prix': 27,
               'lusail': 28,
               'qatar grand prix': 28,
               'yas_marina': 29,
               'abu dhabi grand prix': 29,
               'nurburgring': 30,
               'anniversary grand prix': 30,
               'imola': 31,
               'emilia romagna grand prix': 31}

driver_profs = {'HAM': [248, 1.7167630057803467], 'HEI': [13, 1.0706521739130435], 'ROS': [57, 1.2766990291262137],
                'ALO': [100, 1.2597402597402598], 'KOV': [4, 1.0357142857142858], 'NAK': [0, 1.0], 'BOU': [0, 1.0],
                'RAI': [114, 1.3131868131868132], 'KUB': [12, 1.12], 'GLO': [3, 1.0315789473684212],
                'SAT': [1, 1.010989010989011], 'PIQ': [1, 1.0357142857142858], 'MAS': [41, 1.151291512915129],
                'COU': [62, 1.2510121457489878], 'TRU': [11, 1.04296875], 'SUT': [0, 1.0],
                'WEB': [42, 1.1935483870967742], 'BUT': [50, 1.161812297734628], 'DAV': [0, 1.0],
                'VET': [122, 1.388535031847134], 'FIS': [19, 1.0822510822510822], 'BAR': [68, 1.2085889570552146],
                'SCH': [27, 1.15], 'LIU': [0, 1.0], 'WUR': [3, 1.0434782608695652], 'SPE': [0, 1.0], 'ALB': [0, 1.0],
                'WIN': [0, 1.0], 'YAM': [0, 1.0], 'MSC': [155, 1.4378531073446328], 'MON': [30, 1.3157894736842106],
                'KLI': [0, 1.0], 'TMO': [1, 1.027027027027027], 'IDE': [0, 1.0], 'VIL': [23, 1.1393939393939394],
                'FMO': [0, 1.0], 'DLR': [1, 1.0093457943925233], 'DOO': [0, 1.0], 'KAR': [0, 1.0], 'FRI': [0, 1.0],
                'ZON': [0, 1.0], 'PIZ': [0, 1.0], 'BAD': [0, 1.0],
                'MAG': [1, 1.0049751243781095], 'BUE': [0, 1.0], 'ALG': [0, 1.0], 'GRO': [10, 1.0518134715025906],
                'BIA': [1, 1.0188679245283019], 'GAS': [0, 1.0], 'SEN': [0, 1.0], 'HUL': [1, 1.0068027210884354],
                'DIG': [0, 1.0], 'CHA': [0, 1.0], 'DIR': [0, 1.0], 'DAM': [0, 1.0], 'MAL': [1, 1.0104166666666667],
                'PER': [30, 1.1176470588235294], 'RIC': [28, 1.1133603238866396], 'VER': [75, 1.29296875],
                'PIC': [0, 1.0], 'GUT': [0, 1.0], 'BOT': [66, 1.2796610169491525], 'CHI': [0, 1.0], 'VDG': [0, 1.0],
                'KVY': [3, 1.025], 'ERI': [0, 1.0], 'LOT': [0, 1.0], 'STE': [0, 1.0], 'NAS': [0, 1.0],
                'SAI': [21, 1.106060606060606], 'MER': [0, 1.0], 'RSS': [0, 1.0], 'PAL': [0, 1.0], 'WEH': [0, 1.0],
                'HAR': [0, 1.0], 'VAN': [0, 1.0], 'OCO': [1, 1.0069444444444444], 'GIO': [0, 1.0],
                'STR': [1, 1.0063694267515924], 'LEC': [56, 1.3916083916083917], 'SIR': [0, 1.0],
                'NOR': [20, 1.1739130434782608], 'RUS': [3, 1.0260869565217392], 'LAT': [0, 1.0], 'TSU': [0, 1.0],
                'MAZ': [0, 1.0], 'ZHO': [0, 1.0], 'PIA': [0, 1.0], 'DEV': [0, 1.0], 'SAR': [0, 1.0]}


def driver_proficiency(results):
    """
    Takes in a file path that contains a list of results. Parses the number
    of podiums for each unique driver and calculates a relative podium probability.

    N.B. a podium suggests a result of 1st, 2nd or 3rd.
    """
    drivers = pd.read_csv('archive/drivers_new.csv')
    profs = {}
    with open(results) as f:
        ref = csv.DictReader(f)
        for line in ref:
            cur_id = int(line['driverId'])
            cur_code = drivers.loc[cur_id - 1]['code']
            if cur_code not in profs.keys():
                profs[cur_code] = [0, 0]
            if line['position'] in podium:
                profs[cur_code][0] += 1
            profs[cur_code][1] += 1
    return profs


def driver_prof_ff1():
    result = driver_proficiency('archive/results.csv')
    for year in years:
        events = ff1.get_event_schedule(year, include_testing=False)
        for event in events:
            session = ff1.get_session(year, event, 'R')
            session.load()
            results = session.results
            for driver in results['Abbreviation']:
                if driver not in result.keys():
                    result[driver] = [0, 0]
                pos = int(results.loc[results['Abbreviation'] == driver]['Position'])
                if pos in podium_int:
                    result[driver][0] += 1
                result[driver][1] += 1
    for key in result.keys():
        result[key] = [result[key][0], result[key][0] / result[key][1] + 1]
    return result


def get_driver_proficiency(driver, coded):
    """

    :param driver:
    :return:
    """
    drivers = pd.read_csv('archive/drivers_new.csv')
    if not coded:
        first, last = driver.split(sep=None)[0], driver.split(sep=None)[1]
        code = ''
        try:
            code = np.array(drivers.loc[(drivers['surname'] == last) & (drivers['forename'] == first)]['code'])[0]
        except:
            print("Driver Code Not Found!")
    else:
        code = driver
    # prof = driver_prof_ff1()[code][1]
    prof = driver_profs[code][1]
    print('Proficiency of ', code, 'is: ', prof)
    return prof


def idx_to_diff(circuit_n, data_path="archive/circuits_full_stripped.csv"):
    """
    Yes
    """
    df = pd.read_csv(data_path)
    return df.loc[circuit_n]['score']


def get_circuit_difficulty(circuit):
    result = ''
    for char in circuit:
        if char.isalpha():
            result += char.lower()
        elif char == ' ':
            result += char
    if result in circ_to_num.keys():
        print('Difficulty of ', circuit, 'is: ',
              idx_to_diff(circ_to_num[result], data_path='archive/circuits_full_stripped.csv'))
        return idx_to_diff(circ_to_num[result], data_path='archive/circuits_full_stripped.csv')
    else:
        print('No circuit found with name ', circuit, '!')


def check_nat(x):
    return bool(x is not pd.NaT)


def parse_results(circuit):
    X = []
    Y_pit_int = []
    Y_pit_num = []
    for year in years:
        session = ff1.get_session(year, circuit, 'R')
        session.load()
        laps = session.laps
        cur_weather = session.weather_data
        hum, press, rain, tt, ws = np.average(cur_weather['Humidity']), np.average(cur_weather['Pressure']), \
            int(any(cur_weather['Rainfall'])), np.average(cur_weather['TrackTemp']), np.average(cur_weather['WindSpeed'])
        req_laps = session.total_laps
        for driver in session.drivers:
            print(driver)
            results = session.results
            driver_code = np.array(results.loc[results['DriverNumber'] == driver, 'Abbreviation'])[0]
            driver_laps = laps.pick_driver(driver)
            driver_pos = int(results.loc[results['DriverNumber'] == str(driver), 'Position'].iloc[0])
            if np.array(driver_laps).shape[0] != req_laps:
                print(driver_code, "did not complete race!")
                break
            try:
                proficiency = get_driver_proficiency(driver_code, coded=True)
            except:
                proficiency = 1
            difficulty = get_circuit_difficulty(circuit)
            result = np.array(driver_laps['PitInTime'].apply(check_nat)).astype(float)
            in_laps = np.array(driver_laps.loc[laps['PitInTime'].apply(check_nat)]).shape[0]
            X.append(np.array([((year - 2017) / 8), hum, press, rain, tt, ws, proficiency, difficulty, 1 / driver_pos]))
            Y_pit_int.append(result)
            Y_pit_num.append(int(in_laps))
    lengths = {}
    for data in Y_pit_int:
        length = data.size
        if length not in lengths:
            lengths[length] = 0
        lengths[length] += 1
    top_length = max(lengths, key=lengths.get)
    arglist = []
    for i in range(len(Y_pit_int)):
        if Y_pit_int[i].size == top_length:
            arglist.append(i)
    X_new, Y_pit_int_new, Y_pit_num_new = [X[_] for _ in arglist], [Y_pit_int[_] for _ in arglist], \
        [Y_pit_num[_] for _ in arglist]
    return[np.array(X_new), np.array(Y_pit_int_new), np.array(Y_pit_num_new)]


def prune_laps(predictions: np.array):
    n = len(predictions)
    result = []
    for i in range(n):
        if predictions[i] >= (n / 6) and n - predictions[i] >= (n / 6):
            result.append(predictions[i])
    return result


def process_laps(laps, threshold):
    for i in range(len(laps)):
        if laps[i] <= threshold:
            laps[i] = 0
        else:
            laps[i] = 1
    return laps


def parse_results_single(driver, circuit, year):
    X = []
    Y_pit_int = []
    session = ff1.get_session(year, circuit, 'R')
    session.load()
    laps = session.laps
    cur_weather = session.weather_data
    hum, press, rain, tt, ws = np.average(cur_weather['Humidity']), np.average(cur_weather['Pressure']), \
        int(any(cur_weather['Rainfall'])), np.average(cur_weather['TrackTemp']), np.average(cur_weather['WindSpeed'])
    req_laps = session.total_laps
    print(driver)
    results = session.results
    driver_num = np.array(results.loc[results['Abbreviation'] == driver, 'DriverNumber'])[0]
    driver_laps = laps.pick_driver(driver_num)
    driver_pos = int(results.loc[results['DriverNumber'] == str(driver_num), 'Position'].iloc[0])
    try:
        proficiency = get_driver_proficiency(driver, coded=True)
    except:
        proficiency = 1
    difficulty = get_circuit_difficulty(circuit)
    result = np.array(driver_laps['PitInTime'].apply(check_nat)).astype(float)
    X.append(np.array([((year - 2017) / 8), hum, press, rain, tt, ws, proficiency, difficulty, 1 / driver_pos]))
    Y_pit_int.append(result)

    return np.array(Y_pit_int)


def main():
    driver = 'Lewis Hamilton'
    circuit = 'Las Vegas'

    # print(driver_prof_ff1())
    # get_driver_proficiency(driver, coded=False)
    # get_circuit_difficulty(circuit)
    # print(parse_results('Australian Grand Prix'))]
    # print(parse_results_single('HAM', 'British Grand Prix', 2020))


if __name__ == '__main__':
    main()

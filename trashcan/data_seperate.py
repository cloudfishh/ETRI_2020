import os
import pandas as pd

dir_data = 'D:/2020_ETRI/data_'
list_loc = ['SG_data_광주_비식별화',
            'SG_data_나주_비식별화',
            'SG_data_대전_비식별화',
            'SG_data_서울_비식별화',
            'SG_data_인천_비식별화']

# list_apt = pd.read_csv(dir_data + '/SG사업단_아파트명단.csv', encoding='CP949', index_col=0)
# list_apt = list_apt.drop(list_apt.iloc[-1].name)

for loc in range(5):                                # location
    print('=== '+list_loc[loc]+' ===')

    dir_loc = dir_data + '/' + list_loc[loc]
    list_apt_raw = os.listdir(dir_loc)
    list_apt = [file for file in list_apt_raw if file.endswith('.csv')]

    print('   # of apartment: ' + str(len(list_apt)))

    for apt in range(len(list_apt)):              # apartment
        raw = pd.read_csv(dir_loc + '/' + list_apt[apt], index_col=0)
        print('   # of households: ' + str(len(raw.columns)-2))

        dir_new = 'D:/2020_ETRI/data_sy/' + list_loc[loc] + '/' + list_apt[apt][:10]
        if not os.path.isdir(dir_new):   # make apt folder
            os.mkdir(dir_new)

        for h in range(len(raw.columns)-2):
            temp = pd.DataFrame(raw.iloc[:, h+2], index=raw.index)
            temp.to_csv(dir_new + '/' + temp.columns[0]+'.csv')
        print('   ' + list_apt[apt] + 'saved successfully')
    print('===== ' + 'DONE' + ' =====\n')

print('EVERY HOUSEHOLDS SAVED IN CSV SEPARATELY')

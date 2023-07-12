import pickle

loaded_model_1 = pickle.load(open('models/MLmodels/lab_rep/lr_module_P1_Blood Sugar.sav', 'rb'))
loaded_model_2 = pickle.load(open('models/MLmodels/lab_rep/lr_module_P1_FastingPlasmaGlucoseLevel.sav', 'rb'))
loaded_model_3 = pickle.load(open('models/MLmodels/lab_rep/lr_module_P1_Total cholestrol.sav', 'rb'))
loaded_model_4 = pickle.load(open('models/MLmodels/lab_rep/lr_moduleP1_Total cholestrol.sav', 'rb'))
loaded_model_5 = pickle.load(open('models/MLmodels/lab_rep/lr_moduleP1_Total cholestrol.sav', 'rb'))
# loaded_model_5 = pickle.load(open('models/MLmodels/lab_rep/LabReportPrediction_gluecose', 'rb'))

def lab_report_precautions(results):
    results = results.split(',\n')
    for i in results:
        print(i)
        data = ""
        data = i.split(" ")
        print(data)
        count = 0
        all_data = {}
        for da in data:
            # print(da)
            count += 1
            if da == "Cholesterol" or da=="Normal\nCholesterol":
                all_data['Cholesterol'] = data[count+1]
            if da == "HDL":
                all_data['HDL'] = data[count]
            if da == "LDL":
                all_data['LDL'] = data[count]
            if da == "(TG)":
                all_data['(TG)'] =data[count]
            if da == "GLUCOSE" or da == "(PPBS)":
                all_data['GLUCOSE'] = data[count]
            if da == "Sugar":
                all_data['GLUCOSE'] = data[count+1]
                
    try:
        val1 = loaded_model_1.predict([[float(all_data['Cholesterol'])]])
    except:
        val1 = [0]
    try:
        val2 = loaded_model_2.predict([[float(all_data['HDL'])]])
    except:
        val2 = [0]
    try:
        val3 = loaded_model_3.predict([[float(all_data['LDL'])]])
    except:
        val3 = [0]
    try:
        val4 = loaded_model_4.predict([[float(all_data['(TG)'])]])
    except:
        val4 = [0]
    try:
        val5 = loaded_model_5.predict([[float(all_data['GLUCOSE'])]])
    except:
        val5 = [0]
        
    print(val1, val2, val3, val4, val5)

    return all_data, val1[0], val2[0], val3[0], val4[0], val5[0]

def main():
    results = "Cholesterol  200 mg/dl\nHDL  40 mg/dl\nLDL  100 mg/dl\n(TG)  100 mg/dl\nGLUCOSE  100 mg/dl"
    lab_report_precautions(results)

if __name__ == "__main__":
    main()
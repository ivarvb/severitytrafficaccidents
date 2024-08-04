__author__ = "Ivar Vargas Belizario and Liz Maribel Huancapaza Hilasaca"
__copyright__ = "Copyright 2024, registration in process"
__credits__ = ["Ivar Vargas Belizario", "Liz Maribel Huancapaza Hilasaca"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ivar Vargas Belizario"
__email__ = "ivargasbelizario@gmail.com"
__status__ = "development"


import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report 

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SMOTEN, BorderlineSMOTE, KMeansSMOTE 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import classification_report, confusion_matrix 
import seaborn as sns
from matplotlib import pyplot as plt

import pickle
import joblib
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from imblearn.over_sampling import ADASYN

def vis_confusion_matrix(cfm, cats, nme, oversami):

    labels = ["" for i in range(len(cats))]
    for k, label in cats.items():
        labels[k] = label
    
    fig, ax = plt.subplots(figsize=(4,3.5))
    sns.heatmap(cfm, xticklabels=labels, yticklabels=labels, annot=True, fmt='g', cmap='Greens', ax=ax)  
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix - '+nme+' - '+oversami)
    fig.savefig("./plots/"+nme+"_"+oversami+'.png', format='png', dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close()
    

def getdata():
    dfU = pd.read_csv('USUARIOS VULNERABLES 2021-2022.csv')
    dfU = dfU[["CÓDIGO SINIESTRO", "CÓDIGO VEHÍCULO","CLASE DE SINIESTRO","TIPO PERSONA","GRAVEDAD","EDAD","DIA","MES","SEXO","POSEE LICENCIA","ESTADO LICENCIA"]]
    dfU = dfU.dropna()
    # print(df.head()) 
    print(dfU) 


    dfV = pd.read_csv('VEHICULOS 2021-2022.csv')
    dfV = dfV[["CÓDIGO SINIESTRO","CÓDIGO VEHÍCULO","VEHÍCULO","ESTADO SOAT","ESTADO CITV"]]
    dfV = dfV.dropna()
    # print(df.head()) 
    print(dfV) 


    dfS = pd.read_csv('SINIESTROS 2021-2022.csv')
    dfS = dfS[["CÓDIGO SINIESTRO","FECHA SINIESTRO","HORA SINIESTRO","DEPARTAMENTO","PROVINCIA","DISTRITO","ZONA","RED VIAL","ZONIFICACIÓN","CARACTERÍSTICAS DE VÍA","PERFIL LONGITUDINAL VÍA","SUPERFICIE DE CALZADA","COORDENADAS LATITUD","COORDENADAS  LONGITUD","CONDICIÓN CLIMÁTICA"]]

    dfS = dfS.dropna()
    # print(df.head()) 
    print(dfS) 


    dfUV = pd.merge(dfU, dfV, on=['CÓDIGO SINIESTRO','CÓDIGO VEHÍCULO'])
    dfUVS = pd.merge(dfUV, dfS, on=['CÓDIGO SINIESTRO'])

    # print(dfUVS)

    rsdf = dfUVS[(dfUVS['GRAVEDAD'] == "FALLECIDO") | (dfUVS['GRAVEDAD'] == "LESIONADO") | (dfUVS['GRAVEDAD'] == "ILESO")] 
    rsdf['GRAVEDAD'] = np.where( (rsdf["GRAVEDAD"] == "ILESO") | (rsdf["GRAVEDAD"] == "LESIONADO"), "SOBREVIVIENTE", rsdf['GRAVEDAD'])

    rsdf = rsdf.drop(rsdf[rsdf['EDAD'] == "N/I"].index)
    # rsdf = rsdf.drop(rsdf[(rsdf['TIPO PERSONA'] != "CONDUCTOR") & (rsdf['TIPO PERSONA'] != "PASAJERO") ].index)
    rsdf = rsdf.drop(rsdf[(rsdf['TIPO PERSONA'] == "SIN CONDUCTOR") | (rsdf['TIPO PERSONA'] == "CONDUCTOR FUGADO")].index)

    rsdf["COORDENADAS LATITUD"] = rsdf["COORDENADAS LATITUD"].replace(regex={',': '.'}).astype(float)
    rsdf["COORDENADAS  LONGITUD"] = rsdf["COORDENADAS  LONGITUD"].replace(regex={',': '.'}).astype(float)

    rsdf['CONDICIÓN CLIMÁTICA'] = np.where( (rsdf["CONDICIÓN CLIMÁTICA"] == "DESPEJADO") | (rsdf["CONDICIÓN CLIMÁTICA"] == "SOLEADO"), "ILUMINADO", rsdf['CONDICIÓN CLIMÁTICA'])
    rsdf['CONDICIÓN CLIMÁTICA'] = np.where( (rsdf["CONDICIÓN CLIMÁTICA"] == "NUBLADO") | (rsdf["CONDICIÓN CLIMÁTICA"] == "LLUVIOSO") | (rsdf["CONDICIÓN CLIMÁTICA"] == "NIEBLA"), "OSCURO", rsdf['CONDICIÓN CLIMÁTICA'])


    rsdf['SUPERFICIE DE CALZADA'] = np.where( (rsdf["SUPERFICIE DE CALZADA"] == "TROCHA") |
            (rsdf["SUPERFICIE DE CALZADA"] == "AFIRMADO") |
            (rsdf["SUPERFICIE DE CALZADA"] == "CONCRETO") |
            (rsdf["SUPERFICIE DE CALZADA"] == "CASCAJO/RIPIO") |
            (rsdf["SUPERFICIE DE CALZADA"] == "EMPEDRADO") |
            (rsdf["SUPERFICIE DE CALZADA"] == "ADOQUINADO"), "NO ASFALTADA",
            rsdf['SUPERFICIE DE CALZADA'])


    rsdf = rsdf.drop(rsdf[rsdf['CARACTERÍSTICAS DE VÍA'] == "OTRO"].index)
    rsdf['CARACTERÍSTICAS DE VÍA'] = np.where( (rsdf["CARACTERÍSTICAS DE VÍA"] == "CURVA") |
            (rsdf["CARACTERÍSTICAS DE VÍA"] == "INTERSECCIÓN") |
            # (rsdf["CARACTERÍSTICAS DE VÍA"] == "OTRO") |
            (rsdf["CARACTERÍSTICAS DE VÍA"] == "SINUOSA") |
            (rsdf["CARACTERÍSTICAS DE VÍA"] == "PASE A DESNIVEL") |
            (rsdf["CARACTERÍSTICAS DE VÍA"] == "PUENTE") |
            (rsdf["CARACTERÍSTICAS DE VÍA"] == "ÓVALO") |
            (rsdf["CARACTERÍSTICAS DE VÍA"] == "TÚNEL"), "IRREGULAR", rsdf['CARACTERÍSTICAS DE VÍA'])



    rsdf['RED VIAL'] = np.where( (rsdf["RED VIAL"] == "URBANO") |
            (rsdf["RED VIAL"] == "VECINAL") |
            (rsdf["RED VIAL"] == "DEPARTAMENTAL") |
            (rsdf["RED VIAL"] == "SIN CLASIFICAR"), "DEPARTAMENTAL",
            rsdf['RED VIAL'])


    # rsdf['HORA SINIESTRO'] = pd.to_datetime(rsdf['HORA SINIESTRO'], format='%H:%M').dt.hour
    rsdf["HORA SINIESTRO"] = rsdf["HORA SINIESTRO"].str.split(":").str.get(0).astype(int)

    rsdf["DIANOCHE"] = "category"
    rsdf['DIANOCHE'] = np.where( (rsdf["HORA SINIESTRO"] < 12), "DIA", rsdf['DIANOCHE'])
    rsdf['DIANOCHE'] = np.where( (rsdf["HORA SINIESTRO"] >= 12) & (rsdf["HORA SINIESTRO"] < 18), "TARDE", rsdf['DIANOCHE'])
    rsdf['DIANOCHE'] = np.where( (rsdf["HORA SINIESTRO"] >= 18), "NOCHE", rsdf['DIANOCHE'])
    # rsdf = rsdf.drop(rsdf[rsdf['DIANOCHE'] == "category"].index)


    # rsdf["EDAD"] = rsdf["EDAD"].str.get(0).astype(int)
    rsdf['EDAD'] = rsdf['EDAD'].astype('int')

    rsdf["JOVENADULTO"] = "category"
    # rsdf['JOVENADULTO'] = np.where( (rsdf["EDAD"] < 15), "C1", rsdf['JOVENADULTO'])
    # rsdf['JOVENADULTO'] = np.where( (rsdf["EDAD"] >= 15) & (rsdf["EDAD"] < 30), "C2", rsdf['JOVENADULTO'])
    # rsdf['JOVENADULTO'] = np.where( (rsdf["EDAD"] >= 30) & (rsdf["EDAD"] < 40), "C3", rsdf['JOVENADULTO'])
    # rsdf['JOVENADULTO'] = np.where( (rsdf["EDAD"] >= 40) & (rsdf["EDAD"] < 50), "C3", rsdf['JOVENADULTO'])
    # rsdf['JOVENADULTO'] = np.where( (rsdf["EDAD"] >= 50), "C4", rsdf['JOVENADULTO'])
    rsdf['JOVENADULTO'] = np.where( (rsdf["EDAD"] < 40), "C1", rsdf['JOVENADULTO'])
    rsdf['JOVENADULTO'] = np.where( (rsdf["EDAD"] >= 40), "C2", rsdf['JOVENADULTO'])


    rsdf["COORDENADAS LATITUD"] = rsdf["COORDENADAS LATITUD"].astype('float')
    rsdf["COORDENADAS  LONGITUD"] = rsdf["COORDENADAS  LONGITUD"].astype('float')

    rsdf['ZONIFICACIÓN'] = np.where( (rsdf["ZONIFICACIÓN"] == "COMERCIAL") |
                (rsdf["ZONIFICACIÓN"] == "RESIDENCIAL") |
                (rsdf["ZONIFICACIÓN"] == "INDUSTRIAL") |
                (rsdf["ZONIFICACIÓN"] == "ESCOLAR"), "URBANA",
                 rsdf['ZONIFICACIÓN'])




    rsdf = rsdf.drop(rsdf[rsdf['VEHÍCULO'] == "VEHÍCULO NO IDENTIFICADO"].index)
    rsdf = rsdf.drop(rsdf[rsdf['VEHÍCULO'] == "OTRO"].index)

    # rsdf['VEHÍCULO'] = np.where( (rsdf["VEHÍCULO"] == "MOTOCICLETA") 
    #                     | (rsdf["VEHÍCULO"] == "TRIMOTO PASAJERO") 
    #                     | (rsdf["VEHÍCULO"] == "TRIMOTO PASAJERO")
    #                     | (rsdf["VEHÍCULO"] == "TRIMOTO CARGA")
    #                     | (rsdf["VEHÍCULO"] == "BICICLETA")
    #                     | (rsdf["VEHÍCULO"] == "TRICICLO NO MOTORIZADO")
    #                     | (rsdf["VEHÍCULO"] == "TRICICLO MOTORIZADO")
    #                     | (rsdf["VEHÍCULO"] == "TRICICLO MOTORIZADO")
    #                     | (rsdf["VEHÍCULO"] == "TRICICLO MOTORIZADO")
    #                     | (rsdf["VEHÍCULO"] == "TRICICLO MOTORIZADO"),
    #                     "C1", rsdf['VEHÍCULO'])

    # rsdf['VEHÍCULO'] = np.where( (rsdf["VEHÍCULO"] == "AUTOMÓVIL") 
    #                     | (rsdf["VEHÍCULO"] == "CAMIONETA RURAL") 
    #                     | (rsdf["VEHÍCULO"] == "CAMIÓN") 
    #                     | (rsdf["VEHÍCULO"] == "CAMIONETA PICK UP") 
    #                     | (rsdf["VEHÍCULO"] == "ÓMNIBUS") 
    #                     | (rsdf["VEHÍCULO"] == "REMOLCADOR") 
    #                     | (rsdf["VEHÍCULO"] == "REMOLCADOR - SEMIRREMOLQUE") 
    #                     | (rsdf["VEHÍCULO"] == "STATION WAGON") 
    #                     | (rsdf["VEHÍCULO"] == "MINIBUS") 
    #                     | (rsdf["VEHÍCULO"] == "CAMIONETA PANEL") 
    #                     | (rsdf["VEHÍCULO"] == "SEMIRREMOLQUE"), 
    #                     "C2", rsdf['VEHÍCULO'])


    rsdf["ISWEEKEND"] = 'category'
    rsdf["ISWEEKEND"] = rsdf["DIA"].isin(['VIERNES', 'SÁBADO', 'DOMINGO'])



    # print(rsdf["GRAVEDAD"].value_counts())
    # print(rsdf["CONDICIÓN CLIMÁTICA"].value_counts())
    # print(rsdf["CARACTERÍSTICAS DE VÍA"].value_counts())
    # print(rsdf["SUPERFICIE DE CALZADA"].value_counts())
    # print(rsdf["RED VIAL"].value_counts())
    # print(rsdf["ZONIFICACIÓN"].value_counts())
    # print(rsdf["ESTADO SOAT"].value_counts())


    chooseus = ["GRAVEDAD","COORDENADAS LATITUD","COORDENADAS  LONGITUD", "EDAD"]


    choosefe = ["TIPO PERSONA","EDAD","DIA","MES","SEXO",
    # "POSEE LICENCIA",
    # "ESTADO LICENCIA",
    # "ESTADO SOAT",
    # "ESTADO CITV",
    "VEHÍCULO",
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
    # "ZONIFICACIÓN",
    # "ZONA",
    "RED VIAL",
    "CARACTERÍSTICAS DE VÍA","PERFIL LONGITUDINAL VÍA",
    "SUPERFICIE DE CALZADA","CONDICIÓN CLIMÁTICA","HORA SINIESTRO",
    "DIANOCHE",
    "JOVENADULTO",
    "COORDENADAS LATITUD","COORDENADAS  LONGITUD",
    "ISWEEKEND",
    "GRAVEDAD"]
    # ,"JOVENADULTO", "COORDENADAS LATITUD","COORDENADAS  LONGITUD"
    rsdf = rsdf[choosefe]

    cat_columns = rsdf.select_dtypes(['object']).columns
    rsdf[cat_columns] = rsdf[cat_columns].astype('category')


    rsdf.to_csv("dataset/data_untransformed.csv", index=False)

    for col in cat_columns:
        print(rsdf[col].value_counts())

    for col in cat_columns:
        if not col in chooseus:
            rsdf[col] = rsdf[col].cat.codes
    print(rsdf)

    rsdf.to_csv("dataset/data.csv", index=False)

    rsdf = pd.read_csv("dataset/data.csv")



    X = rsdf.drop(['GRAVEDAD'], axis=1)
    y = rsdf["GRAVEDAD"]

    # categorical to number
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    label_mapping = {index:label for index, label in enumerate(encoder.classes_)}
    print(label_mapping)
    
    return X, y, label_mapping

def printresults(ml, oversami, method, y_test, rfc_pred_test, labelmaping):

    print(method)

    filename = "./models/"+method+"_"+oversami+'.h5'
    pickle.dump(ml, open(filename, 'wb')) 
    # joblib.dump(ml, open(filename, 'wb')) 

    acc = balanced_accuracy_score(y_test, rfc_pred_test)
    f1 =f1_score(y_test, rfc_pred_test)
    pre = precision_score(y_test, rfc_pred_test)
    rec = recall_score(y_test, rfc_pred_test)

    acc = np.round(acc, 4)
    f1 = np.round(f1, 4)
    pre = np.round(pre, 4)
    rec = np.round(rec, 4)
    print(f'accuracy: {acc}')
    print(f'pre: {pre}')
    print(f'rec: {rec}')
    print(f'f1: {f1}')

    print(classification_report(y_test, rfc_pred_test))

    # Confusion matrix
    #cfm = confusion_matrix(y_test_true, y_test_pred, normalize='true')
    cfm = confusion_matrix(y_test, rfc_pred_test)
    #cfm = np.around(cfm, decimals=2)
    vis_confusion_matrix(cfm, labelmaping, method, oversami)

    
# make and read the new dataset
X, y, labelmaping = getdata()

# spliting dataset, train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# --- BEGIN BALANCED
### random over sampling to minoriritary classes
sm = None
# oversami = "SMOTE"
# oversami = "SMOTEN"
# oversami = "RANDOM"
oversami = "BorderlineSMOTE"
## oversami = "ADASYN"


print("tech:", oversami)
if oversami=="SMOTE":
    sm = SMOTE(random_state=42,  k_neighbors=5)
if oversami=="SMOTEN":
    sm = SMOTEN(sampling_strategy = 'minority', random_state=42)
elif oversami=="RANDOM":
    sm = RandomOverSampler(random_state=42)
elif oversami=="BorderlineSMOTE":
    sm = BorderlineSMOTE(random_state=42)
# elif oversami=="ADASYN":
#     sm = ADASYN(random_state=42)


X_train, y_train = sm.fit_resample(X_train, y_train)
X_test, y_test = sm.fit_resample(X_test, y_test)
print("End balanced")
# --- END BALANCED



# train and test tast to each models (classifier)

############### GBC ###############
model = xgb.XGBClassifier(
    max_depth=15,
    learning_rate=0.0001,
    subsample=0.5,
    n_estimators=70)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
printresults(model, oversami, "GBC", y_test, y_pred_test, labelmaping)


############### RFC ###############
rfc = RandomForestClassifier(
        criterion='entropy',
        max_depth=10,
        max_features=8,
        n_estimators=150,
        random_state=42,
        )
rfc.fit(X_train, y_train)
y_pred_test = rfc.predict(X_test)
printresults(rfc, oversami, "RFC", y_test, y_pred_test, labelmaping)


# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_validation = scaler.fit_transform(X_validation)
X_test = scaler.transform(X_test)



############### KNNC ###############

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
printresults(model, oversami, "KNNC", y_test, y_pred_test, labelmaping)


############### SVMC ###############
clf = SVC(C=20, gamma=0.001, kernel='rbf')
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
printresults(clf, oversami, "SVMC", y_test, y_pred_test, labelmaping)


############### MLPC ###############
clf = MLPClassifier(
                        hidden_layer_sizes=(4),
                        max_iter = 2000,activation='relu',
                        solver = 'lbfgs',
                        alpha=0.0001,
                        random_state=42
                        )
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
printresults(clf, oversami, "MLPC", y_test, y_pred_test, labelmaping)



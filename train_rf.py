""" 
Implementation of a RandomForest classifier to discriminate ttW 
---------------------------------------------------------------

This code is designed to work using RDataFrame snapshots made
with CMGRDF framework. It assummes that most of the postprocessing
part comes in a preprocessing step (which is much easier to do in RDF).

The code below is quite hardcoded on purpose, to make it easier for new
students to follow.
""" 
# -- Import libraries 
import ROOT
from ROOT.VecOps import RVec
from ROOT import gROOT as grt 

import uproot
import pandas
import numpy as np
import matplotlib.pyplot as plt

from optparse import OptionParser
from copy import deepcopy

# -- ML libraries 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV


import seaborn as sns

ROOT.EnableImplicitMT(1)

def add_parsing_opts():
    """ Function with base parsing arguments used by any script """
    parser = OptionParser(usage = "python3 run_ttw_run3.py [options]",
                                    description = "Main options for running ttW analysis.")

    parser.add_option("--inpath", dest = "inpath", default = None,
                help = "Path to samples. ")
    parser.add_option("--hyper", dest = "hyper", default = False,
                help = "Path to samples. ")
    return parser.parse_args()

def load_data( 
        filename,
        vars_for_train
    ):   
    
    """
    This function returns a pandas dataframe where each row corresponds to
    an event, and each column is a variable.
    """

    # -- Load the main tree
    # Convert to pandas DataFrame
    tfile = uproot.open( filename )
    events = tfile.get( "Events" )
    
    #print(events.arrays( vars_for_train ) )
    # -- Convert to pandas dataframe
    df_events = pandas.DataFrame( events.arrays( vars_for_train, library = "pd" ) ) 

    tfile.close()
    return df_events 



if __name__ == "__main__":
    opts, args = add_parsing_opts()
    
    inpath = opts.inpath
    
    
    
    TTLNu_1Jets_name = "TTLNu_1Jets"
    TTLNu_EWK_name = "TTLNu_EWK"
    TTHtoNon2B_name = "TTHtoNon2B"
    
    # Define the variables that will be actually used in the training
    vars_for_train = [
        "LepFO1_conePt", "LepFO1_eta", "LepFO1_phi",
        "LepFO2_conePt", "LepFO2_eta", "LepFO2_phi",
        "Jet0_pt", "Jet0_eta", "Jet0_phi",
        "Jet1_pt", "Jet1_eta", "Jet1_phi",
        #"nJet25", "nBJetMedium25", "nBJetLoose25",
        ]
    datasets = {
        "2022": ["22", "2022"],
        "2022EE": ["22EE", "2022EE"],
        "2023": ["23", "2023"],
        "2023BPix": ["23BPix", "2023BPix"]
        }

    samples = {
        "ttlnu_qcd": TTLNu_1Jets_name,
        "ttlnu_ewk": TTLNu_EWK_name,
        "tthtonon2b": TTHtoNon2B_name
        }

    is_signal_map = {
        "ttlnu_qcd": 1,
        "ttlnu_ewk": 1,
        "tthtonon2b": 0
        }
    loaded_data = {}
    for year_key, (suffix, path) in datasets.items():
        for sample_key, sample_name in samples.items():
            var_name = f"{sample_key}_{suffix}"
            file_path = f"{inpath}/{path}/{sample_name}.root"
            loaded_data[var_name] = load_data(file_path, vars_for_train)
            loaded_data[var_name]["is_signal"] = is_signal_map[sample_key]
            print(f"{var_name}: {len(loaded_data[var_name])} filas")
    # Create the dataframes
    #ttlnu_qcd      = load_data( f"{inpath}/{TTLNu_1Jets_name}.root", vars_for_train )
    #ttlnu_ewk      = load_data( f"{inpath}/{TTLNu_EWK_name}.root", vars_for_train )
    #tthtonon2b     = load_data( f"{inpath}/{TTHtoNon2B_name}.root", vars_for_train )
    training_data = pandas.concat([data for key, data in loaded_data.items() if "tthtonon2b" in key or "ttlnu" in key],
                                  axis=0, ignore_index=False)
    print(len(training_data))

    #print(training_data)
    print("total",len(training_data[training_data.isna()==True]))  # Conteo de NaNs por columna
    nan_rows = training_data[training_data.isna().any(axis=1)]
    print(nan_rows)
    print("===============================")
    
    #Here is the code for the training
    #Initial split--> 70% train - 30% test
    x_train, x_test, y_train, y_test = train_test_split(training_data[vars_for_train], training_data['is_signal'], test_size=0.3, random_state=1)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    #print(x_train.isna().sum())  # Conteo de NaNs por columna
    #Call to the RF and score
    #RF=RandomForestClassifier(n_jobs=-1, random_state=42)
    RF=RandomForestClassifier(class_weight='balanced', n_jobs=-1,
                              random_state=42, max_depth=3,
                              min_samples_leaf=5, min_samples_split=2, n_estimators=1000)
    RF.fit(x_train[vars_for_train], y_train)  #Trainig of the RF

    if opts.hyper:
        param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'class_weight': [None, 'balanced']
        }   
        grid_search = GridSearchCV(RF, param_grid, cv=5, scoring='roc_auc', verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Mejores hiperparámetros: {grid_search.best_params_}")
        print(f"Mejor AUC en validación: {grid_search.best_score_:.4f}")

    probs_test = RF.predict_proba(x_test[vars_for_train])
    probs_train = RF.predict_proba(x_train[vars_for_train])

    train_predictions = RF.predict(x_train)
    test_predictions = RF.predict(x_test)

    AUC_test = roc_auc_score(y_test, probs_test[:, 1])
    AUC_train = roc_auc_score(y_train, probs_train[:, 1])

    print(f"Area Under Curve (AUC) test = {AUC_test:.4f}")
    print(f"Area Under Curve (AUC) train = {AUC_train:.4f}")

    fpr_test, tpr_test, _ = roc_curve(y_test, probs_test[:,1])
    fpr_train, tpr_train, _ = roc_curve(y_train, probs_train[:,1])

    plt.figure(figsize=(7,7))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.plot(fpr_test, tpr_test, color='crimson', lw=2, label='ROC curve test (area = {0:.4f})'.format(AUC_test))
    plt.plot(fpr_train, tpr_train, color='blue', lw=1, label='ROC curve train (area = {0:.4f})'.format(AUC_train))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
    name="ttW_RF"
    plt.savefig('training/%s_ROCcurve_CMS.pdf'%name)
    plt.savefig('training/%s_ROCcurve_CMS.png'%name)

    #Probabilities
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    back_test = probs_test[:, 1][y_test == 0]
    sign_test = probs_test[:, 1][y_test == 1]
    back_train = probs_train[:, 1][y_train == 0]
    sign_train = probs_train[:, 1][y_train == 1]
    plt.hist(back_test, 20, color='blue', edgecolor='blue', lw=2, label='Background (test)', alpha=0.3, density=True)
    plt.hist(sign_test, 20, color='red', edgecolor='red', lw=2, label='Signal (test)', alpha=0.3, density=True)
    plt.hist(back_train, 20, color='green',histtype='step', lw=2, label='Background (train)', density=True)
    plt.hist(sign_train, 20, color='brown',histtype='step', lw=2, label='Signal (train)',  density=True)
    plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
    plt.xlim([0, 1])
    plt.xlabel('Event probability of being classified as signal')
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig('training/%s_probs.pdf'%name)
    plt.savefig('training/%s_probs.png'%name)
    
    df_corr = training_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, cmap='coolwarm', annot=True, fmt='.2f', annot_kws={"size": 8})
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Mapa de correlacion')
    plt.savefig('training/heatmap.png')


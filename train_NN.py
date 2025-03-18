""" 
Implementation of a Neural Network classifier to discriminate ttW 
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


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
    parser.add_option("--multi", dest = "multi", default= False,
                help = "If you want to do a multiclassifier. " )
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
   
    # -- Convert to pandas dataframe
    df_events = pandas.DataFrame( events.arrays( vars_for_train, library = "pd" ) ) 

    tfile.close()
    return df_events 



if __name__ == "__main__":
    opts, args = add_parsing_opts()
    opts.multi == opts.multi == "True"
    inpath = opts.inpath

    TTLNu_1Jets_name = "TTLNu_1Jets"
    TTLNu_EWK_name = "TTLNu_EWK"
    TTHtoNon2B_name = "TTHtoNon2B"
    #WZto2L2Q_name   = "WZto2L2Q"
    TTtoLNu2Q_name = "TTtoLNu2Q"
    # Define the variables that will be actually used in the training
    vars_for_train = [
        "LepFO1_conePt", "LepFO1_eta", "LepFO1_phi",
        "LepFO2_conePt", "LepFO2_eta", "LepFO2_phi",
        "Jet0_pt", "Jet0_eta", "Jet0_phi", "Jet0_scoreL", "Jet0_scoreM",
        "Jet1_pt", "Jet1_eta", "Jet1_phi", "Jet1_scoreL", "Jet1_scoreM",
        "Jet2_pt", "Jet2_eta", "Jet2_phi", "Jet2_scoreL", "Jet2_scoreM",
        "Jet3_pt", "Jet3_eta", "Jet3_phi", "Jet3_scoreL", "Jet3_scoreM",
        "Jet4_pt", "Jet4_eta", "Jet4_phi", "Jet4_scoreL", "Jet4_scoreM",
        "Jet5_pt", "Jet5_eta", "Jet5_phi", "Jet5_scoreL", "Jet5_scoreM",
        #"Jet6_pt", "Jet6_eta", "Jet6_phi", "Jet6_scoreL", "Jet6_scoreM",
        "nJet25", "nBJetMedium25", "nBJetLoose25",
        "mll", "ht",
        "PuppiMET_pt", "PuppiMET_phi"
        ]
    datasets = {
        "2022": ["22", "2022"],
        "2022EE": ["22EE", "2022EE"],
        "2023": ["23", "2023"],
        "2023BPix": ["23BPix", "2023BPix"]
        }

    samples = {
        "ttlnu_qcd" : TTLNu_1Jets_name,
        "ttlnu_ewk" : TTLNu_EWK_name,
        "tthtonon2b": TTHtoNon2B_name,
        "ttbarsemi" : TTtoLNu2Q_name,
        }
    is_signal_map = {
        "ttlnu_qcd" : 1,
        "ttlnu_ewk" : 1,
        "tthtonon2b": 0,
        "ttbarsemi" : 0,
        }
    class_map = {
        "ttlnu_qcd" : 0,
        "ttlnu_ewk" : 0,
        "tthtonon2b": 1,
        "ttbarsemi" : 2,
    }
    loaded_data = {}
    for year_key, (suffix, path) in datasets.items():
        for sample_key, sample_name in samples.items():
            var_name = f"{sample_key}_{suffix}"
            file_path = f"{inpath}/{path}/{sample_name}.root"
            loaded_data[var_name] = load_data(file_path, vars_for_train)
            loaded_data[var_name]["is_signal"] = is_signal_map[sample_key]
            loaded_data[var_name]["class"] = class_map[sample_key]
            #Reduce background stat (balanced with the signal)
            if sample_key == "tthtonon2b" or sample_key == "ttbarsemi" and len(loaded_data[var_name]) > 17000:
                loaded_data[var_name] = loaded_data[var_name][:17000]
            ##
            print(f"{var_name}: {len(loaded_data[var_name])} filas")
    # Create the dataframes
    if opts.multi:
        training_data = pandas.concat([data for key, data in loaded_data.items() if "ttbarsemi" in key or "ttlnu" in key or "tthtonon2b" in key],
                                      axis=0, ignore_index=False)
        x_train, x_test, y_train, y_test = train_test_split(training_data[vars_for_train], training_data['class'], test_size=0.3, random_state=1)
    else:
        training_data = pandas.concat([data for key, data in loaded_data.items() if "ttbarsemi" in key or "ttlnu" in key],
        #training_data = pandas.concat([data for key, data in loaded_data.items() if "tthtonon2b" in key or "ttlnu" in key],
                                      axis=0, ignore_index=False)
        training_data['class'] = training_data['class'].apply(lambda x: 1 if x == "ttlnu_qcd" else 0)
        x_train, x_test, y_train, y_test = train_test_split(training_data[vars_for_train], training_data['is_signal'], test_size=0.3, random_state=1)

    #Codify the labels of the classes
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    #This is my NN def
    model = keras.models.Sequential()
    #normal_ini=keras.initializers.glorot_normal(seed=None)
    model.add(keras.layers.Dense(64,  input_shape = (len(vars_for_train),), activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))

    n_classes = len(le.classes_)
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    model.summary()

    #sgd = keras.optimizers.SGD(learning_rate=0.01)
    adam=keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ["accuracy"])
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])
    #Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)


    histObj = model.fit(x_train[vars_for_train],
                        keras.utils.to_categorical(y_train, num_classes=n_classes),
                        epochs=30,
                        batch_size=128,
                        validation_data=(x_test, keras.utils.to_categorical(y_test, num_classes=n_classes)),
                        callbacks=[early_stopping, model_checkpoint],
                        )

    probs_test  = model.predict(x_test[vars_for_train])
    probs_train = model.predict(x_train[vars_for_train])
    train_predictions = np.argmax(probs_train, axis=1)
    n_classes = probs_test.shape[1]
    colors = ['purple', 'green', 'red']
    if opts.multi:
        plt.figure(figsize=(7, 7))
        plt.rcParams.update({'font.size': 15})
        for i, name in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_test == i, probs_test[:, i]) #this is ovr
            roc_auc = auc(fpr, tpr)
            print("el otro auc", roc_auc)
            name = "ttW" if i == 0 else "ttH" if i == 1 else "ttbar"
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC curve {name}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
        name = "ttW_NN"
        plt.savefig(f'trainingNN/{name}_ROCcurve_CMS.pdf')
        plt.savefig(f'trainingNN/{name}_ROCcurve_CMS.png')
    if not opts.multi:
        plt.figure(figsize=(7, 7))
        plt.rcParams.update({'font.size': 15})
        fpr, tpr, _ = roc_curve(y_test, probs_test[:, 1])  # Only two classes (ttlnu vs tth)
        roc_auc = auc(fpr, tpr)
        print("AUC:", roc_auc)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve ttlnu vs tth (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
        plt.savefig(f'trainingNN/ttlnu_vs_tth_ROCcurve_CMS.pdf')
        plt.savefig(f'trainingNN/ttlnu_vs_tth_ROCcurve_CMS.png')

    if opts.multi:
        plt.figure(figsize=(10,5))
        plt.rcParams.update({'font.size': 15}) #Larger font size
        back_test = probs_test[:, 0][y_test != 0]
        sign_test = probs_test[:, 0][y_test == 0]
        back_train = probs_train[:, 0][y_train != 0]
        sign_train = probs_train[:, 0][y_train == 0]
        plt.hist(back_test, 20, color='blue', edgecolor='blue', lw=2, label='Background (test)', alpha=0.3, density=True)
        plt.hist(sign_test, 20, color='red', edgecolor='red', lw=2, label='Signal (test)', alpha=0.3, density=True)
        plt.hist(back_train, 20, color='green',histtype='step', lw=2, label='Background (train)', density=True)
        plt.hist(sign_train, 20, color='brown',histtype='step', lw=2, label='Signal (train)',  density=True)
        plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
        plt.xlim([0, 1])
        plt.xlabel('Event probability of being classified as signal')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.savefig('trainingNN/%s_probs.pdf'%name)
        plt.savefig('trainingNN/%s_probs.png'%name)
    if not opts.multi:
        plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 15})
        back_test = probs_test[:, 1][y_test == 0]  # Background (tth)
        sign_test = probs_test[:, 1][y_test == 1]  # Signal (ttlnu)
        back_train = probs_train[:, 1][y_train == 0]  # Background (tth)
        sign_train = probs_train[:, 1][y_train == 1]  # Signal (ttlnu)

        plt.hist(back_test, 20, color='blue', edgecolor='blue', lw=2, label='Background (test)', alpha=0.3, density=True)
        plt.hist(sign_test, 20, color='red', edgecolor='red', lw=2, label='Signal (test)', alpha=0.3, density=True)
        plt.hist(back_train, 20, color='green', histtype='step', lw=2, label='Background (train)', density=True)
        plt.hist(sign_train, 20, color='brown', histtype='step', lw=2, label='Signal (train)', density=True)
        plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
        plt.xlim([0, 1])
        plt.xlabel('Event probability of being classified as signal')
        plt.legend(loc="upper left")
        plt.grid(True)
        print("Llego aqui")
        plt.savefig('trainingNN/ttlnu_vs_tth_probs.pdf')
        plt.savefig('trainingNN/ttlnu_vs_tth_probs.png')
    

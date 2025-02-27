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
from optparse import OptionParser
from copy import deepcopy

# -- ML libraries 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

ROOT.EnableImplicitMT(1)

def add_parsing_opts():
    """ Function with base parsing arguments used by any script """
    parser = OptionParser(usage = "python3 run_ttw_run3.py [options]",
                                    description = "Main options for running ttW analysis.")

    parser.add_option("--inpath", dest = "inpath", default = None,
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
    
    print(events.arrays( vars_for_train ) )
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
        "LepFO1_conePt"    
    ]
    
    # Create the dataframes
    ttlnu_qcd = load_data( f"{inpath}/{TTLNu_1Jets_name}.root", vars_for_train )
    print(ttlnu_qcd)

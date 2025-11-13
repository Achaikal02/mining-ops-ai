import pandas as pd

def load_config():
    config = {}
    config['financial_params'] = pd.read_csv('parameter_finansial.csv').set_index('Parameter')['Nilai'].to_dict()
    config['trucks'] = pd.read_csv('armada_truck.csv')
    config['excavators'] = pd.read_csv('armada_excavator.csv')
    config['routes'] = pd.read_csv('rute.csv')
    config['variability_rules'] = pd.read_csv('aturan_variabilitas.csv')
    config['breakdown_rules'] = pd.read_csv('aturan_breakdown.csv')
    
    print("Konfigurasi dimuat.")
    return config

# Muat konfigurasi di awal
CONFIG = load_config()
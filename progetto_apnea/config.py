import os
import platform

def get_base_path():
    system = platform.system()
    if system == 'Windows':
        return r'C:\Windows\System32\progetto_apnea'
    elif system == 'Darwin':  # MacOS
        return os.path.expanduser('~/progetto_apnea')
    else:
        raise OSError(f"Sistema operativo {system} non supportato")

def get_data_path():
    data_path = os.path.join(get_base_path(), 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path

def get_models_path():
    models_path = os.path.join(get_base_path(), 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    return models_path

def get_raw_data_path():
    raw_path = os.path.join(get_data_path(), 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    return raw_path

def get_processed_data_path():
    processed_path = os.path.join(get_data_path(), 'processed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    return processed_path
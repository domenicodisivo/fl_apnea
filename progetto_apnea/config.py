import os
import platform

def get_base_path():
    system = platform.system()
    if system == 'Windows':
        return r'c:\Users\domen\OneDrive\Desktop\Template\apps\progetto_apnea'
    elif system == 'Darwin':  # MacOS
        return os.path.expanduser('~/Desktop/Template/apps/progetto_apnea')
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
    # Force creation of raw directory
    os.makedirs(raw_path, exist_ok=True)
    return raw_path

def get_processed_data_path():
    processed_path = os.path.join(get_data_path(), 'processed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    return processed_path

def get_figures_path():
    figures_path = os.path.join(get_base_path(), 'figures')
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    return figures_path

def get_logs_path():
    logs_path = os.path.join(get_base_path(), 'logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    return logs_path

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    print("Creating directories...")
    print(f"Raw data path: {get_raw_data_path()}")
    print(f"Data path: {get_data_path()}")
    print(f"Models path: {get_models_path()}")
    print(f"Processed path: {get_processed_data_path()}")
    print(f"Figures path: {get_figures_path()}")
    print(f"Logs path: {get_logs_path()}")
    print("Done!")
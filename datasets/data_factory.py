from datasets.simulated import load_simulated_data

def source_select(cfg):
    data_type = cfg['data_type']
    if data_type == 'simulated':
        return load_simulated_data
    #You can add your own datasets here
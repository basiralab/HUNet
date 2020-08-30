from models.models import *

def model_select(activate_model):
    model_dict = {
                  'HUNET':HUNET
                    #You can add additional models here
                  }
    return model_dict[activate_model]
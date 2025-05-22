import os
import sys
sys.path.append(os.path.abspath("util"))

import util.img_util as util

def path_finder():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_path_to_data = os.path.join(current_directory, './data')
    data_folder_path = os.path.normpath(relative_path_to_data)
    return data_folder_path

test = util.ImageDataLoader(path_finder())

test.__iter__()


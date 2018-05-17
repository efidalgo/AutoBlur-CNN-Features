# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:08:41 2018

@author: Eduardo Fidalgo Fernández (efidf@unileon.es)
Functions to be used during Machine Learning experimentation

"""

#%%
def check_if_directory_exists(name_folder):
    """
    check_if_directory_exists(name_folder)
    INPUT:
        name_folder: name of the directory to be checked
    OUTPUT:
        a message indicating that the directory does not exist and if it is created
        
    @author: Eduardo Fidalgo (EFF)
    """
    import os
    if not os.path.exists(name_folder):
        print(name_folder + " directory does not exist, created")
        os.makedirs(name_folder) 
    else:
        print(name_folder + " directory exists, no action performed")

#%%
def load_json_file(name_json):
    """
    load_json_file(name_json)
    INPUT:
        name_json: name of the json file to be loaded
    OUTPUT:
        variable with the json file loaded
        
    @author: Eduardo Fidalgo (EFF)
    """
    import json
    with open(name_json) as f:    
      config = json.load(f)
      
    return config

#%%
def os_path_separator():
    """
    load_json_file(name_json)
    INPUT:
        None
    OUTPUT:
        path_sep: variable containing the windows or linux path separator
        
    @author: Eduardo Fidalgo (EFF)
    """
    from sys import platform as _platform
    
    # print start time
    print ("[INFO] Platform where script is being used- " + str(_platform))
    
    # Patch separator
    if _platform  == "win64" or _platform  == "win32":
        path_sep = "\\"
    else:
        path_sep = "/"
    
    return path_sep

#%%
def read_image_names(path_ima='./images',img_ext='*.png',verbose=False):
    """
    read_image_names(path_ima='./images', img_ext='*.png',verbose=False)

    Given a path and an image file extension, this function introduce all
    the image name contained in the folder passed in the path in a list and
    it returns it. The introduced name will contain the relative path
    (the directory concatenate with the image name).

    INPUT:
        path_imag: string containing the absolute or relative path.
            By default path_imag= './images'
        img_ext: string containing the extension of the image files to be read.
            By default img_ext='*.png'
        verbose: boolean. If True, messages are shown.
            By default verbose=False

    OUTPUT:
        full_names_img: list containing file names with its path (in a sigle
        string).

    Extracted from ktools
    """
    import os
    import glob

    # List to be returned
    full_names_img = list()

    #% % 1_ Print the original path and check if the folder with images exists!
    prev_path = os.getcwd()
    if verbose:
        print('(1) Original path was:  {} \n'.format(prev_path))

    if os.path.isdir(path_ima):
        if verbose:
            print('(2) The directory {} does exists!\n'.format(path_ima))
    else:
        if verbose:
            print('(2) WARNING: There is not such directory\n')


    #% % 2_ Read all the files with the specified "extension" in the current
    # working directory
    full_names_img = glob.glob(os.path.join(path_ima,img_ext))
    # glob.glob(os.path.join(path_img, img_ext))

    if verbose:
        print('(3) The "{0}" directory has {1} images with {2} extension.\n'
              .format(os.getcwd(),len(full_names_img),img_ext))

    return full_names_img




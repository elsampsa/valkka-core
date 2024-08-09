"""
tools.py : helper functions

 * Copyright 2017-2023 Valkka Security Ltd. and Sampsa Riikonen
 * Copyright 2024 Sampsa Riikonen
 *
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 *
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

@file    tools.py
@author  Sampsa Riikonen
@date    2017
@version 1.6.1 

@brief helper functions
"""

import copy
import json
import types
import sys
import os
import inspect
import logging
import traceback

is_py3 = (sys.version_info >= (3, 0))

loggers = {}

def getLogger(name):
    global loggers
    logger = loggers.get(name)
    if logger: return logger
    # https://docs.python.org/2/howto/logging.html
    # log levels here : https://docs.python.org/2/howto/logging.html#when-to-use-logging
    # in the future, migrate this to a logger config file
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    loggers[name] = logger 
    return logger
    
    
def setLogger(name, level):
    """Give either logger name or the logger itself
    """
    if (isinstance(name,str)):
        logger = getLogger(name)
    else:
        logger = name 

    logger.setLevel(level)
    
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(ch)


def import_file(full_name, path):
    """Import a python module from a path. 3.4+ only.

    Does not call sys.modules[full_name] = path
    
    WARNING: this only works for single-file modules, not for packages
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)

    spec.loader.exec_module(mod)
    return mod


def get_system_numpy():
    """This doesn't work always..
    """
    from distutils import sysconfig
    # stupid hack: temporarily modify package search order
    sys.path.insert(0, sysconfig.get_python_lib())
    import importlib
    mod = importlib.import_module("numpy")
    # print(mod)
    sys.path.pop(0)
    return mod


# this is module specific!
def getModulePath():
    lis = inspect.getabsfile(inspect.currentframe()).split("/")
    st = "/"
    for l in lis[:-1]:
        st = os.path.join(st, l)
    return st


def getTestDataPath():
    return os.path.join(getModulePath(), "test_data")


def getTestDataFile(fname):
    return os.path.join(getTestDataPath(), fname)


def getDataPath():
    return os.path.join(getModulePath(), "data")


def getDataFile(fname):
    """Return complete path to datafile fname.  Data files are in the directory vainu_aux/vainu_aux/data
    """
    return os.path.join(getDataPath(), fname)


def typeCheck(obj, typ):
    """Check type of obj, for example: typeCheck(x,int)
    """
    if (obj.__class__ != typ):
        raise(AttributeError("Object should be of type " + typ.__name__))


def dictionaryCheck(definitions, dic):
    """ Checks that dictionary has certain values, according to definitions

    :param definitions: Dictionary defining the parameters and their types (dic should have at least these params)
    :param dic:         Dictionary to be checked

    An example definitions dictionary:

    |{
    |"age"     : int,         # must have attribute age that is an integer
    |"name"    : str,         # must have attribute name that is a string
    | }
    """

    for key in definitions:
        # print("dictionaryCheck: key=",key)
        required_type = definitions[key]
        try:
            attr = dic[key]
        except KeyError:
            raise(AttributeError("Dictionary missing key " + key))
        # print("dictionaryCheck:","got: ",attr,"of type",attr.__class__,"should be",required_type)
        if (attr.__class__ != required_type):
            raise(
                AttributeError(
                    "Wrong type of parameter " +
                    key +
                    " : is " +
                    attr.__class__.__name__ +
                    " should be " +
                    required_type.__name__))
            return False  # eh.. program quits anyway
    return True


def objectCheck(definitions, obj):
    """ Checks that object has certain attributes, according to definitions

    :param definitions: Dictionary defining the parameters and their types (obj should have at least these attributes)
    :param obj:         Object to be checked

    An example definitions dictionary:

    |{
    |"age"     : int,         # must have attribute age that is an integer
    |"name"    : str,         # must have attribute name that is a string
    | }
    """

    for key in definitions:
        required_type = definitions[key]
        # this raises an AttributeError of object is missing the attribute -
        # but that is what we want
        attr = getattr(obj, key)
        # print("objectCheck:","got: ",attr,"of type",attr.__class__,"should be",required_type)
        if (attr.__class__ != required_type):
            raise(
                AttributeError(
                    "Wrong type of parameter " +
                    key +
                    " : should be " +
                    required_type.__name__))
            return False  # eh.. program quits anyway
    return True


def parameterInitCheck(definitions, parameters, obj, undefined_ok=False):
    """ Checks that parameters are consistent with a definition

    :param definitions: Dictionary defining the parameters, their default values, etc.
    :param parameters:  Dictionary having the parameters to be checked
    :param obj:         Checked parameters are attached as attributes to this object

    An example definitions dictionary:

    |{
    |"age"     : (int,0),                 # parameter age defaults to 0 if not specified
    |"height"  : int,                     # parameter height **must** be defined by the user
    |"indexer" : some_module.Indexer,     # parameter indexer must of some user-defined class some_module.Indexer
    |"cleaner" : checkAttribute_cleaner,  # parameter cleaner is check by a custom function named "checkAttribute_cleaner" (that's been defined before)
    |"weird"   : None                     # parameter weird is passed without any checking - this means that your API is broken  :)
    | }

    """
    definitions = copy.copy(definitions)
    # parameters =getattr(obj,"kwargs")
    # parameters2=copy.copy(parameters)
    #print("parameterInitCheck: definitions=",definitions)
    for key in parameters:
        try:
            definition = definitions.pop(key)
        except KeyError:
            if (undefined_ok):
                continue
            else:
                print("parameterInitCheck: unknown parameter: stack trace:\n")
                traceback.print_stack()
                print("\n")
                raise AttributeError("Unknown parameter " + str(key))

        parameter = parameters[key]
        if (definition.__class__ ==
                tuple):   # a tuple defining (parameter_class, default value)
            #print("parameterInitCheck: tuple")
            required_type = definition[0]
            if (parameter.__class__ != required_type):
                raise(
                    AttributeError(
                        "Wrong type of parameter " +
                        key +
                        " : should be " +
                        required_type.__name__))
            else:
                setattr(obj, key, parameter)  # parameters2.pop(key)
        elif isinstance(definition, types.FunctionType):
            # object is checked by a custom function
            #print("parameterInitCheck: callable")
            ok = definition(parameter)
            if (ok):
                setattr(obj, key, parameter)  # parameters2.pop(key)
            else:
                raise(
                    AttributeError(
                        "Checking of parameter " +
                        key +
                        " failed"))
        elif (definition is None):            # this is a generic object - no checking whatsoever
            #print("parameterInitCheck: None")
            setattr(obj, key, parameter)  # parameters2.pop(key)
        elif isinstance(definition.__class__, type):  # Check the type
            #print("parameterInitCheck: type")
            required_type = definition
            if (parameter.__class__ != required_type):
                raise(
                    AttributeError(
                        "Wrong type '" + parameter.__class__.__name__ + "' of parameter " +
                        key +
                        " : should be of type '" +
                        required_type.__name__ + "'"))
            else:
                setattr(obj, key, parameter)  # parameters2.pop(key)
        else:
            raise(AttributeError("Check your definitions syntax for " + key))

    # in definitions, there might still some leftover parameters the user did
    # not bother to give
    for key in definitions.keys():
        definition = definitions[key]
        if (definition.__class__ ==
                tuple):   # a tuple defining (parameter_class, default value)
            setattr(obj, key, definition[1])  # parameters2.pop(key)
        elif (definition is None):            # parameter that can be void
            # parameters2.pop(key)
            # pass
            setattr(obj, key, None)
        else:
            raise(AttributeError("Missing a mandatory parameter " + key))

    # setattr(obj,"kwargs", parameters2)


def noCheck(obj):
    return True


def gen_getter(obj, key):
    def func():
        return getattr(obj, key)
    return func
        

def generateGetters(definitions, obj):
    for key in definitions:
        getter_name = "get_"+key
        if not hasattr(obj, getter_name):
            getter = gen_getter(obj, key)
            # print("generateGetters :", getter_name, getter())
            setattr(obj, getter_name, getter)
        



def getH264V4l2(verbose=False):
    """Find all V4l2 cameras with H264 encoding, and returns a list of tuples with ..
    
    (device file, device name), e.g. ("/dev/video2", "HD Pro Webcam C920")
    """
    import glob
    from subprocess import Popen, PIPE
    
    cams=[]

    for device in glob.glob("/sys/class/video4linux/*"):
        devname=device.split("/")[-1]
        devfile=os.path.join("/dev",devname)
        
        lis=("v4l2-ctl --list-formats -d "+devfile).split()

        p = Popen(lis, stdout=PIPE, stderr=PIPE)
        # p.communicate()
        # print(dir(p))
        # print(p.returncode)
        # print(p.stderr.read().decode("utf-8"))
        st = p.stdout.read().decode("utf-8")
        # print(st)
        
        if (st.lower().find("h264")>-1):
            namefile=os.path.join(device, "name")
            # print(namefile)
            f=open(namefile, "r"); name=f.read(); f.close()
            cams.append((devfile, name.strip()))

    if (verbose):
        for cam in cams:
            print(cam)
        
    return cams


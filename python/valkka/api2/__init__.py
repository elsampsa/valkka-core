__all__=["logging","chains","threads","multiprocess","shmem","tools","raise_numpy_version", "warn_numpy_version", "check_numpy_version"]

# import importlib
# import imp
# from distutils import sysconfig

# import everything from from valkka.api2.* to valkka.api2

from valkka.core import get_numpy_version, numpy_version_ok
from valkka.api2.tools import *

# from .logging import setValkkaLogLevel, loglevel_silent, loglevel_normal, loglevel_debug, loglevel_crazy
from valkka.api2.logging import *

# setValkkaLogLevel(loglevel_silent) # this should be default for production
setValkkaLogLevel(loglevel_normal) # default for development # informs about frame drops
# setValkkaLogLevel(loglevel_debug) # default for more serious development

from valkka.api2.chains import BasicFilterchain, BasicFilterchain1, ShmemFilterchain, ShmemFilterchain1, ManagedFilterchain, ViewPort, ValkkaFSLiveFilterchain, ValkkaFSFileFilterchain
from valkka.api2.threads import LiveThread, USBDeviceThread, FileThread, OpenGLThread, Namespace
from valkka.api2.multiprocess import ValkkaProcess, ValkkaShmemRGBProcess, safe_select
from valkka.api2.shmem import ShmemClient, ShmemRGBClient
from valkka.api2.valkkafs import ValkkaFS, findBlockDevices, ValkkaFSManager, formatMstimestamp


def raise_numpy_version():
    """
    import numpy
    if numpy.__version__ != get_numpy_version():
    """
    if not numpy_version_ok():
        raise(AssertionError("Inconsistent numpy versions: libValkka compiled with %s / your system is using %s" % \
            (get_numpy_version(), numpy.__version__)))
    
    
def warn_numpy_version():
    """
    import numpy
    if numpy.__version__ != get_numpy_version():
    """
    if not numpy_version_ok():
        print("WARNING : Inconsistent numpy versions: libValkka compiled with %s / your system is using %s" % \
            (get_numpy_version(), numpy.__version__))
    


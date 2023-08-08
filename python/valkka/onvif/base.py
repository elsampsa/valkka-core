"""base.py : Simple classes for Zeep-based OnVif service clients

Copyright 2017-2019 Valkka Security Ltd. and Sampsa Riikonen.

Authors: Petri Eränkö <petri.eranko@dasys.fi>

This particular file, referred below as "Software", is licensed under the MIT LICENSE:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

@file    base.py
@author  Petri Eränkö
@date    2019
@version 1.5.2 

@brief   Simple classes for Zeep-based OnVif service clients
"""
import sys
import os
import inspect
import datetime as dt
import lxml
import os

from zeep.client import Client, CachingClient, Settings
import zeep.helpers
from zeep.wsse.username import UsernameToken


settings = Settings()


def getWSDLPath(filename):
    lis=inspect.getabsfile(inspect.currentframe()).split("/")
    st="/"
    for l in lis[:-1]:
        st=os.path.join(st,l)
    return os.path.join(st, "wsdl", filename)
    
    

class OnVif:
    """In subclasses, re-define class variables
    """
    namespace = None # this is also your link to specifications
    wsdl_file = None # local file
    sub_xaddr = None # http://ipaddr:port/onvif/sub_xaddr
    port      = None # as defined in the wsdl file

    
    def __init__(self, ip = None, port = 80, user = "admin", password = "12345", use_async = False, sub_xaddr_ = None):
        assert(isinstance(port, int))
    
        if sub_xaddr_ is not None: # overwrite xaddr
            self.xaddr = "http://%s:%i/onvif/%s" % (ip, port, sub_xaddr_)
        else:
            self.xaddr = "http://%s:%i/onvif/%s" % (ip, port, self.sub_xaddr)
            
        self.user = user
        self.password = password
        self.use_async = use_async
        self.service_addr = "{%s}%s" % (self.namespace, self.port)
        
        if self.use_async:
            self.async_transport = AsyncTransport(asyncio.get_event_loop(), cache = None)
            self.zeep_client = Client(wsdl = self.wsdl_file, wsse = UsernameToken(self.user, self.password, use_digest = True), settings = settings, transport = self.async_transport)
        else:
            self.zeep_client = Client(wsdl = self.wsdl_file, wsse = UsernameToken(self.user, self.password, use_digest = True), settings = settings)
        
        self.ws_client = self.zeep_client.create_service(self.service_addr, self.xaddr)

        self.factory = self.zeep_client.type_factory("http://www.onvif.org/ver10/schema")
        

    async def close(self):
        await self.async_transport.session.close()
    
    
    def openSpecs(self):
        os.system("firefox %s" % (self.namespace))
        
        
    def getVariable(self, name):
        return self.factory.CapabilityCategory(name)

        
        
class DeviceManagement(OnVif):
    namespace = "http://www.onvif.org/ver10/device/wsdl"
    wsdl_file = getWSDLPath("devicemgmt.wsdl")
    sub_xaddr = "device_service"
    port      = "DeviceBinding"
    
            
class Media(OnVif):
    wsdl_file = getWSDLPath("media.wsdl")
    namespace = "http://www.onvif.org/ver10/media/wsdl"
    sub_xaddr = "Media"
    port      = "MediaBinding"


class Events(OnVif):
    wsdl_file = getWSDLPath("events.wsdl")
    namespace = "http://www.onvif.org/ver10/events/wsdl"
    sub_xaddr = "Events"
    port      = "EventPortType"


class PullPointEvents(Events):
    port      = "PullPointSubscription"


class PTZ(OnVif):
    wsdl_file = getWSDLPath("ptz.wsdl")
    namespace = "http://www.onvif.org/ver20/ptz/wsdl"
    sub_xaddr = "PTZ"
    port      = "PTZBinding"
    

class Imaging(OnVif):
    wsdl_file = getWSDLPath("imaging.wsdl")
    namespace = "http://www.onvif.org/ver20/imaging/wsdl"
    sub_xaddr = "Imaging"
    port      = "ImagingPort"


class DeviceIO(OnVif):
    wsdl_file = getWSDLPath("deviceio.wsdl")
    namespace = "http://www.onvif.org/ver10/deviceIO/wsdl"
    sub_xaddr = "DeviceIO"
    port      = "DeviceIOBinding"
    
    
class AnalyticsEngine(OnVif):
    wsdl_file = getWSDLPath("analytics.wsdl")
    namespace = "http://www.onvif.org/ver20/analytics/wsdl"
    sub_xaddr = "Analytics"
    port      = "AnalyticsEnginePort"


class RuleEngine(OnVif):
    wsdl_file = getWSDLPath("analytics.wsdl")
    namespace = "http://www.onvif.org/ver20/analytics/wsdl"
    sub_xaddr = "Analytics"
    port      = "RuleEnginePort"


# OnVif, DeviceManagement, Media, Events, PullPointEvents, PTZ, Imaging, DeviceIO, AnalyticsEngine, RuleEngine

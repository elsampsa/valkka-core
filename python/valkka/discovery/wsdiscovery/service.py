"""
Scoped Onvif service implementation.
"""


from .util import _getNetworkAddrs


class Service:

    def __init__(self, types, scopes, xAddrs, epr, instanceId):
        self._types = types
        self._scopes = scopes
        self._xAddrs = xAddrs
        self._epr = epr
        self._instanceId = instanceId
        self._messageNumber = 0
        self._metadataVersion = 1

    def getTypes(self):
        return self._types

    def setTypes(self, types):
        self._types = types

    def getScopes(self):
        return self._scopes

    def setScopes(self, scopes):
        self._scopes = scopes

    def getXAddrs(self):
        ret = []
        ipAddrs = None
        for xAddr in self._xAddrs:
            if '{ip}' in xAddr:
                if ipAddrs is None:
                    ipAddrs = _getNetworkAddrs()
                for ipAddr in ipAddrs:
                    if ipAddr != '127.0.0.1':
                        ret.append(xAddr.format(ip=ipAddr))
            else:
                ret.append(xAddr)
        return ret

    def setXAddrs(self, xAddrs):
        self._xAddrs = xAddrs

    def getEPR(self):
        "get endpoint reference"
        return self._epr

    def setEPR(self, epr):
        "set endpoint reference"
        self._epr = epr

    def getInstanceId(self):
        return self._instanceId

    def setInstanceId(self, instanceId):
        self._instanceId = instanceId

    def getMessageNumber(self):
        return self._messageNumber

    def setMessageNumber(self, messageNumber):
        self._messageNumber = messageNumber

    def getMetadataVersion(self):
        return self._metadataVersion

    def setMetadataVersion(self, metadataVersion):
        self._metadataVersion = metadataVersion

    def incrementMessageNumber(self):
        self._messageNumber = self._messageNumber + 1



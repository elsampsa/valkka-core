"""
A threaded discovery daemon implementation.
"""

# Python2 compatibility
from __future__ import print_function
from __future__ import unicode_literals

import logging
import random
import time
import uuid
import socket
import struct
import threading
import select

from .udp import UDPMessage
from .envelope import SoapEnvelope
from .actions import *
from .uri import URI
from .util import _getNetworkAddrs, matchScope
from .util import _generateInstanceId, extractSoapUdpAddressFromURI
from .message import createSOAPMessage, parseSOAPMessage
from .service import Service
from .namespaces import NS_D


logger = logging.getLogger("ws-discovery")

BUFFER_SIZE = 0xffff
_NETWORK_ADDRESSES_CHECK_TIMEOUT = 5
MULTICAST_PORT = 3702
MULTICAST_IPV4_ADDRESS = "239.255.255.250"

APP_MAX_DELAY = 500 # miliseconds

ADDRESS_ALL = "urn:schemas-xmlsoap-org:ws:2005:04:discovery"
ADDRESS_UNKNOWN = "http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous"


class _StoppableDaemonThread(threading.Thread):
    """Stoppable daemon thread.

    run() method shall exit, when self._quitEvent.wait() returned True
    """
    def __init__(self):
        self._quitEvent = threading.Event()
        super(_StoppableDaemonThread, self).__init__()
        self.daemon = True

    def schedule_stop(self):
        """Schedule stopping the thread.
        Use join() to wait, until thread really has been stopped
        """
        self._quitEvent.set()


class AddressMonitorThread(_StoppableDaemonThread):
    def __init__(self, wsd):
        self._addrs = set()
        self._wsd = wsd
        super(AddressMonitorThread, self).__init__()
        self._updateAddrs()

    def _updateAddrs(self):
        addrs = set(_getNetworkAddrs())

        disappeared = self._addrs.difference(addrs)
        new = addrs.difference(self._addrs)

        for addr in disappeared:
            self._wsd._networkAddressRemoved(addr)

        for addr in new:
            self._wsd._networkAddressAdded(addr)

        self._addrs = addrs

    def run(self):
        while not self._quitEvent.wait(_NETWORK_ADDRESSES_CHECK_TIMEOUT):
            self._updateAddrs()


class NetworkingThread(_StoppableDaemonThread):
    def __init__(self, observer, capture=None):
        super(NetworkingThread, self).__init__()

        self.setDaemon(True)
        self._queue = []    # FIXME synchronisation

        self._knownMessageIds = set()
        self._iidMap = {}
        self._observer = observer
        self._capture = observer._capture
        self._seqnum = 1 # capture sequence number
        self._poll = select.poll()

    @staticmethod
    def _makeMreq(addr):
        return struct.pack("4s4s", socket.inet_aton(MULTICAST_IPV4_ADDRESS), socket.inet_aton(addr))

    @staticmethod
    def _createMulticastOutSocket(addr, ttl):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(0)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        if addr is None:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.INADDR_ANY)
        else:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(addr))

        return sock

    @staticmethod
    def _createMulticastInSocket():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        sock.bind(('', MULTICAST_PORT))

        sock.setblocking(0)

        return sock

    def addSourceAddr(self, addr):
        """None means 'system default'"""
        try:
            self._multiInSocket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, self._makeMreq(addr))
        except socket.error:  # if 1 interface has more than 1 address, exception is raised for the second
            pass

        sock = self._createMulticastOutSocket(addr, self._observer.ttl)
        self._multiOutUniInSockets[addr] = sock
        self._poll.register(sock, select.POLLIN)

    def removeSourceAddr(self, addr):
        try:
            self._multiInSocket.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, self._makeMreq(addr))
        except socket.error:  # see comments for setsockopt(.., socket.IP_ADD_MEMBERSHIP..
            pass

        sock = self._multiOutUniInSockets[addr]
        self._poll.unregister(sock)
        sock.close()
        del self._multiOutUniInSockets[addr]

    def addUnicastMessage(self, env, addr, port, initialDelay=0):
        msg = UDPMessage(env, addr, port, UDPMessage.UNICAST, initialDelay)

        self._queue.append(msg)
        self._knownMessageIds.add(env.getMessageId())

    def addMulticastMessage(self, env, addr, port, initialDelay=0):
        msg = UDPMessage(env, addr, port, UDPMessage.MULTICAST, initialDelay)

        self._queue.append(msg)
        self._knownMessageIds.add(env.getMessageId())

    def run(self):
        while not self._quitEvent.is_set() or self._queue:
            self._sendPendingMessages()
            self._recvMessages()

    def _recvMessages(self):
        for fd, event in self._poll.poll(0):
            sock = socket.fromfd(fd, socket.AF_INET, socket.SOCK_DGRAM)
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)
            except socket.error as e:
                time.sleep(0.01)
                continue

            env = parseSOAPMessage(data, addr[0])

            if env is None: # fault or failed to parse
                continue

            _own_addrs = self._observer._addrsMonitorThread._addrs
            if addr[0] not in _own_addrs:
                if env.getAction() == ACTION_PROBE_MATCH:
                    prms = "\n ".join((str(prm) for prm in env.getProbeResolveMatches()))
                    msg = "probe response from %s:\n --- begin ---\n%s\n--- end ---\n"
                    logger.debug(msg, addr[0], prms)

                if self._capture:
                    self._capture.write("%i RECV %s:%s\n" % (self._seqnum, addr[0], addr[1]))
                    self._capture.write(data.decode("utf-8") + "\n")
                    self._seqnum += 1

            mid = env.getMessageId()
            if mid in self._knownMessageIds:
                continue
            else:
                self._knownMessageIds.add(mid)

            iid = env.getInstanceId()
            mid = env.getMessageId()
            if len(iid) > 0 and int(iid) > 0:
                mnum = env.getMessageNumber()
                key = addr[0] + ":" + str(addr[1]) + ":" + str(iid)
                if mid is not None and len(mid) > 0:
                    key = key + ":" + mid
                if key not in self._iidMap:
                    self._iidMap[key] = iid
                else:
                    tmnum = self._iidMap[key]
                    if mnum > tmnum:
                        self._iidMap[key] = mnum
                    else:
                        continue

            self._observer.envReceived(env, addr)

    def _sendMsg(self, msg):
        data = createSOAPMessage(msg.getEnv()).encode("UTF-8")

        if msg.msgType() == UDPMessage.UNICAST:
            self._uniOutSocket.sendto(data, (msg.getAddr(), msg.getPort()))
            if self._capture:
                self._capture.write("%i SEND %s:%s\n" % (self._seqnum, msg.getAddr(), msg.getPort()))
                self._capture.write(data.decode("utf-8") + "\n")
                self._seqnum += 1
        else:
            for sock in list(self._multiOutUniInSockets.values()):
                sock.sendto(data, (msg.getAddr(), msg.getPort()))
                if self._capture:
                    self._capture.write("%i SEND %s:%s\n" % (self._seqnum, msg.getAddr(), msg.getPort()))
                    self._capture.write(data.decode("utf-8") + "\n")
                    self._seqnum += 1

    def _sendPendingMessages(self):
        """Method sleeps, if nothing to do"""
        if len(self._queue) == 0:
            time.sleep(0.1)
            return
        msg = self._queue.pop(0)
        if msg.canSend():
            self._sendMsg(msg)
            msg.refresh()
            if not (msg.isFinished()):
                self._queue.append(msg)
        else:
            self._queue.append(msg)
            time.sleep(0.01)

    def start(self):
        super(NetworkingThread, self).start()

        self._uniOutSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._multiInSocket = self._createMulticastInSocket()
        self._poll.register(self._multiInSocket)

        self._multiOutUniInSockets = {}  # FIXME synchronisation

    def join(self):
        super(NetworkingThread, self).join()
        self._uniOutSocket.close()

        self._poll.unregister(self._multiInSocket)
        self._multiInSocket.close()


class WSDiscovery:

    def __init__(self, uuid_=None, capture=None, ttl=1):

        self._networkingThread = None
        self._serverStarted = False
        self._remoteServices = {}
        self._localServices = {}

        self._dpActive = False
        self._dpAddr = None
        self._dpEPR = None

        self._remoteServiceHelloCallback = None
        self._remoteServiceHelloCallbackTypesFilter = None
        self._remoteServiceHelloCallbackScopesFilter = None
        self._remoteServiceByeCallback = None
        self._capture = capture

        if uuid_ is not None:
            self.uuid = uuid_
        else:
            self.uuid = uuid.uuid4().urn

        self.ttl = ttl

    def setRemoteServiceHelloCallback(self, cb, types=None, scopes=None):
        """Set callback, which will be called when new service appeared online
        and sent Hi message

        typesFilter and scopesFilter might be list of types and scopes.
        If filter is set, callback is called only for Hello messages,
        which match filter

        Set None to disable callback
        """
        self._remoteServiceHelloCallback = cb
        self._remoteServiceHelloCallbackTypesFilter = types
        self._remoteServiceHelloCallbackScopesFilter = scopes

    def setRemoteServiceByeCallback(self, cb):
        """Set callback, which will be called when new service appeared online
        and sent Hi message
        Service is passed as a parameter to the callback
        Set None to disable callback
        """
        self._remoteServiceByeCallback = cb

    def setRemoveServiceDisappearedCallback(self, cb):
        """Set callback, which will be called when new service disappears
        Service uuid is passed as a parameter to the callback
        Set None to disable callback
        """
        self._remoteServiceDisppearedCallback = cb

    def _addRemoteService(self, service):
        self._remoteServices[service.getEPR()] = service

    def _removeRemoteService(self, epr):
        if epr in self._remoteServices:
            del self._remoteServices[epr]

    def handleEnv(self, env, addr):
        if (env.getAction() == ACTION_PROBE_MATCH):
            for match in env.getProbeResolveMatches():
                self._addRemoteService(Service(match.getTypes(), match.getScopes(), match.getXAddrs(), match.getEPR(), 0))
                if match.getXAddrs() is None or len(match.getXAddrs()) == 0:
                    self._sendResolve(match.getEPR())

        elif env.getAction() == ACTION_RESOLVE_MATCH:
            for match in env.getProbeResolveMatches():
                self._addRemoteService(Service(match.getTypes(), match.getScopes(), match.getXAddrs(), match.getEPR(), 0))

        elif env.getAction() == ACTION_PROBE:
            services = self._filterServices(list(self._localServices.values()), env.getTypes(), env.getScopes())
            self._sendProbeMatch(services, env.getMessageId(), addr)

        elif env.getAction() == ACTION_RESOLVE:
            if env.getEPR() in self._localServices:
                service = self._localServices[env.getEPR()]
                self._sendResolveMatch(service, env.getMessageId(), addr)

        elif env.getAction() == ACTION_HELLO:
            #check if it is from a discovery proxy
            rt = env.getRelationshipType()
            if rt is not None and rt.getLocalname() == "Suppression" and rt.getNamespace() == NS_D:
                xAddr = env.getXAddrs()[0]
                #only support 'soap.udp'
                if xAddr.startswith("soap.udp:"):
                    self._dpActive = True
                    self._dpAddr = extractSoapUdpAddressFromURI(URI(xAddr))
                    self._dpEPR = env.getEPR()

            service = Service(env.getTypes(), env.getScopes(), env.getXAddrs(), env.getEPR(), 0)
            self._addRemoteService(service)
            if self._remoteServiceHelloCallback is not None:
                if self._matchesFilter(service,
                                        self._remoteServiceHelloCallbackTypesFilter,
                                        self._remoteServiceHelloCallbackScopesFilter):
                    self._remoteServiceHelloCallback(service)

        elif env.getAction() == ACTION_BYE:
            #if the bye is from discovery proxy... revert back to multicasting
            if self._dpActive and self._dpEPR == env.getEPR():
                self._dpActive = False
                self._dpAddr = None
                self._dpEPR = None

            self._removeRemoteService(env.getEPR())
            if self._remoteServiceByeCallback is not None:
                self._remoteServiceByeCallback(env.getEPR())

    def envReceived(self, env, addr):
        self.handleEnv(env, addr)

    def _sendResolveMatch(self, service, relatesTo, addr):
        service.incrementMessageNumber()

        env = SoapEnvelope()
        env.setAction(ACTION_RESOLVE_MATCH)
        env.setTo(ADDRESS_UNKNOWN)
        env.setMessageId(uuid.uuid4().urn)
        env.setInstanceId(str(service.getInstanceId()))
        env.setMessageNumber(str(service.getMessageNumber()))
        env.setRelatesTo(relatesTo)

        env.getProbeResolveMatches().append(ProbeResolveMatch(service.getEPR(), \
                                                              service.getTypes(), service.getScopes(), \
                                                              service.getXAddrs(), str(service.getMetadataVersion())))
        self._networkingThread.addUnicastMessage(env, addr[0], addr[1])

    def _sendProbeMatch(self, services, relatesTo, addr):
        env = SoapEnvelope()
        env.setAction(ACTION_PROBE_MATCH)
        env.setTo(ADDRESS_UNKNOWN)
        env.setMessageId(uuid.uuid4().urn)
        random.seed((int)(time.time() * 1000000))
        env.setInstanceId(_generateInstanceId())
        env.setMessageNumber("1")
        env.setRelatesTo(relatesTo)

        for service in services:
            env.getProbeResolveMatches().append(ProbeResolveMatch(service.getEPR(), \
                                                                  service.getTypes(), service.getScopes(), \
                                                                  service.getXAddrs(), str(service.getMetadataVersion())))

        self._networkingThread.addUnicastMessage(env, addr[0], addr[1], random.randint(0, APP_MAX_DELAY))

    def _sendProbe(self, types=None, scopes=None):
        env = SoapEnvelope()
        env.setAction(ACTION_PROBE)
        env.setTo(ADDRESS_ALL)
        env.setMessageId(uuid.uuid4().urn)
        env.setTypes(types)
        env.setScopes(scopes)

        if self._dpActive:
            self._networkingThread.addUnicastMessage(env, self._dpAddr[0], self._dpAddr[1])
        else:
            self._networkingThread.addMulticastMessage(env, MULTICAST_IPV4_ADDRESS, MULTICAST_PORT)

    def _sendResolve(self, epr):
        env = SoapEnvelope()
        env.setAction(ACTION_RESOLVE)
        env.setTo(ADDRESS_ALL)
        env.setMessageId(uuid.uuid4().urn)
        env.setEPR(epr)

        if self._dpActive:
            self._networkingThread.addUnicastMessage(env, self._dpAddr[0], self._dpAddr[1])
        else:
            self._networkingThread.addMulticastMessage(env, MULTICAST_IPV4_ADDRESS, MULTICAST_PORT)

    def _sendHello(self, service):
        service.incrementMessageNumber()

        env = SoapEnvelope()
        env.setAction(ACTION_HELLO)
        env.setTo(ADDRESS_ALL)
        env.setMessageId(uuid.uuid4().urn)
        env.setInstanceId(str(service.getInstanceId()))
        env.setMessageNumber(str(service.getMessageNumber()))
        env.setTypes(service.getTypes())
        env.setScopes(service.getScopes())
        env.setXAddrs(service.getXAddrs())
        env.setEPR(service.getEPR())

        random.seed((int)(time.time() * 1000000))

        self._networkingThread.addMulticastMessage(env, MULTICAST_IPV4_ADDRESS, MULTICAST_PORT, random.randint(0, APP_MAX_DELAY))

    def _sendBye(self, service):
        env = SoapEnvelope()
        env.setAction(ACTION_BYE)
        env.setTo(ADDRESS_ALL)
        env.setMessageId(uuid.uuid4().urn)
        env.setInstanceId(str(service.getInstanceId()))
        env.setMessageNumber(str(service.getMessageNumber()))
        env.setEPR(service.getEPR())

        service.incrementMessageNumber()
        self._networkingThread.addMulticastMessage(env, MULTICAST_IPV4_ADDRESS, MULTICAST_PORT)

    def start(self):
        'start the discovery server - should be called before using other functions'
        self._startThreads()
        self._serverStarted = True

    def stop(self):
        'cleans up and stops the discovery server'

        self.clearRemoteServices()
        self.clearLocalServices()

        self._stopThreads()
        self._serverStarted = False

    def  _networkAddressAdded(self, addr):
        self._networkingThread.addSourceAddr(addr)
        for service in list(self._localServices.values()):
            self._sendHello(service)

    def _networkAddressRemoved(self, addr):
        self._networkingThread.removeSourceAddr(addr)

    def _startThreads(self):
        if self._networkingThread is not None:
            return

        self._networkingThread = NetworkingThread(self)
        self._networkingThread.start()

        self._addrsMonitorThread = AddressMonitorThread(self)
        self._addrsMonitorThread.start()


    def _stopThreads(self):
        if self._networkingThread is None:
            return

        self._networkingThread.schedule_stop()
        self._addrsMonitorThread.schedule_stop()

        self._networkingThread.join()
        self._addrsMonitorThread.join()

        self._networkingThread = None

    def _isTypeInList(self, ttype, types):
        for entry in types:
            if ttype.getFullname() == entry.getFullname():
                return True

        return False

    def _isScopeInList(self, scope, scopes):
        for entry in scopes:
            if matchScope(scope.getValue(), entry.getValue(), scope.getMatchBy()):
                return True

        return False

    def _matchesFilter(self, service, types, scopes):
        if types is not None:
            for ttype in types:
                if not self._isTypeInList(ttype, service.getTypes()):
                    return False
        if scopes is not None:
            for scope in scopes:
                if not self._isScopeInList(scope, service.getScopes()):
                    return False
        return True

    def _filterServices(self, services, types, scopes):
        return [service for service in services \
                    if self._matchesFilter(service, types, scopes)]

    def clearRemoteServices(self):
        'clears remotely discovered services'

        self._remoteServices.clear()

    def clearLocalServices(self):
        'send Bye messages for the services and remove them'

        for service in list(self._localServices.values()):
            self._sendBye(service)

        self._localServices.clear()

    def searchServices(self, types=None, scopes=None, timeout=3):
        'search for services given the TYPES and SCOPES in a given TIMEOUT'

        if not self._serverStarted:
            raise Exception("Server not started")

        self._sendProbe(types, scopes)

        time.sleep(timeout)

        return self._filterServices(list(self._remoteServices.values()), types, scopes)

    def publishService(self, types, scopes, xAddrs):
        """Publish a service with the given TYPES, SCOPES and XAddrs (service addresses)

        if xAddrs contains item, which includes {ip} pattern, one item per IP addres will be sent
        """

        if not self._serverStarted:
            raise Exception("Server not started")

        instanceId = _generateInstanceId()

        service = Service(types, scopes, xAddrs, self.uuid, instanceId)
        self._localServices[self.uuid] = service
        self._sendHello(service)

        time.sleep(0.001)


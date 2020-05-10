"""
Message implementation used by the daemon. NOT a SOAP / ONVIF / WSDiscovery message.
"""

import random
import time


UNICAST_UDP_REPEAT=2
UNICAST_UDP_MIN_DELAY=50
UNICAST_UDP_MAX_DELAY=250
UNICAST_UDP_UPPER_DELAY=500

MULTICAST_UDP_REPEAT=4
MULTICAST_UDP_MIN_DELAY=50
MULTICAST_UDP_MAX_DELAY=250
MULTICAST_UDP_UPPER_DELAY=500


class UDPMessage:
    MULTICAST = 'multicast'
    UNICAST = 'unicast'

    def __init__(self, env, addr, port, msgType, initialDelay=0):
        """msgType shall be Message.MULTICAST or Message.UNICAST"""
        self._env = env
        self._addr = addr
        self._port = port
        self._msgType = msgType

        if msgType == self.UNICAST:
            udpRepeat, udpMinDelay, udpMaxDelay, udpUpperDelay = \
                    UNICAST_UDP_REPEAT, \
                    UNICAST_UDP_MIN_DELAY, \
                    UNICAST_UDP_MAX_DELAY, \
                    UNICAST_UDP_UPPER_DELAY
        else:
            udpRepeat, udpMinDelay, udpMaxDelay, udpUpperDelay = \
                    MULTICAST_UDP_REPEAT, \
                    MULTICAST_UDP_MIN_DELAY, \
                    MULTICAST_UDP_MAX_DELAY, \
                    MULTICAST_UDP_UPPER_DELAY

        self._udpRepeat = udpRepeat
        self._udpUpperDelay = udpUpperDelay
        self._t = (udpMinDelay + ((udpMaxDelay - udpMinDelay) * random.random())) / 2
        self._nextTime = int(time.time() * 1000) + initialDelay

    def getEnv(self):
        return self._env

    def getAddr(self):
        return self._addr

    def getPort(self):
        return self._port

    def msgType(self):
        return self._msgType

    def isFinished(self):
        return self._udpRepeat <= 0

    def canSend(self):
        ct = int(time.time() * 1000)
        return self._nextTime < ct

    def refresh(self):
        self._t = self._t * 2
        if self._t > self._udpUpperDelay:
            self._t = self._udpUpperDelay
        self._nextTime = int(time.time() * 1000) + self._t
        self._udpRepeat = self._udpRepeat - 1



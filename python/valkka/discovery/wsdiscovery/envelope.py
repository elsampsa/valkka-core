"""
SOAP envelope object implementation.
"""


class SoapEnvelope:

    def __init__(self):
        self._action = ""
        self._messageId = ""
        self._relatesTo = ""
        self._relationshipType = None
        self._to = ""
        self._replyTo = ""
        self._instanceId = ""
        self._sequenceId = ""
        self._messageNumber = ""
        self._epr = ""
        self._types = []
        self._scopes = []
        self._xAddrs = []
        self._metadataVersion = "1"
        self._probeResolveMatches = []

    def getAction(self):
        return self._action

    def setAction(self, action):
        self._action = action

    def getMessageId(self):
        return self._messageId

    def setMessageId(self, messageId):
        self._messageId = messageId

    def getRelatesTo(self):
        return self._relatesTo

    def setRelatesTo(self, relatesTo):
        self._relatesTo = relatesTo

    def getRelationshipType(self):
        return self._relationshipType

    def setRelationshipType(self, relationshipType):
        self._relationshipType = relationshipType

    def getTo(self):
        return self._to

    def setTo(self, to):
        self._to = to

    def getReplyTo(self):
        return self._replyTo

    def setReplyTo(self, replyTo):
        self._replyTo = replyTo

    def getInstanceId(self):
        return self._instanceId

    def setInstanceId(self, instanceId):
        self._instanceId = instanceId

    def getSequenceId(self):
        return self._sequenceId

    def setSequenceId(self, sequenceId):
        self._sequenceId = sequenceId

    def getEPR(self):
        return self._epr

    def setEPR(self, epr):
        self._epr = epr

    def getMessageNumber(self):
        return self._messageNumber

    def setMessageNumber(self, messageNumber):
        self._messageNumber = messageNumber

    def getTypes(self):
        return self._types

    def setTypes(self, types):
        self._types = types

    def getScopes(self):
        return self._scopes

    def setScopes(self, scopes):
        self._scopes = scopes

    def getXAddrs(self):
        return self._xAddrs

    def setXAddrs(self, xAddrs):
        self._xAddrs = xAddrs

    def getMetadataVersion(self):
        return self._metadataVersion

    def setMetadataVersion(self, metadataVersion):
        self._metadataVersion = metadataVersion

    def getProbeResolveMatches(self):
        return self._probeResolveMatches

    def setProbeResolveMatches(self, probeResolveMatches):
        self._probeResolveMatches = probeResolveMatches



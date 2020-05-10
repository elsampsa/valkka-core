
from ..namespaces import NS_A, NS_D
from ..envelope import SoapEnvelope
from ..util import createSkelSoapMessage, getBodyEl, getHeaderEl, addElementWithText, \
                   addTypes, addScopes, getDocAsString, getScopes, _parseAppSequence, \
                   addEPR, getXAddrs, addXAddrs, getTypes


ACTION_PROBE_MATCH = "http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches"


def createProbeMatchMessage(env):
    doc = createSkelSoapMessage(ACTION_PROBE_MATCH)

    bodyEl = getBodyEl(doc)
    headerEl = getHeaderEl(doc)

    addElementWithText(doc, headerEl, "a:MessageID", NS_A, env.getMessageId())
    addElementWithText(doc, headerEl, "a:RelatesTo", NS_A, env.getRelatesTo())
    addElementWithText(doc, headerEl, "a:To", NS_A, env.getTo())

    appSeqEl = doc.createElementNS(NS_D, "d:AppSequence")
    appSeqEl.setAttribute("InstanceId", env.getInstanceId())
    appSeqEl.setAttribute("MessageNumber", env.getMessageNumber())
    headerEl.appendChild(appSeqEl)

    probeMatchesEl = doc.createElementNS(NS_D, "d:ProbeMatches")
    probeMatches = env.getProbeResolveMatches()
    for probeMatch in probeMatches:
        probeMatchEl = doc.createElementNS(NS_D, "d:ProbeMatch")
        addEPR(doc, probeMatchEl, probeMatch.getEPR())
        addTypes(doc, probeMatchEl, probeMatch.getTypes())
        addScopes(doc, probeMatchEl, probeMatch.getScopes())
        addXAddrs(doc, probeMatchEl, probeMatch.getXAddrs())
        addElementWithText(doc, probeMatchEl, "d:MetadataVersion", NS_D, probeMatch.getMetadataVersion())
        probeMatchesEl.appendChild(probeMatchEl)


    bodyEl.appendChild(probeMatchesEl)

    return getDocAsString(doc)


def parseProbeMatchMessage(dom):
    env = SoapEnvelope()
    env.setAction(ACTION_PROBE_MATCH)

    env.setMessageId(dom.getElementsByTagNameNS(NS_A, "MessageID")[0].firstChild.data.strip())
    env.setRelatesTo(dom.getElementsByTagNameNS(NS_A, "RelatesTo")[0].firstChild.data.strip())
    env.setTo(dom.getElementsByTagNameNS(NS_A, "To")[0].firstChild.data.strip())

    _parseAppSequence(dom, env)

    pmNodes = dom.getElementsByTagNameNS(NS_D, "ProbeMatch")
    for node in pmNodes:
        epr = node.getElementsByTagNameNS(NS_A, "Address")[0].firstChild.data.strip()

        types = []
        typeNodes = node.getElementsByTagNameNS(NS_D, "Types")
        if len(typeNodes) > 0:
            types = getTypes(typeNodes[0])

        scopes = []
        scopeNodes = node.getElementsByTagNameNS(NS_D, "Scopes")
        if len(scopeNodes) > 0:
            scopes = getScopes(scopeNodes[0])

        xAddrs = []
        xAddrNodes = node.getElementsByTagNameNS(NS_D, "XAddrs")
        if len(xAddrNodes) > 0:
            xAddrs = getXAddrs(xAddrNodes[0])

        mdv = node.getElementsByTagNameNS(NS_D, "MetadataVersion")[0].firstChild.data.strip()
        env.getProbeResolveMatches().append(ProbeResolveMatch(epr, types, scopes, xAddrs, mdv))

    return env



class ProbeResolveMatch:

    def __init__(self, epr, types, scopes, xAddrs, metadataVersion):
        self._epr = epr
        self._types = types
        self._scopes = scopes
        self._xAddrs = xAddrs
        self._metadataVersion = metadataVersion

    def getEPR(self):
        return self._epr

    def getTypes(self):
        return self._types

    def getScopes(self):
        return self._scopes

    def getXAddrs(self):
        return self._xAddrs

    def getMetadataVersion(self):
        return self._metadataVersion

    def __repr__(self):
        return "EPR: %s\nTypes: %s\nScopes: %s\nXAddrs: %s\nMetadata Version: %s" % \
            (self.getEPR(), self.getTypes(), self.getScopes(),
             self.getXAddrs(), self.getMetadataVersion())




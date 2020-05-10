
from ..namespaces import NS_A, NS_D
from ..envelope import SoapEnvelope
from ..util import createSkelSoapMessage, getBodyEl, getHeaderEl, addElementWithText, \
                   addTypes, getTypes, addScopes, getDocAsString, getScopes, addEPR, \
                   addXAddrs, getXAddrs, _parseAppSequence

from .probematch import ProbeResolveMatch


ACTION_RESOLVE_MATCH = "http://schemas.xmlsoap.org/ws/2005/04/discovery/ResolveMatches"


def createResolveMatchMessage(env):
    doc = createSkelSoapMessage(ACTION_RESOLVE_MATCH)

    bodyEl = getBodyEl(doc)
    headerEl = getHeaderEl(doc)

    addElementWithText(doc, headerEl, "a:MessageID", NS_A, env.getMessageId())
    addElementWithText(doc, headerEl, "a:RelatesTo", NS_A, env.getRelatesTo())
    addElementWithText(doc, headerEl, "a:To", NS_A, env.getTo())

    appSeqEl = doc.createElementNS(NS_D, "d:AppSequence")
    appSeqEl.setAttribute("InstanceId", env.getInstanceId())
    appSeqEl.setAttribute("MessageNumber", env.getMessageNumber())
    headerEl.appendChild(appSeqEl)

    resolveMatchesEl = doc.createElementNS(NS_D, "d:ResolveMatches")
    if len(env.getProbeResolveMatches()) > 0:
        resolveMatch = env.getProbeResolveMatches()[0]
        resolveMatchEl = doc.createElementNS(NS_D, "d:ResolveMatch")
        addEPR(doc, resolveMatchEl, resolveMatch.getEPR())
        addTypes(doc, resolveMatchEl, resolveMatch.getTypes())
        addScopes(doc, resolveMatchEl, resolveMatch.getScopes())
        addXAddrs(doc, resolveMatchEl, resolveMatch.getXAddrs())
        addElementWithText(doc, resolveMatchEl, "d:MetadataVersion", NS_D, resolveMatch.getMetadataVersion())

        resolveMatchesEl.appendChild(resolveMatchEl)

    bodyEl.appendChild(resolveMatchesEl)

    return getDocAsString(doc)


def parseResolveMatchMessage(dom):
    env = SoapEnvelope()
    env.setAction(ACTION_RESOLVE_MATCH)

    env.setMessageId(dom.getElementsByTagNameNS(NS_A, "MessageID")[0].firstChild.data.strip())
    env.setRelatesTo(dom.getElementsByTagNameNS(NS_A, "RelatesTo")[0].firstChild.data.strip())
    env.setTo(dom.getElementsByTagNameNS(NS_A, "To")[0].firstChild.data.strip())

    _parseAppSequence(dom, env)

    nodes = dom.getElementsByTagNameNS(NS_D, "ResolveMatch")
    if len(nodes) > 0:
        node = nodes[0]
        epr = node.getElementsByTagNameNS(NS_A, "Address")[0].firstChild.data.strip()

        typeNodes = node.getElementsByTagNameNS(NS_D, "Types")
        types = []
        if len(typeNodes) > 0:
            types = getTypes(typeNodes[0])

        scopeNodes = node.getElementsByTagNameNS(NS_D, "Scopes")
        scopes = []
        if len(scopeNodes) > 0:
            scopes = getScopes(scopeNodes[0])

        xAddrs = getXAddrs(node.getElementsByTagNameNS(NS_D, "XAddrs")[0])
        mdv = node.getElementsByTagNameNS(NS_D, "MetadataVersion")[0].firstChild.data.strip()
        env.getProbeResolveMatches().append(ProbeResolveMatch(epr, types, scopes, xAddrs, mdv))

    return env






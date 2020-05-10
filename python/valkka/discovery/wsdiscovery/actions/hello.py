
from ..namespaces import NS_A, NS_D
from ..envelope import SoapEnvelope
from ..util import createSkelSoapMessage, getBodyEl, getHeaderEl, addElementWithText, \
                   addTypes, addScopes, getDocAsString, getScopes, getQNameFromValue, \
                   addEPR, addXAddrs, _parseAppSequence, getTypes, getXAddrs


ACTION_HELLO = "http://schemas.xmlsoap.org/ws/2005/04/discovery/Hello"


def createHelloMessage(env):
    doc = createSkelSoapMessage(ACTION_HELLO)

    bodyEl = getBodyEl(doc)
    headerEl = getHeaderEl(doc)

    addElementWithText(doc, headerEl, "a:MessageID", NS_A, env.getMessageId())

    if len(env.getRelatesTo()) > 0:
        addElementWithText(doc, headerEl, "a:RelatesTo", NS_A, env.getRelatesTo())
        relatesToEl = headerEl.getElementsByTagNameNS(NS_A, "RelatesTo")[0]
        relatesToEl.setAttribute("RelationshipType", "d:Suppression")

    addElementWithText(doc, headerEl, "a:To", NS_A, env.getTo())

    appSeqEl = doc.createElementNS(NS_D, "d:AppSequence")
    appSeqEl.setAttribute("InstanceId", env.getInstanceId())
    appSeqEl.setAttribute("MessageNumber", env.getMessageNumber())
    headerEl.appendChild(appSeqEl)

    helloEl = doc.createElementNS(NS_D, "d:Hello")
    addEPR(doc, helloEl, env.getEPR())
    addTypes(doc, helloEl, env.getTypes())
    addScopes(doc, helloEl, env.getScopes())
    addXAddrs(doc, helloEl, env.getXAddrs())
    addElementWithText(doc, helloEl, "d:MetadataVersion", NS_D, env.getMetadataVersion())

    bodyEl.appendChild(helloEl)

    return getDocAsString(doc)


def parseHelloMessage(dom):
    env = SoapEnvelope()
    env.setAction(ACTION_HELLO)

    env.setMessageId(dom.getElementsByTagNameNS(NS_A, "MessageID")[0].firstChild.data.strip())
    env.setTo(dom.getElementsByTagNameNS(NS_A, "To")[0].firstChild.data.strip())

    _parseAppSequence(dom, env)

    relatesToNodes = dom.getElementsByTagNameNS(NS_A, "RelatesTo")
    if len(relatesToNodes) > 0:
        env.setRelatesTo(relatesToNodes[0].firstChild.data.strip())
        env.setRelationshipType(getQNameFromValue( \
            relatesToNodes[0].getAttribute("RelationshipType"), relatesToNodes[0]))

    env.setEPR(dom.getElementsByTagNameNS(NS_A, "Address")[0].firstChild.data.strip())

    typeNodes = dom.getElementsByTagNameNS(NS_D, "Types")
    if len(typeNodes) > 0:
        env.setTypes(getTypes(typeNodes[0]))

    scopeNodes = dom.getElementsByTagNameNS(NS_D, "Scopes")
    if len(scopeNodes) > 0:
        env.setScopes(getScopes(scopeNodes[0]))

    xNodes = dom.getElementsByTagNameNS(NS_D, "XAddrs")
    if len(xNodes) > 0:
        env.setXAddrs(getXAddrs(xNodes[0]))

    env.setMetadataVersion(dom.getElementsByTagNameNS(NS_D, "MetadataVersion")[0].firstChild.data.strip())

    return env

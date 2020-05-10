
from xml.dom import minidom
from ..namespaces import NS_A, NS_D
from ..envelope import SoapEnvelope
from ..util import createSkelSoapMessage, getBodyEl, getHeaderEl, addElementWithText, \
                   addTypes, getTypes, addScopes, getDocAsString, getScopes


ACTION_PROBE = "http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe"


def createProbeMessage(env):
    doc = createSkelSoapMessage(ACTION_PROBE)

    bodyEl = getBodyEl(doc)
    headerEl = getHeaderEl(doc)

    addElementWithText(doc, headerEl, "a:MessageID", NS_A, env.getMessageId())
    addElementWithText(doc, headerEl, "a:To", NS_A, env.getTo())

    if len(env.getReplyTo()) > 0:
        addElementWithText(doc, headerEl, "a:ReplyTo", NS_A, env.getReplyTo())

    probeEl = doc.createElementNS(NS_D, "d:Probe")
    bodyEl.appendChild(probeEl)

    addTypes(doc, probeEl, env.getTypes())
    addScopes(doc, probeEl, env.getScopes())

    return getDocAsString(doc)


def parseProbeMessage(dom):
    env = SoapEnvelope()
    env.setAction(ACTION_PROBE)
    env.setMessageId(dom.getElementsByTagNameNS(NS_A, "MessageID")[0].firstChild.data.strip())

    replyToNodes = dom.getElementsByTagNameNS(NS_A, "ReplyTo")
    if len(replyToNodes) > 0 and \
       isinstance(replyToNodes[0].firstChild, minidom.Text):
        env.setReplyTo(replyToNodes[0].firstChild.data.strip())

    env.setTo(dom.getElementsByTagNameNS(NS_A, "To")[0].firstChild.data.strip())

    typeNodes = dom.getElementsByTagNameNS(NS_D, "Types")
    if len(typeNodes) > 0:
        env.getTypes().extend(getTypes(typeNodes[0]))

    scopeNodes = dom.getElementsByTagNameNS(NS_D, "Scopes")
    if len(scopeNodes) > 0:
        env.getScopes().extend(getScopes(scopeNodes[0]))

    return env





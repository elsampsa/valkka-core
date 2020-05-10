
from ..namespaces import NS_A, NS_D
from ..envelope import SoapEnvelope
from ..util import createSkelSoapMessage, getBodyEl, getHeaderEl, addElementWithText, \
                   addEPR, addTypes, addScopes, getDocAsString, getScopes


ACTION_RESOLVE = "http://schemas.xmlsoap.org/ws/2005/04/discovery/Resolve"


def createResolveMessage(env):
    doc = createSkelSoapMessage(ACTION_RESOLVE)

    bodyEl = getBodyEl(doc)
    headerEl = getHeaderEl(doc)

    addElementWithText(doc, headerEl, "a:MessageID", NS_A, env.getMessageId())
    addElementWithText(doc, headerEl, "a:To", NS_A, env.getTo())

    if len(env.getReplyTo()) > 0:
        addElementWithText(doc, headerEl, "a:ReplyTo", NS_A, env.getReplyTo())

    resolveEl = doc.createElementNS(NS_D, "d:Resolve")
    addEPR(doc, resolveEl, env.getEPR())
    bodyEl.appendChild(resolveEl)

    return getDocAsString(doc)


def parseResolveMessage(dom):
    env = SoapEnvelope()
    env.setAction(ACTION_RESOLVE)

    env.setMessageId(dom.getElementsByTagNameNS(NS_A, "MessageID")[0].firstChild.data.strip())

    replyToNodes = dom.getElementsByTagNameNS(NS_A, "ReplyTo")
    if len(replyToNodes) > 0:
        env.setReplyTo(replyToNodes[0].firstChild.data.strip())

    env.setTo(dom.getElementsByTagNameNS(NS_A, "To")[0].firstChild.data.strip())
    env.setEPR(dom.getElementsByTagNameNS(NS_A, "Address")[0].firstChild.data.strip())

    return env



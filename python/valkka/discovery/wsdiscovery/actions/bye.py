
from ..namespaces import NS_A, NS_D
from ..envelope import SoapEnvelope
from ..util import createSkelSoapMessage, getBodyEl, getHeaderEl, addElementWithText, \
                   addTypes, addScopes, getDocAsString, getScopes, addEPR, \
                   _parseAppSequence


ACTION_BYE = "http://schemas.xmlsoap.org/ws/2005/04/discovery/Bye"


def createByeMessage(env):
    doc = createSkelSoapMessage(ACTION_BYE)

    bodyEl = getBodyEl(doc)
    headerEl = getHeaderEl(doc)

    addElementWithText(doc, headerEl, "a:MessageID", NS_A, env.getMessageId())
    addElementWithText(doc, headerEl, "a:To", NS_A, env.getTo())

    appSeqEl = doc.createElementNS(NS_D, "d:AppSequence")
    appSeqEl.setAttribute("InstanceId", env.getInstanceId())
    appSeqEl.setAttribute("MessageNumber", env.getMessageNumber())
    headerEl.appendChild(appSeqEl)

    byeEl = doc.createElementNS(NS_D, "d:Bye")
    addEPR(doc, byeEl, env.getEPR())
    bodyEl.appendChild(byeEl)

    return getDocAsString(doc)


def parseByeMessage(dom):
    env = SoapEnvelope()
    env.setAction(ACTION_BYE)

    env.setMessageId(dom.getElementsByTagNameNS(NS_A, "MessageID")[0].firstChild.data.strip())
    env.setTo(dom.getElementsByTagNameNS(NS_A, "To")[0].firstChild.data.strip())

    _parseAppSequence(dom, env)

    env.setEPR(dom.getElementsByTagNameNS(NS_A, "Address")[0].firstChild.data.strip())

    return env





import io
import string
import random
import netifaces
from xml.dom import minidom
from .scope import Scope
from .uri import URI
from .namespaces import NS_A, NS_D, NS_S
from .qname import QName


def createSkelSoapMessage(soapAction):
    doc = minidom.Document()

    envEl = doc.createElementNS(NS_S, "s:Envelope")

    envEl.setAttribute("xmlns:a", NS_A)  # minidom does not insert this automatically
    envEl.setAttribute("xmlns:d", NS_D)
    envEl.setAttribute("xmlns:s", NS_S)

    doc.appendChild(envEl)

    headerEl = doc.createElementNS(NS_S, "s:Header")
    envEl.appendChild(headerEl)

    addElementWithText(doc, headerEl, "a:Action", NS_A, soapAction)

    bodyEl = doc.createElementNS(NS_S, "s:Body")
    envEl.appendChild(bodyEl)

    return doc


def addElementWithText(doc, parent, name, ns, value):
    el = doc.createElementNS(ns, name)
    text = doc.createTextNode(value)
    el.appendChild(text)
    parent.appendChild(el)


def addEPR(doc, node, epr):
    eprEl = doc.createElementNS(NS_A, "a:EndpointReference")
    addElementWithText(doc, eprEl, "a:Address", NS_A, epr)
    node.appendChild(eprEl)


def addScopes(doc, node, scopes):
    if scopes is not None and len(scopes) > 0:
        addElementWithText(doc, node, "d:Scopes", NS_D, " ".join([x.getQuotedValue() for x in scopes]))
        if scopes[0].getMatchBy() is not None and len(scopes[0].getMatchBy()) > 0:
            node.getElementsByTagNameNS(NS_D, "Scopes")[0].setAttribute("MatchBy", scopes[0].getMatchBy())


def addTypes(doc, node, types):
    if types is not None and len(types) > 0:
        envEl = getEnvEl(doc)
        typeList = []
        prefixMap = {}
        for type in types:
            ns = type.getNamespace()
            localname = type.getLocalname()
            if prefixMap.get(ns) == None:
                prefix = getRandomStr()
                prefixMap[ns] = prefix
            else:
                prefix = prefixMap.get(ns)
            addNSAttrToEl(envEl, ns, prefix)
            typeList.append(prefix + ":" + localname)
        addElementWithText(doc, node, "d:Types", NS_D, " ".join(typeList))


def addXAddrs(doc, node, xAddrs):
    if xAddrs is not len(xAddrs) > 0:
        addElementWithText(doc, node, "d:XAddrs", NS_D, " ".join([x for x in xAddrs]))


def getDocAsString(doc):
    outStr = None
    stream = io.StringIO(outStr)
    stream.write(doc.toprettyxml())
    return stream.getvalue()


def getBodyEl(doc):
    return doc.getElementsByTagNameNS(NS_S, "Body")[0]


def getHeaderEl(doc):
    return doc.getElementsByTagNameNS(NS_S, "Header")[0]


def getEnvEl(doc):
    return doc.getElementsByTagNameNS(NS_S, "Envelope")[0]


def addNSAttrToEl(el, ns, prefix):
    el.setAttribute("xmlns:" + prefix, ns)


def _parseAppSequence(dom, env):
    nodes = dom.getElementsByTagNameNS(NS_D, "AppSequence")
    if nodes:
        appSeqNode = nodes[0]
        env.setInstanceId(appSeqNode.getAttribute("InstanceId"))
        env.setSequenceId(appSeqNode.getAttribute("SequenceId"))
        env.setMessageNumber(appSeqNode.getAttribute("MessageNumber"))


def _parseSpaceSeparatedList(node):
    if node.childNodes:
        return [item.replace('%20', ' ') \
            for item in node.childNodes[0].data.split()]
    else:
        return []


def extractSoapUdpAddressFromURI(uri):
    val = uri.getPathExQueryFragment().split(":")
    part1 = val[0][2:]
    part2 = None
    if val[1].count('/') > 0:
        part2 = int(val[1][:val[1].index('/')])
    else:
        part2 = int(val[1])
    addr = [part1, part2]
    return addr


def getXAddrs(xAddrsNode):
    return _parseSpaceSeparatedList(xAddrsNode)


def getTypes(typeNode):
    return [getQNameFromValue(item, typeNode) \
                for item in _parseSpaceSeparatedList(typeNode)]


def getScopes(scopeNode):
    matchBy = scopeNode.getAttribute("MatchBy")
    return [Scope(item, matchBy) \
                for item in _parseSpaceSeparatedList(scopeNode)]


def matchScope(src, target, matchBy):

    MATCH_BY_LDAP = "http://schemas.xmlsoap.org/ws/2005/04/discovery/ldap"
    MATCH_BY_URI = "http://schemas.xmlsoap.org/ws/2005/04/discovery/rfc2396"
    MATCH_BY_UUID = "http://schemas.xmlsoap.org/ws/2005/04/discovery/uuid"
    MATCH_BY_STRCMP = "http://schemas.xmlsoap.org/ws/2005/04/discovery/strcmp0"

    if matchBy == "" or matchBy == None or matchBy == MATCH_BY_LDAP or matchBy == MATCH_BY_URI or matchBy == MATCH_BY_UUID:
        src = URI(src)
        target = URI(target)
        if src.getScheme().lower() != target.getScheme().lower():
            return False
        if src.getAuthority().lower() != target.getAuthority().lower():
            return False
        srcPath = src.getPathExQueryFragment()
        targetPath = target.getPathExQueryFragment()
        if srcPath == targetPath:
            return True
        elif targetPath.startswith(srcPath):
            n = len(srcPath)
            if targetPath[n - 1] == srcPath[n - 1] == '/':
                return True
            if targetPath[n] == '/':
                return True
            return False
        else:
            return False
    elif matchBy == MATCH_BY_STRCMP:
        return src == target
    else:
        return False


def getNamespaceValue(node, prefix):
    while node != None:
        if node.nodeType == minidom.Node.ELEMENT_NODE:
            attr = node.getAttributeNode("xmlns:" + prefix)
            if attr != None:
                return attr.nodeValue
        node = node.parentNode
    return ""


def getDefaultNamespace(node):
    while node != None:
        if node.nodeType == minidom.Node.ELEMENT_NODE:
            attr = node.getAttributeNode("xmlns")
            if attr != None:
                return attr.nodeValue
        node = node.parentNode
    return ""


def getQNameFromValue(value, node):
    vals = value.split(":")
    ns = ""
    if len(vals) == 1:
        localName = vals[0]
        ns = getDefaultNamespace(node)
    else:
        localName = vals[1]
        ns = getNamespaceValue(node, vals[0])
    return QName(ns, localName)


def _getNetworkAddrs():
    result = []

    for if_name in netifaces.interfaces():
        iface_info = netifaces.ifaddresses(if_name)
        if netifaces.AF_INET in iface_info:
            for addrDict in iface_info[netifaces.AF_INET]:
                addr = addrDict['addr']
                if addr != '127.0.0.1':
                    result.append(addr)
    return result


def _generateInstanceId():
    return str(random.randint(1, 0xFFFFFFFF))


def getRandomStr():
    return "".join([random.choice(string.ascii_letters) for x in range(10)])


def showEnv(env):
    print("-----------------------------")
    print("Action: %s" % env.getAction())
    print("MessageId: %s" % env.getMessageId())
    print("InstanceId: %s" % env.getInstanceId())
    print("MessageNumber: %s" % env.getMessageNumber())
    print("Reply To: %s" % env.getReplyTo())
    print("To: %s" % env.getTo())
    print("RelatesTo: %s" % env.getRelatesTo())
    print("Relationship Type: %s" % env.getRelationshipType())
    print("Types: %s" % env.getTypes())
    print("Scopes: %s" % env.getScopes())
    print("EPR: %s" % env.getEPR())
    print("Metadata Version: %s" % env.getMetadataVersion())
    print("Probe Matches: %s" % env.getProbeResolveMatches())
    print("-----------------------------")


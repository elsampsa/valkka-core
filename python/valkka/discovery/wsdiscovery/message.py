"""
Top-level SOAP message creation and parsing
"""

import io
from .namespaces import NS_A, NS_S
from .actions import *
from xml.dom import minidom


def createSOAPMessage(env):
    "construct a a raw SOAP XML string, given a prepared SoapEnvelope object"
    if env.getAction() == ACTION_PROBE:
        return createProbeMessage(env)
    if env.getAction() == ACTION_PROBE_MATCH:
        return createProbeMatchMessage(env)
    if env.getAction() == ACTION_RESOLVE:
        return createResolveMessage(env)
    if env.getAction() == ACTION_RESOLVE_MATCH:
        return createResolveMatchMessage(env)
    if env.getAction() == ACTION_HELLO:
        return createHelloMessage(env)
    if env.getAction() == ACTION_BYE:
        return createByeMessage(env)


def parseSOAPMessage(data, ipAddr):
    "parse raw XML data string, return a (minidom) xml document"

    try:
        dom = minidom.parseString(data)
    except Exception:
        #print('Failed to parse message from %s\n"%s": %s' % (ipAddr, data, ex), file=sys.stderr)
        return None

    if dom.getElementsByTagNameNS(NS_S, "Fault"):
        #print('Fault received from %s:' % (ipAddr, data), file=sys.stderr)
        return None

    soapAction = dom.getElementsByTagNameNS(NS_A, "Action")[0].firstChild.data.strip()
    if soapAction == ACTION_PROBE:
        return parseProbeMessage(dom)
    elif soapAction == ACTION_PROBE_MATCH:
        return parseProbeMatchMessage(dom)
    elif soapAction == ACTION_RESOLVE:
        return parseResolveMessage(dom)
    elif soapAction == ACTION_RESOLVE_MATCH:
        return parseResolveMatchMessage(dom)
    elif soapAction == ACTION_BYE:
        return parseByeMessage(dom)
    elif soapAction == ACTION_HELLO:
        return parseHelloMessage(dom)

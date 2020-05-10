import os
import sys
import logging
import click
import netifaces
from wsdiscovery.daemon import WSDiscovery
from wsdiscovery.scope import Scope

try:
    from urlparse import urlparse
except:
    from urllib.parse import urlparse

logging.basicConfig()
logger = logging.getLogger("ws-discovery")
logger.setLevel(logging.INFO)

# an external interface is needed
# darwin has en0, linux has eth0 by default
external_ifaces = ("en0", "eth0")


def get_valid_interface():
    ifaces = netifaces.interfaces()
    valid = list(set(external_ifaces).intersection(ifaces))
    return valid[0] if len(valid) == 1 else None

def get_ip_address(ifname):
    ipv4_addresses = netifaces.ifaddresses(ifname)[netifaces.AF_INET]
    return ipv4_addresses[0]["addr"]


def run(scope=None, capture=None):

    ifname = get_valid_interface()
    if not ifname:
        logger.error("no interface found; use one of: %s", ", ".join(external_ifaces))
        return

    ipaddr = get_ip_address(ifname)
    logger.debug("using %s interface with IPv4 address %s", ifname, ipaddr)

    wsd = WSDiscovery(capture=capture)
    wsd.start()

    if not scope:
        svcs = wsd.searchServices()
    else:
        # we support giving just one scope, for now
        svcs = wsd.searchServices(scopes=[Scope(scope)])

    print("\nDiscovered:\n")
    for service in svcs:
        url = urlparse(service.getXAddrs()[0])
        print(" address: %s" % url.netloc)
        print("  - %s\n" % "\n  - ".join([str(s) for s in service.getScopes()]))
    wsd.stop()


@click.command()
@click.option('--scope', '-s', help='Full scope URI, eg. onvif://www.onvif.org/Model/')
@click.option('--loglevel', '-l',  help='Log level; one of INFO, DEBUG, WARNING, ERROR')
@click.option('--capture', '-c', nargs=1, type=click.File('w'), help='Capture messages to a file')
def discover(scope, loglevel, capture):
    "Discover systems using WS-Discovery"

    if loglevel:
        level = getattr(logging, loglevel, None)
        if not level:
           print("Invalid log level '%s'" % loglevel)
           return
        logger.setLevel(level)

    run(scope=scope, capture=capture)

if __name__ == '__main__':
    discover()

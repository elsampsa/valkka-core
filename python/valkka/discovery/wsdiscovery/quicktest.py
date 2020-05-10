from duranc.gateway.wsdiscovery import WSDiscovery, QName, Scope
wsd = WSDiscovery()
wsd.start()
ret = wsd.searchServices()
for r in ret:
    print(r.getXAddrs())
wsd.stop()

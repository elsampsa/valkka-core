"""
Simple QName implementation
"""


class QName:

    def __init__(self, namespace, localname):
        self._namespace = namespace
        self._localname = localname

    def getNamespace(self):
        return self._namespace

    def getLocalname(self):
        return self._localname

    def getFullname(self):
        return self.getNamespace() + ":" + self.getLocalname()

    def __repr__(self):
        return self.getFullname()




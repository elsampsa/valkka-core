
"""
Simple URI implementation, just for the purposes of service scope matching
"""

try:
    from urllib.parse import unquote  # Python 3
except ImportError:
    from urllib2 import unquote  # Python 2


class URI:

    def __init__(self, uri):
        uri = unquote(uri)
        i1 = uri.find(":")
        i2 = uri.find("@")
        self._scheme = uri[:i1]
        if i2 != -1:
            self._authority = uri[i1 + 1: i2]
            self._path = uri[i2 + 1:]
        else:
            self._authority = ""
            self._path = uri[i1 + 1:]

    def getScheme(self):
        return self._scheme

    def getAuthority(self):
        return self._authority

    def getPath(self):
        return self._path

    def getPathExQueryFragment(self):
        i = self._path.find("?")
        path = self.getPath()
        if i != -1:
            return path[:self._path.find("?")]
        else:
            return path


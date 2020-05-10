"""
Simple scope object implementation.

"""


class Scope:

    def __init__(self, value, matchBy=None):
        self._matchBy = matchBy
        self._value = value

    def getMatchBy(self):
        return self._matchBy

    def getValue(self):
        return self._value

    def getQuotedValue(self):
        return self._value.replace(' ', '%20')

    def __repr__(self):
        if self.getMatchBy() == None or len(self.getMatchBy()) == 0:
            return self.getValue()
        else:
            return self.getMatchBy() + ":" + self.getValue()



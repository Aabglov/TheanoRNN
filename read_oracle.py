EOS = '\xa4'

def cleanOracle():
    with open('/Users/keganrabil/Desktop/oracle.txt','r') as f:
        data = f.read()
    f.close()

    cleaned = data.lower().replace('\n\n',EOS)
    return cleaned

def getVocab(cleaned):
    s = list(set([j for j in cleaned]))

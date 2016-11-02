###############################################################
#                        Data Utility
###############################################################
import os


# Toy problem:
# first couple paragraphs from a random Federalist paper.
# I had listened to Hamilton a lot before beginning this project.
def loadText(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path,"data",filename),'r') as f:
        data = f.read()
    f.close()
    corpus = data#.lower()
    corpus_len = len(corpus)
    print("data loaded: {}".format(corpus_len))
    return corpus



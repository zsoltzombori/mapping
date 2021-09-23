import rdflib
import re

class Query:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'r') as f:
            text = f.read().replace("\\n\\\n", " ")

        namespace_pattern = "prefix (\w*): <([^>]*)>"
        matches = re.findall(namespace_pattern, text)
        self.namespaces = {}
        for m in matches:
            self.namespaces[m[0]] = rdflib.Namespace(m[1])

        sql_pattern = "sql=(.*)\n"
        matches = re.findall(sql_pattern, text)
        assert len(matches) == 1
        self.sql_query = matches[0]

        sparql_pattern = "sparql.*(SELECT.*)\n"
        matches = re.findall(sparql_pattern, text)
        assert len(matches) == 1
        self.sparql_query = matches[0]

        triples_pattern = "sparql.*WHERE {(.*)}\n"
        matches = re.findall(triples_pattern, text)
        assert len(matches) == 1
        self.triples = recover_triples(matches[0])
        self.logic = triples_to_logic(self.triples)

        selects_pattern = "sparql.*SELECT(.*)WHERE"
        matches = re.findall(selects_pattern, text)
        assert len(matches) == 1
        self.selects = recover_selects(matches[0])




def recover_triples(triplestring):
    parts = triplestring.split()
    triples = []
    subject = None
    predicate = None
    position = 0
    for p in parts:
        if p == ".":
            continue
        if position == 0: # expecting subject
            subject = p
            position = 1
        elif position == 1: # expecting object
            predicate = p
            position = 2
        elif position == 2:
            if p[-1] == ";":
                triples.append((subject, predicate, p[:-1]))
                position = 1
            else:
                triples.append((subject, predicate, p))
                position = 0
    return triples

def recover_selects(selectstring):
    selects = selectstring.split()

    # TODO only support list of ?varname at the moment
    for s in selects:
        if s[0] != "?":
            return None
    return selects

def triples_to_logic(triples):
    print(triples)
    logic = []
    reformulated = []
    identities = []
    for t in triples:
        if t[1] == "rdf:type":
            logic.append([t[2],t[0]])
            h = hash(t[2])
            reformulated.
        else:
            logic.append([t[1],t[0], t[2]])
    print("logic: ", logic)
    return

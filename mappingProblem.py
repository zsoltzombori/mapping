import rdflib

import util

class MappingProblem:
    def __init__(self, schema, ontology, mappings):
        self.schema = schema
        self.ontology = ontology
        self.mappings = mappings
        self.graph = rdflib.Graph()
        self.graph.parse(ontology)
        self.queries = []

        self.namespaces = namespaces = {
            'owl': rdflib.Namespace("http://www.w3.org/2002/07/owl#"),
            'rdfs': rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#"),
        }

        


    def update_namespaces(self):
        for q in self.queries:
            for key in q.namespaces:
                self.namespaces[key] = q.namespaces[key]

    def get_classes(self):
        s = "SELECT ?c WHERE {?c rdf:type owl:Class}"
        qres = self.graph.query(s, initNs = self.namespaces)
        return util.sparql_result_to_list(qres, "c")

    def get_properties(self):
        s = "SELECT ?c WHERE {?c rdf:type owl:ObjectProperty}"
        qres = self.graph.query(s, initNs = self.namespaces)
        return util.sparql_result_to_list(qres, "c")

    def functional_property(self, property):
        if property in ("rdfs:label", "rdfs:comment"):
            return 1
        
        assert property[0] == ":", property
        property = "<{}{}>".format(self.namespaces[""], property[1:])
        s = "SELECT ?c WHERE {?c rdf:type owl:FunctionalProperty}"
        qres = self.graph.query(s, initNs = self.namespaces)
        for row in qres:
            if row.c.n3() == property:
                return 1

        s = "SELECT ?c WHERE {?c rdf:type owl:DatatypeProperty}"
        qres = self.graph.query(s, initNs = self.namespaces)
        for row in qres:
            if row.c.n3() == property:
                return 1

        s = "SELECT ?c WHERE {?c rdf:type owl:InverseFunctionalProperty}"
        qres = self.graph.query(s, initNs = self.namespaces)
        for row in qres:
            if row.c.n3() == property:
                return -1

        return 0

    def domain_of_functional_property(self, property):
        s = "SELECT ?c WHERE {" + property + " rdf:type owl:FunctionalProperty; rdfs:domain ?c}"
        qres = self.graph.query(s, initNs = self.namespaces)
        domains = []
        for row in qres:
            if isinstance(row.c, rdflib.term.URIRef):
                domains.append(row.c.n3())
        return domains

    def transform2sql(self, query):
        
        tables = [] # sql tables used to be joined
        objects = {} # keep track of where variables are to be found

        # collect all type edges
        for t in query.triples:
            if t[1] == "rdf:type":
                if t[2] in self.mappings:
                    table = self.mappings[t[2]]
                    tables.append(table)
                    objects[t[0]] = (table, "id")
                else:
                    assert False, "Missing mapping for " + t[2]

        wheres = []
        for t in query.triples:
            if t[1] == "rdf:type":
                continue

            if t[1] + ".to" in self. mappings:
                property_to = self.mappings[t[1] + ".to"]
            else:
                assert False, "Missing mapping for " + t[1] + ".to"
            
            functional = self.functional_property(t[1])
            if t[1] in self.mappings: # separate table for the property
                property_table = self.mappings[t[1]]
                tables.append(property_table)
                property_from = self.mappings[t[1] + ".from"]
                from_table = objects[t[0]][0]
                if t[2] not in objects:
                    objects[t[2]] = (property_table, property_to)
                else:
                    to_table = objects[t[2]][0]
                    wheres.append("{}.{} = {}.id".format(property_table, property_to, to_table))
                wheres.append("{}.id = {}.{}".format(from_table, property_table, property_from))
                # wheres.append("{}.{} = {}.id".format(property_table, property_from, to_table))

            elif functional == 1:
                property_table = objects[t[0]][0]
                if t[2] not in objects:
                    objects[t[2]] = (property_table, property_to)
                else:
                    to_table = objects[t[2]][0]
                    wheres.append("{}.{} = {}.id".format(property_table, property_to, to_table))
            elif functional == -1:
                property_table = objects[t[2]][0]
                if t[0] not in objects:
                    objects[t[0]] = (property_table, property_to)
                else:
                    to_table = objects[t[0]][0]
                    wheres.append("{}.{} = {}.id".format(property_table, property_to, to_table))
            else:
                assert False, "Non functional property should have a mapping {}".format(t[1])                

            

        tables = list(set(tables))
        if query.selects == None:
            sel = "count(*)"
        else:
            sel = []
            for s in query.selects:
                sel.append("{}.{}".format(objects[s][0], objects[s][1]))
            sel = ",".join(sel)
        sql_query = "SELECT " + sel + " FROM " +",".join(tables)
        if len(wheres) > 0:
            sql_query += " WHERE " + " AND ".join(wheres)
        return sql_query

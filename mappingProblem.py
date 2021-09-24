import rdflib
import psycopg2
import os

import util
import query

class MappingProblem:
    def __init__(self, schema, ontology, mappings, use_db=True):
        self.schema = schema
        self.ontology = ontology
        self.mappings = mappings
        self.graph = rdflib.Graph()
        self.graph.parse(ontology)
        self.queries = []
        self.use_db = use_db

        self.namespaces = namespaces = {
            'owl': rdflib.Namespace("http://www.w3.org/2002/07/owl#"),
            'rdfs': rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#"),
        }

        if self.use_db:
            self.cursor = util.init_db()
            util.inspect_database(self.cursor)
            util.inspect_schema(self.cursor, self.schema)
            self.cursor.execute("SET search_path TO {}, public;".format(self.schema))

            self.db_tables = util.table_names(self.cursor, self.schema)
            print(self.db_tables)
        else:
            self.cursor = None

        self.classes = self.get_classes()
        print(self.classes)
        self.properties = self.get_properties()
        print(self.properties)


    def add_query_dir(self, query_dir):
        for filename in os.listdir(query_dir):
            if filename.endswith(".qpair"):
                fullname = os.path.join(query_dir, filename)
                # print(fullname)
                q = query.Query(fullname)
                self.queries.append(q)
        self.update_namespaces()
        self.queries.sort(key=lambda x: x.filename)

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

    def sparql2sql(self, query):
        froms= [] # sql tables used to be joined
        objects = util.DictOfList() # keep track of where variables are to be found
        
        for t in query.logic:
            if t[0] in self.mappings:
                m = self.mappings[t[0]]
            else:
                print("Missing mapping for class/predicate: ", t[0])
                m = ("dummyT_"+t[0], "dummyA1_"+t[0], "dummyA2_"+t[0])
            froms.append(m[0])

            objects.add(t[1], (m[0], m[1]))
            if len(t) == 3:
                objects.add(t[2], (m[0], m[2]))

        selects = ["{}.{}".format(table, attribute) for (table, attribute) in objects.values()]
        wheres = []
        for k in objects.keys():
            columns = objects.get(k)
            for i in range(len(columns)):
                t1, a1 = columns[i]
                for j in range(i+1, len(columns)):
                    t2, a2 = columns[j]
                    if (t1 != t2 or a1 != a2):
                        wheres.append("{}.{}={}.{}".format(t1,a1,t2,a2))

        froms = list(set(froms))
        selects = list(set(selects))
        wheres = list(set(wheres))
        sql = "SELECT " + ", ".join(selects) + " FROM " + ", ".join(froms)
        if len(wheres) > 0:
            sql += " WHERE " + " AND ".join(wheres)
        return sql

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


    

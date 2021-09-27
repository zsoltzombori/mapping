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

            self.db_attributes = util.attributes(self.cursor, self.schema)
            self.db_tables = self.db_attributes.keys()
        else:
            self.cursor = None
            self. db_tables = [
                "persons",
                "conferences",
                "reviews",
                "reviewers",
                "conference_members",
                "authors",
                "co_authors",
                "documents",
                "paper_full_versions",
                "paper_abstracts",
                "program_committees",
                "pc_members",
                "program_committee_chairs",
                "co_author_paper",
                "paper_reviewer",
                "program_committee_member"
            ]
            self.db_attributes={
                "persons": ["id", "name", "email"],
                "conferences": ["id", "site_url", "accepts_hardcopy_submissions", "logo_url", "date", "name", "reviews_per_paper"],
                "reviews": ["id"],
                "reviewers": ["id"],
                "conference_members": ["id"],
                "authors": ["id"],
                "co_authors": ["id"],
                "documents": ["id"],
                "paper_full_versions": ["id"],
                "paper_abstracts": ["id"],
                "program_committees": ["id"],
                "pc_members": ["id"],
                "program_committee_chairs": ["id"],
                "co_author_paper": ["id"],
                "paper_reviewer": ["id"],
                "program_committee_member": ["id"],
            }


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

    def expand_uri(self, uri):
        uri_parts = uri.split(":")
        if len(uri_parts) == 1:
            return uri
        elif (len(uri_parts) == 2):
            if uri_parts[0] in self.namespaces:
                return "<"+self.namespaces[uri_parts[0]] + uri_parts[1] + ">"
            else:
                assert False, uri

        else:
            assert False, uri

    def remove_uri(self, uri):
        if uri[0] != "<" or uri[-1] != ">":
            return uri

        uri2 = uri[1:-1]
        for ns in self.namespaces:
            value = self.namespaces[ns]
            if uri2.find(value) == 0:
                return ns + ":" + uri2[len(value):]
        return uri


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

    def domain_of_property(self, property):
        s1 = "SELECT ?c WHERE {" + property + " rdf:type owl:ObjectProperty; rdfs:domain ?c}"
        qres1 = self.graph.query(s1, initNs = self.namespaces)
        s2 = "SELECT ?c WHERE {" + property + " rdf:type owl:DatatypeProperty; rdfs:domain ?c}"
        qres2 = self.graph.query(s2, initNs = self.namespaces)
        domains = []
        rows = list(qres1) + list(qres2)
        for row in rows:
            if isinstance(row.c, rdflib.term.URIRef):
                domains.append(row.c.n3())
            # else:
            #     bnodes = list(row)
            #     i = 0
            #     while i< len(bnodes):
            #         bnode = bnodes[i]
            #         all_edges = list(self.graph[bnode])
            #         union_edges = list(self.graph[bnode:rdflib.term.URIRef('http://www.w3.org/2002/07/owl#unionOf')])
            #         print("all: ", all_edges)
            #         print("union: ", union_edges)
            #         bnodes += union_edges
            #         i += 1
        return domains

    def range_of_property(self, property):
        s = "SELECT ?c WHERE {" + property + " rdf:type owl:ObjectProperty; rdfs:range ?c}"
        qres = self.graph.query(s, initNs = self.namespaces)
        ranges = []
        for row in qres:
            if isinstance(row.c, rdflib.term.URIRef):
                ranges.append(row.c.n3())
        return ranges

    def get_candidates(self, t, classes):
            candidates = []

            if len(t) == 2: # this is a class membership triple
                tables = util.top_candidates(t[0], self.db_tables)
                for T in tables:
                    attributes = util.top_candidates("id", self.db_attributes.get(T))
                    for A in attributes:
                        candidates.append((T, A))
                return candidates

            if t[0] in ("rdfs:label", "rdfs:comment"):
                predname = t[0].split(":")[1]
                if t[1] in classes.keys():
                    current_class = classes.get(t[1])
                    tables = util.top_candidates(current_class, self.db_tables)
                    for T in tables:
                        attributes = util.top_candidates("predname", self.db_attributes.get(T))
                        for A in attributes:
                            candidates.append((T, A))
                return candidates
            

            # other property triple                
            domains = self.domain_of_property(t[0])
            domains = [self.remove_uri(d) for d in domains]
            ranges = self.range_of_property(t[0])
            ranges = [self.remove_uri(r) for r in ranges]
            
            # case 1: relation lives in the domain table
            for d in domains:
                tables = util.top_candidates(d, self.db_tables)
                for T in tables:
                    attributes_from = util.top_candidates("id", self.db_attributes.get(T))
                    attributes_to = util.top_candidates(t[0], self.db_attributes.get(T))
                    for A_from in attributes_from:
                        for A_to in attributes_to:
                            candidates.append((T, A_from, A_to))
                            
            # case 2: relation lives in the range table
            for r in ranges:
                tables = util.top_candidates(r, self.db_tables)
                for T in tables:
                    attributes_from = util.top_candidates(t[0], self.db_attributes.get(T))
                    attributes_to = util.top_candidates("id", self.db_attributes.get(T))
                    for A_from in attributes_from:
                        for A_to in attributes_to:
                            candidates.append((T, A_from, A_to))
                
            # case 3: relation has a separate table
            tables = util.top_candidates(t[0], self.db_tables)
            for T in tables:
                attributes_from = []
                attributes_to = []
                for d in domains:
                    attributes_from += util.top_candidates(d, self.db_attributes.get(T))
                for r in ranges:
                    attributes_to += util.top_candidates(r, self.db_attributes.get(T))
                for A_from in attributes_from:
                    for A_to in attributes_to:
                        candidates.append((T, A_from, A_to))


            return candidates
            
        

    def sparql2sql(self, query):

        # collect all class memberships:
        classes = util.DictOfList()
        for t in query.logic:
            if len(t) == 2:
                classes.add(t[1], t[0])

        froms= [] # sql tables used to be joined
        objects = util.DictOfList() # keep track of where variables are to be found
                
        for t in query.logic:
            candidates = self.get_candidates(t, classes)
            print("Triple: ", t)
            for c in candidates:
                print("    ", c)

            # if t[0] in self.mappings:
            #     m = self.mappings[t[0]]
            # else:
            #     print("Missing mapping for class/predicate: ", t[0])
            #     m = ("dummyT_"+t[0], "dummyA1_"+t[0], "dummyA2_"+t[0])

            m = candidates[0] # TODO more candidates
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

        selects = list(set(selects))
        froms = list(set(froms))
        wheres = list(set(wheres))

        if query.selects is None:
            sql = "SELECT COUNT(*)"
        else:
            sql = "SELECT " + ", ".join(selects)
        sql += " FROM " + ", ".join(froms)
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


    

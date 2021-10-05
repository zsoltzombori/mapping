import rdflib
import psycopg2

import util
import mappingProblem

schema = "cmt_renamed"
schema = "cmt_structured"
#schema = "conference_structured"
#schema = "conference_renamed"
#schema = "npd_user_tests"

ontology = "RODI/data/{}/ontology.ttl".format(schema)
query_dir = "RODI/data/{}/queries".format(schema)

problem = mappingProblem.MappingProblem(schema, ontology, use_db=True)
problem.add_query_dir(query_dir)

success = 0
for query in problem.queries:
    print("\n**********************")
    print(query.filename)
    print("sparql query: ", query.sparql_query)
    # qres = problem.graph.query(query.sparql_query, initNs = problem.namespaces)
    # for row in qres:
    #     print(f"{row}")
    
    print("\nsql query: ", query.sql_query)
    problem.cursor.execute(query.sql_query)
    result = problem.cursor.fetchall()

    sql_candidates = problem.sparql2sql(query)
    curr_succ = False
    for sql2 in sql_candidates:
        print("\n transformed query: ", sql2)
        problem.cursor.execute(sql2)
        result2 = problem.cursor.fetchall()

        if result == result2:
            success+=1
            curr_succ = True
            print("!!!!PASSED!!!!")
            break
    if not curr_succ:
        print("!!!!FAILED!!!!")
        # util.print_result(result)
        # util.print_result(result2)
        # assert False
print("Passed {} out of {}".format(success, len(problem.queries)))
        
        





# print("All triples with 'acceptedBy' in subject position")
# result = list(g[rdflib.URIRef("http://cmt#acceptedBy")])
# for i, r in enumerate(result):
#     print(i, r)

# print("All triples with 'acceptedBy' in predicate position")
# result = list(g[:rdflib.URIRef("http://cmt#acceptedBy")])
# for i, r in enumerate(result):
#     print(i, r)

# print("Entities with range 'Bid'")
# result = list(g[:rdflib.term.URIRef('http://www.w3.org/2000/01/rdf-schema#range'): rdflib.URIRef("http://cmt#Bid")])
# for i, r in enumerate(result):
#     print(i, r)

# print("persons")
# result = list(g[:rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'): rdflib.URIRef("http://cmt#Person")])
# for i, r in enumerate(result):
#     print(i, r)

# qres = g.query("""SELECT ?s WHERE { ?s a <c:> }""")
# for row in qres:
#     print(f"{row.s}")





# ##########################






# xxx

# try connecting to the database
# conn = None
# try:
#     params = config()
#     conn = psycopg2.connect(**params)
#     cursor = conn.cursor()

#     inspect(cur)
#     xxx

#     # execute a statement
#     print('PostgreSQL database version:')
#     cur.execute('SELECT version()')

#     # display the PostgreSQL database server version
#     db_version = cur.fetchone()
#     print(db_version)
       
#     # close the communication with the PostgreSQL
#     cur.close()
# except (Exception, psycopg2.DatabaseError) as error:
#     print(error)
# finally:
#     if conn is not None:
#         conn.close()
#         print('Database connection closed.')
        

# # restore a database from dump

# try:
#     params = config()
#     conn = psycopg2.connect(**params)
#     cur = conn.cursor()

#     cur.execute("pg_restore {}".format(dump_file))
       
#     # close the communication with the PostgreSQL
#     cur.close()
# except (Exception, psycopg2.DatabaseError) as error:
#     print(error)
# finally:
#     if conn is not None:
#         conn.close()
#         print('Database connection closed.')

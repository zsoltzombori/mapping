import rdflib
import psycopg2

import util

cursor = util.init_db()
util.inspect_database(cursor)

# from cmt_renamed import problem
from cmt_structured import problem
util.inspect_schema(cursor, problem.schema)
cursor.execute("SET search_path TO {}, public;".format(problem.schema))

tables = util.table_names(cursor, problem.schema)
print(tables)

classes = problem.get_classes()
print(classes)

properties = problem.get_properties()
print(properties)

for query in problem.queries:
    print("\n**********************")
    print(query.filename)
    print("sparql query: ", query.sparql_query)
    # qres = problem.graph.query(query.sparql_query, initNs = problem.namespaces)
    # for row in qres:
    #     print(f"{row}")
    
    print("\nsql query: ", query.sql_query)
    cursor.execute(query.sql_query)
    result = cursor.fetchall()

    sql2 = problem.transform2sql(query)
    print("\n transformed query: ", sql2)
    cursor.execute(sql2)
    result2 = cursor.fetchall()

    if result == result2:
        print("!!!!PASSED!!!!")
    else:
        util.print_result(result)
        util.print_result(result2)
        assert False
        
        
        





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

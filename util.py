from configparser import ConfigParser
import psycopg2
import rdflib

# parse db configurations from file
def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db

def init_db():
    params = config()
    conn = psycopg2.connect(**params)
    cursor = conn.cursor()
    return cursor

# print table
def print_result(table):
    print("Found {} rows:".format(len(table)))
    for t in table:
        print(t)

def inspect_schema(cursor, schema):
    s = "select table_name, column_name from information_schema.columns where table_schema='{}'".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    print("\nTables for schema '{}':".format(schema))
    print_result(result)

def table_names(cursor, schema):
    s = "select table_name from information_schema.columns where table_schema='{}'".format(schema)
    cursor.execute(s)
    result = cursor.fetchall()
    return [r[0] for r in result]

def inspect_database(cursor):
    s = "SELECT * FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';"
    cursor.execute(s)
    result = cursor.fetchall()
    print("\nAll tables for all schemas:")
    schema = ""
    for row in result:
        if row[0] != schema:
            schema = row[0]
            print(schema)
        print("    {}".format(row[1]))


def sparql_result_to_list(qres, colname):
    result = []
    for row in qres:
        if isinstance(row[colname], rdflib.term.URIRef): # TODO UNDERSTAND WHAT TO DO WITH BNode-s (blank nodes)
            result.append(row[colname].n3())
    return result

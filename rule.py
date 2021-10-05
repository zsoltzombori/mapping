import datetime
import decimal

class Variable:
    def __init__(self, number):
        self.n = number

# an atom is a list
# the first element is the predicate
# subsequent elements are constants or Variables 
class Rule:
    def __init__(self, spec, cursor, schema, preds):
        self.head = spec[0]
        self.body = spec[1:]
        self.cursor = cursor
        self.schema = schema
        self.preds = preds

    def apply_subst(self, atom, subst):
        args = atom[1:]
        args2 = []
        varcnt = 0
        for a in args:
            if isinstance(a, Variable) and a.n in subst:
                a2 = subst[a.n]
            else:
                a2 = a
            if isinstance(a2, Variable):
                varcnt += 1
            args2.append(a2)
        return varcnt, atom[:1] + args2

    def type_mismatch(self, sql_type, obj):
        if sql_type == "NULL":
            result = obj is not None
        elif sql_type == "bool":
            result = not isinstance(obj, bool)
        elif sql_type in ("real", "double"):
            result = not isinstance(obj, float)
        elif sql_type in ("smallint", "integer", "bigint"):
            result = not isinstance(obj, int)
        elif sql_type == "numeric":
             result = not isinstance(obj, decimal.Decimal)
        elif sql_type in ("varchar", "text", "character varying", "character"):
            result = not isinstance(obj, str)
        elif sql_type == "date":
            result = not isinstance(obj, datetime.date)
        elif sql_type in ("time", "timetz"):
            result = not isinstance(obj, datetime.time)
        elif sql_type in ("datetime", "datetimetz"):
            result = not isinstance(obj, datetime.datetime)
        elif sql_type == "interval":
            result = not isinstance(obj, datetime.timedelta)
        elif sql_type == "ARRAY":
            result = not isinstance(obj, list)
        else:
            result = False
        return result

    def unifying_facts(self, atom):
        args = atom[1:]
        if len(args) == 1: # unary predicate
            assert not isinstance(args[0], Variable), "Should not be variable"
            targets = self.preds["unary"]
            unary = True
        elif len(args) == 2: #binary predicates
            assert not (isinstance(args[0], Variable) and isinstance(args[1], Variable)), "Should not be variable"
            targets = self.preds["binary"]
            unary = False
        else:
            assert False, "Only unary/binary predicates are supported"

        matches = []
        if unary:
            for t in targets:
                table0, column0, type0 = t
                sql = "SELECT {} from \"{}\" where {}=%s".format(column0, table0, column0)
                if self.type_mismatch(type0, args[0]):
                    continue
                
                self.cursor.execute(sql, (args[0],))
                result = self.cursor.fetchall()
                for r in result:
                    if r[0] is not None:
                        matches.append( (atom[0], (table0, column0), (r[0],)))
        else:
            for t in targets:
                ((table0, column0, type0), (table1, column1, type1)) = t
                assert table0 == table1
                sql = "SELECT {}, {} from \"{}\"".format(column0, column1, table0)
                
                if isinstance(args[0], Variable) and not self.type_mismatch(type1, args[1]):
                    self.cursor.execute(sql + " where {}=%s".format(column1),  (args[1],))
                elif isinstance(args[1], Variable) and not self.type_mismatch(type0, args[0]):
                    self.cursor.execute(sql + " where {}=%s".format(column0),  (args[0],))
                elif not (self.type_mismatch(type0, args[0]) or self.type_mismatch(type1, args[1])):
                    self.cursor.execute(sql + " where {}=%s and {}=%s".format(column0, column1),  (args[0], args1))
                    self.cursor.execute(sql, (args[0], args[1]))

                result = self.cursor.fetchall()
                for r in result:
                    if r[0] is not None and r[1] is not None:
                        matches.append((atom[0], (table0, column0, column1), (r[0], r[1])))

        return matches

    def candidate_mappings(self, body, subst, mapping):

        if len(body) == 0:
            return [mapping]
        
        body_subst = [self.apply_subst(b, subst) for b in body]
        # todo select the one with the least amount of variables
        B = body_subst[0][1]
        body2 = body[1:]

        result = []

        # collect all facts that unify with B
        # have the same arity as B and has the same constant in the same position
        # since Subst(B) contains some constants, Facts is much smaller than the dataset
        facts = self.unifying_facts(B)
        for f in facts:

            #     Subs2 is Subst + mgu(Subst(B), F)
            subst2 = subst.copy() # shallow copy is fine here, I think
            for arg, value in zip(B[1:], f[2]):
                if isinstance(arg, Variable):
                    subst2[arg.n] = value

            #     Mapping2 is Mapping + {pred(B) -> pred(F)}
            mapping2 = mapping.copy()
            mapping2[f[0]] = (f[1], f[2])

            new_mappings = self.candidate_mappings(body2, subst2, mapping2)
            result += new_mappings
        return result

   
   #     score = embeddingScore(Mapping2)
   #     # if there are too many candidate facts in Facts, sample from them based on score
   #     # for negative samples, it is ok to use a threshold for score
   #     # for positive samples, farther candidates should have a chance too


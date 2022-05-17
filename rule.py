import datetime
import decimal
import re

import util

class Variable:
    def __init__(self, number):
        self.n = number

class Constant:
    def __init__(self, number):
        self.n = number

class StringFromCols:
    def __init__(self, number):
        self.n = number

    def find_split(self, text, columns, row):
        
        # find all matches
        matches = []
        for v, column in zip(row, columns):
            if v is None:
                continue
            # we assume '/' delimiting
            index = text.find('/' + v + '/') # first check middle match
            if index > 0:
                matches.append((index+1, len(v), column, v))
            else:
                index = text.find(v + '/')
                if index == 0:
                    matches.append((index, len(v), column, v))
                else:
                    index = text.find('/' + v)
                    if index > 0 and (index + len(v) + 1 == len(text)):
                        matches.append((index+1, len(v), column, v))            
        matches.sort()
        result_cols = []
        result_vals = []
        match_count = 0
        curr_index = 0
        for m in matches:
            if m[0] < curr_index:
                continue

            padding = text[curr_index:m[0]]
            if len(padding) > 0:
                # padding should not contain "/{number}/" pattern
                if re.search("/[0-9]+/", padding):
                    return False, None
                if re.search("/[0-9]+-[0-9]+-[0-9]+/", padding):
                    return False, None
                result_cols.append(padding)
                result_vals.append(padding)
                
            result_cols.append(m[2])
            result_vals.append(m[3])            
            match_count += 1
            curr_index = m[0] + m[1]

        if curr_index < len(text)-1: # padding at the end
            padding = text[curr_index:]
            if re.search("/[0-9]+", padding):
                return False, None
            if re.search("/[0-9]+-[0-9]+-[0-9]+/", padding):
                return False, None
            result_cols.append(padding)
            result_vals.append(padding)

        if match_count < 1:
            return False, None
        
        return True, (match_count, tuple(result_cols), tuple(result_vals))
        
            
    def align_unary(self, text, cursor, tables):
        matches = []
        for table in tables: 
            columns = tables[table]
            columns = [c[0] for c in columns]
            colstrings = ["CAST({} as CHARACTER VARYING)".format(c) for c in columns]
            sql = "select distinct {} from {};".format(", ".join(colstrings), table)
            cursor.execute(sql)
            result = cursor.fetchall()
            duplicate_checker = {}
            matches_for_table = []
            succ_cnt = 0
            best_count = 0
            for r in result:
                success, match = self.find_split(text, columns, r)
                succ_cnt += success
                if success and match not in duplicate_checker:
                    duplicate_checker[match] = True
                    match_count, matched_cols, matched_vals = match
                    best_count = max(best_count ,match_count)
                    if match_count >=1:
                        matches_for_table.append((match_count, table, matched_cols, matched_vals))

            matches_for_table.sort(reverse=True)
            
            for cnt, table, cols, vals in matches_for_table:
                if cnt >= best_count - 1:
                    cols = list(cols)
                    cols.insert(0, table)
                    matches.append((self.n, cols, vals))
        return matches

    def align_binary(self, text1, text2, cursor, tables):
        matches = []
        for table in tables:
            columns = tables[table]
            columns = [c[0] for c in columns]
            colstrings = ["CAST({} as CHARACTER VARYING)".format(c) for c in columns]
            sql = "select distinct {} from {};".format(", ".join(colstrings), table)
            cursor.execute(sql)
            result = cursor.fetchall()
            duplicate_checker = {}
            matches_for_table = []
            best_count = 0
            for r in result:
                success1, match1 = self.find_split(text1, columns, r)
                success2, match2 = self.find_split(text2, columns, r)
                if success1 and success2 and (match1, match2) not in duplicate_checker:
                    duplicate_checker[(match1, match2)] = True
                    match_count1, matched_cols1, matched_vals1 = match1
                    match_count2, matched_cols2, matched_vals2 = match2
                    best_count = max(match_count1+match_count2, best_count)
                    if match_count1 >=1 and match_count2 >=1: # TODO match_count threshold
                        matches_for_table.append((match_count1+match_count2, table, matched_cols1, matched_vals1, matched_cols2, matched_vals2))

            matches_for_table.sort(reverse=True)
            for cnt, table, cols1, vals1, cols2, vals2 in matches_for_table:
                if cnt >= best_count -1:
                    cols1 = list(cols1)
                    cols1.insert(0, table)
                    cols2 = list(cols2)
                    cols2.insert(0, table)
                    matches.append((self.n, cols1+cols2, vals1+vals2))
        return matches
    

# an atom is a list
# the first element is the predicate
# subsequent elements are constants or Variables 
class Rule:
    def __init__(self, spec, cursor, preds):
        self.head = spec[0]
        self.body = spec[1:]
        self.cursor = cursor
        self.preds = preds

    # return all the mappings that yield a proof of fact
    def get_support(self, fact):
        if len(fact) != len(self.head):
            return [], []
        if fact[0] != self.head[0]:
            return [], []

        subst = {}
        # target = ["SOS"] + fact # TODO
        target = ["SOS", self.head[0], "PREDEND"]
        for h_arg, f_arg in zip(self.head[1:], fact[1:]):
            if isinstance(h_arg, Variable):
                subst[h_arg.n] = f_arg
                target.append("var_{}".format(h_arg.n))
            elif h_arg != f_arg:
                return [], []
        target.append("EOH")

        return self.candidate_mappings(self.body, subst, {}, target)


    def unifying_facts(self, atom):
        args = atom[1:]
        if isinstance(atom[0], StringFromCols):
            if not isinstance(args[0], str):
                return []

            if len(args) == 1:
                matches = atom[0].align_unary(args[0], self.cursor, self.preds["table"])
            else:
                if not isinstance(args[1], str):
                    return []
                matches = atom[0].align_binary(args[0], args[1], self.cursor, self.preds["table"])
            return matches
        
        if len(args) == 1: # unary predicate
            assert not isinstance(args[0], (Variable, Constant)), "Unary argument should not be variable/constant"
            targets = self.preds["unary"]
            unary = True
        elif len(args) == 2: #binary predicates
            assert (not isinstance(args[0], (Variable, Constant))) or (not isinstance(args[1], (Variable, Constant))), "Both arguments should not be variable/constant"
            targets = self.preds["binary"]
            unary = False
        else:
            assert False, "Only unary/binary predicates are supported"

        temp = []
        for t in targets:
            temp += targets[t]
        targets = temp

        matches = []
        if unary:
            for t in targets:
                table0, column0, type0 = t
                sql = "SELECT DISTINCT \"{}\" from \"{}\" where \"{}\"=%s".format(column0, table0, column0)
                if util.type_match(type0, args[0]):
                    self.cursor.execute(sql, (args[0],))
                    result = self.cursor.fetchall()
                    for r in result:
                        if r[0] is not None:
                            matches.append( (atom[0], (table0, column0), (r[0],)))
        else:
            for t in targets:
                ((table0, column0, type0), (table1, column1, type1)) = t
                assert table0 == table1
                sql = "SELECT DISTINCT \"{}\", \"{}\" from \"{}\"".format(column0, column1, table0)

                result = None
                if isinstance(args[0], (Variable, Constant)):
                    if util.type_match(type1, args[1]):
                        self.cursor.execute(sql + " where \"{}\"=%s".format(column1),  (args[1],))
                        result = self.cursor.fetchall()
                elif isinstance(args[1], (Variable, Constant)):
                    if util.type_match(type0, args[0]):
                        self.cursor.execute(sql + " where \"{}\"=%s".format(column0),  (args[0],))
                        result = self.cursor.fetchall()
                elif util.type_match(type0, args[0]) and util.type_match(type1, args[1]):
                    self.cursor.execute(sql + " where \"{}\"=%s and \"{}\"=%s".format(column0, column1),  (args[0], args[1]))
                    result = self.cursor.fetchall()

                if result is not None:
                    for r in result:
                        if r[0] is not None and r[1] is not None:
                            matches.append((atom[0], (table0, column0, column1), (r[0], r[1])))

        return matches

    def candidate_mappings(self, body, subst, mapping, target):
        if len(body) == 0:
            return [mapping], [target + ["EOS"]]
        
        body_subst = [apply_subst(b, subst) for b in body]
        # TODO select the one with the least amount of variables
        B = body_subst[0][1]
        B2 = body[0]
        body2 = body[1:]

        mappings_out = []
        targets_out = []

        # collect all facts that unify with B
        # have the same arity as B and has the same constant in the same position
        # since Subst(B) contains some constants, Facts is much smaller than the dataset
        facts = self.unifying_facts(B)
        for f in facts:
            constants_match = True

            # we don't want an unary predicate that is just the projection of an existing binary predicate
            redundant_pred = False
            for m in mapping:
               p,v = mapping[m]
               if f[1] == p and f[2] == v:
                   redundant_pred = True
                   break
               if len(p) == 3:
                   if f[1] == (p[0], p[1]) and f[2][0] == v[0]:
                       redundant_pred = True
                       break
                   if f[1] == (p[0], p[2]) and f[2][0] == v[1]:
                       redundant_pred = True
                       break
            if redundant_pred:
                break
            
            #     Mapping2 is Mapping + {pred(B) -> pred(F)}
            mapping2 = mapping.copy()
            mapping2[f[0]] = (f[1], f[2])

            args = []
            for a, v in zip(B2[1:], f[2]):
                if isinstance(a, Variable):
                    args.append("var_{}".format(a.n))
                elif isinstance(a, Constant):
                    if isinstance(v, str):
                        v_parts = v.split()
                        v = "_".join(v_parts)
                    args.append(v)

            target2 = target + [util.pred2name(f[1])] + args + ["EOP"]

            #     Subs2 is Subst + mgu(Subst(B), F)
            subst2 = subst.copy() # shallow copy is fine here, I think
            for arg, value in zip(B[1:], f[2]):
                if isinstance(arg, Variable):
                    subst2[arg.n] = value
                if isinstance(arg, Constant):
                    if arg.n in mapping2:
                        if mapping2[arg.n] != value:
                            constants_match = False
                            break
                    else:
                        mapping2[arg.n] = (("const",), (value,))

            if constants_match:
                new_mappings, new_targets = self.candidate_mappings(body2, subst2, mapping2, target2)
                mappings_out += new_mappings
                targets_out += new_targets
        return mappings_out, targets_out

   
   #     score = embeddingScore(Mapping2)
   #     # if there are too many candidate facts in Facts, sample from them based on score
   #     # for negative samples, it is ok to use a threshold for score
   #     # for positive samples, farther candidates should have a chance too

def apply_subst(atom, subst):
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


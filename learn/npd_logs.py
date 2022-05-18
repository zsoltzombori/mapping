import sys, os
import re

def parse_result(text, ispositive=True):
    if ispositive:
        prefix="Positive"
        size_prefix="pos"
    else:
        prefix = "Negative"
        size_prefix="neg"

    size_pattern = size_prefix + " sizes:  [0-9]* [0-9]* ([0-9]*)"
    size_matches = re.findall(size_pattern, text)
    if len(size_matches) == 0:
        return None
    else:
        size = int(size_matches[0])
    
    pattern = prefix + " top1: (.*), top5: (.*), top10"
    matches = re.findall(pattern, text)
    if len(matches) == 0:
        return None
    else:
        match = matches[-1]
        top1 = float(match[0])
        top5 = float(match[1])
        return size, top1, top5
    

def process_file(filename):
    with open(filename, 'r') as f:
        text = f.read().replace("\\n\\\n", " ")

    pos_result = parse_result(text, ispositive=True)
    neg_result = parse_result(text, ispositive=False)
    return pos_result, neg_result
    


directory = "out/exp115/"

cnt = 0
neg_cnt = 0
pos_cnt = 0
pos_top1 = 0
pos_top5 = 0
neg_top1 = 0
neg_top5 = 0
pos_top1_weighted = 0
pos_top5_weighted = 0
neg_top1_weighted = 0
neg_top5_weighted = 0
missing_pos = []
missing_neg = []
done_files = []

for file in os.listdir(os.fsencode(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".cout"):
         fullname = os.path.join(directory, filename)
         pos_result, neg_result = process_file(fullname)

         if pos_result is None:
             missing_pos.append(filename)
             continue
         else:
             done_files.append(filename)
             
         cnt += 1

         posc, post1, post5 = pos_result
         pos_cnt += posc
         pos_top1 += post1
         pos_top5 += post5
         pos_top1_weighted += posc * post1
         pos_top5_weighted += posc * post5

         if neg_result is None:
             missing_neg.append(filename)
             continue
          
         negc, negt1, negt5 = neg_result
         neg_cnt += negc
         neg_top1 += negt1
         neg_top5 += negt5
         neg_top1_weighted += negc * negt1
         neg_top5_weighted += negc * negt5

cnt = max(1, cnt)
pos_cnt = max(1, pos_cnt)
neg_cnt = max(1, neg_cnt)
         
pos_top1 /= cnt
pos_top5 /= cnt
pos_top1_weighted /= pos_cnt
pos_top5_weighted /= pos_cnt

neg_top1 /= cnt
neg_top5 /= cnt
neg_top1_weighted /= neg_cnt
neg_top5_weighted /= neg_cnt


print("Missing: ", len(missing_pos))
print("Predicates: ", cnt)
print("Positive: ", pos_top1, pos_top5, pos_top1_weighted, pos_top5_weighted)
print("Negative: ", neg_top1, neg_top5, neg_top1_weighted, neg_top5_weighted)

missing_pos = sorted(missing_pos)
for f in missing_pos:
    print(f)

# done_files = sorted(done_files)
# for f in done_files:
#     print(f)

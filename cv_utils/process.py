import os

from object_list import categories as cat

with open('another_list.txt', 'r') as file:
    lines = file.readlines()

nms = []
for line in lines:
    nm = line.split()[0].strip()
    nms.append(nm.lower())


objs = []
for c in cat:
    nm = c['name'].lower()
    objs.append(nm)

cnt = len(objs) + 1

for nm in nms:
    if nm not in objs:
        cat.append({"id": cnt, "name": nm})
        cnt += 1

with open('new_object_list.py', 'w') as file:
    file.write("categories = [\n")
    for c in cat:
        file.write(str(c) + ",\n")
    file.write("]\n")
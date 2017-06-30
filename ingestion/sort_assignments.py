from string import ascii_lowercase
from collections import defaultdict
from glob import glob

def dictionary_from_files(search_path):
    d = defaultdict(set)
    print("Searching %s" % search_path)
    for ii in glob(search_path):
        filename = ii.split('/')[-1]
        if filename in ascii_lowercase or filename == "other":
            print("Skipping %s" % filename)
            continue
        with open(ii) as infile:
            for jj in infile:
                fields = jj.split('\t')
                if '\t' not in jj:
                    continue
                wiki = fields[1]
                if not wiki:
                    continue

                key = wiki[0].lower()
                if key not in ascii_lowercase:
                    key = "other"
                reduced = jj.strip()
                if '\t' in reduced:
                    d[key].add(reduced)
    return d

def write_files(base, lines):
    for ii in lines:
        with open("%s/%s" % (base, ii), 'w') as outfile:
            for jj in sorted(lines[ii], key=lambda x: x.split('\t')[1]):
                outfile.write("%s\n" % jj)

if __name__ == "__main__":
    for path in ["data/internal/page_assignment/" + x for x in
                 ["direct", "unambiguous", "ambiguous"]]:
        lines = dictionary_from_files("%s/*" % path)
        write_files(path, lines)

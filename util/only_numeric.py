
if __name__ == "__main__":
    import sys
    infile = sys.argv[1]
    outfile = sys.argv[2]
                                        
    with open(infile, 'r') as infile, open(outfile, 'w') as outfile:
        for ii in infile:
            outfile.write("%s\n" % " ".join(x for x in ii.split()
                                            if "|" in x or ":" in x))
    

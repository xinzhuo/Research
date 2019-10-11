import codecs

infile = "hl_software_2019-09-27_14-25-52_live.log"

phrases = ["lock pos", "today", "yday", "new order", "trade"]
with codecs.open(infile, "rb", encoding='utf-16') as f:
    for line in f:
        for p in phrases:
            if p in line:
                print(line)


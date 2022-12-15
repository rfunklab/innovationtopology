import re
from ast import literal_eval

break_str = 'persistent homology intervals in dim {}:'

def replace_set_notation(s):
    return (s.replace('{', '[')
            .replace('}', ']'))

def replace_list_notation(s):
    return (s.replace('[', '(')
            .replace(']', ')'))

def replace_set_and_list_notation(s):
    return replace_list_notation(replace_set_notation(s))

def unpack_representatives_dim0(p):
    # inclusion time included
    print(p)
    if ':' in p:
        return literal_eval(replace_list_notation(p.replace('\n','')
                                                    .replace(' ', '')))
    # otherwise, no inclusion time included, so parse and add inclusion of 0
    evaled = literal_eval(replace_set_and_list_notation(p.replace('\n','')
                                                        .replace(' ', '')))
    # add in inclusion times which for dim 0 is trivial
    return {evaled : 0}

def unpack_representatives_high_order(p):
    # split on carriage returns and chop off the start and end carriage returns
    splits = [s for s in p.split('\n') if (s != ' ' and s != '')]
    splits = [s if s[0] != ' ' else s[1:] for s in splits]
    # replace space between simplex list and simplex inclusion time with a colon then 
    # reformat into a dict of simplices:inclusion time
    second_splits = [literal_eval('{'+','.join([replace_list_notation((spl.replace(' ', ':')
                                                            .replace('{', '')
                                                            .replace('}', ''))) for spl in s.split(', ')])+'}')
                    for s in splits]
    
    return second_splits

def unpack_representatives(p, d):
    if d == 0:
        return unpack_representatives_dim0(p)
    if d > 0:
        return unpack_representatives_high_order(p)

def unpack_barcode_with_representatives(p, d):
    
    # regpat = '(\[(\d+.\d+|\d{1}),(\d+\.\d+| )\):)'
    regpat = '(\[(-?[\d.]+(?:e-?\d+)?|\d{1}),(-?[\d.]+(?:e-?\d+)?| )\):)'
    bar_locs = [(m.start(0), m.end(0)) for m in re.finditer(regpat, p)]

    if len(bar_locs) == 0:
        return []

    barcode = []
    for i in range(len(bar_locs)-1):
        # this interval
        this_bar = bar_locs[i]
        # next interval
        next_bar = bar_locs[i+1]

        # find the data setting between this interval and the next
        this_ix = this_bar[1] + 1
        next_ix = next_bar[0]
        to_parse = p[this_ix:next_ix]

        this_bar_str = p[this_bar[0]:this_bar[1]]

        # literal evaluation of bar into tuple of (birth, death)
        this_literal_bar = literal_eval(this_bar_str.replace('[','(').replace(':',''))
        # unpack representative data within this bar
        reps = unpack_representatives(to_parse,d)

        barcode.append([this_literal_bar, reps])

    # parse final bar
    # last bars often have weird structure
    final_bar = bar_locs[-1]
    final_ix = final_bar[1] + 1
    to_parse = p[final_ix:]

    final_bar_str = p[final_bar[0]:final_bar[1]]
    last_literal_bar = literal_eval(final_bar_str.replace('[','(').replace(':',''))
    # unpack representative data within this bar
    reps = unpack_representatives(to_parse,d)

    barcode.append([last_literal_bar, reps])

    return barcode

def unpack_ripser_stdout(result_str, d):
    this_break_str = break_str.format(d)
    next_break_str = break_str.format(d+1)

    this_break = result_str.find(this_break_str) + len(this_break_str)
    next_break = result_str.find(next_break_str)

    parseable = result_str[this_break:next_break]

    if break_str not in parseable:
        return unpack_barcode_with_representatives(parseable, d)
    return []
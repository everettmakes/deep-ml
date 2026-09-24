def find_duplicates(records):
    # records: list of hashable items (ints or strings)
    # return a list of values that appear more than once,
    # each listed once, ordered by the position of its second occurrence
    seen = set()
    duplicates = set()
    result = []
    for r in records:
        if r in seen:
            if r not in duplicates:
                result.append(r)
                duplicates.add(r)
        else:
            seen.add(r)
            
        
    return result

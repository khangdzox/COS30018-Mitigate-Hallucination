
def find_all_subset_index(subset, sequence):
    """
    Find all the start indices of the subset in the sequence.
    """
    ans = []
    for i in range(len(sequence) - len(subset) + 1):
        if sequence[i:i + len(subset)] == subset:
            ans.append(i)
    return ans

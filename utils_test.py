from utils import partial_levenshtein, levenshtein


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2
    assert levenshtein("", "") == 0
    assert levenshtein("a", "") == 1
    assert levenshtein("", "a") == 1
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("Fort Washington Avenue", "Washington Avenue") == 5
    print("levenshtein tests passed.")

def test_partial_levenshtein():
    assert partial_levenshtein("Fort Washington Avenue", "Washington Avenue") == (0, 5)
    assert partial_levenshtein("Fort Washington Avenue", "Washington Ave") == (0, 5)
    assert partial_levenshtein("Fort Washington Avenue", "Washington Ave.") == (1, 5)
    assert partial_levenshtein("Fort Washington Avenue", "Washington Street") == (5, 5)
    assert partial_levenshtein("Fort Washington Avenue, New York", "Washington Street") == (5, 5)
    assert partial_levenshtein("Fort Washington Avenue, New York", "washington Street") == (5, 5)
    assert partial_levenshtein("Fort Washington Avenue, New York", "washington Street", case_insensitive=False) == (6, 5)
    print("partial_levenshtein tests passed.")

if __name__ == "__main__":
    test_levenshtein()
    test_partial_levenshtein()
    print("All tests passed.")
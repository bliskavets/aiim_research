import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unsloth  # noqa: must precede trl
from src.grpo_posstats_trainer import new_accumulator, accumulate_posstats

def test_accumulate():
    acc = new_accumulator(8)
    C = np.array([[1.,2.,3.],[4.,5.,6.]])
    lp = np.array([[-1.,-2.,-3.],[-4.,-5.,-6.]])
    mask = np.array([[1,1,0],[1,1,1]])
    correct = np.array([1.,0.])
    accumulate_posstats(acc, C, lp, mask, correct)
    a = acc["all"]
    assert list(a["n"][:3]) == [2,2,1]
    assert list(a["C"][:3]) == [5,7,6]           # pos0:1+4, pos1:2+5, pos2:6(row1 only)
    assert list(a["C2"][:3]) == [17,29,36]       # 1+16, 4+25, 36
    assert list(a["lp"][:3]) == [-5,-7,-6]
    cor = acc["correct"]                          # row0 only
    assert list(cor["n"][:3]) == [1,1,0]
    assert list(cor["C"][:3]) == [1,2,0]
    inc = acc["incorrect"]                        # row1 only
    assert list(inc["n"][:3]) == [1,1,1]
    assert list(inc["C"][:3]) == [4,5,6]
    # accumulation is additive across batches
    accumulate_posstats(acc, C, lp, mask, correct)
    assert list(acc["all"]["n"][:3]) == [4,4,2]
    print("PASS test_accumulate")

if __name__ == "__main__":
    test_accumulate(); print("ALL PASSED")

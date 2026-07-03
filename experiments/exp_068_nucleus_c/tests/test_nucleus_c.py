import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nucleus_c import nucleus_C

def test_basic():
    lp = torch.log(torch.tensor([[0.6,0.3,0.1]]))          # one token, N=3
    C,n = nucleus_C(lp, top_p=0.7, min_k=1)                 # cum=[.6,.9,1]; ≤.7 -> n=1
    assert int(n[0])==1 and abs(C[0].item() - (-math.log(0.6)))<1e-5
    C,n = nucleus_C(lp, top_p=0.95, min_k=1)                # ≤.95 -> {.6,.9} n=2
    assert int(n[0])==2
    assert abs(C[0].item() - (-(math.log(0.6)+math.log(0.3))/2))<1e-5
    C,n = nucleus_C(lp, top_p=1.5, min_k=1)                 # all 3 (cum<=1.5)
    assert int(n[0])==3
    print("PASS basic")

def test_min_k():
    lp = torch.log(torch.tensor([[0.98,0.01,0.01]]))       # top-1 already > top_p
    C,n = nucleus_C(lp, top_p=0.7, min_k=1)                 # count(≤.7)=0 -> clamp 1
    assert int(n[0])==1 and abs(C[0].item() - (-math.log(0.98)))<1e-5
    print("PASS min_k")

def test_batch_and_bounds():
    lp = torch.log(torch.tensor([[[0.9,0.05,0.05]],[[0.2,0.2,0.2]]]))  # (2,1,3)
    C,n = nucleus_C(lp, top_p=0.8, min_k=1)
    # row0: cum .9>.8 -> count 0 -> n=1 ; row1: cum .2,.4 ≤.8, .6? wait sums:.2,.4,.6 all ≤.8 -> n=3
    assert int(n[0,0])==1 and int(n[1,0])==3
    assert (n>=1).all()
    print("PASS batch")

if __name__=="__main__":
    test_basic(); test_min_k(); test_batch_and_bounds(); print("ALL PASSED")

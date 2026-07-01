import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.adaptive_discount import g_p1, g_pc1, g_c1, g_pc2

def test_p1():
    g = g_p1(2048, tau=1024.0, floor=0.3, device=torch.device("cpu"))
    assert abs(g[0].item()-1.0) < 1e-6          # t=0 -> 1
    assert (g[1:] < g[:-1]).all()               # decreasing
    assert (g >= 0.3-1e-6).all() and (g <= 1.0+1e-6).all()
    assert g[1024].item() > 0.6                 # gentle: floor+.5*(1-floor)=0.65
    print("PASS p1")

def test_pc1():
    G,T=4,6
    ema=torch.randn(G,T)*2+11
    g=g_pc1(ema, m=torch.tensor(11.0), sd=torch.tensor(2.0), tau=1024.0, floor=0.3)
    assert g.shape==(G,T)
    assert (g>=0).all() and (g<=1.0+1e-6).all()
    # positional decrease when sigmoid factor held ~equal: check with constant ema
    emac=torch.full((1,T),11.0)
    gc=g_pc1(emac, torch.tensor(11.0), torch.tensor(2.0), 1024.0, 0.3)[0]
    assert (gc[1:] < gc[:-1]).all()
    print("PASS pc1")

def test_c1():
    s=torch.tensor([[2.0,1.0,0.5,0.1]])
    g=g_c1(s, s_ref=torch.tensor(1.0), g_min=0.2)
    # s>=s_ref -> clip to 1 ; s<s_ref -> s/s_ref, floored at 0.2
    assert abs(g[0,0].item()-1.0)<1e-6          # 2.0 -> clip 1
    assert abs(g[0,1].item()-1.0)<1e-6          # 1.0 -> 1
    assert abs(g[0,2].item()-0.5)<1e-6          # 0.5/1
    assert abs(g[0,3].item()-0.2)<1e-6          # 0.1 -> floor 0.2
    print("PASS c1")

def test_pc2():
    C=torch.tensor([[22.0, 11.0, 1.0]])         # C_ref=11
    g=g_pc2(C, C_ref=torch.tensor(11.0), tau=1e9, g_min=0.2, g_max=1.5)  # tau huge -> pos~1
    assert abs(g[0,0].item()-1.5)<1e-3          # 22/11=2 -> clip to g_max 1.5 (BOOST)
    assert abs(g[0,1].item()-1.0)<1e-3          # 11/11=1
    assert abs(g[0,2].item()-0.2)<1e-3          # 1/11~0.09 -> floor 0.2
    print("PASS pc2")

if __name__=="__main__":
    test_p1();test_pc1();test_c1();test_pc2();print("ALL PASSED")

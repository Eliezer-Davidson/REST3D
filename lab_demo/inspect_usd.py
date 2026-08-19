from pxr import Usd, UsdGeom, Gf
import sys
def info(path, label):
    print(f"\n===== {label} =====\n{path}")
    st = Usd.Stage.Open(path)
    if not st: print("  FAILED to open"); return
    up = UsdGeom.GetStageUpAxis(st); mpu = UsdGeom.GetStageMetersPerUnit(st)
    dp = st.GetDefaultPrim()
    print(f"  upAxis={up}  metersPerUnit={mpu}  defaultPrim={dp.GetPath() if dp else None}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = dp if dp else st.GetPseudoRoot()
    try:
        b = cache.ComputeWorldBound(root).ComputeAlignedRange()
        mn, mx = b.GetMin(), b.GetMax()
        ext = mx - mn
        print(f"  world bbox min={tuple(round(v,3) for v in mn)} max={tuple(round(v,3) for v in mx)}")
        print(f"  extent (m if mpu=1) = {tuple(round(v,3) for v in ext)}")
    except Exception as e:
        print("  bbox failed:", e)
    print("  top-level prims:")
    for p in list(root.GetChildren())[:25]:
        print(f"    {p.GetName()}  <{p.GetTypeName()}>")
info("/home/user/work/unitree_sim_isaaclab/assets/objects/small_warehouse/small_warehouse_digital_twin.usd", "WAREHOUSE")
info("/home/user/work/unitree_sim_isaaclab/assets/objects/PackingTable/PackingTable.usd", "PACKING TABLE")

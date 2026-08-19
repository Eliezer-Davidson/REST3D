"""Build a demonstrable lab scene in MuJoCo: floor + walls + 4 benches + a shelf unit
+ a G1-scale stand-in + authored labware (beakers, Erlenmeyer flasks, vials, test tubes,
rack). Renders headless PNGs. Labware are dynamic (freejoint) so the G1 can knock them
about; tables/shelf/walls are static. Meters, Z-up."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

ROOT = "/home/user/work/REST3D/lab_demo"
PROPS = f"{ROOT}/props"; os.makedirs(PROPS, exist_ok=True)


def revolve(profile, nseg=48):
    """profile: list of (r,z) tracing the silhouette; start/end on axis (r=0) -> watertight."""
    prof = np.asarray(profile, float)
    V, F = [], []
    ring_start = []  # index of first vert of each ring, or single pole index
    for (r, z) in prof:
        if r <= 1e-9:
            ring_start.append(("pole", len(V))); V.append((0.0, 0.0, z))
        else:
            ring_start.append(("ring", len(V)))
            for k in range(nseg):
                a = 2 * np.pi * k / nseg
                V.append((r * np.cos(a), r * np.sin(a), z))
    for i in range(len(prof) - 1):
        t0, s0 = ring_start[i]; t1, s1 = ring_start[i + 1]
        if t0 == "ring" and t1 == "ring":
            for k in range(nseg):
                k2 = (k + 1) % nseg
                a, b, c, d = s0 + k, s0 + k2, s1 + k2, s1 + k
                F.append((a, b, c)); F.append((a, c, d))
        elif t0 == "pole" and t1 == "ring":
            for k in range(nseg):
                k2 = (k + 1) % nseg
                F.append((s0, s1 + k, s1 + k2))
        elif t0 == "ring" and t1 == "pole":
            for k in range(nseg):
                k2 = (k + 1) % nseg
                F.append((s0 + k, s0 + k2, s1))
    return np.asarray(V, float), np.asarray(F, int)


def save_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V: fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for f in F: fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


# ---- labware profiles (meters), base at z=0 ----------------------------------
LABWARE = {
    # beaker: slight taper, ~7cm dia, 10cm tall
    "beaker": [(0,0),(0.034,0),(0.034,0.006),(0.036,0.10),(0.030,0.10),(0.030,0.006),(0,0.006)],
    # Erlenmeyer flask: conical body -> narrow neck, ~11cm tall
    "flask":  [(0,0),(0.048,0),(0.048,0.012),(0.013,0.078),(0.013,0.11),(0.016,0.115),
               (0.012,0.115),(0.009,0.078),(0.040,0.010),(0,0.006)],
    # vial: ~1.6cm dia, 4.5cm tall, small cap
    "vial":   [(0,0),(0.008,0),(0.008,0.040),(0.009,0.040),(0.009,0.047),(0.006,0.047),(0,0.043)],
    # test tube: rounded bottom, ~1.5cm dia, 10cm
    "tube":   [(0,0),(0.004,0.001),(0.0072,0.006),(0.0075,0.10),(0.0060,0.10),(0.0058,0.008),(0,0.002)],
}
for name, prof in LABWARE.items():
    V, F = revolve(prof, 40)
    save_obj(f"{PROPS}/{name}.obj", V, F)
    print(f"  {name}.obj  {len(V)}v {len(F)}f  z_top={max(p[1] for p in prof):.3f}m")

# tube rack: a slab with 2x4 sockets (approx as a solid block; tubes stand in front)
rk_V = np.array([[x,y,z] for x in(-0.06,0.06) for y in(-0.025,0.025) for z in(0,0.03)])
# build a box mesh (12 tris)
def box_mesh(hx,hy,hz):
    c=np.array([[sx*hx,sy*hy,sz*hz] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)])
    faces=[(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),(2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    return c, np.array(faces)
rV,rF = box_mesh(0.065,0.028,0.018); rV[:,2]+=0.018
save_obj(f"{PROPS}/rack.obj", rV, rF)
print(f"  rack.obj  {len(rV)}v {len(rF)}f")

# =============================================================================
# scene
# =============================================================================
def table(name, x, y):
    # solid bench: body block + overhanging top slab; top surface at 0.90m
    return f"""    <body name="{name}" pos="{x} {y} 0">
      <geom type="box" size="1.10 0.34 0.44" pos="0 0 0.44" rgba="0.62 0.63 0.66 1"/>
      <geom type="box" size="1.20 0.39 0.02" pos="0 0 0.90" rgba="0.30 0.32 0.36 1"/>
    </body>"""

def item(mesh, x, y, z, rgba, idx):
    return f"""    <body name="{mesh}_{idx}" pos="{x} {y} {z}">
      <freejoint/>
      <geom type="mesh" mesh="{mesh}" rgba="{rgba}" density="900"
            friction="1.0 0.02 0.001" condim="4"/>
    </body>"""

TABLES = {"benchA":(-1.55,-1.20), "benchB":(1.55,-1.20), "benchC":(-1.55,1.20), "benchD":(1.55,1.20)}
TOP = 0.925  # place labware base just above the 0.90 top slab

items = []
red="0.80 0.18 0.16 0.80"; blue="0.16 0.42 0.85 0.80"; grn="0.25 0.70 0.42 0.80"
amb="0.85 0.62 0.20 0.85"; clr="0.72 0.80 0.85 0.55"; pur="0.55 0.30 0.70 0.80"
i=0
def place(mesh,bx,by,dx,dy,rgba):
    global i; i+=1; items.append(item(mesh, bx+dx, by+dy, TOP, rgba, i))
# bench A: beakers + flasks
bx,by=TABLES["benchA"]
place("beaker",bx,by,-0.5,0.05,blue); place("beaker",bx,by,-0.2,-0.05,grn)
place("flask",bx,by,0.2,0.05,red); place("flask",bx,by,0.55,-0.05,blue)
# bench B: rack + tubes + vials
bx,by=TABLES["benchB"]
items.append(f"""    <body name="rack_1" pos="{bx-0.4} {by} {TOP}"><freejoint/>
      <geom type="mesh" mesh="rack" rgba="0.22 0.22 0.26 1" density="700" condim="4"/></body>"""); i+=1
for j,(dx,col) in enumerate([(-0.45,red),(-0.4,grn),(-0.35,amb),(-0.30,blue)]):
    place("tube",bx,by,dx,0.0,col)
place("vial",bx,by,0.1,0.05,amb); place("vial",bx,by,0.22,-0.03,grn); place("vial",bx,by,0.33,0.04,pur)
place("beaker",bx,by,0.6,0.0,clr)
# bench C: flasks + beaker
bx,by=TABLES["benchC"]
place("flask",bx,by,-0.4,0.0,grn); place("flask",bx,by,0.0,0.05,amb); place("beaker",bx,by,0.45,-0.03,red)
# bench D: mixed
bx,by=TABLES["benchD"]
place("beaker",bx,by,-0.5,0.0,blue); place("vial",bx,by,-0.1,0.05,red)
place("vial",bx,by,0.0,-0.05,blue); place("flask",bx,by,0.4,0.0,clr)

# shelf unit against back wall (static): 3 planks on 2 uprights + a few boxes
shelf = """    <body name="shelf" pos="0 3.4 0">
      <geom type="box" size="0.02 0.30 0.90" pos="-1.3 0 0.90" rgba="0.35 0.35 0.38 1"/>
      <geom type="box" size="0.02 0.30 0.90" pos="1.3 0 0.90" rgba="0.35 0.35 0.38 1"/>
      <geom type="box" size="1.32 0.30 0.02" pos="0 0 0.55" rgba="0.5 0.5 0.53 1"/>
      <geom type="box" size="1.32 0.30 0.02" pos="0 0 1.05" rgba="0.5 0.5 0.53 1"/>
      <geom type="box" size="1.32 0.30 0.02" pos="0 0 1.55" rgba="0.5 0.5 0.53 1"/>
      <geom type="box" size="0.14 0.14 0.11" pos="-0.8 0 0.68" rgba="0.7 0.6 0.35 1"/>
      <geom type="box" size="0.12 0.12 0.10" pos="-0.2 0 0.67" rgba="0.6 0.35 0.3 1"/>
      <geom type="box" size="0.13 0.13 0.11" pos="0.7 0 1.18" rgba="0.4 0.5 0.6 1"/>
    </body>"""

asset_meshes = "\n".join(f'    <mesh name="{m}" file="{m}.obj"/>' for m in ["beaker","flask","vial","tube","rack"])
table_xml = "\n".join(table(n,x,y) for n,(x,y) in TABLES.items())
items_xml = "\n".join(items)

xml = f"""<mujoco model="lab_demo">
  <compiler meshdir="{PROPS}" angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.82 0.86 0.92" rgb2="0.4 0.45 0.55" width="256" height="256"/>
    <texture name="ft" type="2d" builtin="checker" rgb1="0.72 0.72 0.70" rgb2="0.64 0.64 0.62" width="512" height="512"/>
    <material name="fm" texture="ft" texrepeat="10 10" reflectance="0.05"/>
    <material name="wall" rgba="0.86 0.87 0.88 1"/>
{asset_meshes}
  </asset>
  <worldbody>
    <light pos="0 0 4.5" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <light pos="-3 -3 3.5" dir="0.6 0.6 -1" directional="true" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="12 12 0.1" material="fm"/>
    <geom name="wall_back"  type="box" size="6 0.05 1.6" pos="0 3.8 1.6" material="wall"/>
    <geom name="wall_left"  type="box" size="0.05 4 1.6" pos="-4.2 0 1.6" material="wall"/>
    <geom name="wall_right" type="box" size="0.05 4 1.6" pos="4.2 0 1.6" material="wall"/>
    <!-- G1-scale stand-in (1.32 m) in the aisle -->
    <body name="G1_standin" pos="0 -0.2 0">
      <geom type="capsule" fromto="0 0 0.10 0 0 1.10" size="0.14" rgba="0.15 0.16 0.18 1"/>
      <geom type="sphere" size="0.12" pos="0 0 1.22" rgba="0.15 0.16 0.18 1"/>
    </body>
{table_xml}
{shelf}
{items_xml}
  </worldbody>
</mujoco>"""

open(f"{ROOT}/lab_demo.xml","w").write(xml)
print("wrote lab_demo.xml")

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print(f"model OK: {m.nbody} bodies, {m.ngeom} geoms, {m.nmesh} meshes")

from PIL import Image
r = mujoco.Renderer(m, 1080, 1920)
views = [("A_overview", 55, -22, 11.0, (0,0.4,0.9)),
         ("B_aisle",   90, -12, 7.5, (0,0.2,0.9)),
         ("C_bench",   35, -28, 5.0, (-1.55,-1.2,0.95))]
for tag,az,el,dist,look in views:
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(f"{ROOT}/lab_{tag}.png")
    print("wrote", f"lab_{tag}.png")
print("DONE")

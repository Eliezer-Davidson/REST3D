"""Lab scene v2: denser authored glassware + REST3D-harvested equipment (centrifuges,
under-counter fridge, rolling stool, rack) rescaled to realistic sizes. Renders headless."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, glob
import mujoco

ROOT = "/home/user/work/REST3D/lab_demo"
PROPS = f"{ROOT}/props"
HARV = "/home/user/work/REST3D/output/lab-room-design-2/_render_meshes"

def load_obj(path):
    V, F = [], []
    for l in open(path):
        if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(t.split("/")[0])-1 for t in l.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, int)

def save_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V: fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for f in F: fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

# ---- preprocess harvested props: recenter (xy->0, base z->0) + scale to target ----
# key: (filename substring, target_size, scale_mode)  mode 'max' or 'height'
HARVSPEC = {
    "centrifugeL": ("large_white_centrifuge", 0.34, "max"),
    "centrifugeS": ("small_white_centrifuge", 0.24, "max"),
    "centrifugeW": ("white_centrifuge_on_left", 0.30, "max"),
    "fridge":      ("white_under-counter_refrigerator", 0.85, "height"),
    "stool":       ("black_rolling_stool", 0.62, "height"),
    "labrack":     ("pink_plastic_rack", 0.16, "max"),
}
harv_dims = {}
for key, (sub, target, mode) in HARVSPEC.items():
    src = glob.glob(f"{HARV}/*{sub}*.obj")
    if not src:
        print(f"  [skip] {key}: no mesh for '{sub}'"); continue
    V, F = load_obj(src[0])
    mn, mx = V.min(0), V.max(0); ext = mx - mn
    s = target / (ext[2] if mode == "height" else ext.max())
    V = (V - np.array([ (mn[0]+mx[0])/2, (mn[1]+mx[1])/2, mn[2] ])) * s  # xy-center, base at 0
    save_obj(f"{PROPS}/h_{key}.obj", V, F)
    harv_dims[key] = (V.max(0) - V.min(0))
    print(f"  h_{key}.obj  scale={s:.4f}  final_ext={tuple(round(v,3) for v in harv_dims[key])}")

# =============================================================================
absmesh = lambda name: f'{PROPS}/{name}.obj'
def M(name): return f'    <mesh name="{name}" file="{PROPS}/{name}.obj"/>'

def table(name, x, y):
    return f"""    <body name="{name}" pos="{x} {y} 0">
      <geom type="box" size="1.10 0.34 0.44" pos="0 0 0.44" rgba="0.62 0.63 0.66 1"/>
      <geom type="box" size="1.20 0.39 0.02" pos="0 0 0.90" rgba="0.28 0.30 0.34 1"/>
    </body>"""

_i = [0]
def dyn(mesh, x, y, z, rgba, dens=900):
    _i[0]+=1
    return f"""    <body name="{mesh}_{_i[0]}" pos="{x} {y} {z}"><freejoint/>
      <geom type="mesh" mesh="{mesh}" rgba="{rgba}" density="{dens}" friction="1.0 0.02 0.001" condim="4"/></body>"""

def static_prop(mesh, x, y, z, rgba, yaw=0.0):
    return f"""    <body name="{mesh}_s" pos="{x} {y} {z}" euler="0 0 {yaw}">
      <geom type="mesh" mesh="{mesh}" rgba="{rgba}"/></body>"""

TABLES = {"benchA":(-1.55,-1.20),"benchB":(1.55,-1.20),"benchC":(-1.55,1.20),"benchD":(1.55,1.20)}
TOP=0.925; BT=0.92
red="0.80 0.18 0.16 0.82"; blue="0.16 0.42 0.85 0.82"; grn="0.25 0.70 0.42 0.82"
amb="0.85 0.62 0.20 0.85"; clr="0.72 0.80 0.85 0.55"; pur="0.55 0.30 0.70 0.82"; cyan="0.2 0.7 0.75 0.82"
glass_colors=[red,blue,grn,amb,clr,pur,cyan]

items=[]
# denser glassware: scatter beakers/flasks/vials/tubes across each bench
rng=[(-0.85,-0.5),(-0.5,-0.15),(-0.15,0.2),(0.2,0.55),(0.55,0.9)]
kinds=["beaker","flask","beaker","flask","vial","tube","beaker","vial"]
ci=0
for bn,(bx,by) in TABLES.items():
    ci+=1
    xs=np.linspace(-0.85,0.85,7)
    for j,x in enumerate(xs):
        mesh=["beaker","flask","vial","beaker","flask","tube","beaker"][ (j+ci)%7 ]
        col=glass_colors[(j+ci)%len(glass_colors)]
        dy=0.10*np.sin(j*1.7+ci)
        items.append(dyn(mesh, bx+x, by+dy, TOP, col))
# a couple test-tube racks with tubes
bx,by=TABLES["benchB"]
items.append(dyn("rack", bx-0.6, by+0.02, TOP, "0.22 0.22 0.26 1", dens=700))
for k,dx in enumerate([-0.66,-0.60,-0.54]):
    items.append(dyn("tube", bx+dx, by+0.02, TOP, glass_colors[k], dens=600))

# harvested equipment placement (static)
white="0.88 0.89 0.90 1"; pink="0.86 0.5 0.62 1"; dark="0.15 0.16 0.18 1"
harv_xml=[]
def H(key,x,y,z,rgba,yaw=0):
    if key in harv_dims: harv_xml.append(static_prop(f"h_{key}",x,y,z,rgba,yaw))
H("centrifugeL", TABLES["benchC"][0]+0.6, TABLES["benchC"][1], BT, white)
H("centrifugeW", TABLES["benchA"][0]-0.6, TABLES["benchA"][1], BT, white)
H("centrifugeS", TABLES["benchD"][0]+0.55, TABLES["benchD"][1], BT, white)
H("labrack",     TABLES["benchD"][0]-0.55, TABLES["benchD"][1], BT, pink)
H("fridge", -3.4, -1.2, 0.0, white)          # against left wall
H("stool",  -0.7, -0.35, 0.0, dark)          # in the aisle

mesh_names=["beaker","flask","vial","tube","rack"]+[f"h_{k}" for k in harv_dims]
asset_meshes="\n".join(M(n) for n in mesh_names)
table_xml="\n".join(table(n,x,y) for n,(x,y) in TABLES.items())

shelf="""    <body name="shelf" pos="0 3.4 0">
      <geom type="box" size="0.02 0.30 0.90" pos="-1.3 0 0.90" rgba="0.35 0.35 0.38 1"/>
      <geom type="box" size="0.02 0.30 0.90" pos="1.3 0 0.90" rgba="0.35 0.35 0.38 1"/>
      <geom type="box" size="1.32 0.30 0.02" pos="0 0 0.55" rgba="0.5 0.5 0.53 1"/>
      <geom type="box" size="1.32 0.30 0.02" pos="0 0 1.05" rgba="0.5 0.5 0.53 1"/>
      <geom type="box" size="1.32 0.30 0.02" pos="0 0 1.55" rgba="0.5 0.5 0.53 1"/>
      <geom type="box" size="0.14 0.14 0.11" pos="-0.8 0 0.68" rgba="0.7 0.6 0.35 1"/>
      <geom type="box" size="0.12 0.12 0.10" pos="-0.2 0 0.67" rgba="0.6 0.35 0.3 1"/>
      <geom type="box" size="0.13 0.13 0.11" pos="0.7 0 1.18" rgba="0.4 0.5 0.6 1"/>
    </body>"""

xml=f"""<mujoco model="lab_demo_v2">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>
    <quality shadowsize="4096"/></visual>
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
    <geom name="wall_back" type="box" size="6 0.05 1.6" pos="0 3.8 1.6" material="wall"/>
    <geom name="wall_left" type="box" size="0.05 4 1.6" pos="-4.2 0 1.6" material="wall"/>
    <geom name="wall_right" type="box" size="0.05 4 1.6" pos="4.2 0 1.6" material="wall"/>
    <body name="G1_standin" pos="0 -0.2 0">
      <geom type="capsule" fromto="0 0 0.10 0 0 1.10" size="0.14" rgba="0.15 0.16 0.18 1"/>
      <geom type="sphere" size="0.12" pos="0 0 1.22" rgba="0.15 0.16 0.18 1"/>
    </body>
{table_xml}
{shelf}
{chr(10).join(harv_xml)}
{chr(10).join(items)}
  </worldbody>
</mujoco>"""

open(f"{ROOT}/lab_demo_v2.xml","w").write(xml)
print("wrote lab_demo_v2.xml")
m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
print(f"model OK: {m.nbody} bodies, {m.ngeom} geoms, {m.nmesh} meshes")

from PIL import Image
r=mujoco.Renderer(m,1080,1920)
for tag,az,el,dist,look in [("v2_overview",55,-24,11.0,(0,0.4,0.9)),
                            ("v2_benchC",40,-22,3.0,(-1.55,1.20,1.0)),
                            ("v2_benchA",50,-24,2.8,(-1.55,-1.20,0.95))]:
    c=mujoco.MjvCamera(); c.type=mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:]=look; c.distance=dist; c.azimuth=az; c.elevation=el
    r.update_scene(d,camera=c); Image.fromarray(r.render()).save(f"{ROOT}/lab_{tag}.png"); print("wrote",tag)
print("DONE")

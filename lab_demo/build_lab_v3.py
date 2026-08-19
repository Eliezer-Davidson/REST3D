"""Lab scene v3 — realism pass. Bigger room + ceiling + light fixtures + doorway,
perimeter benches + center island, metal/laminate materials, transparent glass,
clean tiled floor (no reflectance splotches), authored microscopes / Bunsen burners /
fume hood / sink / reagent bottles / centrifuges, stocked shelves + wall cabinets."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

ROOT = "/home/user/work/REST3D/lab_demo"
PROPS = f"{ROOT}/props"; os.makedirs(PROPS, exist_ok=True)

def revolve(profile, nseg=48):
    prof = np.asarray(profile, float); V, F, rs = [], [], []
    for (r, z) in prof:
        if r <= 1e-9: rs.append(("p", len(V))); V.append((0,0,z))
        else:
            rs.append(("r", len(V)))
            for k in range(nseg):
                a = 2*np.pi*k/nseg; V.append((r*np.cos(a), r*np.sin(a), z))
    for i in range(len(prof)-1):
        t0,s0 = rs[i]; t1,s1 = rs[i+1]
        if t0=="r" and t1=="r":
            for k in range(nseg):
                k2=(k+1)%nseg; F += [(s0+k,s0+k2,s1+k2),(s0+k,s1+k2,s1+k)]
        elif t0=="p" and t1=="r":
            for k in range(nseg): F.append((s0,s1+k,s1+(k+1)%nseg))
        elif t0=="r" and t1=="p":
            for k in range(nseg): F.append((s0+k,s0+(k+1)%nseg,s1))
    return np.asarray(V,float), np.asarray(F,int)

def save_obj(p,V,F):
    with open(p,"w") as fh:
        for v in V: fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for f in F: fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

MESHES = {
 "beaker":[(0,0),(0.034,0),(0.034,0.006),(0.036,0.10),(0.030,0.10),(0.030,0.006),(0,0.006)],
 "flask":[(0,0),(0.048,0),(0.048,0.012),(0.013,0.078),(0.013,0.11),(0.016,0.115),(0.012,0.115),(0.009,0.078),(0.040,0.010),(0,0.006)],
 "vial":[(0,0),(0.008,0),(0.008,0.040),(0.009,0.040),(0.009,0.047),(0.006,0.047),(0,0.043)],
 "tube":[(0,0),(0.004,0.001),(0.0072,0.006),(0.0075,0.10),(0.0060,0.10),(0.0058,0.008),(0,0.002)],
 "bottle":[(0,0),(0.032,0),(0.032,0.10),(0.020,0.125),(0.011,0.14),(0.011,0.165),(0.014,0.165),(0.014,0.172),(0,0.172)],
 "gradcyl":[(0,0),(0.026,0),(0.026,0.15),(0.030,0.155),(0.022,0.155),(0.022,0.006),(0,0.006)],
}
for n,p in MESHES.items():
    V,F = revolve(p,40); save_obj(f"{PROPS}/{n}.obj", V,F)
print("labware meshes:", ", ".join(MESHES))

# ---------- helpers to emit primitive-built equipment ----------
_id=[0]
def uid(t): _id[0]+=1; return f"{t}{_id[0]}"

def bench(cx,cy,L,W,yaw=0):
    n=uid("bench")
    return f"""    <body name="{n}" pos="{cx} {cy} 0" euler="0 0 {yaw}">
      <geom type="box" material="metal" size="{L/2-0.04} {W/2-0.04} 0.44" pos="0 0 0.44"/>
      <geom type="box" material="laminate" size="{L/2} {W/2} 0.02" pos="0 0 0.90"/>
    </body>"""

def microscope(cx,cy,z,yaw=0):
    n=uid("scope")
    return f"""    <body name="{n}" pos="{cx} {cy} {z}" euler="0 0 {yaw}">
      <geom type="box" material="black" size="0.075 0.11 0.018" pos="0 0 0.018"/>
      <geom type="box" material="black" size="0.03 0.035 0.13" pos="0 0.07 0.15"/>
      <geom type="box" material="steel" size="0.075 0.055 0.010" pos="0 -0.01 0.10"/>
      <geom type="cylinder" material="black" size="0.028 0.03" pos="0 0.0 0.16"/>
      <geom type="cylinder" material="steel" size="0.012 0.02" pos="0 0.01 0.125" euler="0.3 0 0"/>
      <geom type="cylinder" material="black" size="0.014 0.06" pos="0 -0.05 0.24" euler="-0.9 0 0"/>
    </body>"""

def centrifuge(cx,cy,z,yaw=0):
    n=uid("cf")
    return f"""    <body name="{n}" pos="{cx} {cy} {z}" euler="0 0 {yaw}">
      <geom type="cylinder" material="white" size="0.17 0.09" pos="0 0 0.09"/>
      <geom type="ellipsoid" material="white" size="0.17 0.17 0.06" pos="0 0 0.185"/>
      <geom type="box" material="black" size="0.06 0.015 0.02" pos="0 -0.16 0.10"/>
    </body>"""

def burner(cx,cy,z):
    n=uid("burner")
    return f"""    <body name="{n}" pos="{cx} {cy} {z}">
      <geom type="cylinder" material="steel" size="0.05 0.012" pos="0 0 0.012"/>
      <geom type="cylinder" material="steel" size="0.011 0.06" pos="0 0 0.08"/>
      <geom type="capsule" size="0.014 0.02" pos="0 0 0.16" rgba="0.25 0.45 0.95 0.7"/>
      <geom type="capsule" size="0.008 0.015" pos="0 0 0.185" rgba="0.95 0.6 0.15 0.7"/>
    </body>"""

def fume_hood(cx,cy,yaw=0):
    # counter 0.9 + enclosure to ~2.3; open front with a raised glass sash
    n=uid("hood")
    return f"""    <body name="{n}" pos="{cx} {cy} 0" euler="0 0 {yaw}">
      <geom type="box" material="metal" size="0.75 0.42 0.45" pos="0 0 0.45"/>
      <geom type="box" material="white" size="0.75 0.03 0.70" pos="0 0.40 1.60"/>
      <geom type="box" material="white" size="0.03 0.42 0.70" pos="-0.73 0 1.60"/>
      <geom type="box" material="white" size="0.03 0.42 0.70" pos="0.73 0 1.60"/>
      <geom type="box" material="white" size="0.75 0.42 0.04" pos="0 0 2.30"/>
      <geom type="box" material="glass" size="0.70 0.01 0.28" pos="0 -0.38 2.00"/>
      <geom type="box" material="steel" size="0.70 0.42 0.015" pos="0 0 0.92"/>
    </body>"""

def cabinet(cx,cy,z,w,h,d,mat="white"):
    n=uid("cab")
    return f"""    <body name="{n}" pos="{cx} {cy} {z}">
      <geom type="box" material="{mat}" size="{w} {d} {h}"/>
      <geom type="box" material="steel" size="{w*0.48} {d*0.02} {h*0.85}" pos="{-w*0.5} {-d} 0"/>
      <geom type="box" material="steel" size="{w*0.48} {d*0.02} {h*0.85}" pos="{w*0.5} {-d} 0"/>
    </body>"""

def shelfunit(cx,cy,yaw=0):
    n=uid("shelf"); planks=""
    for zz in (0.5,0.95,1.4,1.85):
        planks+=f'\n      <geom type="box" material="steel" size="0.9 0.22 0.015" pos="0 0 {zz}"/>'
    return f"""    <body name="{n}" pos="{cx} {cy} 0" euler="0 0 {yaw}">
      <geom type="box" material="metal" size="0.02 0.22 1.0" pos="-0.88 0 1.0"/>
      <geom type="box" material="metal" size="0.02 0.22 1.0" pos="0.88 0 1.0"/>{planks}
    </body>"""

def dyn(mesh,x,y,z,rgba,dens=800):
    n=uid(mesh)
    return f"""    <body name="{n}" pos="{x} {y} {z}"><freejoint/>
      <geom type="mesh" mesh="{mesh}" rgba="{rgba}" density="{dens}" friction="1 0.02 0.001" condim="4"/></body>"""

# transparent-glass liquid tints (low alpha => see-through)
RED="0.85 0.20 0.18 0.45"; BLUE="0.20 0.45 0.9 0.45"; GRN="0.25 0.75 0.45 0.45"
AMB="0.9 0.65 0.2 0.5"; CLR="0.80 0.88 0.93 0.28"; PUR="0.6 0.35 0.75 0.45"; CYN="0.25 0.75 0.8 0.45"
LIQ=[RED,BLUE,GRN,AMB,PUR,CYN]

items=[]; equip=[]
# ---------------- LAYOUT (room interior x:[-6,6] y:[-4.5,4.5]) ----------------
TOP=0.925
# back-wall counter (long)
equip.append(bench(0, 4.0, 10.0, 0.7))
# center island
equip.append(bench(0, 0.0, 3.2, 1.0))
# right-wall sink bench
equip.append(bench(5.2, 0.0, 0.8, 4.0, yaw=0))
# left-wall short bench
equip.append(bench(-5.3, 1.6, 0.8, 3.0))

# equipment on back counter
equip.append(microscope(-3.6, 4.05, TOP))
equip.append(microscope(-2.9, 4.05, TOP, yaw=0.3))
equip.append(centrifuge(-1.6, 4.02, TOP))
equip.append(centrifuge(3.7, 4.02, TOP, yaw=0.5))
# fume hood back-right corner (standalone against wall)
equip.append(fume_hood(4.9, 3.9))
# burners on island
for bx in (-1.0, 0.2, 1.2): equip.append(burner(bx, -0.25, TOP))

# sink on right bench (basin recess + faucet)
equip.append(f"""    <body name="sink" pos="5.2 1.4 0">
      <geom type="box" material="steel" size="0.30 0.30 0.03" pos="0 0 0.90"/>
      <geom type="box" material="black" size="0.22 0.22 0.06" pos="0 0 0.86"/>
      <geom type="cylinder" material="steel" size="0.012 0.12" pos="-0.18 0.18 1.02"/>
      <geom type="cylinder" material="steel" size="0.010 0.10" pos="-0.08 0.18 1.12" euler="1.4 0 0"/>
    </body>""")

# wall cabinets over the back counter
for cxp in (-3.5,-1.0,1.5,4.0):
    equip.append(cabinet(cxp, 4.35, 1.75, 0.6, 0.35, 0.2))
# stocked shelf units on left wall
equip.append(shelfunit(-5.7, -2.2, yaw=0))

# harvested fridge + stool (kept; they read fine) — reuse v2 preprocessed if present
import glob
for key,(sub,tgt,mode,px,py) in {
  "fridge":("white_under-counter_refrigerator",0.85,"height",-5.5,3.6),
  "stool":("black_rolling_stool",0.62,"height",-1.6,-2.2),
}.items():
    src=glob.glob(f"/home/user/work/REST3D/output/lab-room-design-2/_render_meshes/*{sub}*.obj")
    if src:
        V,F=[],[]
        for l in open(src[0]):
            if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
            elif l.startswith("f "): F.append([int(t.split('/')[0])-1 for t in l.split()[1:4]])
        V=np.array(V); F=np.array(F); mn,mx=V.min(0),V.max(0); ext=mx-mn
        s=tgt/(ext[2] if mode=="height" else ext.max())
        V=(V-np.array([(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,mn[2]]))*s
        save_obj(f"{PROPS}/h_{key}.obj",V,F)
        equip.append(f'    <body name="h_{key}" pos="{px} {py} 0"><geom type="mesh" mesh="h_{key}" material="white"/></body>')

# glassware clusters
def cluster(bx,by,n0,zbase=TOP):
    xs=np.linspace(-0.9,0.9,7)
    for j,x in enumerate(xs):
        mesh=["beaker","flask","gradcyl","beaker","flask","tube","beaker"][(j+n0)%7]
        col=LIQ[(j+n0)%len(LIQ)]; dy=0.12*np.sin(j*1.6+n0)
        items.append(dyn(mesh,bx+x,by+dy,zbase,col))
cluster(0,4.0,0); cluster(-1.6,-0.05,2); cluster(1.4,0.05,4)
# vials + tubes on island ends
for k,dx in enumerate(np.linspace(-1.4,-1.0,4)): items.append(dyn("tube",dx,0.28,TOP,LIQ[k%6],600))
for k,dx in enumerate(np.linspace(1.5,1.7,3)): items.append(dyn("vial",dx,0.28,TOP,LIQ[k%6],600))
# reagent bottles on shelves (left wall shelfunit at -5.7,-2.2)
for si,zz in enumerate((0.5,0.95,1.4)):
    for k,xx in enumerate(np.linspace(-0.7,0.7,5)):
        items.append(dyn("bottle",-5.68,-2.2+ (xx*0.0), 0.5+si*0.45, ["0.7 0.5 0.3 1","0.3 0.5 0.7 1","0.6 0.7 0.4 1","0.7 0.7 0.75 1","0.6 0.4 0.5 1"][k], 500))
# (place bottles along the shelf x-span)
items=[]  # rebuild items cleanly below to fix bottle placement
def dynp(mesh,x,y,z,rgba,dens=800): return dyn(mesh,x,y,z,rgba,dens)
items+= [dynp(*a) for a in []]
cluster(0,4.0,0); cluster(-1.6,-0.05,2); cluster(1.4,0.05,4)
for k,dx in enumerate(np.linspace(-1.4,-1.0,4)): items.append(dynp("tube",dx,0.28,TOP,LIQ[k%6],600))
for k,dx in enumerate(np.linspace(1.5,1.7,3)): items.append(dynp("vial",dx,0.28,TOP,LIQ[k%6],600))
for si,zz in enumerate((0.52,0.97,1.42)):
    for k,xx in enumerate(np.linspace(-0.7,0.7,5)):
        items.append(dynp("bottle",-5.7+0.0, -2.2+xx, zz, ["0.7 0.5 0.3 1","0.3 0.5 0.7 1","0.6 0.7 0.4 1","0.75 0.75 0.78 1","0.6 0.4 0.5 1"][k],500))
# some boxes on the top shelf & wall cabinets area
for k,xx in enumerate(np.linspace(-0.7,0.7,4)):
    equip.append(f'    <geom type="box" material="{["white","metal","steel","white"][k]}" size="0.09 0.14 0.10" pos="-5.7 {-2.2+xx} 1.97"/>')

meshset=["beaker","flask","vial","tube","bottle","gradcyl","h_fridge","h_stool"]
asset_meshes="\n".join(f'    <mesh name="{n}" file="{PROPS}/{n}.obj"/>' for n in meshset if os.path.exists(f"{PROPS}/{n}.obj"))

# ---------------- room: floor(clean tile) + walls(doorway) + ceiling + lights ----------------
XH,YH,WH = 6.0,4.5,1.5
room=f"""
    <geom name="floor" type="plane" material="floor" size="{XH} {YH} 0.1" pos="0 0 0"/>
    <geom name="wall_back"  type="box" material="wall" size="{XH} 0.08 {WH}" pos="0 {YH} {WH}"/>
    <geom name="wall_left"  type="box" material="wall" size="0.08 {YH} {WH}" pos="{-XH} 0 {WH}"/>
    <geom name="wall_right" type="box" material="wall" size="0.08 {YH} {WH}" pos="{XH} 0 {WH}"/>
    <geom name="wall_frontL" type="box" material="wall" size="2.5 0.08 {WH}" pos="{-XH+2.5} {-YH} {WH}"/>
    <geom name="wall_frontR" type="box" material="wall" size="2.5 0.08 {WH}" pos="{XH-2.5} {-YH} {WH}"/>
    <geom name="doorhdr" type="box" material="wall" size="1.0 0.08 0.25" pos="0 {-YH} 2.75"/>
    <geom name="ceiling" type="box" material="ceiling" size="{XH} {YH} 0.06" pos="0 0 3.0"/>
"""
fixtures=""; lights=""
for fx in (-3.0,0.0,3.0):
    for fy in (-2.2,2.2):
        fixtures+=f'\n    <geom type="box" material="fixture" size="0.55 0.28 0.02" pos="{fx} {fy} 2.93"/>'
        lights+=f'\n    <light pos="{fx} {fy} 2.9" dir="0 0 -1" diffuse="0.35 0.35 0.34" specular="0.05 0.05 0.05" cutoff="60"/>'

xml=f"""<mujoco model="lab_demo_v3">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.42 0.42 0.43" diffuse="0.25 0.25 0.25" specular="0.0 0.0 0.0"/>
    <quality shadowsize="8192"/>
    <map shadowclip="6"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.7 0.75 0.82" rgb2="0.35 0.4 0.5" width="256" height="256"/>
    <texture name="tile" type="2d" builtin="checker" rgb1="0.84 0.85 0.86" rgb2="0.79 0.80 0.81" width="512" height="512"/>
    <material name="floor" texture="tile" texrepeat="18 14" reflectance="0" specular="0.05" shininess="0.1"/>
    <material name="wall" rgba="0.90 0.91 0.92 1" specular="0.05" shininess="0.1"/>
    <material name="ceiling" rgba="0.95 0.95 0.96 1" specular="0.0"/>
    <material name="fixture" rgba="0.98 0.98 0.90 1" emission="0.6"/>
    <material name="laminate" rgba="0.17 0.19 0.23 1" specular="0.25" shininess="0.4"/>
    <material name="metal" rgba="0.60 0.62 0.66 1" specular="0.6" shininess="0.5"/>
    <material name="steel" rgba="0.76 0.78 0.81 1" specular="0.75" shininess="0.6"/>
    <material name="white" rgba="0.90 0.90 0.92 1" specular="0.25" shininess="0.3"/>
    <material name="black" rgba="0.12 0.13 0.15 1" specular="0.35" shininess="0.3"/>
    <material name="glass" rgba="0.80 0.88 0.93 0.22" specular="0.9" shininess="0.9"/>
{asset_meshes}
  </asset>
  <worldbody>
{room}{fixtures}{lights}
    <body name="G1_standin" pos="0 -2.6 0">
      <geom type="capsule" fromto="0 0 0.10 0 0 1.10" size="0.14" material="black"/>
      <geom type="sphere" size="0.12" pos="0 0 1.22" material="black"/>
    </body>
{chr(10).join(equip)}
{chr(10).join(items)}
  </worldbody>
</mujoco>"""

open(f"{ROOT}/lab_demo_v3.xml","w").write(xml)
print("wrote lab_demo_v3.xml")
m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
print(f"model OK: {m.nbody} bodies, {m.ngeom} geoms, {m.nmesh} meshes")

from PIL import Image
r=mujoco.Renderer(m,1080,1920)
for tag,az,el,dist,look in [("v3_overview",50,-20,13.0,(0,0.3,1.0)),
                            ("v3_backcounter",70,-16,4.5,(-2.5,4.0,1.1)),
                            ("v3_island",120,-18,4.0,(0,0,1.0))]:
    c=mujoco.MjvCamera(); c.type=mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:]=look; c.distance=dist; c.azimuth=az; c.elevation=el
    r.update_scene(d,camera=c); Image.fromarray(r.render()).save(f"{ROOT}/lab_{tag}.png"); print("wrote",tag)
print("DONE")

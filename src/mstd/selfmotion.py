"""Pinhole-rendered static dots under known camera translation and yaw.

World x=right, y=up, z=initial forward. Positive yaw turns right.
These are image sequences, not drawings of flow vectors.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from math import atan2, ceil, cos, degrees, exp, floor, hypot, isfinite, pi, radians, sin, tan
from pathlib import Path
import random
import shutil
import subprocess

from .calibration import write_stimulus


@dataclass(frozen=True)
class Camera:
    width: int = 80
    height: int = 80
    fov_x_deg: float = 90.0
    height_m: float = 1.61
    near_m: float = 2.0
    far_m: float = 50.0

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or min(self.width,self.height)<8:
            raise ValueError('integer image dimensions >=8 required')
        if not all(isfinite(v) for v in (self.fov_x_deg,self.height_m,self.near_m,self.far_m)):
            raise ValueError('finite camera parameters required')
        if not 10 <= self.fov_x_deg < 150 or not 0 < self.near_m < self.far_m or self.height_m <= 0:
            raise ValueError('invalid FOV, camera height or depth range')

    @property
    def fx(self):
        return self.width/(2*tan(radians(self.fov_x_deg)/2))

    @property
    def cx(self):
        return (self.width-1)/2

    @property
    def cy(self):
        return (self.height-1)/2

    @property
    def intrinsics(self):
        return {'fx_px':self.fx,'fy_px':self.fx,'cx_px':self.cx,'cy_px':self.cy}


@dataclass(frozen=True)
class Motion:
    mode: str = 'straight'
    speed_mps: float = 3.0
    heading_deg: float = 0.0
    yaw_rate_deg_s: float = 0.0
    radius_m: float = 45.0
    turn_sign: int = 1
    gaze_offset_deg: float = 0.0

    def __post_init__(self):
        if self.mode not in ('static','straight','backward','yaw','curve'):
            raise ValueError('unsupported motion mode')
        if not all(isfinite(v) for v in (self.speed_mps,self.heading_deg,self.yaw_rate_deg_s,
                                         self.radius_m,self.gaze_offset_deg)):
            raise ValueError('finite motion parameters required')
        if self.speed_mps < 0 or self.radius_m <= 0 or self.turn_sign not in (-1,1):
            raise ValueError('invalid motion speed/radius/sign')

    def pose(self,t):
        if not isfinite(t) or t < 0:
            raise ValueError('nonnegative finite time required')
        if self.mode == 'curve':
            omega = self.turn_sign*self.speed_mps/self.radius_m
            psi = omega*t
            if omega:
                x,z = self.speed_mps/omega*(1-cos(psi)), self.speed_mps/omega*sin(psi)
            else:
                x,z = 0.0,0.0
            vx,vz = self.speed_mps*sin(psi),self.speed_mps*cos(psi)
            yaw = psi+radians(self.gaze_offset_deg)
        elif self.mode in ('straight','backward'):
            angle = radians(self.heading_deg)
            speed = -self.speed_mps if self.mode=='backward' else self.speed_mps
            vx,vz = speed*sin(angle),speed*cos(angle)
            x,z,yaw,omega = vx*t,vz*t,0.0,0.0
        else:
            omega = radians(self.yaw_rate_deg_s) if self.mode=='yaw' else 0.0
            x,z,vx,vz,yaw = 0.0,0.0,0.0,0.0,omega*t
        # Relative heading is defined for translation, including 180 deg backward.
        bearing = degrees(atan2(vx*cos(yaw)-vz*sin(yaw),vx*sin(yaw)+vz*cos(yaw))) if hypot(vx,vz)>0 else None
        return {'x_m':x,'z_m':z,'yaw_rad':yaw,'vx_mps':vx,'vz_mps':vz,
                'yaw_rate_rad_s':omega,'heading_camera_deg':bearing,
                'signed_curvature_per_m':self.turn_sign/self.radius_m if self.mode=='curve' else 0.0}


def project(camera,point,pose):
    x,y,z = point
    dx,dz = x-pose['x_m'],z-pose['z_m']
    c,s = cos(pose['yaw_rad']),sin(pose['yaw_rad'])
    xc,zc = c*dx-s*dz,s*dx+c*dz
    if zc < camera.near_m or zc > camera.far_m:
        return None
    u = camera.cx+camera.fx*xc/zc
    v = camera.cy-camera.fx*(y-camera.height_m)/zc
    if not (0 <= u < camera.width and 0 <= v < camera.height):
        return None
    return u,v,zc


def spawn(camera,pose,scene,rng):
    if scene not in ('ground','cloud'):
        raise ValueError('scene must be ground or cloud')
    if scene=='ground':
        # Uniform image elevation in the visible part of a horizontal ground plane.
        low = camera.height_m/camera.far_m
        high = min(camera.height_m/camera.near_m,(camera.height-1-camera.cy)/camera.fx)
        if high <= low:
            raise ValueError('ground is outside the image; change camera height/FOV/depth')
        z = camera.height_m/rng.uniform(low,high)
        yc = -camera.height_m
    else:
        z = rng.uniform(camera.near_m,camera.far_m)
        yc = rng.uniform(-.9,.9)*camera.height/(2*camera.fx)*z
    xc = rng.uniform(-.9,.9)*camera.width/(2*camera.fx)*z
    c,s = cos(pose['yaw_rad']),sin(pose['yaw_rad'])
    return (pose['x_m']+c*xc+s*z,camera.height_m+yc,pose['z_m']-s*xc+c*z)


def render(camera,projected):
    image = [16.0]*(camera.width*camera.height)
    # Constant retinal-size soft dots, common in random-dot stimuli. Opaque intensity max.
    for u,v,_ in projected:
        for y in range(max(0,floor(v)-1),min(camera.height,ceil(v)+2)):
            for x in range(max(0,floor(u)-1),min(camera.width,ceil(u)+2)):
                value = 16+220*exp(-((x-u)**2+(y-v)**2)/(2*.65**2))
                image[y*camera.width+x] = max(image[y*camera.width+x],value)
    return bytes(max(0,min(255,floor(value+.5))) for value in image)


def sequence(camera,motion,scene,frames=80,frame_ms=50,points=250,seed=0):
    if any(type(v) is not int or v <= 0 for v in (frames,frame_ms,points)):
        raise ValueError('positive integer frames, frame_ms and point count required')
    if scene not in ('ground','cloud'):
        raise ValueError('scene must be ground or cloud')
    rng = random.Random(seed)
    world = [spawn(camera,motion.pose(0),scene,rng) for _ in range(points)]
    ids = list(range(points))
    next_id = points
    for index in range(frames):
        pose = motion.pose(index*frame_ms/1000)
        following = motion.pose((index+1)*frame_ms/1000)
        positions,sparse = [],[]
        replacements = 0
        for p,point in enumerate(world):
            projection = project(camera,point,pose)
            if projection is None:
                point = spawn(camera,pose,scene,rng)
                world[p],ids[p] = point,next_id
                next_id += 1
                replacements += 1
                projection = project(camera,point,pose)
            if projection is None:
                raise ValueError('could not spawn a visible point')
            positions.append(projection)
            later = project(camera,point,following) if index+1<frames else None
            u,v,z = projection
            sparse.append({'id':ids[p],'u_px':u,'v_px':v,'depth_m':z,
                           'du_px':later[0]-u if later else None,
                           'dv_px':later[1]-v if later else None})
        truth = {'frame':index,'time_s':index*frame_ms/1000,'camera':pose,
                 'respawned_points':replacements,'sparse_flow_to_next_frame':sparse}
        yield render(camera,positions),truth


def cases():
    return [
        ('cloud_static','cloud',Motion('static')),
        ('cloud_forward','cloud',Motion()),
        ('cloud_backward','cloud',Motion('backward')),
        ('ground_forward','ground',Motion()),
        ('ground_heading_left10','ground',Motion(heading_deg=-10)),
        ('ground_heading_right10','ground',Motion(heading_deg=10)),
        ('ground_yaw_left','ground',Motion('yaw',yaw_rate_deg_s=-10)),
        ('ground_yaw_right','ground',Motion('yaw',yaw_rate_deg_s=10)),
        ('ground_turn_left_r45','ground',Motion('curve',turn_sign=-1)),
        ('ground_turn_right_r45','ground',Motion('curve',turn_sign=1)),
        ('ground_turn_right_r20','ground',Motion('curve',radius_m=20)),
        ('ground_turn_right_r85','ground',Motion('curve',radius_m=85)),
        ('ground_turn_right_r45_gaze_left20','ground',Motion('curve',gaze_offset_deg=-20)),
        ('ground_turn_right_r45_gaze_right20','ground',Motion('curve',gaze_offset_deg=20)),
    ]


def generate_selfmotion(output,frames=80,frame_ms=50,points=250,seed=0,video=False):
    if any(type(v) is not int or v <= 0 for v in (frames,frame_ms,points)):
        raise ValueError('positive integer frames, frame_ms and point count required')
    ffmpeg = shutil.which('ffmpeg') if video else None
    if video and not ffmpeg:
        raise ValueError('ffmpeg is required for lossless AVI previews')
    camera = Camera()
    directory = Path(output)
    directory.mkdir(parents=True,exist_ok=False)
    runs = []
    for name,scene,motion in cases():
        truth_path = directory/f'{name}.truth.jsonl'
        preview = None
        with truth_path.open('x',encoding='utf-8') as truth_stream:
            def images():
                nonlocal preview
                for index,(image,truth) in enumerate(sequence(camera,motion,scene,frames,frame_ms,points,seed)):
                    if index==0:
                        preview = image
                    truth_stream.write(json.dumps(truth,allow_nan=False)+'\n')
                    yield image
            path = directory/f'{name}.dat'
            write_stimulus(path,camera.width,camera.height,images())
        (directory/f'{name}.pgm').write_bytes(f'P5\n{camera.width} {camera.height}\n255\n'.encode()+preview)
        if video:
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo',
                            '-pixel_format','gray','-video_size',f'{camera.width}x{camera.height}',
                            '-framerate',str(1000/frame_ms),'-i','pipe:0','-an','-c:v','ffv1',
                            str(directory/f'{name}.avi')],input=path.read_bytes()[25:],check=True)
        runs.append({'name':name,'scene':scene,'motion':asdict(motion),'stimulus':path.name,
                     'stimulus_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'truth':truth_path.name,'video':f'{name}.avi' if video else None,'spikes':None,
                     'current_backend':'translation_baseline' if motion.mode=='straight' else 'requires_extension_or_control'})
    meta = {'schema':'selfmotion-stimulus-v1','width':camera.width,'height':camera.height,
            'frames':frames,'frame_ms':frame_ms,'duration_ms':frames*frame_ms,'seed':seed,
            'points':points,'camera':asdict(camera),'intrinsics_px':camera.intrinsics,'runs':runs,
            'coordinates':'world x-right y-up z-initial-forward; image x-right y-down; positive yaw=right',
            'rendering':'persistent static world dots; respawn only after depth/viewport exit; soft constant-retinal-size dots',
            'paper_relation':'inspired by 2022 ground-dot paradigm; not exact dataset reproduction',
            'notes':['No arrows, labels or rendered flow overlays in V1 input.',
                     'AVI FFV1 and DAT contain the same grayscale image samples.',
                     'Use actual frame_ms in CARLsim; 50ms is 20Hz, not paper 30Hz.',
                     'Current MSTd CLI has expansion templates only; no backward, yaw or curvature decoder.']}
    (directory/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (directory/'intrinsics.json').write_text(json.dumps(camera.intrinsics,indent=2),encoding='utf-8')
    return {'output':str(directory),'runs':len(runs),'video':video}

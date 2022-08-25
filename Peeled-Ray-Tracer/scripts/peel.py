from concurrent.futures import process
from this import d
import trimesh
import numpy as np
import os
import sys
import cv2
import PIL.Image
import argparse
import cv2
import cv2
from ray_trace_rendering_with_texture import *
import multiprocessing as mp
import time
import matplotlib.pyplot as pyplot
import natsort
from tqdm import tqdm
from functools import partial
# from sklearn.neighbors import KDTree
from PIL import Image
import pymeshlab

scene = None
mesh = None
objects = None
mesh_smpl = None
texture = None

lighting = True

def point_cloud(raw_depth,K):
    pts = []
    for i in range(0,512):
        for j in range(0,512):
            if raw_depth[i][j]>0:
                Z = raw_depth[i][j]
                X = (Z*(i))/K[0][0]
                Y = (Z*(j))/K[0][0]
                pts.append([X,Y,Z])
    return trimesh.PointCloud(np.array(pts).reshape(-1,3))

def pre_process(mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path[:-4]+'.obj')
    ms.save_current_mesh(mesh_path[:-4]+'.ply')


def depth_images_norm(depth1, depth2, depth3, depth4):
    vals1 = np.unique(depth1)
    vals2 = np.unique(depth2)
    vals3 = np.unique(depth3)
    vals4 = np.unique(depth4)
    np.sort(vals1)
    np.sort(vals2)
    np.sort(vals3)
    np.sort(vals4)
    depth_zero1 = np.copy(depth1)
    depth_zero2 = np.copy(depth2)
    depth_zero3 = np.copy(depth3)
    depth_zero4 = np.copy(depth4)
    max_ = max(vals1[-1], vals2[-1], vals3[-1], vals4[-1])
    min_ = min(vals1[1], vals2[1], vals3[1], vals4[1])
    depth1 = (depth1 - min_)/(max_-min_)
    depth2 = (depth2 - min_)/(max_-min_)
    depth3 = (depth3 - min_)/(max_-min_)
    depth4 = (depth4 - min_)/(max_-min_)
    depth1 = (1.0-depth1)*255.0
    depth2 = (1.0-depth2)*255.0
    depth3 = (1.0-depth3)*255.0
    depth4 = (1.0-depth4)*255.0
    depth1[depth_zero1==0.0] = 0.0
    depth2[depth_zero2==0.0] = 0.0
    depth3[depth_zero3==0.0] = 0.0
    depth4[depth_zero4==0.0] = 0.0
    return depth1.astype('uint8'), depth2.astype('uint8'), depth3.astype('uint8'), depth4.astype('uint8')


def int_frame(frame):
    if 'Frame' in frame:
        return frame[5:]
    else:
        return 0


""" parallise"""


def parallise(light, camera, index_ray_all, index_tri_all, locations_all, ray_origins, ray_directions, texture, idx):
    where = np.where(index_ray_all == idx)
    index_tri = index_tri_all[where]
    locations = locations_all[where]
    index_ray = list(np.zeros(len(where[0].tolist())))

    face_color1 = np.array([0, 0, 0])
    face_color2 = np.array([0, 0, 0])
    face_color3 = np.array([0, 0, 0])
    face_color4 = np.array([0, 0, 0])
    iters = len(index_tri)
    if (iters):

        """ Separate ray hits """
        unique_index = np.unique(np.array(index_ray))
        occurences = [list(np.where(index_ray == unique)[0]) for unique in unique_index]
        intersections_1 = list(filter(lambda x: len(x) > 0, occurences))
        intersections_2 = list(filter(lambda x: len(x) > 1, occurences))
        intersections_3 = list(filter(lambda x: len(x) > 2, occurences))
        intersections_4 = list(filter(lambda x: len(x) > 3, occurences))

        first = [x[0] for x in intersections_1]
        second = [x[1] for x in intersections_2]
        third = [x[2] for x in intersections_3]
        fourth = [x[3] for x in intersections_4]

        # depth_maps
        z_vec = np.array([0, 0, -1])
        depth = trimesh.util.diagonal_dot(locations - ray_origins[0], z_vec)

        depth1 = 0.0
        depth2 = 0.0
        depth3 = 0.0
        depth4 = 0.0
        if len(first) > 0:
            depth1 = depth[first][0]
        if len(second) > 0:
            depth2 = depth[second][0]
        if len(third) > 0:
            depth3 = depth[third][0]
        if len(fourth) > 0:
            depth4 = depth[fourth][0]


        ind = [2, 1, 0]

        UV = mesh.visual.uv
        UV = ( UV-UV.min() ) / ( UV.max()-UV.min() )

        # barycentric interpolation of uv co-ordinates for intersection points
        bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[index_tri], points=locations)
        points_uv = UV[mesh.faces[index_tri]]
        points_uv[:,:,0] = points_uv[:,:,0]*bary
        points_uv[:,:,1] = points_uv[:,:,1]*bary

        # quering colors from texture map
        texels = points_uv.sum(1)*texture.shape[0]
        texels = texels.astype('int32')
        points_rgb = texture[texels[:,1]-1, texels[:,0]-1]

        rgb_colors = points_rgb[:, ind]

        face_color1 = np.array([0, 0, 0])
        face_color2 = np.array([0, 0, 0])
        face_color3 = np.array([0, 0, 0])
        face_color4 = np.array([0, 0, 0])

        if (len(rgb_colors[first])):
            face_color1 = rgb_colors[first][0] / 255
        if (len(rgb_colors[second])):
            face_color2 = rgb_colors[second][0] / 255
        if (len(rgb_colors[third])):
            face_color3 = rgb_colors[third][0] / 255
        if (len(rgb_colors[fourth])):
            face_color4 = rgb_colors[fourth][0] / 255

    if (iters):
        if lighting==True:
            color1 = np.array([0, 0, 0])
            color2 = np.array([0, 0, 0])
            color3 = np.array([0, 0, 0])
            color4 = np.array([0, 0, 0])
            if len(first) > 0:
                color1 = RecursiveRayTracing(objects, ray_origins[idx], ray_directions[idx], light, camera,
                                            np.array(face_color1), 4, 1.0, 1.0, 1)
            if len(second) > 0:
                color2 = RecursiveRayTracing(objects, ray_origins[idx], ray_directions[idx], light, camera,
                                            np.array(face_color2), 4, 1.0, 1.0, 2)
            if len(third) > 0:
                color3 = RecursiveRayTracing(objects, ray_origins[idx], ray_directions[idx], light, camera,
                                            np.array(face_color3), 4, 1.0, 1.0, 3)
            if len(fourth) > 0:
                color4 = RecursiveRayTracing(objects, ray_origins[idx], ray_directions[idx], light, camera,
                                            np.array(face_color4), 4, 1.0, 1.0, 4)
        
        else:
            color1 = np.array([0, 0, 0])
            color2 = np.array([0, 0, 0])
            color3 = np.array([0, 0, 0])
            color4 = np.array([0, 0, 0])
            if len(first) > 0:
                color1 = face_color1
            if len(second) > 0:
                color2 = face_color2
            if len(third) > 0:
                color3 = face_color3
            if len(fourth) > 0:
                color4 = face_color3

    else:
        color1 = np.array([0, 0, 0])
        color2 = np.array([0, 0, 0])
        color3 = np.array([0, 0, 0])
        color4 = np.array([0, 0, 0])

        depth1 = 0.0
        depth2 = 0.0
        depth3 = 0.0
        depth4 = 0.0

    return color1, color2, color3, color4, depth1, depth2, depth3, depth4

def main(mesh, out_dir, rot_angle, mass, nproc, tex_map):
    global scene
    global objects
    global texture

    scene = None
    objects = None
    texture = tex_map

    vertices = mesh.vertices

    image_res = 512


    """ rotation """
    rot_mat = trimesh.transformations.rotation_matrix(angle=np.radians(rot_angle), direction=[0, 1, 0])
    mesh.apply_transform(rot_mat)

    """ add random tilt """
    rand_tilt = np.random.randint(-10,10)
    tilt = trimesh.transformations.rotation_matrix(angle=np.radians(rand_tilt), direction=[0, 0, 1])
    mesh.vertices -= com
    mesh.apply_transform(tilt)
    mesh.vertices += com

    """ create Scene """
    scene = trimesh.scene.Scene()

    """ add mesh to scene """
    scene.add_geometry(mesh)

    """ Define camera parameters """
    focal_length = 5000
    res = 512
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 12])
    scene.camera_transform = camera_pose
    scene.camera.resolution = [res, res]
    K = scene.camera.K
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    scene.camera.K = K

    # convert the camera to rays with one ray per pixel
    ray_origins, ray_directions, pixels = scene.camera_rays()

    """create light area"""
    len_x = np.abs(np.max(mesh.vertices[:, 0]) - np.min(mesh.vertices[:, 0]))
    len_y = np.abs(np.max(mesh.vertices[:, 1]) - np.min(mesh.vertices[:, 1]))
    len_z = np.abs(np.max(mesh.vertices[:, 2]) - np.min(mesh.vertices[:, 2]))

    light_area = trimesh.creation.box([len_x / 3, len_y / 3, 10e-5])
    scene.camera_transform[:, 3][:3] = scene.camera_transform[:, 3][:3]
    translate = np.copy(scene.camera_transform[:, 3][:3])
    translate[2] = translate[2] + len_z / 100
    light_area.apply_translation(translate)

    # randomly rotate & translate light source
    random_angle = np.random.randint(-70,70)
    light_rot = trimesh.transformations.rotation_matrix(random_angle,direction=[0,1,0])
    light_area.vertices -= light_area.center_mass
    light_area.apply_transform(light_rot)
    light_area.vertices += light_area.center_mass
    light_area.vertices -= np.array([0,0,np.random.uniform(-50,50)])

    scene.add_geometry(light_area)

    """parameters for lighting """
    width, height = scene.camera.resolution
    camera = scene.camera_transform[:, 3][:3]
    pos = 4

    color = np.random.uniform(0.5,1.0,3)

    light = {'position': np.array(translate) + pos, 'ambient': np.array([0.7,0.7,0.7]),
             'diffuse': color, 'specular': color }
    obj = SolidObjects(mesh=mesh, name='human', ambient=np.array([1.0, 1.0, 1.0]), diffuse=np.array([1.0, 1.0, 1.0]),
                       specular=np.array([1.0, 1.0, 1.0]), shininess=100, reflection=0.0,texture=texture)
    objects = [obj]



    """Render scene via ray tracing"""
    image_res = scene.camera.resolution
    locations_all, index_ray_all, index_tri_all = mesh.ray.intersects_location(ray_origins=ray_origins,
                                                                               ray_directions=ray_directions,
                                                                               multiple_hits=True)

    """multi-processing"""
    pooled_row = mp.Pool(nproc)
    func = partial(parallise, light, camera, index_ray_all, index_tri_all, locations_all, ray_origins, ray_directions, tex_map)
    results = pooled_row.map(func, range(0, image_res[0] * image_res[1]))
    pooled_row.close()
    pooled_row.join()

    color1 = []
    color2 = []
    color3 = []
    color4 = []

    depth1 = []
    depth2 = []
    depth3 = []
    depth4 = []

    for res in results:
        color1.append(res[0])
        color2.append(res[1])
        color3.append(res[2])
        color4.append(res[3])
        depth1.append(res[4])
        depth2.append(res[5])
        depth3.append(res[6])
        depth4.append(res[7])

    # depth1= np.array(depth1)
    # depth2= np.array(depth2)
    # depth3= np.array(depth3)
    # depth4= np.array(depth4)

    # print(min(depth1[depth1>0]),max(depth1))
    # print(min(depth2[depth3>0]),max(depth2))
    # print(min(depth3[depth3>0]),max(depth3))
    # print(min(depth4[depth4>0]),max(depth4))

    color1 = np.array(color1).reshape(image_res[0], image_res[1], 3)
    color2 = np.array(color2).reshape(image_res[0], image_res[1], 3)
    color3 = np.array(color3).reshape(image_res[0], image_res[1], 3)
    color4 = np.array(color4).reshape(image_res[0], image_res[1], 3)

    color1 = ((color1 - color1.min()) / (color1.max() - color1.min())) * 255

    color2 = ((color2 - color2.min()) / (color2.max() - color2.min())) * 255

    color3 = ((color3 - color3.min()) / (color3.max() - color3.min())) * 255

    color4 = ((color4 - color4.min()) / (color4.max() - color4.min())) * 255

    depth1 = np.array(depth1, dtype=np.float32).reshape(image_res[0], image_res[1])
    depth2 = np.array(depth2, dtype=np.float32).reshape(image_res[0], image_res[1])
    depth3 = np.array(depth3, dtype=np.float32).reshape(image_res[0], image_res[1])
    depth4 = np.array(depth4, dtype=np.float32).reshape(image_res[0], image_res[1])

    depth1 = np.rot90(depth1,-1)
    depth2 = np.rot90(depth2,-1)
    depth3 = np.rot90(depth3,-1)
    depth4 = np.rot90(depth4,-1)
    

    if lighting==True:
        color1 = cv2.cvtColor(np.uint8(color1), cv2.COLOR_BGR2RGB)
        color2 = cv2.cvtColor(np.uint8(color2), cv2.COLOR_BGR2RGB)
        color3 = cv2.cvtColor(np.uint8(color3), cv2.COLOR_BGR2RGB)
        color4 = cv2.cvtColor(np.uint8(color4), cv2.COLOR_BGR2RGB)


    cv2.imwrite(out_dir + '/rgb_1.png', cv2.rotate(color1,0))
    cv2.imwrite(out_dir + '/rgb_2.png', cv2.rotate(color2,0))
    cv2.imwrite(out_dir + '/rgb_3.png', cv2.rotate(color3,0))
    cv2.imwrite(out_dir + '/rgb_4.png', cv2.rotate(color4,0))


    np.savez_compressed(out_dir + '/dep1', a=depth1)
    np.savez_compressed(out_dir + '/dep2', a=depth2)
    np.savez_compressed(out_dir + '/dep3', a=depth3)
    np.savez_compressed(out_dir + '/dep4', a=depth4)

    np.savez_compressed(out_dir + '/com', a=mesh.center_mass, b=mass)
    np.save(out_dir + '/rotation.npy', rot_mat)

    # saving depth maps for visualization
    depth1, depth2, depth3, depth4 = depth_images_norm(depth1, depth2, depth3, depth4)
    cv2.imwrite(out_dir + '/dep_1.png', depth1)
    cv2.imwrite(out_dir + '/dep_2.png', depth2)
    cv2.imwrite(out_dir + '/dep_3.png', depth3)
    cv2.imwrite(out_dir + '/dep_4.png', depth4)

    scene = None
    mesh = None
    mesh_smpl = None
    objects = None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seq', default='0', type=str, help='Starting sequence processing')
    parser.add_argument('--end_seq', default='10', type=str, help='Ending sequence processing')

    args = parser.parse_args()
    start_seq = int(args.start_seq)
    end_seq = int(args.end_seq)

    dataroot = '../sample' + '/'

    saveroot = '../PEELMAPS' + '/'

    ERROR = []
    meshes = natsort.natsorted(os.listdir(dataroot))
    for mesh in meshes:
        try:
            meshname = mesh
            savedir = saveroot + mesh
            os.makedirs(savedir, exist_ok=True)
            mesh_path = dataroot + mesh + '/models/model_normalized.ply'
            pre_process(mesh_path)
            mesh = trimesh.load(mesh_path,process=False)
            com = mesh.center_mass

            tex_map = np.array(Image.open(dataroot + meshname + '/images/texture0.jpg'))
            tex_map = cv2.resize(tex_map,(512,512))
            tex_map = cv2.flip(tex_map,0)
            print(meshname, end=' --> ') 
            r = 0
            while(r<360):
                print(r, end=' ')
                out_dir = saveroot + meshname + '/' + str(r)
                os.makedirs(out_dir, exist_ok=True)
                main(mesh, out_dir, r, com, nproc=37, tex_map=tex_map)
                r += 36
            print()
        except:
            print("ERROR")
            ERROR.append(meshname)

#functions to load colmap scenes and ply meshes, and render them 

import bpy
import os
import glob
import os
import time
import sys

Path = '/media/gimms/ABitXtra/Image_2_Mesh'

#create a vertex material blender material nodes if there none
material = bpy.data.materials.get("Material")

if material is None:
    # create material
    material = bpy.data.materials.new(name="Material")
    material.use_nodes = True
    #add nodes
    Diffuse_output = material.node_tree.nodes.get('Principled BSDF')
    VertexCol = material.node_tree.nodes.new('ShaderNodeVertexColor')


    # link colour to emission so our objects doesnt have any shadow 
    material.node_tree.links.new(Diffuse_output.inputs[17], VertexCol.outputs[0])


# use the photogrametry addon to import colmap scene with cameras+point cloud,  
def load_colmap(path, keep_points=False):
    bpy.context.scene.photogrammetry.input = 'in_colmap'
    bpy.context.scene.photogrammetry.output = 'out_blender'
    bpy.context.scene.photogrammetry.in_colmap.dirpath = path
    bpy.context.scene.photogrammetry.out_blender.relative_paths = True
    bpy.context.scene.photogrammetry.out_blender.update_render_size = True
    bpy.context.scene.photogrammetry.out_blender.camera_alpha = 0
    bpy.ops.photogrammetry.process()
    
    #by default i set it to delet the point cloud
    if keep_points == False:
        
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['PhotogrammetryPoints'].select_set(True)
        bpy.ops.object.delete() 


# a simple function to add a mesh and then add vertex material to it
def load_mesh(path, addMat = True):
    #find name from directory name and load mesh
    #name = os.path.basename(os.path.normpath(path))
    #print(name)
    bpy.ops.import_mesh.ply(filepath=path+"_Mesh+Col_HD.ply")
    
    ob = bpy.data.objects[path+'_Mesh+Col_HD']
    
    material = bpy.data.materials.get("Material")
   
    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = material
    else:
        # no slots
        ob.data.materials.append(material)
    
    


                
# simple function to load scene if it exists or returns false if not
def loadScene(path):
    name = os.path.basename(os.path.normpath(path))
    
    if os.path.exists(path+"/sparse/0") and  os.path.exists(path+name+"_Mesh+Col.ply"):
        
        load_colmap(path+"/sparse/0")  
        load_mesh(path)
        
        return True
    else:
        return False

def setuv_from_camera(Camera):
    
   
    #set uv from camera
    for ob in bpy.context.scene.objects:
            if ob.type == 'MESH':
                ob.data.uv_layers.active.data[0].uv = (10,10)
                ob.select_set(True)
                bpy.context.view_layer.objects.active  = ob
                proj_uv = ob.modifiers.new(type = 'UV_PROJECT', name = 'proj_uv')
                proj_uv.projectors[0].object = bpy.data.objects[Camera.name]
                proj_uv.uv_layer = "UVMap"
                bpy.ops.object.modifier_apply(apply_as='DATA', modifier=proj_uv.name)
    
    
#searches through all cameras and renders them the eve, making it fast and easy
def render_all_cam(path, ProjUV = False):
    #black background
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0,0,0,1)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'JPEG' 
    
    num=0
    for ob in bpy.context.scene.objects:
            if ob.type == 'CAMERA':
                bpy.context.scene.camera = ob
                
                if ProjUV:
                    setuv_from_camera(ob)
             
                num+=1
              
                number = str(num).zfill(5)
                scene.render.filepath = path+"/renderImages/" + number
                bpy.ops.render.render(write_still=True) # render still

def render_section(path, anim_texture = False):   
    
    
    
    start = bpy.data.scenes['Scene'].frame_start
    end = bpy.data.scenes['Scene'].frame_end
    
     
    
    for n, frame in enumerate(range(start,end)):
        bpy.context.scene.frame_set(frame)
        
       
        if anim_texture:
            file = str(n).zfill(5)
            new_img = bpy.data.images.load(filepath = path+'generated/'+file+'.png')

            gen_material = bpy.data.materials.get("generated_texture")
            nodes = [n for n in gen_material.node_tree.nodes if n.type == 'TEX_IMAGE']
            for node in nodes:
                node.image = new_img
            
            renderPath = path+"GenRenImages/"
            
        else:
            bpy.context.scene.render.image_settings.file_format = 'JPEG'
            renderPath = path+"renderImages/"
            
        bpy.context.scene.render.filepath = renderPath + str(frame).zfill(5)
        bpy.ops.render.render(write_still=True) # render still
        
    
    
#clear scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete() 
    

# the colmap importer gives 
sysPath = sys.__stdout__
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__





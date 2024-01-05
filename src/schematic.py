import litemapy as lt
import numpy as np

block_file = "voxels.txt"
block_ids = "blocks/ids.txt"

ids = {}

with open(block_ids, "r") as f:
    while True:
        line = f.readline()
        if line == "":
            break
        
        id = line.split(" ")[0].split("\n")[0]
        ids[id] = True

blocks = {}

with open(block_file, "r") as f:
    while True:
        line = f.readline()
        if line == "":
            break
        
        line = line.split(" ")
        coords = [int(x) for x in line[:-1]]
        tex_name = line[3].split("\n")[0]
        tex_name.strip()
        blocks[tuple(coords)] = tex_name

def getBounds(blocks):
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf

    for coords in blocks.keys():
        x,y,z = coords
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x) 
        max_y = max(max_y, y)
        max_z = max(max_z, z)

    max_x += 1
    max_y += 1
    max_z += 1
    return min_x, min_y, min_z, max_x, max_y, max_z

bounds = getBounds(blocks)
region = lt.Region(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
schematic = region.as_schematic(name = "voxels", author = "a1pha1", description = "Perspective Art")

for coords, tex_name in blocks.items():
    x,y,z = coords

    blockID = "minecraft:" + tex_name

    if blockID not in ids:
        print("Unknown block ID: " + blockID)
        continue
    
    state = lt.BlockState(blockID)
    region.setblock(x,y,z,state)

print("Schematic contains " + str(len(blocks)) + " blocks")
print("Saving schematic...")
schematic.save("voxels.litematic")



        
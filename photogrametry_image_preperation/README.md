# Photogrametry Dataset Pipeline

### For this to work you will need the bellow:

1. Colmap inatlled as a binary or compiled from raw.
2. Houdini installed, the free aprentice version will work.
3. blender installed with the jupyter notebook kernel in the main README doc.

## Steps for conversion:

1. Place stills of each object in their own subdirectory inside of rocks, named to prefrence, see the before example for refrence.
2. Run the Colmap_batch_convert.ipynb to convert and process each object into coloured simplified meshes, using Colmap and Houdini.
3. Run the Render_Colmap+PLY.ipynb to render out the forms from the original camera positions and merge them into a single dataset.

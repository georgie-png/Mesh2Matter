
# Mesh 2 Matter

![a 3d rendered and ML generated rock sits clistening in the dark](https://github.com/georgie-png/Mesh2Matter/blob/master/blenderflow/GenRenImages/00022.jpg)

## An exploration generative texturing using a Pix2Pix model!!!

Video example : https://youtu.be/IQ3YRHw_D6M

### The project folder is split into three parts:

1. Data prep using photgrametry to recreat the minerals as 3d mesh and then re-render them from the same camera positions.

2. Development and Training of a pix2pix model, one a regular 512x512 .jpg2.jpg model

3. A render pipeline using the trained Generator part of the pix2pix model in blender to retxture and animate objects.



This Project was built mainly in jupyter notebook, using tensorflow and blender. To use blender in jupyter notebook you can use this kernel modification bellow.

https://github.com/cameronfr/BlenderKernel




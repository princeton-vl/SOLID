import blenderproc as bproc
import numpy as np

from rendering.utils.object import load_object, setup_object, get_rows_and_cols 
from rendering.utils.argparse import ConvertDegreeToRadian, add_default_arguments, add_render_func


def get_camera_pos(init_location, num_rows, num_cols, tile_size):
    init_x = init_location[0]
    init_z = init_location[2]

    x = (2 * init_x + ((num_cols - 1) * tile_size)) / 2
    z = (2 * init_z + ((num_rows - 1) * tile_size)) / 2
    return [x, 3, z]


@add_default_arguments("query")
def add_parser(parser):
    parser.add_argument("--tile_size", default=1.2, type=float)
    parser.add_argument(
        "--angle_delta",
        default=360,
        type=float,
        action=ConvertDegreeToRadian,
    )


@add_render_func("query")
def render(
    objects,
    res=384,
    tile_size=1.2,
    angle_delta=np.pi * 2,
    **kwargs,
):
    assert len(objects) == 1, "only support one object"
    bproc.init()

    # Sample a random object
    num_objs = len(objects)
    objects  = [load_object(i, obj, **kwargs) for i, obj in enumerate(objects)]
    objects  = [setup_object(obj, **kwargs) for obj in objects]

    num_rows, num_cols = get_rows_and_cols(num_objs)
    location = [0, 0, 0]
    for i in range(num_rows):
        for j in range(num_cols):
            objects[i * num_cols + j].set_location(list(location))
            location[0] += tile_size
        location[0]  = 0
        location[2] += tile_size

    bproc.camera.set_resolution(res, res)

    # Set the camera to be in front of the object
    angle = 0
    nth_frame = 0

    camera_pos = get_camera_pos([0, 0, 0], num_rows, num_cols, tile_size)
    camera_mat = bproc.math.build_transformation_mat(
        camera_pos, [-np.pi / 2, np.pi, angle]
    )
    light = bproc.types.Light()
    light.set_location(camera_pos, frame=nth_frame)
    light.set_energy(1000, frame=nth_frame)
    bproc.camera.add_camera_pose(camera_mat, frame=nth_frame)

    angle += angle_delta
    while angle < np.pi * 2:
        nth_frame += 1
        x, y, z = camera_pos

        x_new = np.cos(angle) * x - np.sin(angle) * y
        y_new = np.sin(angle) * x + np.cos(angle) * y

        new_camera_pos = [x_new, y_new, z]
        new_camera_mat = bproc.math.build_transformation_mat(
            new_camera_pos, [-np.pi / 2, np.pi, angle]
        )
        bproc.camera.add_camera_pose(new_camera_mat, frame=nth_frame)
        light.set_location(new_camera_pos, frame=nth_frame)

        angle += angle_delta

    # Render the scene
    data = bproc.renderer.render()

    # Write the rendering into an hdf5 file
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
    del light
    return data["colors"], seg_data["instance_segmaps"], seg_data["instance_attribute_maps"]

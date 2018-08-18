
import numpy as np
import tensorflow as tf
from inpaint_model import InpaintCAModel


class Inpainting(object):
    _model_cache = {}

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir):
        if checkpoint_dir in cls._model_cache:
            return cls._model_cache
        else:
            model = Inpainting(checkpoint_dir)
            cls._model_cache[checkpoint_dir] = model
            return model

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    @staticmethod
    def gen_mask(img_array):
        xx, yy = np.where((img_array[:, :, 0] == 0) & (
            img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 0))
        mask = np.zeros(img_array.shape, dtype=int)
        for i, j in zip(xx, yy):
            mask[i, j, :] = [255, 255, 255]
        return mask

    def predict(self, image):
        # ng.get_gpus(1)

        model = InpaintCAModel()
        mask = self.gen_mask(image)

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(
                    self.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)
            return result[0][:, :, ::-1]

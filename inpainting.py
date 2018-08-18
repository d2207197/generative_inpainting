
import numpy as np
import tensorflow as tf
from inpaint_model import InpaintCAModel

cat_to_checkpoint_map = {
    '小籠包': 'model_logs/20180816142207530631_instance-1_pixfood20-dumpling_NORMAL_wgan_gp_pixfood20_dumpling',  # noqa: E501
    '麵包': 'model_logs/20180816142933277373_instance-1_pixfood20-bread_NORMAL_wgan_gp_pixfood20_bread',  # noqa: E501
    '義大利麵': 'model_logs/20180816032219684085_instance-1_pixfood20-spaghetti_NORMAL_wgan_gp_full_model_pixfood20_spaghetti_256',  # noqa: E501
    '沙拉': 'model_logs/20180816032219636259_instance-1_pixfood20-salad_NORMAL_wgan_gp_full_model_pixfood20_salad_256',  # noqa: E501
    '串燒': 'model_logs/20180816171421313325_pixnet-instance1_pixfood20-kebab_NORMAL_wgan_gp_full_model_pixfood20_kebab_256',  # noqa: E501
    '牛排': 'model_logs/20180816045241440611_pixnet-instance1_pixfood20-steak_NORMAL_wgan_gp_full_model_pixfood20_steak_256',  # noqa: E501
    '拉麵': 'model_logs/20180817070326778069_instance-1_pixfood20-拉麵_NORMAL_wgan_gp_pixfood20-拉麵',  # noqa: E501
    '壽司': 'model_logs/20180817070519853688_instance-1_pixfood20-壽司_NORMAL_wgan_gp_pixfood20-壽司',  # noqa: E501
    '丼飯': 'model_logs/20180817085029837150_pixnet-instance1_pixfood20-丼飯_NORMAL_wgan_gp_pixfood20-丼飯',  # noqa: E501
    '牛肉麵': 'model_logs/20180817172532454644_instance-1_pixfood20-牛肉麵_NORMAL_wgan_gp_pixfood20-牛肉麵',  # noqa: E501
    '生魚片': 'model_logs/20180817172109423433_instance-1_pixfood20-生魚片_NORMAL_wgan_gp_pixfood20-生魚片',  # noqa: E501
    '鬆餅': 'model_logs/20180817172259513799_instance-1_pixfood20-鬆餅_NORMAL_wgan_gp_pixfood20-鬆餅',  # noqa: E501
    '蛋糕': 'model_logs/20180817172435274069_instance-1_pixfood20-蛋糕_NORMAL_wgan_gp_pixfood20-蛋糕',  # noqa: E501
    '滷肉飯': 'model_logs/20180817171517861902_pixnet-instance1_pixfood20-滷肉飯_NORMAL_wgan_gp_pixfood20-滷肉飯',  # noqa: E501
    '火鍋': 'model_logs/20180817172146342675_pixnet-instance1_pixfood20-火鍋_NORMAL_wgan_gp_pixfood20-火鍋',  # noqa: E501
    '手搖杯': 'model_logs/20180817193428453541_pixnet-instance1_pixfood20-手搖杯_NORMAL_wgan_gp_pixfood20-手搖杯',  # noqa: E501
    '咖啡': 'model_logs/20180818031704951283_pixnet-instance1_pixfood20-咖啡_NORMAL_wgan_gp_pixfood20-咖啡',  # noqa: E501
    '冰淇淋':  'model_logs/20180818060331227116_pixnet-instance1_pixfood20-冰淇淋_NORMAL_wgan_gp_pixfood20-冰淇淋',  # noqa: E501
    '薯條': 'model_logs/20180818063952643387_instance-1_pixfood20-薯條_NORMAL_wgan_gp_pixfood20-薯條',  # noqa: E501
    '漢堡': 'model_logs/20180818064159676192_instance-1_pixfood20-漢堡_NORMAL_wgan_gp_pixfood20-漢堡'  # noqa: E501
}


class Inpainting(object):
    _model_cache = {}

    @classmethod
    def from_cat(cls, cat):
        checkpoint_dir = cat_to_checkpoint_map[cat]
        print(checkpoint_dir)
        return cls.from_checkpoint_dir(checkpoint_dir)

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir):
        if checkpoint_dir in cls._model_cache:
            return cls._model_cache[checkpoint_dir]
        else:
            model = Inpainting(checkpoint_dir)
            cls._model_cache[checkpoint_dir] = model
            return model

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.model = InpaintCAModel()

    @staticmethod
    def gen_mask(img_array):
        xx, yy = np.where((img_array[:, :, 0] == 0) & (
            img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 0))
        mask = np.zeros(img_array.shape, dtype=int)
        for i, j in zip(xx, yy):
            mask[i, j, :] = [255, 255, 255]
        return mask

    def predict(self, image):
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
            output = self.model.build_server_graph(input_image)
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


if __name__ == '__main__':
    import sys
    import cv2

    print(f'loading {sys.argv[1]}')
    model = Inpainting.from_cat(sys.argv[1])
    image = cv2.imread(sys.argv[2])
    print(image.shape, image.dtype)
    result = model.predict(image)
    print(f'writing {sys.argv[3]}')
    cv2.imwrite(sys.argv[3], result)

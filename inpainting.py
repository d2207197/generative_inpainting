
import base64
import logging
import os
import sys
from io import BytesIO

import attr
import cv2
import numpy as np
import requests
import skimage.io as ski_io
import tensorflow as tf
from carriage import Row, StreamTable
from matplotlib import pyplot as plt

from inpaint_model import InpaintCAModel
from IPython.display import display
from mobile_net.clf import load_graph, read_tensor_from_image_file, load_labels

logger = logging.getLogger(__name__)

output_p = "examples/output.png"
checkpoint_dir_p = "model_logs/coffee"

model = InpaintCAModel()


def display_grid_plot(image_dict):
    nr = 5
    nc = 4
    id_list = []
    for i in range(nc):
        for j in range(nr):
            id_list.append((i, j))

    fig, axs = plt.subplots(nc, nr, figsize=(16,16))
    fig.suptitle('Multiple images')
    for i in range(len(image_dict)):
        ci, ri = id_list[i]
        axs[ci, ri].imshow(image_dict[i])
        axs[ci, ri].set_title(str(i))

    plt.show()

def display_image_BGR(image_bgr):
    image_rgb = convert_BGR_to_RGB(image_bgr)
    return display_image_RGB(image_rgb)


def display_image_RGB(image_rgb):
    plt.imshow(image_rgb)
    return plt.show()


def convert_BGR_to_RGB(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def convert_RGB_to_BGR(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


@attr.s
class FoodQuiz:
    question_id = attr.ib()
    raw_image = attr.ib()
    bbox = attr.ib()
    description = attr.ib()


cat_id_to_cat_map = {
    0: '小籠包',
    1: '麵包',
    2: '義大利麵',
    3: '沙拉',
    4: '串燒',
    5: '牛排',
    6: '拉麵',
    7: '壽司',
    8: '丼飯',
    9: '牛肉麵',
    10: '生魚片',
    11: '鬆餅',
    12: '蛋糕',
    13: '滷肉飯',
    14: '火鍋',
    15: '手搖杯',
    16: '咖啡',
    17: '冰淇淋',
    18: '薯條',
    19: '漢堡',
}


def display_cat_id_map():
    cat_id_cat_stbl = StreamTable(
        [Row(cat_id=cat_id, cat=cat)
         for cat_id, cat in cat_id_to_cat_map.items()])
    cat_id_cat_stbl.show(len(cat_id_to_cat_map))


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
    '蛋糕': 'model_logs/20180818120257878036_pixnet-instance1_pixfood20-蛋糕_NORMAL_wgan_gp_pixfood20-蛋糕',  # noqa: E501
    '滷肉飯': 'model_logs/20180817171517861902_pixnet-instance1_pixfood20-滷肉飯_NORMAL_wgan_gp_pixfood20-滷肉飯',  # noqa: E501
    '火鍋': 'model_logs/20180817172146342675_pixnet-instance1_pixfood20-火鍋_NORMAL_wgan_gp_pixfood20-火鍋',  # noqa: E501
    '手搖杯': 'model_logs/20180817193428453541_pixnet-instance1_pixfood20-手搖杯_NORMAL_wgan_gp_pixfood20-手搖杯',  # noqa: E501
    '咖啡': 'model_logs/20180818031704951283_pixnet-instance1_pixfood20-咖啡_NORMAL_wgan_gp_pixfood20-咖啡',  # noqa: E501
    '冰淇淋':  'model_logs/20180818060331227116_pixnet-instance1_pixfood20-冰淇淋_NORMAL_wgan_gp_pixfood20-冰淇淋',  # noqa: E501
    '薯條': 'model_logs/20180818063952643387_instance-1_pixfood20-薯條_NORMAL_wgan_gp_pixfood20-薯條',  # noqa: E501
    '漢堡': 'model_logs/20180818064159676192_instance-1_pixfood20-漢堡_NORMAL_wgan_gp_pixfood20-漢堡'  # noqa: E501
}


class Inpainting(object):
    _sess_cache = {}

    @classmethod
    def from_cat_id(cls, cat_id):
        cat = cat_id_to_cat_map[cat_id]
        checkpoint_dir = cat_to_checkpoint_map[cat]
        return cls.from_checkpoint_dir(checkpoint_dir)

    @classmethod
    def from_cat(cls, cat):
        checkpoint_dir = cat_to_checkpoint_map[cat]
        return cls.from_checkpoint_dir(checkpoint_dir)

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir):
        if checkpoint_dir in cls._sess_cache:
            return cls._sess_cache[checkpoint_dir]
        else:
            model = Inpainting(checkpoint_dir)
            cls._sess_cache[checkpoint_dir] = model
            return model

    def __init__(self, checkpoint_dir):
        model = InpaintCAModel()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        input_image_ph = tf.placeholder(
            tf.float32, shape=(1, 256, 512, 3))
        output = model.build_server_graph(input_image_ph, reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # === END OF BUILD GRAPH ===
        sess = tf.Session(config=sess_config)

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(
                checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        # sess.run(assign_ops)
        self.sess = sess
        self.graph = output
        self.placeholder = input_image_ph
        self.assign_ops = assign_ops

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

        self.sess.run(self.assign_ops)
        result = self.sess.run(self.graph,
                               feed_dict={self.placeholder: input_image})
        return result[0][:, :, ::-1]


# 從 PIXNET 拿到比賽題目
def get_image(question_id, img_header=True):
    endpoint = 'http://pixnethackathon2018-competition.events.pixnet.net/api/question'  # noqa: E501
    payload = dict(question_id=question_id, img_header=img_header)
    print('Step 1: 從 PIXNET 拿比賽題目\n')
    response = requests.get(endpoint, params=payload).json()

    try:
        data = response['data']
        question_id = data['question_id']
        description = data['desc']
        bbox = data['bounding_area']

        # Assign image format
        encoded_image = data['image']
        raw_image = ski_io.imread(
            BytesIO(base64.b64decode(
                encoded_image[encoded_image.find(',')+1:]))
        )
        # raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        header = encoded_image[:encoded_image.find(',')]
        if 'bmp' not in header:
            raise ValueError('Image should be BMP format')

        print('題號：', question_id)
        print('文字描述：', description)
        print('Bounding Box:', bbox)
        print('影像物件：', type(raw_image), raw_image.dtype,
              ', 影像大小：', raw_image.shape)

        quiz = FoodQuiz(question_id, raw_image, bbox, description)

    except Exception as err:
        # Catch exceptions here...
        print(response)
        raise err

    print('=====================')

    return quiz

# 上傳答案到 PIXNET


def submit_image(image, question_id, key=None):
    print('Step 3: 上傳答案到 PIXNET\n')

    endpoint = 'http://pixnethackathon2018-competition.events.pixnet.net/api/answer'  # noqa: E501

    if key is None:
        key = os.environ.get('PIXNET_FOODAI_KEY')

    # Assign image format
    image_format = 'jpeg'
    with BytesIO() as f:
        ski_io.imsave(f, image, format_str=image_format)
        f.seek(0)
        data = f.read()
        encoded_image = base64.b64encode(data)
    image_b64string = 'data:image/{};base64,'.format(
        image_format) + encoded_image.decode('utf-8')

    payload = dict(question_id=question_id,
                   key=key,
                   image=image_b64string)
    response = requests.post(endpoint, json=payload)
    try:
        rdata = response.json()
        return rdata
        if response.status_code == 200 and not rdata['error']:
            print('上傳成功')
        print('題號：', question_id)
        print('回答截止時間：', rdata['data']['expired_at'])
        print('所剩答題次數：', rdata['data']['remain_quota'])

    except Exception as err:
        print(rdata)
        raise err
    print('=====================')


class Secret:
    pass


# def answer_question(question_id):
def answer_question(raw_image):
    # quiz = get_image(question_id)
    # print('題號：', quiz.question_id)
    # print('文字描述：', quiz.description)
    # print('Bounding Box:', quiz.bbox)
    # print('影像物件：', type(quiz.raw_image), quiz.raw_image.dtype,
    #       ', 影像大小：', quiz.raw_image.shape)
    quiz = FoodQuiz(3, raw_image, (2, 2), 'blah balh')
    display_image_RGB(quiz.raw_image)

    raw_image = convert_RGB_to_BGR(quiz.raw_image)

    display_cat_id_map()
    result_images = {}
    for i in range(20):
        try:
            model = Inpainting.from_cat_id(i)
            result_images[i] = model.predict(raw_image)
        except Exception as err:
            logger.exception(f'cat_id {i} not found')
            continue

        print(f'{i} {cat_id_to_cat_map[i]}')
        display_image_BGR(result_images[i])

    cat_id = int(input())

    result_image = convert_BGR_to_RGB(result_images[cat_id])
    display('final choice')
    display_image_RGB(result_image)

    # return submit_image(result_image, question_id)


def mobile_net_clf(raw_image):
    model_file = "mobile_net/tf_files/retrained_graph.pb"
    label_file = "mobile_net/tf_files/retrained_labels.txt"
    input_name = "import/input"
    output_name = "import/final_result"
    top = 5
    graph = load_graph(model_file)

    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    file_name = 'tmp_image.jpg'
    cv2.imwrite(file_name, raw_image)

    t = read_tensor_from_image_file(file_name,
                                    input_height=224,
                                    input_width=224,
                                    input_mean=128,
                                    input_std=128)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
    results = np.squeeze(results)
    labels = load_labels(label_file)

    return labels, results



if __name__ == '__main__':

    print(f'loading {sys.argv[1]}')
    model = Inpainting.from_cat(sys.argv[1])
    image = cv2.imread(sys.argv[2])
    print(image.shape, image.dtype)
    result = model.predict(image)
    print(f'writing {sys.argv[3]}')
    cv2.imwrite(sys.argv[3], result)

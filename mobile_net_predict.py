from mobile_net.clf import load_graph, read_tensor_from_image_file, load_labels

model_file = "mobile_net/tf_files/retrained_graph.pb"
label_file = "mobile_net/tf_files/retrained_labels.txt"
input_name = "import/input"
output_name = "import/final_result"
top = 5
graph = load_graph(model_file)

input_operation = graph.get_operation_by_name(input_name);
output_operation = graph.get_operation_by_name(output_name);

q_id = 3
file_name = 'raw_image.jpg'
# raw_image = get_image(q_id) 
# ski_io.imsave(file_name, raw_image.raw_image.copy(), quality=100)


t = read_tensor_from_image_file(file_name,
                                input_height=224,
                                input_width=224,
                                input_mean=128,
                                input_std=128)

with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
    end=time.time()
results = np.squeeze(results)

top_k = results.argsort()[-1 * top:][::-1]
labels = load_labels(label_file)

print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
template = "{} (score={:0.5f})"
for i in top_k:
    print(template.format(labels[i], results[i]))

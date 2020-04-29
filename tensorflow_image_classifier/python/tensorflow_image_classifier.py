"""
DOCSTRING
"""
import sys
import tensorflow

image_path = sys.argv[1]
image_data = tensorflow.gfile.FastGFile(image_path, 'rb').read()
label_lines = [line.rstrip() for line in tensorflow.gfile.GFile("/tf_files/retrained_labels.txt")]

with tensorflow.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tensorflow.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tensorflow.import_graph_def(graph_def, name='')

with tensorflow.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))


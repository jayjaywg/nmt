from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf
import prediction_service_pb2_grpc

import requests

from tensorflow_serving.apis import predict_pb2

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

def _get_result_rpc_callback():
    def _callback(result_future):
        """Callback function.

        Calculates the statistics for the prediction result.

        Args:
          result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(
                result_future.result().outputs['seq_output'])
        #     prediction = numpy.argmax(response)
        #     if label != prediction:
        #         result_counter.inc_error()
        # result_counter.inc_done()
        # result_counter.dec_active()
            print(response)

    return _callback

def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      if label != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference(host):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  # test_data_set = ["预 测 一 下 今 年 世 界 杯 谁 会 赢", "今 年 世 界 杯"]
  with open(r"D:\guwang\src\ConsoleAppTemp1\Seq2Seq.TFServing.Client\[QnA] [cn] Label of Similar pair Zhidao all cat part1.tsv", "r", encoding="utf-8") as f:
      test_data_set = f.readlines()
  test_data_set = list(q.split('\t')[0] for q in test_data_set)
  test_data_set = test_data_set[:10]
  channel = grpc.insecure_channel(host+":8500")
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # result_counter = _ResultCounter(num_tests, concurrency)
  for _ in range(1):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'seq2seq_cnsim'
    # request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.model_spec.signature_name = "encode"
    # image, label = test_data_set.next_batch(1)
    request.inputs['seq_input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(test_data_set, dtype=tf.string))
    # result_counter.throttle()
    result = stub.Predict(request, 5.0)  # 5 seconds
    print(result)
    # result_future.add_done_callback(
    #     _get_result_rpc_callback())
        # _create_rpc_callback(label[0], result_counter))

  return

def do_inference_restful(host):
    with open(r"D:\temp\[QnA] [cn] Label of Similar pair Zhidao 体育_0.tsv", "r", encoding="utf-8") as f:
        test_data_set = f.readlines()
    test_data_set = list(q.split('\t')[0] for q in test_data_set)
    test_data_set = test_data_set[:101]
    req = {
              "signature_name": "encode",
              "instances" : test_data_set
          }
    res = requests.post("http://"+host+":8501"+"/v1/models/seq2seq_cnsim:predict", json=req)
    print(res)

def main(_):
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  do_inference(FLAGS.server)
  # do_inference_restful(FLAGS.server)



if __name__ == '__main__':
  tf.app.run()

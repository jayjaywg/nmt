# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Export pre-trained model."""
import collections
import os
import time
import argparse

import nmt
import tensorflow as tf

import numpy as np

import attention_model as attention_model
import gnmt_model as gnmt_model
import model as nmt_model
import nmt_utils
import vocab_utils
from utils import misc_utils
from tensorflow.python.ops import lookup_ops

class Exporter(object):
  """Export pre-trained model and serve it by tensorflow/serving.
  """

  def __init__(self, hparams, flags):
    """Construct exporter.

    By default, the hparams can be loaded from the `hparams` file
    which saved in out_dir if you enable save_hparams. So if you want to
    export the model, you just add arguments that needed for exporting.
    Arguments are specified in ``nmt.py`` module.
    Go and check that in ``add_export_arugments()`` function.

    Args:
     hparams: Hyperparameter configurations.
     flags: extra flags used for exporting model.
    """
    self.hparams = hparams
    self._model_dir = self.hparams.out_dir
    v = flags.version_number
    self._version_number = v if v else int(round(time.time() * 1000))

    export_path = flags.export_path if flags.export_path else self.hparams.out_dir
    self._export_dir = os.path.join(export_path, str(self._version_number))

    # Decide a checkpoint path
    ckpt_path = self._get_ckpt_path(flags.ckpt_path)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    self._ckpt_path = ckpt.model_checkpoint_path

    # A file contains sequences, used for initializing iterators.
    # A good idea is to use test or dev files as infer_file
    # test_file = self.hparams.test_prefix + "." + self.hparams.src
    # self._infer_file = flags.infer_file if flags.infer_file else test_file

    self._print_params()

  def _print_params(self):
    misc_utils.print_hparams(self.hparams)
    print("Model directory  : %s" % self._model_dir)
    print("Checkpoint path  : %s" % self._ckpt_path)
    print("Export path      : %s" % self._export_dir)
    # print("Inference file   : %s" % self._infer_file)
    print("Version number   : %d" % self._version_number)

  def _get_ckpt_path(self, flags_ckpt_path):
    ckpt_path = None
    if flags_ckpt_path:
      ckpt_path = flags_ckpt_path
    else:
      for metric in self.hparams.metrics:
        p = getattr(self.hparams, "best_" + metric + "_dir")
        if os.path.exists(p):
          if self._has_ckpt_file(p):
            ckpt_path = p
          break
    if not ckpt_path:
      ckpt_path = self.hparams.out_dir
    return ckpt_path

  @staticmethod
  def _has_ckpt_file(p):
    for f in os.listdir(p):
      if str(f).endswith(".meta"):
        return True
    return False



  def _create_infer_model(self,src_seqs_placeholder):
    if not self.hparams.attention:
      model_creator = nmt_model.Model
    elif self.hparams.attention_architecture == "standard":
      model_creator = attention_model.AttentionModel
    elif self.hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
      model_creator = gnmt_model.GNMTModel
    else:
      raise ValueError("Unknown model architecture")
    model = create_infer_model(model_creator=model_creator,
                                            hparams=self.hparams, scope=None, extra_args=None, src_seqs_placeholder=src_seqs_placeholder)
    return model


  def export(self):
    graph = tf.Graph()
    with graph.as_default():
        feature_config = {
          'infer_input': tf.FixedLenSequenceFeature(dtype=tf.string,
                                              shape=[], allow_missing=True),
        }
        input_list_of_query = tf.placeholder(shape=[None], dtype=tf.string, name="tf_example")
        tf_example = tf.parse_example(input_list_of_query, feature_config)
        inference_input = tf_example['infer_input']

        infer_model = self._create_infer_model(src_seqs_placeholder=input_list_of_query)

    with tf.Session(graph=infer_model.graph,
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver = infer_model.model.saver
      saver.restore(sess, self._ckpt_path)

      # initialize tables
      sess.run(tf.tables_initializer())

      # get outputs of model
      nmt_outputs, _, _, encoder_state = infer_model.model.decode(sess=sess)
      cell_direct = hparams.encoder_type
      unit_type = hparams.unit_type
      beam_width = hparams.beam_width
      if cell_direct == "uni":
          if unit_type == "gru":
              encoder_last_hidden_output = encoder_state[-1]
          elif unit_type == "lstm":
              encoder_last_hidden_output = encoder_state[-1].h
          else:
              raise ValueError("Unknown unit_type, only support lstm and gru for now")
      elif cell_direct == "bi":  # encoder last hidden state is concat of last forward layer and last backward layer
          if unit_type == "gru":
              encoder_last_hidden_output = tf.concat([encoder_state[-2], encoder_state[-1]], axis=-1)
          elif unit_type == "lstm":
              encoder_last_hidden_output = tf.concat([encoder_state[-2].h, encoder_state[-1].h], axis=-1)
          else:
              raise ValueError("Unknown unit_type, only support lstm and gru for now")

      # # get text translation
      # assert nmt_outputs.shape[0] == 1
      #

      #
      # translation = nmt_utils.get_translation(
      #     nmt_outputs,
      #     sent_id=0,
      #     tgt_eos=self.hparams.tgt_eos,
      #     subword_option=self.hparams.subword_option)

      # test exported model logic
      test_data_set = ["最 好 的 分 辨 率","1 8 岁 的 人 该 干 些 什 么 事","最 好 的 分 辨 率 w 1"]
      encoder_vec, inference_output = sess.run(
          [encoder_last_hidden_output,
           nmt_outputs],
        feed_dict={
            input_list_of_query: test_data_set
        })

      # create signature def
      # key `seq_input` in `inputs`, `seq_output` in 'outputs', should be should be consistent with this client
      inference_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={
          'seq_input': input_list_of_query
        },
        outputs={
          'seq_output': nmt_outputs
        }
      )

      encode_signature = tf.saved_model.signature_def_utils.predict_signature_def(
          inputs={
              'seq_input': input_list_of_query
          },
          outputs={
              'seq_output': encoder_last_hidden_output
          }
      )

      legacy_ini_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

      builder = tf.saved_model.builder.SavedModelBuilder(self._export_dir)

      builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: inference_signature,
            "encode": encode_signature
        },
        legacy_init_op=legacy_ini_op,
        clear_devices=True,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
      builder.save(as_text=True)
      print("Export Done!")

class InferModel(
      collections.namedtuple("InferModel",
                             ("graph", "model", "src_placeholder",
                              "batch_size_placeholder", "iterator"))):
  pass

def create_infer_model(model_creator, hparams, scope=None, extra_args=None, src_seqs_placeholder=None):
      """Create inference model."""
      # this method is called within a default graph. See caller.

      graph = tf.get_default_graph()
      src_vocab_file = hparams.src_vocab_file
      tgt_vocab_file = hparams.tgt_vocab_file

      src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
          src_vocab_file, tgt_vocab_file, hparams.share_vocab)
      reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
          tgt_vocab_file, default_value=vocab_utils.UNK)

      # max len of sequence
      src_seq_len = tf.map_fn(lambda src: tf.size(tf.string_split([src]).values), src_seqs_placeholder, dtype=tf.int32)
      max_seq_len=tf.reduce_max(src_seq_len)

      # padding sequence with eos
      src_dataset = tf.map_fn(lambda src: _string_padding(str2=src, current_length=tf.size(tf.string_split([src]).values),
                                                          max_length=max_seq_len, padding_const=hparams.eos), src_seqs_placeholder)

      src_dataset = tf.map_fn(lambda src: tf.string_split([src]).values, src_dataset)

      if hparams.src_max_len_infer:
          src_dataset = tf.map_fn(lambda src: src[:hparams.src_max_len_infer], src_dataset)
      # Convert the word strings to ids
      src_dataset = tf.map_fn(
          lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32), src_dataset, dtype=tf.int32)
      batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

      src_seq_len = tf.transpose(src_seq_len)

      model = model_creator(
          hparams,
          iterator=None,
          mode="serving",
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          reverse_target_vocab_table=reverse_tgt_vocab_table,
          scope=scope,
          extra_args=extra_args,
          src_dataset=src_dataset,
          src_seq_len=src_seq_len)  # only for SERVING
      return InferModel(
          graph=graph,
          model=model,
          src_placeholder=src_seqs_placeholder,
          batch_size_placeholder=batch_size_placeholder,
          iterator=None)

def _string_padding(str2, current_length, max_length, padding_const):
    while_condition = lambda current_length, str2: tf.logical_not(tf.greater(current_length + 1, max_length))
    def body(len2, str2):
        return [tf.add(len2, 1), str2 + tf.constant(" "+padding_const)]
    # do the loop:
    r = tf.while_loop(while_condition, body, [current_length, str2])
    return r[1]

def _update_flags(flags, test_name):
  flags.export_path = r"D:\guwang\work\QA\model\6_2\s2s_71_exported" # TODO guwang
  flags.version_number = None
  flags.ckpt_path = None
  flags.infer_file = "nmt/testdata/test_infer_file"

if __name__ == "__main__":
  out_dir = r"D:\guwang\work\QA\model\6_2\s2s_71"
  nmt_parser = argparse.ArgumentParser()
  nmt.add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()

  _update_flags(FLAGS, "exporter_test")
  default_hparams = nmt.create_hparams(FLAGS)
  from nmt import create_or_load_hparams
  hparams = create_or_load_hparams(
      out_dir, default_hparams, FLAGS.hparams_path, save_hparams=False)
  Exporter(hparams, FLAGS).export()
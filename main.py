from absl import app
from absl import flags
from absl import logging
from engine import Engine

FLAGS = flags.FLAGS

flags.DEFINE_enum('model_name', None, ['transformer', 'rnn'], 'transformer')
flags.DEFINE_enum('phase', None, ['train', 'test', 'inference', 'val'], 'Phase to run.')
flags.DEFINE_string('data_path', './data/wmt16', 'Path to dataset.')
flags.DEFINE_string('name_suffix', 'enro', 'What dataset to use.')
flags.DEFINE_string('sentence', 'My name is Jarvis!', 'Sentence to translate.')
flags.DEFINE_string('model_path', './models/model_transformer_enro.pt', 'Path to model.')
flags.DEFINE_bool('use_w2v', False, 'Whether to use word2vec.')
flags.DEFINE_bool('freeze_w2v', True, 'Whether to freeze the w2v embeddings.')
flags.DEFINE_enum('text_preprocessor', None, ['stemming', 'bert_tokenizer'], 'How to preprocess the text. If None, the sentences will be split by space, \
                  if "stemming", the sentence will be split by space and the words will be stemmed, if "tokenizer", the BERT tokenizer will be applied.')

def main(_):
    engine = Engine(FLAGS.model_name, FLAGS.data_path, FLAGS.name_suffix, FLAGS.model_path,
                    FLAGS.use_w2v, FLAGS.freeze_w2v, FLAGS.text_preprocessor)

    if FLAGS.phase == 'train':
        engine.train()
    elif FLAGS.phase == 'test':
        engine.test(model_path=FLAGS.model_path)
    elif FLAGS.phase == 'val':
        engine.val(model_path=FLAGS.model_path)
    elif FLAGS.phase == 'inference':
        print('Original sentence: {}'.format(FLAGS.sentence))
        print('Translated sentence: {}'.format(' '.join(engine.translate_sentence(FLAGS.sentence, model_path=FLAGS.model_path)[0])))

    logging.info('Done!')

if __name__ == '__main__':
    app.run(main)

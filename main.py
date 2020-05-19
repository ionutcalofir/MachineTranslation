from absl import app
from absl import flags
from absl import logging
from engine import Engine

FLAGS = flags.FLAGS

flags.DEFINE_enum('phase', None, ['train', 'test', 'inference', 'val'], 'Phase to run.')
flags.DEFINE_string('data_path', './data/wmt16', 'Path to dataset.')
flags.DEFINE_string('name_suffix', 'enro', 'What dataset to use.')
flags.DEFINE_string('sentence', 'My name is Jarvis!', 'Sentence to translate.')
flags.DEFINE_string('model_path', './models_baseline/model_transformer_enro.pt', 'Path to model.')

def main(_):
    engine = Engine(FLAGS.data_path, FLAGS.name_suffix, FLAGS.model_path)

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

from pathlib import Path
import os
import logging
from tqdm import tqdm
import tensorflow as tf

def setup_logger(log_dir,args):
    Path(log_dir).mkdir(parents=True,exist_ok='temp' in log_dir)
    with open(os.path.join(log_dir,'args.txt'),'w') as f:
        f.write(str(args))

    # TF Summary Writer as Logger
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,'tb'))
    class TFSummaryHandler(logging.StreamHandler):
        def emit(self, record):
            with summary_writer.as_default():
                if record.msg == 'raw':
                    tag, value, it = record.args
                    tf.summary.scalar(tag, value, step=it)
                elif record.msg == 'text':
                    tag, value, it = record.args
                    tf.summary.text(tag, value, step=it)
                else:
                    summary_str, it = record.args

        def flush(self):
            summary_writer.flush()

    handler = TFSummaryHandler()
    handler.setLevel(logging.DEBUG)

    logger = logging.getLogger('summary_writer')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Standard stdout Logger
    class TqdmHandler(logging.Handler):
        def emit(self, record):
            try:
                tqdm.write(self.format(record))
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    handler = TqdmHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger('stdout')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

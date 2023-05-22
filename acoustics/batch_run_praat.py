from concurrent.futures import process
import sys
import logging
import argparse
import pandas as pd
from batch_base.process_job import ProcessJob, batch_task_wrapper, BatchInputType
from acoustics.praat_acoustics import process_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class batch_job(ProcessJob):
    """ Batch job class for raw variable output; Derived from ProcessJob of the batch_base package.
        This job accesses the following member variables
        self.local_input_folder: stores the video frames for inference, organized by videos as folders
        self.downloaded_data: list of the local paths to each video folder
        The overrided run_task returns the inference results as a pandas dataframe
    """
    @batch_task_wrapper
    def run_task(self):
        try:             
            logger.info('Running CDS acoustics')
            process_directory(self.local_input_folder, self.local_output_folder, num_threads=1)
             
        except Exception as e:
            logger.error("Processing CDS Acoustics failed: {}".format(e))  


def run_praat(s3_path_to_input_csv, 
                s3_path_to_result,
                s3_path_to_output_prefix,
                ):
    vad_job = batch_job(s3_path_to_input_csv=s3_path_to_input_csv,
                             s3_path_to_result=s3_path_to_result,
                             input_type=BatchInputType.VIDEO_CSV, 
                             s3_path_to_output_prefix = s3_path_to_output_prefix,
                            )
    vad_job.run_task()

if __name__ == '__main__':
    # --s3_path_to_input_csv s3://cds-vad-test/actor-videos/CDS-actors-dbm-demo-small-set-urls.csv
    # --s3_path_to_result s3://cds-vad-test/results
    # --input_type BatchInputType.S3_OBJECTS
    # --s3_path_to_output_prefix s3://cds-vad-test/results/
    # --s3_path_to_model s3://cds-vad-test/models/pytorch_model_segmentation.bin
    parser = argparse.ArgumentParser(description="Run pyanote-cds-acoustics. save variables by frames")
    parser.add_argument("--s3_path_to_input_csv", type=str, help="s3 path to the input csv for vad")
    parser.add_argument("--s3_path_to_result", type=str, help='s3 path to the result csv')
    parser.add_argument("--s3_path_to_output_prefix", type=str, help="s3 path to upload")
    #parser.add_argument("--s3_path_to_model", type=str, help='path to torch model')

    args = parser.parse_args()
    logger.info("param1: {}".format(args.s3_path_to_input_csv))
    logger.info("param2: {}".format(args.s3_path_to_result))
    logger.info("param3: {}".format(args.s3_path_to_output_prefix))
    #logger.info("param4: {}".format(args.s3_path_to_model))

    run_praat(args.s3_path_to_input_csv, args.s3_path_to_result, args.s3_path_to_output_prefix)
    sys.exit(0)

from acoustics.praat_acoustics import process_directory, acoustics_map
import os

video_directory = "./data/actor_videos"

def test_process_directory(tmp_path):
    tmp_output_dir = tmp_path / "data/tmp_output"
    process_directory(video_directory, tmp_output_dir, num_threads=1)

    # check if output files are created
    for f in os.listdir(video_directory):
        if f.endswith(".mp4"):
            for k in acoustics_map.keys():
                assert os.path.exists(f'{tmp_output_dir}/data/{k}/{f[:-4]}.csv')

# def test_batch_job():
#     # This test requires that the s3_path_to_input_csv file contains urls to encrypted videos
#     # TODO: this test does not currently have an assert statement
#     # To verify the job runs properly, the test should first remove any existing
#     # files from the s3_path_to_result directory, then run the job, then check
#     # if the expected files are in the s3_path_to_result directory
#     # currently, one can do this manually to verify the job runs properly
#     s3_path_to_result = 's3://cds-vad-test/results'
#     s3_path_to_output_prefix = 's3://cds-vad-test/results/'
#     s3_path_to_model = 's3://cds-vad-test/models/pytorch_model_segmentation.bin'
#     s3_path_to_input_csv = 's3://simulated-td-videos/dev_encrtyped_videos.csv' 

#     run_vad(s3_path_to_input_csv, s3_path_to_result, s3_path_to_output_prefix, s3_path_to_model)


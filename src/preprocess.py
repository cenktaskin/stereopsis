from preprocessing.data_preprocessor import Preprocessor

preprocessor = Preprocessor(calibration_data_id="20220610",
                            raw_datasets=["202207011356-cleaned"])

simple_process = 0
if simple_process:
    preprocessor.set_output_path("processed/dataset-20220701test-fullres")
    preprocessor.crop_the_dataset(save_result=False)

undistortion = 0
if undistortion:
    preprocessor.set_output_path("processed/dataset-20220701test-undistorted")
    preprocessor.undistort_dataset(save_results=False)

rectify = 0
if rectify:
    preprocessor.set_output_path("processed/dataset-20220701test-rectified")
    preprocessor.rectify_dataset(save_results=False)

register = 0
if register:
    preprocessor.set_output_path("processed/dataset-20220701test-registered")
    preprocessor.register_dataset(save_results=False)

register_and_register = 0
if register_and_register:
    preprocessor.set_output_path("processed/dataset-20220701test-registered-and-rectified")
    preprocessor.register_and_rectify_dataset(save_results=False)

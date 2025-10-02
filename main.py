from src.DermaCareNet.full_training_pipeline_yolov5m.pipeline_cv import Yolov5TrainingPipeline




pipeline = Yolov5TrainingPipeline(url="https://drive.google.com/file/d/1jVFpALhnDOLuqUj7CQgXw5N2PK2E44lN/view?usp=sharing",
                                  prefix='https://drive.google.com/uc?/export=download&id=')

pipeline.initialize_pipeline()
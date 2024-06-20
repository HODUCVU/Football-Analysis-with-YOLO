# Build Football Analysis System with YOLO
## Data 
| Data | Link 
|---|--|
| Data Training | [https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) |
| Input Videos | [https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data)|
## Output


https://github.com/HODUCVU/Football-Analysis-with-YOLO/assets/73897430/c15b96eb-65f1-4d0a-9734-114a27f94e76



## Usage
* Set up virtual-env (if necessary)
```
> sudo apt install python3-virtualenv
> virtualenv venv
> source venv/bin/activate

```
* Set up Environment
```
> pip install -r requirements.txt
```
* Running
```
> python3 main.py --video_input_path input_videos.mp4 --video_output_path output_video.avi --model models/best.pt
```

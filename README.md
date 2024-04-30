# bowlingAI_pinsDetection
Bowling center owners! this code allows you to detect automatically if the red pin is in the front position.

https://www.youtube.com/shorts/-j8adb1jmmI?feature=share

## how to use it
Git clone the code in main branch
CD to the main repo

docker build -t api_image:latest .

docker run -it --rm \
  --name api \
  --network=host \
  --gpus all \
  -p 5000:5000 \
  -v ${PWD}:/wd \
  --user jovyan \
  api_image:latest

Obtain weights from DepthAnything and insert it in placeholder:

Metric depth --> https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main
 base --> https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth

Make requests: 
path_in_your_machine/bowlingAI_pinsDetection$ python3 request.py

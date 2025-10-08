source .venv/bin/activate


python infer.py   --model "/mnt/c/Users/deepm/OneDrive/Desktop/Wheat_detection/server/model.pth"   --input "/mnt/c/Users/deepm/OneDrive/Desktop/image.png"   --out "/mnt/c/Users/deepm/OneDrive/Desktop/Wheat_detection/server/out"   --img-size 768   --device cpu




# build image (first time)
docker build -t wheat-infer:latest .

# run (model.pth must be present in server/ or change path)
docker run --rm -p 5000:5000 \
  -v "$(pwd)/model.pth:/app/model.pth:ro" \
  -v "$(pwd)/uploads:/app/uploads" \
  -v "$(pwd)/static/results:/app/static/results" \
  -e MODEL_PATH=/app/model.pth \
  wheat-infer:latest


  docker run --rm -p 5000:5000 \
  --mount type=bind,source="$(pwd)/model.pth",target=/app/model.pth,readonly \
  --mount type=bind,source="$(pwd)/uploads",target=/app/uploads \
  --mount type=bind,source="$(pwd)/static/results",target=/app/static/results \
  -e MODEL_PATH=/app/model.pth \
  wheat-infer:latest




  docker run --rm \                                                                                                                     
    -p 5000:5000 \                                                                                                                      
    -v "$(pwd)/model.pth:/app/model.pth:ro" \                                                                                           
    -v "$(pwd)/best.pt:/app/best.pt:ro" \
    -v "$(pwd)/uploads:/app/uploads" \                                                                                                  
    -v "$(pwd)/static/results:/app/static/results" \
    -e MODEL_PATH=/app/model.pth \                                                                                                      
    -e YOLO_MODEL_PATH=/app/best.pt \
    -e IMG_SIZE=1024 \                                                                                                                  
    -e DEVICE=cpu \                                                                                                                     
    -e YOLO_DEVICE=cpu \                                                                                                                
    -e YOLO_CONF=0.25 \                                                                                                                 
    -e YOLO_IOU=0.45 \                                                                                                                  
    -e YOLO_MAX_DET=100 \                                                                                                               
    -e YOLO_DISEASE_CLASS=2\                                                                                                           
    -e YOLO_WHEAT_CLASSES=1,2\                                                                                                         
    wheat-infer:latest 

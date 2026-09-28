docker build --no-cache -t hardware-eval .


docker run --rm --gpus all \
  -v "$(pwd)/mvtec_anomaly_detection:/data:ro" \
  -v "$(pwd)/exported_models:/models:ro" \
  -v "$(pwd)/output:/output" \
  hardware-eval \
  --device-name "Ryzen7_9700X_RTX2080Ti" \
  --accelerator auto \


  device-name："Ryzen7_9700X_RTX2080Ti" , "Ryzen7_7735U", "Intel_Celeron_N5095"
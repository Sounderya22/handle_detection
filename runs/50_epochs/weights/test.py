from ultralytics import YOLO

model = YOLO('best.pt')
results = model('handle3.png', conf=0.1)

# Visualize
results[0].show()
results[0].save(filename='output.jpg')  # Save to disk

# from ultralytics import YOLO
# from pathlib import Path

# # === Configuration ===
# model_path = 'best.pt'
# test_folder = Path('/home/vv/PRG_Assessment/car-door-handle.v2i.yolov12/dataset/test/images')           # folder with test images
# output_folder = Path('/home/vv/PRG_Assessment/test_results')        # output folder

# output_folder.mkdir(parents=True, exist_ok=True)

# # === Load the model
# model = YOLO(model_path)

# # === Loop over images and run inference
# image_extensions = ['.jpg', '.jpeg', '.png']

# for img_path in test_folder.glob("*"):
#     if img_path.suffix.lower() not in image_extensions:
#         continue

#     results = model(str(img_path), conf=0.25)
    
#     # Save to output folder with same filename
#     out_path = output_folder / img_path.name
#     results[0].save(filename=str(out_path))

# print("✅ Inference complete. Results saved to:", output_folder)

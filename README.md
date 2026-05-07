# Concrete-AI-Inspector
Automated Structural Health Monitoring using YOLOv11 and Streamlit

#  AI Structural Health Monitoring (SHM)

An end-to-end Machine Vision pipeline for automated concrete crack segmentation and structural damage quantification. 

##  Features
* **Hybrid Vision Comparison:** Side-by-side real-time analysis comparing Semantic Segmentation (YOLOv11) against traditional Edge Detection (Canny).
* **Digital Image Processing (DIP):** Implements **Bilateral Filtering** for edge-preserving noise reduction.
* **Engineering Analytics:** Automatically calculates the 'Surface Area Defect Percentage' to classify structural integrity.

##  Tech Stack
* **UI/Dashboard:** Streamlit
* **Deep Learning:** PyTorch, Ultralytics (YOLOv11)
* **Machine Vision:** OpenCV

##  How to Run Locally
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Place your trained `best.pt` YOLO weights inside a `weights/` folder.
4. Run the dashboard using `streamlit run app.py`.

# DeepFaceTesting
Documentation for Testing of DeepFace
  
Create conda environment for deepface  
```**conda create --name deepface**  ```

Install deepface using pip  
```**pip install deepface**  ```

Install jupyter notebook dependencies in deepface environment  
```**conda install jupyter**  ```
  
Install matplotlib dependencies  
```**conda install -c conda-forge matplotlib**```
  
Install opencv  
```**conda install -c conda-forge opencv**```

## Initial Jupyter Notebook Setup  
1. Launch conda environment using the command ```conda activate deepface```
2. Launch Jupyter Notebook from conda environment using the command ```jupyter notebook```
3. Create a new jupyter notebook ipynb under **new** on the right hand side
4. Follow example ipynb in repo or look below for important code snippets  

## Initial imports
  ```python
#Import DeepFace object from deepface
from deepface import DeepFace
#Plotting
import matplotlib.pyplot as plt
#Loading in images
import cv2
  ```
## Face Detection Backends & Object Call  
```python
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]

face = DeepFace.detectFace('C:/Users/username/deepfaceimages/namefolder/name.jpg',
                    target_size=(224,224),
                    detector_backend='opencv')
```

  ## Comparing Detection Backends
  ```python
  fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs = axs.flatten()
for i, b in enumerate(backends):
    try:
        face = DeepFace.detectFace('C:/Users/username/deepfaceimages/namefolder/name2.jpg',
                    target_size=(224,224),
                    detector_backend='opencv')
        axs[i].imshow(face)
        axs[i].set_title(b)
        axs[i].axis('off')
    except:
        pass
plt.show()
  ```

  ## Face Verification  
  ```python
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
results = DeepFace.verify(img1_path = "C:/Users/username/deepfaceimages/namefolder/name7.jpg", 
                          img2_path = "C:/Users/username/deepfaceimages/namefolder/name1.jpg", 
                          model_name = models[0])
  ```
Afterwards you can pull the results out with  
```python
results
```
> {'verified': False,
 'distance': 0.8153093721238277,
 'threshold': 0.4,
 'model': 'VGG-Face',
 'detector_backend': 'opencv',
 'similarity_metric': 'cosine',
 'facial_areas': {'img1': {'x': 62, 'y': 37, 'w': 110, 'h': 110},
  'img2': {'x': 41, 'y': 49, 'w': 123, 'h': 123}},
 'time': 1.7}

  ## Simplified Verification Code  
  ```python
for model in models:
    
    model = models[0]
    results = DeepFace.verify(img1_path = "C:/Users/ssala/deepfaceimages/JenniferLopez/Jlo1.jpg", 
                              img2_path = "C:/Users/ssala/deepfaceimages/JenniferLopez/Jlo2.jpg", 
                              model_name = models[0])

    fig, axs = plt.subplots(1, 2, figsize = (15, 5))
    axs[0].imshow(plt.imread("C:/Users/ssala/deepfaceimages/JenniferLopez/Jlo1.jpg"))
    axs[1].imshow(plt.imread("C:/Users/ssala/deepfaceimages/JenniferLopez/Jlo2.jpg"))
    fig.suptitle(f"Verified {results['verified']} - Distance {results['distance']:0.5}: Model {model}")

    axs[0].axis("off")
    axs[1].axis("off")
    plt.show()
  ```  
  ## Object Call DeepFace and Assign to Results  
  ```python
    results = DeepFace.find(img_path="C:/Users/ssala/deepfaceimages/MattDamon/Matt1.jpg",
             db_path="C:/Users/ssala/deepfaceimages/MattDamon/")
  ```  
  Call results  
  ```python
  results
  ```
You now have the output and assignment of each image result given a specific model, in this case the values are from the VGG backend.  
> [                                            identity  source_x  source_y  \
 0  C:/Users/ssala/deepfaceimages/MattDamon//Matt1...        35        56   
 1  C:/Users/ssala/deepfaceimages/MattDamon//Matt2...        35        56   
 2  C:/Users/ssala/deepfaceimages/MattDamon//Matt4...        35        56   
 3  C:/Users/ssala/deepfaceimages/MattDamon//Matt6...        35        56   
 4  C:/Users/ssala/deepfaceimages/MattDamon//Matt5...        35        56   
 5  C:/Users/ssala/deepfaceimages/MattDamon//Matt3...        35        56   
 6  C:/Users/ssala/deepfaceimages/MattDamon//Matt7...        35        56   
 
   >  source_w  source_h  VGG-Face_cosine  
 0       116       116     1.110223e-16  
 1       116       116     8.124922e-02  
 2       116       116     1.970226e-01  
 3       116       116     2.050450e-01  
 4       116       116     2.096324e-01  
 5       116       116     2.254156e-01  
 6       116       116     2.923807e-01  ]
 >  
 # Facial Attribute Analysis  
 ```python
results = DeepFace.analyze(img_path = "C:/Users/ssala/deepfaceimages/MattDamon/Matt1.jpg")
 ```
This will anaylze the facial composition and any attributes that can be associated with them.  
  Call results  
  ```python
  results
  ```  
  > [{'emotion': {'angry': 3.1368814408779144,
   'disgust': 6.192519776959671e-05,
   'fear': 0.2862529596313834,
   'happy': 8.087079972028732,
   'sad': 4.808013513684273,
   'surprise': 0.046116369776427746,
   'neutral': 83.63559246063232},
  'dominant_emotion': 'neutral',
  'region': {'x': 35, 'y': 56, 'w': 116, 'h': 116},
  'age': 27,
  'gender': {'Woman': 0.010882341302931309, 'Man': 99.98911619186401},
  'dominant_gender': 'Man',
  'race': {'asian': 0.0001700479401733901,
   'indian': 0.0005569705535890535,
   'black': 5.607818209796278e-06,
   'white': 99.29988384246826,
   'middle eastern': 0.2788534853607416,
   'latino hispanic': 0.42053312063217163},
  'dominant_race': 'white'}]

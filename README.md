# Deepfake Detector                                                                              
                                                                                                 
  An image-based deepfake detection web app that classifies face images as **Real** or **Deepfake**
   using a fine-tuned Vision Transformer (ViT).                                                  
                                                                                                   
  ## Setup                                                                                       
                                                                                                 
  ```bash
  pip install -r requirements.txt
  python train.py   # fine-tune the model (requires Kaggle auth)                                   
  python app.py     # start the web app at http://127.0.0.1:5000
                                                                                                   
  How It Works                                                                                   
                                                                                                   
  1. Upload a face image to the web interface                                                      
  2. OpenCV decodes and preprocesses the image                                                   
  3. A fine-tuned ViT model classifies it as Real or Deepfake                                      
  4. Confidence score is displayed                                                                 
                                                                                                   
  Credits                                                                                          
                                                                                                   
  Pretrained Model                                                                               
  - Deep-Fake-Detector-v2-Model by prithivMLmods                                                 
  - https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model                               
  - License: Apache 2.0                                             
                                                                                                   
  Dataset                                                                                        
  - Deepfake and Real Images by Manjil Karki                                                       
  - https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
                                                                                                   
  Libraries                                                                                      
  - HuggingFace Transformers — https://github.com/huggingface/transformers                         
  - PEFT (LoRA) — https://github.com/huggingface/peft                     
  - PyTorch — https://pytorch.org                                                                  
  - OpenCV — https://opencv.org                                                                    
                                    
This is code and picture for Skull3D-Net project.
Our data is private thus can't be released. 
The model weight can be found at https://huggingface.co/mengliu007/best_model/tree/main



![image](https://github.com/user-attachments/assets/aa8cb84b-b9a4-4c91-b06d-518745453675)
Case example:
1. get the CT data (dicom FILE);
2. Download the code to your server with GPUs；
3. Run the data.py to preprocess the raw CT data;
4. Download the model weights to the same server;
5. Put the model and weights to the same GPU;
6. Run the model file to put the data into the model (model3d.py);
7. predict the age!

![Fig 3](https://github.com/user-attachments/assets/3e9abf96-ca2f-4390-8cd9-a418dcb857ab)



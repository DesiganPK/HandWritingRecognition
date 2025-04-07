'''import os
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO, WARNING, and ERROR logs
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
warnings.simplefilter("ignore", category=DeprecationWarning)'''
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from transformers.utils.logging import set_verbosity_error
import google.generativeai as genai
genai.configure(api_key="AIzaSyC_IeRcWH9nmYZMtqY9CrObu8hxs0xPHQo")
set_verbosity_error()
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image


# Load processor
processor = TrOCRProcessor.from_pretrained(r"C:\Users\Desigan\Downloads\word_segmentation-master\word_segmentation-master\trocr-base-handwritten")

# Load model - explicitly specifying the weight file
model = VisionEncoderDecoderModel.from_pretrained(
    r"C:\Users\Desigan\Downloads\word_segmentation-master\word_segmentation-master\trocr-base-handwritten",
    ignore_mismatched_sizes=True  # This allows loading even if some weights are missing
)

i = 0
text = ""
while True:
    file_path = r"C:\Users\Desigan\Downloads\word_segmentation-master\word_segmentation-master\segmented\segment"+str(i)+".png"
    if not os.path.exists(file_path):
        #print(f"File not found: {file_path}")
        break
    #image = cv2.imread(file_path)
    image = Image.open(file_path).convert("RGB")
    if image is None:
        break
    else:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():  # No gradient needed for inference
            generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        prediction_text = generated_text
        text += prediction_text+" "
        i+=1
        
model_ai = genai.GenerativeModel("gemini-1.5-flash-8b")
response = model_ai.generate_content("Please give the following sentence with spelling mistakes corrected.Change misspelled words only to the closest valid English word and parse the text word by word.Do not change words that are a valid word in English dictionary. There may be some characters wrong in each word. Please also keep it case sensitive(Do not change the case of the characters.). Also, some words may not have any errors, if so, do not change them. Input Sentence: "+text)
#response = model_ai.generate_content("Please give the following sentence with spelling mistakes corrected.Change misspelled words only to the closest valid English word and parse the text word by word. Do not change words that are a valid word in English dictionary. There may be some characters wrong in each word. Please also keep it case sensitive(Do not change the case of the characters.). Also, some words may not have any errors, if so, do not change them. Input Sentence: "+text)
print("Full predicted text = "+response.text)
text = response.text
text = prediction_text.strip('\n')
#print("Full predicted text = ", text)

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import io

# FastAPI অ্যাপ ইনিশিয়ালাইজ করুন
app = FastAPI(title="Image Upscaling API")

# মডেল লোড করুন
# এখানে আপনার ডাউনলোড করা .pth মডেল ফাইলের পাথ দিন
model_path = 'experiments/pretrained_models/RealESRGAN_x4plus.pth' 
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# আপস্কেলার তৈরি করুন
# half=True দিলে GPU ব্যবহার করে দ্রুত কাজ করবে, না থাকলে False দিন
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False) # আপনার GPU না থাকলে False ব্যবহার করুন

# একটি API এন্ডপয়েন্ট তৈরি করুন
@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    """
    একটি ছবি আপলোড নিয়ে সেটিকে 4K রেজোলিউশনে উন্নত করে।
    """
    # আপলোড করা ফাইল থেকে ইমেজ ডেটা পড়ুন
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Real-ESRGAN মডেল ব্যবহার করে ছবিটি আপস্কেল করুন
    try:
        output, _ = upsampler.enhance(img, outscale=4)
    except RuntimeError as error:
        print('Error', error)
        print('Note: If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        # GPU মেমরি সমস্যা হলে টাইল অপশন ব্যবহার করা যেতে পারে
        return {"error": "Failed to process image due to memory issues."}

    # ফলাফল ছবিটি PNG ফরম্যাটে এনকোড করুন
    _, img_encoded = cv2.imencode(".png", output)
    
    # ছবিটি স্ট্রিম হিসেবে ফেরত দিন
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

# রুট এন্ডপয়েন্ট
@app.get("/")
def read_root():
    return {"message": "Welcome to the 4K Image Upscaling API!"}

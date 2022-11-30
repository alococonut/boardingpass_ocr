import uuid
import cv2
import numpy as np
import os
import zxingcpp
import fitz
from fastapi import APIRouter, Form, UploadFile, File
from paddleocr import PaddleOCR
from starlette.responses import JSONResponse
from routers.ocr_tools import boardingpass_ocr,Barcode_detect, receipt_ocr
from commons import utils
from commons.constants import STATUS_SUCCESS, STATUS_UNKNOWN_ERROR
from commons.logger import logger as logging


router = APIRouter(
    prefix="/ocr",
    tags=["income_ocr"]
)


path = os.getcwd()
model_path = os.path.join(path, 'etc/paddle_model/')

ocr_model = PaddleOCR(det_model_dir=model_path + 'en_PP-OCRv3_det_infer',
                rec_model_dir=model_path + 'en_PP-OCRv3_rec_infer',
                cls_model_dir=model_path + 'ch_ppocr_mobile_v2.0_cls_infer',
                lang='en')
# ocr_model = PaddleOCR(lang="en")


@router.get("/")
async def root():
    return "INCOME OCR"


@router.post("/boardingpass")
def my_mykad_front(client_no: str = Form(alias='clientRefNum'),
                   request_id: str = Form(alias='requestId', default=uuid.uuid4().hex),
                   image_front: UploadFile = File(),
                   resolution_check: bool = Form(default=None),
                   alignment: bool = Form(default=None),
                   landmarks: bool = Form(default=None),
                   security_checks: bool = Form(default=None),
                   debug: bool = Form(default=False)):
    logging.info(f'client_no={client_no}, request_id={request_id}, debug={debug}')
    if resolution_check is None:
        resolution_check = True
    if alignment is None:
        alignment = True
    if landmarks is None:
        landmarks = True
    if security_checks is None:
        security_checks = True
    content_front = image_front.file.read()
    img_front = cv2.imdecode(np.fromstring(content_front, np.uint8), cv2.IMREAD_UNCHANGED)



    return JSONResponse(content={'request_id':2018,'content':'for test'})

    # if STATUS_SUCCESS.code == status_front.code:
    #     return JSONResponse(content=utils.convert_result(output_dict_front, debug))
    # else:
    #     return JSONResponse(content=dict(status_front), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@router.post("/boardingpass2")
def my_mykad_front(image_front: UploadFile = File()):
    content_front = image_front.file.read()
    img_front = cv2.imdecode(np.fromstring(content_front, np.uint8), cv2.IMREAD_UNCHANGED)
    ocr_1 = boardingpass_ocr.ocr(input=img_front,model=ocr_model)
    ocr_result = ocr_1.processs()

    barcode_results = zxingcpp.read_barcodes(img_front)
    if len(barcode_results) == 0:
        cropped_image = Barcode_detect.crop_barcode(img_front)
        barcode_results = zxingcpp.read_barcodes(cropped_image)
    if barcode_results:
        barcode_results = Barcode_detect.extract_info(barcode_results)
    else:
        barcode_results = None

    output = {'ocr':ocr_result,'barcode':barcode_results}

    return JSONResponse(content=output)


@router.post("/receipt")
def car_receipt(image:UploadFile = File(),
                filetype:str = Form(alias='filetype')):
    content = image.file.read()
    name = image.filename
    recep = receipt_ocr.receipt(input=content,model=ocr_model,name=name)
    output = recep.chose_file()


    return JSONResponse(content = {'text':output})






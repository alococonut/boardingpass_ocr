# Status class for holding status code and description
class Status:
    def __init__(self, code, description):
        self.code = code
        self.description = description

    def keys(self):
        return 'code', 'description'

    def __getitem__(self, item):
        return getattr(self, item)


# Supported status codes
STATUS_SUCCESS = Status(200, "Processing successful")
STATUS_UNKNOWN_ERROR = Status(500, "Unknown Error")
ERROR_MIN_RESOLUTION = Status(-2002, "Minimum image resolution requirements not met")
ERROR_INVALID_IMAGE = Status(10001, "Invalid base64 image")
ERROR_UNSUPPORTED_CARD = Status(10002, "Unsupported card type")
ERROR_FIELD_SEGMENTATION = Status(10003, "Field segmentation error")
ERROR_CARD_TYPE = Status(10004, "Error determining card type")
ERROR_ALIGNMENT = Status(10005, "Alignment error")

# Min required dimensions of card
MIN_WIDTH = 1280
MIN_HEIGHT = 807

# Tesseract configurations
TESSERACT_CONFIG_DEFAULT = '--dpi 72 -c OMP_THREAD_LIMIT=1 --tessdata-dir "etc/traineddata/"'
TESSERACT_CONFIG_BLOCK = '--dpi 72 --psm 6 -c OMP_THREAD_LIMIT=1 --tessdata-dir "etc/traineddata/"'
TESSERACT_CONFIG_LINE = '--dpi 72 --psm 7 -c OMP_THREAD_LIMIT=1 --tessdata-dir "etc/traineddata/"'
